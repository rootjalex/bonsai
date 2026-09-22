#include "SSA/PromoteAllocas.h"

#include "SSA/Analysis.h"

#include "IR/Mutator.h"

#include "Utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// Does `v` refer to the value named `name`? A definition and the block
// arguments that thread it onwards share a name, so this is the only way to
// follow one across blocks.
bool refers_to(const Value &v, const string &name) {
    return std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &i) { return i->name == name; },
            [&](const Constant &) { return false; },
            [&](const Argument &a) { return a.name == name; },
        },
        v.data);
}

// Every jump out of `block`, paired with the index its first argument binds
// to in the target's argument list (1 for a call continuation, which is
// handed the returned value first).
vector<std::pair<Terminator::Jump *, size_t>> jumps(Block &block) {
    vector<std::pair<Terminator::Jump *, size_t>> result;
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) { result.push_back({&j, 0}); },
                   [&](Terminator::Dispatch &d) {
                       for (auto &target : d.targets) {
                           result.push_back({&target, 0});
                       }
                   },
                   [&](Terminator::Return &) {},
                   [&](Terminator::ParFor &p) {
                       result.push_back({&p.body, 1});
                       result.push_back({&p.cont, 0});
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       result.push_back({&c.call, 0});
                       result.push_back({&c.cont, c.drop ? size_t(0) : 1});
                   },
                   [&](Terminator::MultiCall &c) {
                       result.push_back({&c.call, 0});
                       result.push_back({&c.cont, c.drop ? size_t(0) : 1});
                   },
               },
               block.terminator.data);
    return result;
}

// Values a terminator uses without passing them on as an argument. Using an
// allocation this way is an escape: it is the pointer itself that is being
// branched on, returned, or handed to a callee.
vector<shared_ptr<Value>> terminator_uses(const Block &block) {
    vector<shared_ptr<Value>> uses;
    std::visit(
        overloads{
            [&](const std::monostate &) {},
            [&](const Terminator::Jump &) {},
            [&](const Terminator::Dispatch &d) { uses.push_back(d.cond); },
            [&](const Terminator::Return &r) {
                if (r.value) {
                    uses.push_back(r.value);
                }
            },
            [&](const Terminator::ParFor &p) {
                uses.push_back(p.start);
                uses.push_back(p.end);
                uses.push_back(p.stride);
            },
            [&](const Terminator::Yield &) {},
            [&](const Terminator::Call &c) {
                for (const auto &a : c.call.args) {
                    uses.push_back(a);
                }
            },
            [&](const Terminator::MultiCall &c) {
                for (const auto &a : c.call.args) {
                    uses.push_back(a);
                }
                // Every varying value is handed to the callee too, in one of
                // the calls, so an allocation appearing among them escapes
                // just as much as one in the shared arguments.
                for (const auto &vs : c.varying) {
                    for (const auto &a : vs) {
                        uses.push_back(a);
                    }
                }
                for (const auto &k : c.keys) {
                    uses.push_back(k);
                }
            },
        },
        block.terminator.data);
    return uses;
}

// The read-modify-write accumulate ops are stores too: `acc.mul p v` is
// `store p (mul (load p) v)`, so a promoted one becomes just that binary op
// on the value reaching it. Argmin/argmax are left out: they combine an index
// with a value rather than two values of the allocated type.
std::optional<Instruction::Op> accumulate_binop(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::AccAdd:
        return Instruction::Op::Add;
    case Instruction::Op::AccMul:
        return Instruction::Op::Mul;
    case Instruction::Op::AccSub:
        return Instruction::Op::Sub;
    case Instruction::Op::AccMin:
        return Instruction::Op::Min;
    case Instruction::Op::AccMax:
        return Instruction::Op::Max;
    default:
        return std::nullopt;
    }
}

// A value's name may be in a *type*: an array whose length is not a constant
// is `f32[n]` for the value `n` -- a queue's storage sized by the sample
// count, say (SSA/Defer.cpp) -- and the code generator finds `n` by that
// name when it allocates or indexes the array. So a load this pass deletes,
// whose uses it points at the value reaching it, has to be renamed inside
// the types too, or the types keep naming a value that no longer exists.
// This rewrites the sizes: the base mutator leaves them alone.
struct RenameInTypes : ir::Mutator {
    const map<string, ir::Expr> &renames;
    explicit RenameInTypes(const map<string, ir::Expr> &renames)
        : renames(renames) {}

    ir::Expr visit(const ir::Var *node) override {
        const auto it = renames.find(node->name);
        return it == renames.end() ? ir::Expr(node) : it->second;
    }
    Type visit(const ir::Array_t *node) override {
        Type etype = mutate(node->etype);
        ir::Expr size = node->size.defined() ? mutate(node->size) : node->size;
        if (etype.same_as(node->etype) && size.same_as(node->size)) {
            return node;
        }
        return ir::Array_t::make(std::move(etype), std::move(size));
    }
    Type visit(const ir::DynArray_t *node) override {
        Type etype = mutate(node->etype);
        ir::Expr capacity =
            node->capacity.defined() ? mutate(node->capacity) : node->capacity;
        if (etype.same_as(node->etype) && capacity.same_as(node->capacity)) {
            return node;
        }
        return ir::DynArray_t::make(std::move(etype), std::move(capacity));
    }
};

// The expression that stands for `v` in a type: the value by name, or the
// constant itself.
ir::Expr expr_of(const Value &v) {
    return std::visit(
        overloads{
            [](const Argument &a) { return ir::Var::make(a.type, a.name); },
            [](const shared_ptr<Instruction> &i) {
                return ir::Var::make(i->type, i->name);
            },
            [](const Constant &c) {
                return std::visit(
                    overloads{
                        [](const bool &b) { return ir::BoolImm::make(b); },
                        [&](const int64_t &i) {
                            return ir::IntImm::make(c.type, i);
                        },
                        [&](const uint64_t &u) {
                            return ir::UIntImm::make(c.type, u);
                        },
                        [&](const double &d) {
                            return ir::FloatImm::make(c.type, d);
                        },
                        [&](const std::string &s) {
                            return ir::StringImm::make(s);
                        },
                        [&](const Undefined &) { return ir::Undef::make(c.type); },
                    },
                    c.data);
            }},
        v.data);
}

// Renames the deleted loads in every type the function holds: the types of
// its instructions and of what they ask the size of, its blocks' parameters,
// and the copies of a parameter that its values carry.
void rename_in_types(Function &func, const map<string, ir::Expr> &renames) {
    RenameInTypes renamer(renames);
    for (const auto &block : func.blocks) {
        for (Argument &arg : block->args) {
            arg.type = renamer.mutate(arg.type);
        }
        for (const auto &instr : block->instrs) {
            instr->type = renamer.mutate(instr->type);
            if (instr->queried_type.defined()) {
                instr->queried_type = renamer.mutate(instr->queried_type);
            }
        }
    }
    for_each_value(func, [&](shared_ptr<Value> &v) {
        if (auto *a = std::get_if<Argument>(&v->data)) {
            a->type = renamer.mutate(a->type);
        }
    });
}

// A promotable stack allocation.
struct Candidate {
    shared_ptr<Instruction> instr;
    string name;
    Type type; // the allocated type, i.e. the pointee
    string block;
};

// The allocations in `region` that can be promoted, in the order they are
// allocated. An allocation is rejected as soon as it is used as anything but
// the address of a Load or a Store, and also if its name is not unique, since
// names are what ties a definition to the block arguments threading it.
vector<Candidate> find_candidates(Function &func, const Cfg &region) {
    vector<Candidate> candidates;
    set<string> seen;
    set<string> rejected;

    for (const auto &block : func.blocks) {
        if (!region.contains(*block)) {
            continue;
        }
        for (const auto &instr : block->instrs) {
            // Both kinds of allocation: which memory a `mut` local was given
            // is a size heuristic (see Allocate::make), and either way it is
            // a slot the function owns, holding one value, that a register
            // can hold instead.
            if (instr->op != Instruction::Op::Alloca &&
                instr->op != Instruction::Op::Alloc) {
                continue;
            }
            // An array's name is bound to its elements rather than to a slot
            // holding a handle (see Type::is_reference), so it is registered
            // under the array type. There is nothing to promote either way:
            // a register cannot hold an array.
            if (instr->type.is_reference()) {
                continue;
            }
            const Ptr_t *ptr = instr->type.as<Ptr_t>();
            internal_assert(ptr) << "Alloca " << instr->name
                                 << " is not pointer-typed: " << instr->type;
            if (!seen.insert(instr->name).second) {
                // Two allocations of the same name cannot be told apart by
                // the block arguments that thread them.
                rejected.insert(instr->name);
                continue;
            }
            candidates.push_back({instr, instr->name, ptr->etype, block->name});
        }
    }

    auto reject_if_named = [&](const Value &v) {
        for (const Candidate &c : candidates) {
            if (refers_to(v, c.name)) {
                rejected.insert(c.name);
            }
        }
    };

    for (const auto &block : func.blocks) {
        const bool inside = region.contains(*block);

        for (const auto &instr : block->instrs) {
            for (size_t k = 0; k < instr->operands.size(); k++) {
                // Anything but "this is the address a Load or Store works on"
                // is an escape: the pointer's value is observed, so it cannot
                // stop existing.
                const bool addressed =
                    inside && k == 0 &&
                    (instr->op == Instruction::Op::Load ||
                     instr->op == Instruction::Op::Store ||
                     accumulate_binop(instr->op).has_value());
                if (!addressed) {
                    reject_if_named(*instr->operands[k]);
                }
            }
        }

        for (const auto &use : terminator_uses(*block)) {
            reject_if_named(*use);
        }

        // Passing the pointer on as a block argument is just threading, and
        // is unwound by the promotion -- but only within the region, and
        // into a ParFor body only for reading: the body ends at a Yield, so
        // a value stored there reaches no use the rename walk can see, and
        // promoting an allocation the body writes would silently drop the
        // store. One the body only reads is a value that reaches the loop
        // and is the same on every iteration -- a `mut` local settled before
        // the loop, a sample count chosen by a match on the sampler -- and
        // promoting it is what lets the body be handed the value rather than
        // the address of a slot on the function's stack, which a body that
        // runs on another machine cannot read.
        if (inside) {
            if (const auto *p =
                    std::get_if<Terminator::ParFor>(&block->terminator.data)) {
                const Cfg body(func, p->body.name);
                for (const auto &arg : p->body.args) {
                    for (const Candidate &c : candidates) {
                        if (!refers_to(*arg, c.name)) {
                            continue;
                        }
                        for (const auto &inner : body.blocks()) {
                            for (const auto &instr : inner->instrs) {
                                const bool writes =
                                    (instr->op == Instruction::Op::Store ||
                                     accumulate_binop(instr->op).has_value()) &&
                                    !instr->operands.empty() &&
                                    refers_to(*instr->operands[0], c.name);
                                if (writes) {
                                    rejected.insert(c.name);
                                }
                            }
                        }
                    }
                }
            }
            // Threading keeps the name (Block::get_value): the argument that
            // carries the pointer on is called what the allocation is, and
            // that name is how its loads and stores in other blocks are
            // found. A jump that hands the pointer to an argument called
            // something else is not threading this can unwind -- the loads
            // through that argument would be missed, and the allocation
            // deleted from under them -- so the allocation stays in memory.
            for (auto &[jump, first_arg] : jumps(*block)) {
                const BlockId to = region.find(jump->name);
                if (to == NO_BLOCK) {
                    continue; // a call's jump to its callee
                }
                const Block &target = region[to];
                for (size_t k = 0; k < jump->args.size(); k++) {
                    const size_t j = k + first_arg;
                    internal_assert(j < target.args.size())
                        << "Jump from " << block->name << " to " << target.name
                        << " passes more arguments than the block takes";
                    for (const Candidate &c : candidates) {
                        if (refers_to(*jump->args[k], c.name) &&
                            target.args[j].name != c.name) {
                            rejected.insert(c.name);
                        }
                    }
                }
            }
            continue;
        }
        for (auto &[jump, _] : jumps(*block)) {
            for (const auto &arg : jump->args) {
                reject_if_named(*arg);
            }
        }
    }

    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const Candidate &c) {
                                        return rejected.count(c.name) > 0;
                                    }),
                     candidates.end());
    return candidates;
}

// Deletes the block arguments that thread `name`, and the matching operands
// in every jump that supplies them. Called before renaming, so that what is
// left of `name` is only its allocation, loads and stores.
void erase_threading(Function &func, const Cfg &region, const string &name) {
    for (const auto &block : func.blocks) {
        if (!region.contains(*block)) {
            continue;
        }
        for (size_t j = block->args.size(); j-- > 0;) {
            if (block->args[j].name != name) {
                continue;
            }
            block->args.erase(block->args.begin() + j);

            for (const auto &pred : func.blocks) {
                for (auto &[jump, first_arg] : jumps(*pred)) {
                    if (jump->name != block->name || j < first_arg) {
                        continue;
                    }
                    const size_t k = j - first_arg;
                    internal_assert(k < jump->args.size())
                        << "Jump from " << pred->name << " to " << block->name
                        << " is missing an argument for " << name;
                    jump->args.erase(jump->args.begin() + k);
                }
            }
        }
        block->lookups.erase(name);
    }
}

} // namespace

size_t promote_allocas(Function &func, const string &entry) {
    const Cfg region(func, entry);
    const DomTree dom = compute_dominator_tree(region);
    const DominanceFrontier frontier = compute_dominance_frontier(region, dom);

    const vector<Candidate> candidates = find_candidates(func, region);
    if (candidates.empty()) {
        return 0;
    }

    size_t promoted = 0;
    for (const Candidate &c : candidates) {
        const BlockId allocated_in = region.id(c.block);
        // Where the value is (re)defined, and hence where the joins that need
        // a block argument for it are.
        BlockSet defs(region.size());
        defs.insert(allocated_in);
        BlockSet stores_in(region.size());
        for (BlockId b = 0; b < region.size(); b++) {
            for (const auto &instr : region[b].instrs) {
                if ((instr->op == Instruction::Op::Store ||
                     accumulate_binop(instr->op).has_value()) &&
                    refers_to(*instr->operands[0], c.name)) {
                    defs.insert(b);
                    stores_in.insert(b);
                }
            }
        }

        // A local that is read, or merged at a join, on a path that never
        // assigned it has no value to hand the read or the join: the program
        // is reading uninitialized memory there, and this form has nothing to
        // stand for that. Such a local stays in memory. Whether every path
        // has assigned it is a must-reach problem -- assigned on entry to a
        // block if assigned on exit from all of its predecessors.
        //
        // The usual case is a slot declared at the top of a function and
        // first written inside a loop: the loop's header merges the slot from
        // the entry, where nothing has written it, with the latch, where
        // something has.
        const vector<bool> assigned_in = solve_dataflow<bool>(
            region, Direction::Forward, /*init=*/true, /*boundary=*/false,
            [&](BlockId b, const bool &in) {
                return in || stores_in.contains(b);
            },
            [](const bool &a, const bool &b) { return a && b; });
        const auto assigned_out = [&](BlockId b) {
            return assigned_in[b] || stores_in.contains(b);
        };
        bool always_assigned = true;
        for (BlockId b = 0; b < region.size(); b++) {
            const Block &block = region[b];
            bool assigned = assigned_in[b];
            for (const auto &instr : block.instrs) {
                const bool addresses = !instr->operands.empty() &&
                                       refers_to(*instr->operands[0], c.name);
                if (!addresses) {
                    continue;
                }
                if (instr->op == Instruction::Op::Store) {
                    assigned = true;
                } else if (!assigned) {
                    always_assigned = false; // a load, or an accumulate
                }
            }
        }
        const BlockSet joins = iterated_dominance_frontier(defs, frontier);
        for (BlockId join : joins) {
            if (!dom.dominates(allocated_in, join)) {
                continue;
            }
            for (BlockId pred : region.preds[join]) {
                if (!assigned_out(pred)) {
                    always_assigned = false;
                }
            }
        }
        if (!always_assigned) {
            continue;
        }
        promoted++;
        // Only the joins the allocation's own block dominates. The value
        // exists nowhere else: a local declared in one arm of an `if` is
        // stored to in that arm and read in that arm, and the join below the
        // `if` is in the frontier of those stores all the same -- but a path
        // reaching it through the other arm never made the allocation, so
        // there is no value to hand the join and nothing past it that could
        // read one.
        BlockSet phis(region.size());
        for (BlockId join : joins) {
            if (dom.dominates(allocated_in, join)) {
                phis.insert(join);
            }
        }

        erase_threading(func, region, c.name);

        // Loads become references to whatever value reaches them.
        map<const Instruction *, shared_ptr<Value>> replacements;

        vector<shared_ptr<Value>> reaching;
        std::function<void(BlockId)> rename = [&](BlockId b) {
            Block &block = region[b];
            const string &name = block.name;
            const size_t depth = reaching.size();

            if (phis.contains(b)) {
                const Argument arg = {c.type, c.name};
                block.args.push_back(arg);
                auto value = std::make_shared<Value>(arg);
                block.lookups[c.name] = value;
                reaching.push_back(std::move(value));
            }

            vector<shared_ptr<Instruction>> kept;
            for (const auto &instr : block.instrs) {
                if (instr.get() == c.instr.get()) {
                    continue; // the allocation itself
                }
                const bool addresses = !instr->operands.empty() &&
                                       refers_to(*instr->operands[0], c.name);

                if (addresses && instr->op == Instruction::Op::Store) {
                    block.lookups[c.name] = instr->operands[1];
                    reaching.push_back(instr->operands[1]);
                    continue;
                }
                if (addresses && accumulate_binop(instr->op).has_value()) {
                    internal_assert(!reaching.empty())
                        << "Accumulate into " << c.name << " in " << name
                        << " before it is ever stored to";
                    auto combined = std::make_shared<Instruction>(
                        func.get_unique_name(), c.type,
                        *accumulate_binop(instr->op),
                        std::vector<shared_ptr<Value>>{reaching.back(),
                                                       instr->operands[1]},
                        region.block(b));
                    auto value = std::make_shared<Value>(combined);
                    kept.push_back(std::move(combined));
                    block.lookups[c.name] = value;
                    reaching.push_back(std::move(value));
                    continue;
                }
                if (addresses && instr->op == Instruction::Op::Load) {
                    internal_assert(!reaching.empty())
                        << "Load of " << c.name << " in " << name
                        << " before it is ever stored to";
                    replacements[instr.get()] = reaching.back();
                    block.lookups[instr->name] = reaching.back();
                    continue;
                }
                kept.push_back(instr);
            }
            block.instrs = std::move(kept);

            // Supply the value to the joins this block flows into.
            for (auto &[jump, _] : jumps(block)) {
                // A jump to a block outside the region -- a call's jump to
                // its callee -- supplies nothing.
                if (!phis.contains(region.find(jump->name))) {
                    continue;
                }
                internal_assert(!reaching.empty())
                    << "No definition of " << c.name << " reaches the jump "
                    << "from " << name << " to " << jump->name;
                jump->args.push_back(reaching.back());
            }

            for (BlockId child : dom.children(b)) {
                rename(child);
            }
            reaching.resize(depth);
        };
        rename(region.entry);

        // Substitute the loads away. Only the block that held the load can
        // name it directly: references from other blocks go through a block
        // argument, which is fed by the jump argument replaced here.
        for (BlockId b = 0; b < region.size(); b++) {
            Block &block = region[b];
            auto substitute = [&](shared_ptr<Value> &v) {
                const auto *instr =
                    std::get_if<shared_ptr<Instruction>>(&v->data);
                if (instr == nullptr) {
                    return;
                }
                const auto it = replacements.find(instr->get());
                if (it != replacements.end()) {
                    v = it->second;
                }
            };

            for (const auto &instr : block.instrs) {
                for (auto &operand : instr->operands) {
                    substitute(operand);
                }
            }
            // The by-name index too: a `let x = *p` renames the load, and
            // any block that then names `x` holds the load's value in its
            // index. clone_function copies the index, and would find a
            // deleted instruction there.
            for (auto &[_, value] : block.lookups) {
                substitute(value);
            }
            for (auto &[jump, _] : jumps(block)) {
                for (auto &arg : jump->args) {
                    substitute(arg);
                }
            }
            if (auto *d =
                    std::get_if<Terminator::Dispatch>(&block.terminator.data)) {
                substitute(d->cond);
            }
            if (auto *r =
                    std::get_if<Terminator::Return>(&block.terminator.data)) {
                if (r->value) {
                    substitute(r->value);
                }
            }
            // A loop's bounds and a multi-call's per-lane values and keys
            // read values too, without being anyone's jump argument. A
            // parfor over `0:n` where `n` was loaded from a promoted local
            // -- the sample count a match on the sampler settled -- kept
            // naming the deleted load here, which nothing then defined.
            if (auto *p =
                    std::get_if<Terminator::ParFor>(&block.terminator.data)) {
                substitute(p->start);
                substitute(p->end);
                substitute(p->stride);
            }
            if (auto *m =
                    std::get_if<Terminator::MultiCall>(&block.terminator.data)) {
                for (auto &vs : m->varying) {
                    for (auto &v : vs) {
                        substitute(v);
                    }
                }
                for (auto &k : m->keys) {
                    substitute(k);
                }
            }
        }

        // And the loads' names inside the types (see RenameInTypes). Per
        // candidate rather than once at the end: a load of this candidate may
        // be what reaches a load of the next, and the next candidate's pass
        // then renames what this one wrote.
        map<string, ir::Expr> renames;
        for (const auto &[load, reaching] : replacements) {
            renames.emplace(load->name, expr_of(*reaching));
        }
        if (!renames.empty()) {
            rename_in_types(func, renames);
        }
    }

    return promoted;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
