#include "SSA/PromoteAllocas.h"

#include "SSA/Analysis.h"

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
vector<Candidate> find_candidates(Function &func, const set<string> &region) {
    vector<Candidate> candidates;
    set<string> seen;
    set<string> rejected;

    for (const auto &block : func.blocks) {
        if (!region.count(block->name)) {
            continue;
        }
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Alloca) {
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
        const bool inside = region.count(block->name) > 0;

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
        // never into a ParFor body: the body ends at a Yield, so a value
        // stored there reaches no use the rename walk can see, and promoting
        // the allocation would silently drop the store.
        if (inside) {
            if (const auto *p =
                    std::get_if<Terminator::ParFor>(&block->terminator.data)) {
                for (const auto &arg : p->body.args) {
                    reject_if_named(*arg);
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
void erase_threading(Function &func, const set<string> &region,
                     const string &name) {
    for (const auto &block : func.blocks) {
        if (!region.count(block->name)) {
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
    const BlockMap blocks = make_block_map(func);
    const AdjacencyMap all_succs = compute_successors(func);
    const set<string> region = reachable_from(entry, all_succs);

    AdjacencyMap succs;
    for (const string &name : region) {
        succs[name];
        for (const string &s : all_succs.at(name)) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }
    const AdjacencyMap preds = compute_predecessors(succs);
    const vector<string> rpo = reverse_postorder(entry, succs);
    const DomTree dom = compute_dominator_tree(entry, succs, preds, rpo);
    const AdjacencyMap dom_children = dom.children();
    const AdjacencyMap frontier = compute_dominance_frontier(preds, dom);

    const vector<Candidate> candidates = find_candidates(func, region);
    if (candidates.empty()) {
        return 0;
    }

    for (const Candidate &c : candidates) {
        // Where the value is (re)defined, and hence where the joins that need
        // a block argument for it are.
        set<string> defs = {c.block};
        for (const string &name : region) {
            for (const auto &instr : blocks.at(name)->instrs) {
                if (instr->op == Instruction::Op::Store &&
                    refers_to(*instr->operands[0], c.name)) {
                    defs.insert(name);
                }
            }
        }
        const set<string> phis = iterated_dominance_frontier(defs, frontier);

        erase_threading(func, region, c.name);

        // Loads become references to whatever value reaches them.
        map<const Instruction *, shared_ptr<Value>> replacements;

        vector<shared_ptr<Value>> reaching;
        std::function<void(const string &)> rename = [&](const string &name) {
            Block &block = *blocks.at(name);
            const size_t depth = reaching.size();

            if (phis.count(name)) {
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
                        blocks.at(name));
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
                if (!phis.count(jump->name)) {
                    continue;
                }
                internal_assert(!reaching.empty())
                    << "No definition of " << c.name << " reaches the jump "
                    << "from " << name << " to " << jump->name;
                jump->args.push_back(reaching.back());
            }

            for (const string &child : dom_children.at(name)) {
                rename(child);
            }
            reaching.resize(depth);
        };
        rename(entry);

        // Substitute the loads away. Only the block that held the load can
        // name it directly: references from other blocks go through a block
        // argument, which is fed by the jump argument replaced here.
        for (const string &name : region) {
            Block &block = *blocks.at(name);
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
        }
    }

    return candidates.size();
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
