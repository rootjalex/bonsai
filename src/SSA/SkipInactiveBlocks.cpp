#include "SSA/SkipInactiveBlocks.h"

#include "Error.h"
#include "SSA/Analysis.h"
#include "SSA/SSA.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <string>
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

const Instruction *instruction_of(const Value &v) {
    const auto *i = std::get_if<shared_ptr<Instruction>>(&v.data);
    return i == nullptr ? nullptr : i->get();
}

// Whether two values are one: the same instruction, or an argument of the
// same name.
bool same_value(const Value &a, const Value &b) {
    if (const Instruction *i = instruction_of(a)) {
        return i == instruction_of(b);
    }
    const auto *x = std::get_if<Argument>(&a.data);
    const auto *y = std::get_if<Argument>(&b.data);
    return x != nullptr && y != nullptr && x->name == y->name;
}

// Applies `f` to every value a terminator refers to, by reference.
template <typename F>
void for_each_terminator_value(Terminator &terminator, F &&f) {
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) {
                       for (auto &a : j.args) {
                           f(a);
                       }
                   },
                   [&](Terminator::Dispatch &d) {
                       f(d.cond);
                       for (Terminator::Jump &t : d.targets) {
                           for (auto &a : t.args) {
                               f(a);
                           }
                       }
                   },
                   [&](Terminator::Return &r) {
                       if (r.value) {
                           f(r.value);
                       }
                   },
                   [&](Terminator::ParFor &p) {
                       f(p.start);
                       f(p.end);
                       f(p.stride);
                       for (auto &a : p.body.args) {
                           f(a);
                       }
                       for (auto &a : p.cont.args) {
                           f(a);
                       }
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       for (auto &a : c.call.args) {
                           f(a);
                       }
                       for (auto &a : c.cont.args) {
                           f(a);
                       }
                   },
                   [&](Terminator::MultiCall &c) {
                       for (auto &a : c.call.args) {
                           f(a);
                       }
                       for (auto &a : c.cont.args) {
                           f(a);
                       }
                       for (auto &vs : c.varying) {
                           for (auto &a : vs) {
                               f(a);
                           }
                       }
                       for (auto &k : c.keys) {
                           f(k);
                       }
                   },
               },
               terminator.data);
}

// Applies `f` to every value referred to from a block outside `inside`.
template <typename F>
void for_each_value_outside(Function &func, const set<const Block *> &inside,
                            F &&f) {
    for (const shared_ptr<Block> &block : func.blocks) {
        if (inside.count(block.get())) {
            continue;
        }
        for (const shared_ptr<Instruction> &instr : block->instrs) {
            for (auto &operand : instr->operands) {
                f(operand);
            }
        }
        for_each_terminator_value(block->terminator, f);
    }
}

// Is every lane on in `narrow` on in `wide`, as far as the masks' own
// definitions say? The same mask, or an AND with it: the mask of a block
// nested inside an arm is the arm's mask AND its own condition (see
// Linearize.cpp), and that is the relation this reads.
bool implies(const Value &narrow, const Value &wide) {
    if (same_value(narrow, wide)) {
        return true;
    }
    const Instruction *i = instruction_of(narrow);
    if (i == nullptr || i->op != Instruction::Op::LAnd) {
        return false;
    }
    return std::any_of(i->operands.begin(), i->operands.end(),
                       [&](const shared_ptr<Value> &operand) {
                           return implies(*operand, wide);
                       });
}

// The run of blocks one guard skips: the blocks, and the mask no lane is on
// in when the guard bypasses them.
struct Run {
    vector<shared_ptr<Block>> blocks;
    set<const Block *> inside;
    shared_ptr<Value> mask;

    // Whether `v` is one of the run's own values: an instruction of one of
    // its blocks, or an argument of one other than the first, whose
    // arguments move to the guard.
    bool defines(const Value &v) const {
        if (const Instruction *i = instruction_of(v)) {
            return inside.count(i->owner.lock().get()) > 0;
        }
        const auto *a = std::get_if<Argument>(&v.data);
        if (a == nullptr) {
            return false;
        }
        for (size_t k = 1; k < blocks.size(); k++) {
            for (const Argument &arg : blocks[k]->args) {
                if (arg.name == a->name) {
                    return true;
                }
            }
        }
        return false;
    }

    // What `v`, one of the run's values, is when the run is bypassed: for a
    // blend the run made -- `select(mask, value, before)` with a mask no
    // lane can be on when the run's own is empty -- it is `before`, and so
    // on down a chain of them, until a value from outside the run. That is
    // exactly the value a lane not in the arm reads from the blend when the
    // arm does run, so the bypass hands the join the same thing the arm
    // would have: nothing downstream can tell the arm was skipped. Null for
    // anything else, which the caller passes a zero for -- a raw value of
    // the arm's, read only under the arm's mask or through such a blend.
    shared_ptr<Value> when_bypassed(const shared_ptr<Value> &v) const {
        shared_ptr<Value> current = v;
        for (;;) {
            if (!defines(*current)) {
                return current;
            }
            const Instruction *i = instruction_of(*current);
            if (i == nullptr || i->op != Instruction::Op::Select ||
                i->operands.size() != 3 || !implies(*i->operands[0], *mask)) {
                return nullptr;
            }
            current = i->operands[2];
        }
    }
};

// The logic a mask is made of: no work worth a branch to skip, and what a
// block's mask may be computed from ahead of the test.
bool is_mask_logic(Instruction::Op op) {
    return op == Instruction::Op::Select || op == Instruction::Op::LAnd ||
           op == Instruction::Op::LOr || op == Instruction::Op::Not ||
           op == Instruction::Op::Any;
}

// Work a skipped block must not do at all, rather than merely do for nothing:
// a load in it may be through an address only an active lane made valid.
bool touches_memory(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Load:
    case Instruction::Op::Store:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::Alloc:
    case Instruction::Op::Alloca:
    case Instruction::Op::Append:
        return true;
    default:
        return false;
    }
}

// Address arithmetic: pure, and so safe to do ahead of the test whether or not
// any lane is on.
bool is_address_arithmetic(Instruction::Op op) {
    return op == Instruction::Op::GEP || op == Instruction::Op::FieldPtr ||
           op == Instruction::Op::Reinterpret;
}

// The instructions of `block` that `mask` is computed from, in the block's
// order: the chain of ORs a join's mask is (see Linearize.cpp), which has to
// be hoisted ahead of the test of it. Empty when the mask is defined
// elsewhere.
vector<shared_ptr<Instruction>> mask_chain(const Block &block,
                                           const Value &mask) {
    set<const Instruction *> wanted;
    std::function<void(const Value &)> visit = [&](const Value &v) {
        const Instruction *i = instruction_of(v);
        if (i == nullptr || i->owner.lock().get() != &block ||
            !wanted.insert(i).second) {
            return;
        }
        for (const auto &operand : i->operands) {
            visit(*operand);
        }
    };
    visit(mask);
    vector<shared_ptr<Instruction>> chain;
    for (const shared_ptr<Instruction> &instr : block.instrs) {
        if (wanted.count(instr.get())) {
            chain.push_back(instr);
        }
    }
    return chain;
}

} // namespace

size_t skip_inactive_blocks(Function &func, const string &entry,
                            BlockMasks &masks,
                            const shared_ptr<Value> &entry_mask) {
    // For measuring what the skipping is worth on a program: with this set,
    // every gang runs every arm, as it did before this pass existed.
    if (std::getenv("BONSAI_NO_SKIP") != nullptr) {
        return 0;
    }

    BlockMap blocks = make_block_map(func);
    const AdjacencyMap succs = compute_successors(func);
    const set<string> region = reachable_from(entry, succs);
    const vector<string> order = reverse_postorder(entry, succs);
    map<string, size_t> index;
    for (size_t i = 0; i < order.size(); i++) {
        index[order[i]] = i;
    }
    AdjacencyMap preds = compute_predecessors(succs);

    // The guards made here, by name, and the run each one skips. A block
    // whose only ways in are the block before it and guards' bypasses is
    // still one live path: the bypasses hand it what the run would have, so
    // it can take a bypass more.
    set<string> guards;
    map<string, Run> runs;

    auto is_header = [&](const string &name) {
        const auto p = preds.find(name);
        if (p == preds.end() || p->second.empty()) {
            return true; // the entry, which is a header of sorts
        }
        return std::any_of(p->second.begin(), p->second.end(),
                           [&](const string &q) {
                               return !index.count(q) ||
                                      index.at(q) >= index.at(name);
                           });
    };

    size_t guarded = 0;
    for (const string &x_name : order) {
        const auto mask_it = masks.find(x_name);
        if (mask_it == masks.end() || guards.count(x_name)) {
            continue;
        }
        const shared_ptr<Value> mask = mask_it->second;
        if (entry_mask != nullptr && same_value(*mask, *entry_mask)) {
            continue;
        }
        const shared_ptr<Block> x = blocks.at(x_name);

        // The block's one way out. A block ending in a call is left alone:
        // its callee is specialized to the mask and tests it on entry (see
        // the `$masked` call guard in CodeGen_LLVM_SSA.cpp), and the
        // relooper takes a call continuation to have the call as its only
        // predecessor.
        if (!std::holds_alternative<Terminator::Jump>(x->terminator.data)) {
            continue;
        }
        // Not the entry, and not a loop header: a guard in front of a header
        // would sit on the back edge, and a header's mask is its loop's
        // business (see SSA/UniformizeLoops.h).
        const vector<string> x_preds = preds[x_name];
        if (is_header(x_name)) {
            continue;
        }

        // The run of blocks the guard skips: from this block along the
        // chain, for as long as the next block runs under a narrowing of
        // this block's mask -- an arm nested inside this one, which no lane
        // can be on when none is on here -- and is reached only from the
        // block before it (or from earlier guards' bypasses). The run ends at
        // the first block that is not such a narrowing, which is where the
        // bypass lands; it too must have no other way in, or it would be a
        // join with several live paths whose arguments are not this run's to
        // zero. A block ending in anything but a jump -- a uniform branch, a
        // call -- ends the run in front of it.
        vector<shared_ptr<Block>> run{x};
        set<const Block *> inside{x.get()};
        string z_name;
        for (shared_ptr<Block> cur = x;;) {
            auto *out = std::get_if<Terminator::Jump>(&cur->terminator.data);
            if (out == nullptr) {
                break;
            }
            const string &y_name = out->name;
            if (!region.count(y_name)) {
                break;
            }
            const vector<string> &y_preds = preds[y_name];
            const bool clean = std::all_of(
                y_preds.begin(), y_preds.end(), [&](const string &p) {
                    return p == cur->name || guards.count(p);
                });
            if (!clean) {
                break;
            }
            const shared_ptr<Block> y = blocks.at(y_name);
            const auto ym = masks.find(y_name);
            const bool nested = ym != masks.end() &&
                                implies(*ym->second, *mask) &&
                                !is_header(y_name) &&
                                std::holds_alternative<Terminator::Jump>(
                                    y->terminator.data);
            if (!nested) {
                z_name = y_name;
                break;
            }
            run.push_back(y);
            inside.insert(y.get());
            cur = y;
        }
        if (z_name.empty()) {
            continue;
        }
        const shared_ptr<Block> last = run.back();
        auto *out = std::get_if<Terminator::Jump>(&last->terminator.data);
        const shared_ptr<Block> z = blocks.at(z_name);

        // Whether a value is one of the run's: an instruction of one of its
        // blocks, or an argument of one other than the first, whose
        // arguments move to the guard.
        auto defined_inside = [&](const Value &v) {
            if (const Instruction *i = instruction_of(v)) {
                return inside.count(i->owner.lock().get()) > 0;
            }
            const auto *a = std::get_if<Argument>(&v.data);
            if (a == nullptr) {
                return false;
            }
            for (size_t k = 1; k < run.size(); k++) {
                for (const Argument &arg : run[k]->args) {
                    if (arg.name == a->name) {
                        return true;
                    }
                }
            }
            return false;
        };

        // What goes ahead of the test: the chain the block's mask is
        // computed from, which must be logic only.
        const vector<shared_ptr<Instruction>> chain = mask_chain(*x, *mask);
        if (!std::all_of(chain.begin(), chain.end(), [](const auto &i) {
                return is_mask_logic(i->op);
            })) {
            continue;
        }
        set<const Instruction *> hoisted;
        for (const auto &i : chain) {
            hoisted.insert(i.get());
        }

        // The run's values a block after it uses, or a block after it has
        // as its mask: its guard will test it, and a masked call in it will
        // be handed it. An argument of an inner block is such a value too --
        // what an earlier guard's bypass handed it -- and a value the run
        // itself passes on to the block after it.
        vector<shared_ptr<Value>> live;
        auto note_live = [&](const shared_ptr<Value> &v) {
            if (!defined_inside(*v) ||
                std::any_of(live.begin(), live.end(),
                            [&](const shared_ptr<Value> &l) {
                                return same_value(*l, *v);
                            })) {
                return;
            }
            const Instruction *i = instruction_of(*v);
            if (i != nullptr && (i->name.empty() || hoisted.count(i))) {
                return;
            }
            live.push_back(v);
        };
        for_each_value_outside(func, inside, note_live);
        for (const auto &[name, m] : masks) {
            if (!inside.count(blocks.at(name).get())) {
                note_live(m);
            }
        }

        // A pointer among them is an address, not a value, and goes ahead of
        // the test too rather than through an argument: an argument standing
        // for it would be null along the bypass, and a `mut` argument of a
        // call is spelled through such a pointer as a place (see Call::make),
        // which a null is not. Address arithmetic is pure, so computing it
        // whether or not a lane is on costs nothing; it can be, when what it
        // is computed from is defined before the run, or hoisted with it. A
        // run computing a pointer that cannot be is left alone.
        bool stuck = false;
        for (const shared_ptr<Block> &b : run) {
            for (const shared_ptr<Instruction> &instr : b->instrs) {
                if (hoisted.count(instr.get()) || !instr->type.is<Ptr_t>() ||
                    std::none_of(live.begin(), live.end(),
                                 [&](const shared_ptr<Value> &l) {
                                     return instruction_of(*l) == instr.get();
                                 })) {
                    continue;
                }
                bool movable = is_address_arithmetic(instr->op);
                for (const auto &operand : instr->operands) {
                    const Instruction *i = instruction_of(*operand);
                    movable = movable &&
                              (i == nullptr || !inside.count(i->owner.lock().get()) ||
                               hoisted.count(i) > 0);
                    if (i == nullptr && defined_inside(*operand)) {
                        movable = false; // an inner block's argument
                    }
                }
                if (!movable) {
                    stuck = true;
                    break;
                }
                hoisted.insert(instr.get());
            }
            if (stuck) {
                break;
            }
        }
        if (stuck) {
            continue;
        }
        // Pointer arguments of inner blocks have nowhere to be hoisted to.
        if (std::any_of(live.begin(), live.end(),
                        [&](const shared_ptr<Value> &l) {
                            return instruction_of(*l) == nullptr &&
                                   l->get_type().is<Ptr_t>();
                        })) {
            continue;
        }
        live.erase(std::remove_if(live.begin(), live.end(),
                                  [&](const shared_ptr<Value> &l) {
                                      const Instruction *i = instruction_of(*l);
                                      return i != nullptr && hoisted.count(i) > 0;
                                  }),
                   live.end());

        // Worth a test, judged on what is left in the run: memory it must
        // not touch, or enough arithmetic to outweigh the test -- six
        // operations, which is where ispc draws the line between
        // predicating both arms straight through and branching around them
        // (PREDICATE_SAFE_IF_STATEMENT_COST).
        {
            bool worth = false;
            size_t work = 0;
            for (const shared_ptr<Block> &b : run) {
                for (const shared_ptr<Instruction> &instr : b->instrs) {
                    if (hoisted.count(instr.get())) {
                        continue;
                    }
                    worth = worth || touches_memory(instr->op);
                    work += is_mask_logic(instr->op) ? 0 : 1;
                }
            }
            if (!worth && work < 6) {
                continue;
            }
        }

        // The guard: the first block's arguments, its mask's chain, the
        // run's addresses, the test.
        auto guard = std::make_shared<Block>();
        guard->name = x_name + "!any";
        guard->owner = x->owner;
        for (const Argument &arg : x->args) {
            guard->add_argument(arg);
        }
        x->args.clear();
        for (const string &p : x_preds) {
            for (Terminator::Jump *jump : jumps_of(*blocks.at(p))) {
                if (jump->name == x_name) {
                    jump->name = guard->name;
                }
            }
        }
        for (const shared_ptr<Block> &b : run) {
            for (const shared_ptr<Instruction> &instr : b->instrs) {
                if (hoisted.count(instr.get())) {
                    instr->owner = guard;
                    guard->instrs.push_back(instr);
                }
            }
            b->instrs.erase(
                std::remove_if(b->instrs.begin(), b->instrs.end(),
                               [&](const shared_ptr<Instruction> &i) {
                                   return hoisted.count(i.get()) > 0;
                               }),
                b->instrs.end());
        }
        auto any = std::make_shared<Instruction>(
            func.get_unique_name(), Bool_t::make(), Instruction::Op::Any,
            vector<shared_ptr<Value>>{mask}, guard);
        guard->instrs.push_back(any);

        // What the bypass hands the block after the run: what the run would
        // have. A blend of the run's is the value from before it (see
        // Run::when_bypassed); anything else only the run computes is a
        // zero, which nothing reads.
        const Run this_run{run, inside, mask};
        auto bypassed = [&](const Run &r, const shared_ptr<Value> &v,
                            const shared_ptr<Block> &into) {
            if (!r.defines(*v)) {
                return v;
            }
            if (shared_ptr<Value> before = r.when_bypassed(v)) {
                return before;
            }
            return zero_value(v->get_type(), func, into);
        };
        Terminator::Jump skip{.name = z_name, .args = {}};
        for (const shared_ptr<Value> &a : out->args) {
            skip.args.push_back(bypassed(this_run, a, guard));
        }

        // A value of the run's that a later block uses becomes an argument
        // of the block after the run, and those uses take the argument. The
        // bypasses of earlier guards that already land there -- around runs
        // this one is inside -- hand it what their own run would have.
        for (const shared_ptr<Value> &v : live) {
            const Type type = v->get_type();
            const string name =
                instruction_of(*v) ? instruction_of(*v)->name
                                   : std::get<Argument>(v->data).name;
            const shared_ptr<Value> passed =
                z->add_argument(Argument{type, name + "!skip"});
            out->args.push_back(v);
            skip.args.push_back(bypassed(this_run, v, guard));
            for (const string &p : preds[z_name]) {
                if (p == last->name) {
                    continue;
                }
                internal_assert(guards.count(p))
                    << p << " reaches " << z_name << " beside the run";
                const shared_ptr<Block> g = blocks.at(p);
                for (Terminator::Jump *jump : jumps_of(*g)) {
                    if (jump->name == z_name) {
                        jump->args.push_back(bypassed(runs.at(p), v, g));
                    }
                }
            }
            for_each_value_outside(func, inside, [&](shared_ptr<Value> &u) {
                if (same_value(*u, *v)) {
                    u = passed;
                }
            });
            for (auto &[name, m] : masks) {
                if (!inside.count(blocks.at(name).get()) &&
                    same_value(*m, *v)) {
                    m = passed;
                }
            }
        }

        // targets[0] is where a false condition goes: past the run.
        guard->terminator.data = Terminator::Dispatch{
            .cond = std::make_shared<Value>(any),
            .targets = {skip, Terminator::Jump{.name = x_name, .args = {}}}};

        const auto at = std::find(func.blocks.begin(), func.blocks.end(), x);
        func.blocks.insert(at, guard);
        blocks[guard->name] = guard;
        guards.insert(guard->name);
        runs[guard->name] = this_run;
        preds[guard->name] = x_preds;
        preds[x_name] = {guard->name};
        preds[z_name].push_back(guard->name);
        index[guard->name] = index.at(x_name);
        guarded++;
    }

    refresh_preds(func);
    return guarded;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
