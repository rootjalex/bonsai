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

// Applies `f` to every value referred to from a block other than `except`.
template <typename F>
void for_each_value_outside(Function &func, const Block *except, F &&f) {
    for (const shared_ptr<Block> &block : func.blocks) {
        if (block.get() == except) {
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

    size_t guarded = 0;
    for (const string &x_name : order) {
        const auto mask_it = masks.find(x_name);
        if (mask_it == masks.end()) {
            continue;
        }
        const shared_ptr<Value> mask = mask_it->second;
        if (entry_mask != nullptr && same_value(*mask, *entry_mask)) {
            continue;
        }
        const shared_ptr<Block> x = blocks.at(x_name);

        // The block's one way out, and where it leads: the block after it in
        // the chain. A block ending in a call is left alone: its callee is
        // specialized to the mask and tests it on entry (see the `$masked`
        // call guard in CodeGen_LLVM_SSA.cpp), and the relooper takes a call
        // continuation to have the call as its only predecessor.
        auto *out = std::get_if<Terminator::Jump>(&x->terminator.data);
        if (out == nullptr) {
            continue;
        }
        const string y_name = out->name;
        if (!region.count(y_name)) {
            continue;
        }

        // Not the entry, and not a loop header: a guard in front of a header
        // would sit on the back edge, and a header's mask is its loop's
        // business (see SSA/UniformizeLoops.h).
        const vector<string> x_preds = preds[x_name];
        if (x_preds.empty()) {
            continue;
        }
        bool header = false;
        for (const string &p : x_preds) {
            header = header || !index.count(p) || index.at(p) >= index.at(x_name);
        }
        if (header) {
            continue;
        }
        // The successor is reached from this block alone, so that the bypass
        // is the only other way in and its arguments are this block's to
        // account for.
        const vector<string> &y_preds = preds[y_name];
        if (y_preds.size() != 1 || y_preds[0] != x_name) {
            continue;
        }
        // Its own values travelling on as arguments already are the slots of
        // a join with several live paths, whose fallback is the previous
        // value and not zero (see SkipInactiveBlocks.h).
        bool passes_own = false;
        for (const auto &a : out->args) {
            const Instruction *i = instruction_of(*a);
            passes_own = passes_own || (i != nullptr && i->owner.lock() == x);
        }
        if (passes_own) {
            continue;
        }

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

        // The block's values a later block uses, or a later block's mask is:
        // its guard will test it, and a masked call in it will be handed it.
        vector<shared_ptr<Instruction>> live;
        for (const shared_ptr<Instruction> &instr : x->instrs) {
            if (instr->name.empty() || hoisted.count(instr.get())) {
                continue;
            }
            bool used = false;
            for_each_value_outside(func, x.get(), [&](shared_ptr<Value> &v) {
                used = used || instruction_of(*v) == instr.get();
            });
            for (const auto &[name, m] : masks) {
                used = used ||
                       (name != x_name && instruction_of(*m) == instr.get());
            }
            if (used) {
                live.push_back(instr);
            }
        }

        // A pointer among them is an address, not a value, and goes ahead of
        // the test too rather than through an argument: an argument standing
        // for it would be null along the bypass, and a `mut` argument of a
        // call is spelled through such a pointer as a place (see Call::make),
        // which a null is not. Address arithmetic is pure, so computing it
        // whether or not a lane is on costs nothing; it can be, when what it
        // is computed from is defined before the block, or hoisted with it.
        // A block computing a pointer that cannot be is left alone.
        vector<shared_ptr<Instruction>> addresses;
        bool stuck = false;
        for (const shared_ptr<Instruction> &instr : x->instrs) {
            if (hoisted.count(instr.get()) ||
                std::find(live.begin(), live.end(), instr) == live.end() ||
                !instr->type.is<Ptr_t>()) {
                continue;
            }
            bool movable = is_address_arithmetic(instr->op);
            for (const auto &operand : instr->operands) {
                const Instruction *i = instruction_of(*operand);
                movable = movable && (i == nullptr || i->owner.lock() != x ||
                                      hoisted.count(i) > 0);
            }
            if (!movable) {
                stuck = true;
                break;
            }
            addresses.push_back(instr);
            hoisted.insert(instr.get());
        }
        if (stuck) {
            continue;
        }
        live.erase(std::remove_if(live.begin(), live.end(),
                                  [&](const shared_ptr<Instruction> &i) {
                                      return hoisted.count(i.get()) > 0;
                                  }),
                   live.end());

        // Worth a test, judged on what is left in the block: memory it must
        // not touch, or enough arithmetic to outweigh the test.
        {
            bool worth = false;
            size_t work = 0;
            for (const shared_ptr<Instruction> &instr : x->instrs) {
                if (hoisted.count(instr.get())) {
                    continue;
                }
                worth = worth || touches_memory(instr->op);
                work += is_mask_logic(instr->op) ? 0 : 1;
            }
            if (!worth && work < 3) {
                continue;
            }
        }

        // The guard: the block's arguments, its mask's chain, its addresses,
        // the test.
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
        for (const shared_ptr<Instruction> &instr : x->instrs) {
            if (hoisted.count(instr.get())) {
                instr->owner = guard;
                guard->instrs.push_back(instr);
            }
        }
        x->instrs.erase(std::remove_if(x->instrs.begin(), x->instrs.end(),
                                       [&](const shared_ptr<Instruction> &i) {
                                           return hoisted.count(i.get()) > 0;
                                       }),
                        x->instrs.end());
        auto any = std::make_shared<Instruction>(
            func.get_unique_name(), Bool_t::make(), Instruction::Op::Any,
            vector<shared_ptr<Value>>{mask}, guard);
        guard->instrs.push_back(any);

        // What the bypass hands the successor: what the block would have,
        // with zero for what only the block computes, the values below.
        const shared_ptr<Block> y = blocks.at(y_name);
        Terminator::Jump skip{.name = y_name, .args = out->args};

        // A value of the block's that a later block uses becomes an argument
        // of the successor, and those uses take the argument.
        for (const shared_ptr<Instruction> &instr : live) {
            const shared_ptr<Value> passed =
                y->add_argument(Argument{instr->type, instr->name + "!skip"});
            out->args.push_back(std::make_shared<Value>(instr));
            skip.args.push_back(zero_value(instr->type, func, guard));
            for_each_value_outside(func, x.get(), [&](shared_ptr<Value> &v) {
                if (instruction_of(*v) == instr.get()) {
                    v = passed;
                }
            });
            for (auto &[name, m] : masks) {
                if (instruction_of(*m) == instr.get()) {
                    m = passed;
                }
            }
        }

        // targets[0] is where a false condition goes: past the block.
        guard->terminator.data = Terminator::Dispatch{
            .cond = std::make_shared<Value>(any),
            .targets = {skip, Terminator::Jump{.name = x_name, .args = {}}}};

        const auto at = std::find(func.blocks.begin(), func.blocks.end(), x);
        func.blocks.insert(at, guard);
        blocks[guard->name] = guard;
        preds[guard->name] = x_preds;
        preds[x_name] = {guard->name};
        preds[y_name].push_back(guard->name);
        index[guard->name] = index.at(x_name);
        guarded++;
    }

    refresh_preds(func);
    return guarded;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
