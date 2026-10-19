#include "SSA/Definitions.h"

#include "Error.h"
#include "Utils.h"

#include <optional>

namespace bonsai {
namespace ir {
namespace ssa {

using std::optional;
using std::shared_ptr;
using std::string;

bool same_definition(const Definition &a, const Definition &b) {
    return a.value && b.value && same_value(*a.value, *b.value) &&
           a.block == b.block;
}

Definitions::Definitions(const Function &func, bool lenient, bool bound)
    : func(func), lenient(lenient), bound(bound), bmap(make_block_map(func)) {}

Definition Definitions::of(const string &block, const shared_ptr<Value> &v) {
    return std::visit(
        overloads{
            [&](const Constant &) { return Definition{v, ""}; },
            [&](const shared_ptr<Instruction> &i) {
                const auto owner = i->owner.lock();
                internal_assert(owner) << "instruction " << i->name
                                       << " outlived its block";
                return Definition{v, owner->name};
            },
            [&](const Argument &a) { return of_argument(block, a); },
        },
        v->data);
}

const Argument *Definitions::parameter(const string &block,
                                       const shared_ptr<Value> &v) {
    Definition d = of(block, v);
    // A `let` is a name for a value, not a computation (`let rs = @278`,
    // the program's name for a split function's parameter): the parameter
    // is the one it names.
    for (;;) {
        const auto *i = d.value ? std::get_if<std::shared_ptr<Instruction>>(&d.value->data)
                                : nullptr;
        if (i == nullptr || d.block.empty() || (*i)->op != Instruction::Op::Set ||
            (*i)->operands.empty()) {
            break;
        }
        d = of(d.block, (*i)->operands[0]);
    }
    const auto *a = std::get_if<Argument>(&d.value->data);
    if (a == nullptr || d.block != func.blocks.front()->name) {
        return nullptr;
    }
    for (const Argument &param : func.blocks.front()->args) {
        if (param.name == a->name) {
            return &param;
        }
    }
    return nullptr;
}

Definition Definitions::of_argument(const string &block_name,
                                    const Argument &a) {
    const auto key = std::make_pair(block_name, a.name);
    if (const auto it = memo.find(key); it != memo.end()) {
        return it->second;
    }
    if (const auto it = provisional.find(key); it != provisional.end()) {
        return it->second;
    }
    const auto biter = bmap.find(block_name);
    internal_assert(biter != bmap.end())
        << block_name << " is not a block of " << func.blocks[0]->name;
    const shared_ptr<Block> &block = biter->second;
    const auto self = std::make_shared<Value>(a);
    const Definition here{self, block_name};

    // The function's parameters are the entry block's arguments.
    if (block.get() == func.blocks.front().get()) {
        return memo[key] = here;
    }
    size_t k = block->args.size();
    for (size_t i = 0; i < block->args.size(); i++) {
        if (block->args[i].name == a.name) {
            k = i;
            break;
        }
    }
    if (k == block->args.size()) {
        internal_assert(lenient)
            << a.name << " is referred to in " << block_name
            << " but is neither one of its arguments nor a parameter";
        if (!bound) {
            return memo[key] = here; // reached by name from above: unknown here
        }
        // Reached by name from above without being passed, and the reader
        // wants the value behind it: the name is bound by the nearest block
        // above that takes it, or by the entry, found back through the
        // predecessors the way the form itself finds a name
        // (Block::get_value). A name in scope is bound once, so any way back
        // reaches the same binding, and the first predecessor serves; the
        // definition is that block's. (Until 10-19 every reader got "unknown
        // here", which the deferral's recipe builder took for a value only
        // the pusher has: the subsurface material, read off an option in a
        // block that never took it, was stored by value for it. The read
        // analysis and the contention analysis had each met the same blind
        // spot: SSA/Analysis.cpp read_paths, Contention.cpp
        // Origins::binding_block.)
        shared_ptr<Block> from = block;
        std::set<const Block *> seen;
        const auto takes = [&](const shared_ptr<Block> &b) {
            return b.get() == func.blocks.front().get() ||
                   std::any_of(b->args.begin(), b->args.end(),
                               [&](const Argument &x) { return x.name == a.name; });
        };
        while (from && !takes(from) && seen.insert(from.get()).second) {
            shared_ptr<Block> next;
            for (const auto &weak : from->preds) {
                if (const auto p = weak.lock()) {
                    next = p;
                    break;
                }
            }
            from = next;
        }
        if (!from || from.get() == block.get()) {
            return memo[key] = here; // bound nowhere above: unknown here
        }
        const Definition bound = of_argument(from->name, a);
        if (provisional.contains({from->name, a.name})) {
            return provisional[key] = bound; // as provisional as the binding's own
        }
        return memo[key] = bound;
    }

    // While this argument is being resolved it is on the stack, so a
    // predecessor that leads back here is a loop: the argument is
    // carried around the loop and defined by the header, which is here.
    if (!visiting.insert(key).second) {
        return here;
    }

    optional<Definition> found;
    optional<Definition> around; // a back edge's answer, if that is all there is
    bool merge = false;
    // Whether an edge was skipped for leading to an argument still being
    // resolved that is not this one: then what the other edges say holds
    // only if that argument comes out the same, which is not known until
    // the question that started it is answered -- so the answer is
    // provisional, never memoized. (Found 10-19 once names reached from
    // above were resolved at their binding: resolving a loop header's
    // `dead`, the analysis reached the body's merge of `dead` with `true`,
    // skipped the edge back to the header as carried round unchanged,
    // answered `true` for the merge and kept it; the simplifier then
    // folded the body's test on it. tests/bonsai/ssa/cse-rebound-argument.)
    bool via_other = false;
    for (const auto &weak : block->preds) {
        const auto pred = weak.lock();
        internal_assert(pred) << "predecessor of " << block_name << " died";
        const shared_ptr<Value> passed = passed_to(*pred, *block, k);
        if (!passed) {
            // Defined by the edge itself: a loop index, a call's result.
            merge = true;
            break;
        }
        const Definition d = of(pred->name, passed);
        // An answer built on a provisional one is as provisional as it is.
        // The skip below marks only the edge skipped at THIS level; an edge
        // whose own resolution, further down, skipped one -- `passed` an
        // argument whose answer is in `provisional` -- hands back what holds
        // only under the outermost question, and an answer made of such
        // edges must not be memoized either. (Found 2026-10-19 in
        // apps/pbrt's light-tree descent, bvh_descend: `done` is set in a
        // leaf arm and in a nested arm of the interior branch, so the merge
        // before the latch takes `true` from one edge and the interior
        // arm's merge of `true` with the header's carried `done` from the
        // other. Asked about a block inside the interior arm first, the
        // analysis answered the inner merge `true` -- provisional, the edge
        // back to the header skipped -- then saw `true` from both edges of
        // the outer merge, memoized it, and substitute_threaded_arguments
        // handed the header the constant: the descent stopped at the root.
        // tests/bonsai/ssa/constant-argument-provisional.)
        if (const auto *pa = std::get_if<Argument>(&passed->data);
            pa != nullptr && provisional.contains({pred->name, pa->name})) {
            via_other = true;
        }
        if (const auto *back = std::get_if<Argument>(&d.value->data);
            back != nullptr && visiting.contains({d.block, back->name})) {
            // Carried round a loop unchanged: the edge leads back to an
            // argument this resolution is already following -- the header's
            // own, or one of a block inside the loop that the question
            // started from -- so along this edge the value is whatever
            // enters the loop, which the other edges say.
            if (!around.has_value()) {
                around = d;
            }
            via_other = via_other || d.block != block_name || back->name != a.name;
            continue;
        }
        if (found.has_value() && !same_definition(*found, d)) {
            merge = true;
            break;
        }
        found = d;
    }
    visiting.erase(key);
    // The outermost question is answered: what was provisional under it is
    // meaningless to the next one.
    const bool outermost = visiting.empty();

    Definition answer;
    if (merge) {
        answer = memo[key] = here;
    } else if (!found.has_value()) {
        if (around.has_value()) {
            // Every edge in leads round the loop to an argument still being
            // resolved: what this is depends on what that turns out to be,
            // so the answer is that argument's, and provisional -- it names
            // a frame above rather than a definition.
            answer = provisional[key] = *around;
        } else {
            answer = memo[key] = here;
        }
    } else if (via_other) {
        // What the edges not skipped say, if the argument the skipped edge
        // leads to comes out the same: an answer for this question alone.
        answer = provisional[key] = *found;
    } else {
        answer = memo[key] = *found;
    }
    if (outermost) {
        provisional.clear();
    }
    return answer;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
