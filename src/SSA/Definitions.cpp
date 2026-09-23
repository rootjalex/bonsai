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

Definitions::Definitions(const Function &func, bool lenient)
    : func(func), lenient(lenient), bmap(make_block_map(func)) {}

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
    const Definition d = of(block, v);
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
    if (k == block->args.size() && lenient) {
        return memo[key] = here; // reached by name from above: unknown here
    }
    internal_assert(k < block->args.size())
        << a.name << " is referred to in " << block_name
        << " but is neither one of its arguments nor a parameter";

    // While this argument is being resolved it is on the stack, so a
    // predecessor that leads back here is a loop: the argument is
    // carried around the loop and defined by the header, which is here.
    if (!visiting.insert(key).second) {
        return here;
    }

    optional<Definition> found;
    optional<Definition> around; // a back edge's answer, if that is all there is
    bool merge = false;
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
