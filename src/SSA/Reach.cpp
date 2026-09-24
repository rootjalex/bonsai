#include "SSA/Reach.h"

#include "Utils.h"

namespace bonsai {
namespace ir {
namespace ssa {

using std::shared_ptr;
using std::vector;

shared_ptr<Value> reach(const shared_ptr<Block> &block,
                        const shared_ptr<Value> &v) {
    return std::visit(
        overloads{
            [&](const Constant &) { return v; },
            [&](const Argument &a) -> shared_ptr<Value> {
                if (const auto it = block->lookups.find(a.name);
                    it != block->lookups.end()) {
                    return it->second;
                }
                return block->get_value(a.name, a.type);
            },
            [&](const shared_ptr<Instruction> &i) -> shared_ptr<Value> {
                if (i->owner.lock().get() == block.get()) {
                    return v;
                }
                return block->get_value(i->name, i->type);
            },
        },
        v->data);
}

vector<shared_ptr<Value>> reach_all(const shared_ptr<Block> &block,
                                    const vector<shared_ptr<Value>> &vs) {
    vector<shared_ptr<Value>> out;
    out.reserve(vs.size());
    for (const auto &v : vs) {
        out.push_back(reach(block, v));
    }
    return out;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
