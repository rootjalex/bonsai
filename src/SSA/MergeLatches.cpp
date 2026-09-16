#include "SSA/MergeLatches.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <memory>
#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

using std::set;
using std::shared_ptr;
using std::string;

string merge_latches(Function &func, const string &header,
                     const set<string> &latches) {
    const BlockMap blocks = make_block_map(func);
    const auto head = blocks.at(header);

    set<string> taken;
    for (const auto &block : func.blocks) {
        taken.insert(block->name);
        for (const Argument &arg : block->args) {
            taken.insert(arg.name);
        }
    }
    auto latch = std::make_shared<Block>();
    latch->name = header + "!latch";
    for (size_t i = 0; taken.count(latch->name); i++) {
        latch->name = header + "!latch" + std::to_string(i);
    }
    latch->owner = head->owner;

    // One argument per argument of the header, to hand on what a back edge
    // brought. Under fresh names rather than the header's: a jump that passes
    // the header an argument of the header's own name is passing the value
    // straight back, which is how insert_preheader() tells a value the loop
    // carries from one it merely reads, and what arrives here is a new value
    // every iteration.
    Terminator::Jump on{header};
    for (const Argument &arg : head->args) {
        Argument own = arg;
        own.name = arg.name + "!latch";
        for (size_t i = 0; taken.count(own.name); i++) {
            own.name = arg.name + "!latch" + std::to_string(i);
        }
        taken.insert(own.name);
        latch->args.push_back(own);
        auto value = std::make_shared<Value>(own);
        latch->lookups[own.name] = value;
        on.args.push_back(std::move(value));
    }
    latch->terminator.data = std::move(on);

    // Every back edge now closes on the new latch, carrying what it carried.
    for (const string &name : latches) {
        for (Terminator::Jump *jump : jumps_of(*blocks.at(name))) {
            if (jump->name == header) {
                internal_assert(jump->args.size() == head->args.size())
                    << "Back edge from " << name << " passes "
                    << jump->args.size() << " values to a header taking "
                    << head->args.size();
                jump->name = latch->name;
            }
        }
    }

    func.blocks.push_back(latch);
    refresh_preds(func);
    return latch->name;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
