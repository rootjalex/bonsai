#include "SSA/ThreadPhases.h"

#include "Error.h"
#include "SSA/Analysis.h"
#include "SSA/Definitions.h"

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

// Whether two loop bounds are one value: the same constant, or the same
// definition as the two loops' blocks refer to it (a capture threaded into
// each through its own block arguments resolves to the one argument of the
// body's head).
bool same_bound(Definitions &defs, const std::string &ablock,
                const std::shared_ptr<Value> &a, const std::string &bblock,
                const std::shared_ptr<Value> &b) {
    return same_definition(defs.of(ablock, a), defs.of(bblock, b));
}

} // namespace

std::vector<ThreadPhase> thread_phases(const Function &func,
                                       const Terminator::ParFor &block_loop) {
    std::vector<ThreadPhase> phases;
    const Cfg body(func, block_loop.body.name);
    for (const auto &block : body.blocks()) {
        const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (p != nullptr && p->binding == Resource::GPUThread) {
            phases.push_back(ThreadPhase{p, block->name, true});
        }
    }
    Definitions defs(func, /*lenient=*/true, /*bound=*/true);
    for (ThreadPhase &phase : phases) {
        for (const ThreadPhase &other : phases) {
            if (&other == &phase) {
                continue;
            }
            if (!same_bound(defs, phase.block, phase.loop->start, other.block,
                            other.loop->start) ||
                !same_bound(defs, phase.block, phase.loop->end, other.block,
                            other.loop->end) ||
                !same_bound(defs, phase.block, phase.loop->stride, other.block,
                            other.loop->stride)) {
                phase.whole_block = false;
                break;
            }
        }
    }
    return phases;
}

bool whole_block(const Function &func, const std::string &block,
                 const Terminator::ParFor &loop) {
    internal_assert(loop.binding == Resource::GPUThread)
        << "whole_block of the loop over " << loop.index
        << ", which is not bound to GPUThread";
    for (const auto &b : func.blocks) {
        const auto *p = std::get_if<Terminator::ParFor>(&b->terminator.data);
        if (p == nullptr || p->binding != Resource::GPUBlock ||
            !Cfg(func, p->body.name).contains(block)) {
            continue;
        }
        // By the block the loop is the terminator of: the caller may hold
        // a copy of the terminator.
        for (const ThreadPhase &phase : thread_phases(func, *p)) {
            if (phase.block == block) {
                return phase.whole_block;
            }
        }
    }
    return true;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
