#pragma once

#include "SSA/SSA.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// The result of the uniform/varying analysis over one ParFor body region: a
// value is *varying* (divergent) when the lanes of a gang may disagree about
// it, and *uniform* otherwise.
//
// Values are keyed two ways because the SSA form uses block arguments in
// place of phis, and an argument's name is only unique within its block (both
// arms of an `if` may take an argument named `i`). Instructions are keyed by
// identity instead, since a block's instruction list owns them.
struct Divergence {
    // Instructions whose result is varying.
    std::set<const Instruction *> instrs;
    // Varying block arguments, as (block name, argument name).
    std::set<std::pair<std::string, std::string>> args;
    // Blocks whose Dispatch condition is varying: the lanes of a gang may
    // disagree about which successor to take, so this branch cannot survive
    // into vector code.
    std::set<std::string> branches;
    // Blocks that may execute with some lanes disabled, i.e. that are control
    // dependent -- transitively -- on a divergent branch. These are exactly
    // the blocks whose side effects have to be masked.
    std::set<std::string> masked;

    // Which varying argument names are in scope in each block: its own, and
    // those of every block that dominates it.
    //
    // A block does not only refer to its own arguments. Linearization drops
    // the arguments it turns into blends, after which the blocks that used to
    // receive them refer to the dominating definition directly -- and a loop
    // header's arguments, which survive, are referred to that way from
    // everywhere in the loop.
    std::map<std::string, std::set<std::string>> in_scope;

    // Is `v`, as referenced from `block`, varying?
    bool is_varying(const std::string &block, const Value &v) const;
};

// Solves divergence over the region of `func` rooted at `entry`, seeded by
// the entry-block arguments named in `varying_seeds` (for a ParFor body, the
// loop index), by the instructions in `varying_instrs`, and by the block
// arguments in `varying_args`.
//
// The second seed is for after linearization, when the index is no longer a
// block argument threaded through the region but a single instruction the
// whole region refers to. The third is for arguments of blocks other than the
// entry, which is what the live and exit masks of a uniformized loop are (see
// SSA/UniformizeLoops.h): nothing they are computed from is varying, and yet
// they hold one bool per lane by construction.
//
// A value is varying if it is seeded, if any operand is varying, or -- for a
// block argument -- if the block is a join that some lanes may reach along
// one predecessor and others along another.
Divergence analyze_divergence(
    const Function &func, const std::string &entry,
    const std::set<std::string> &varying_seeds,
    const std::set<const Instruction *> &varying_instrs = {},
    const std::set<std::pair<std::string, std::string>> &varying_args = {});

} // namespace ssa
} // namespace ir
} // namespace bonsai
