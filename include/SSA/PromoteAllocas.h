#pragma once

#include "SSA/SSA.h"

#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Promotes stack allocations that are local to the region rooted at `entry`
// into SSA values, replacing their loads and stores with block arguments --
// the classic mem2reg, in the formulation of Cytron et al. (1991).
//
// The SSA builder only puts a local in memory when it is declared `mut` (see
// the Allocate visitor in SSA/Convert.cpp); everything else is already a
// value. Vectorization cares about the difference because a `mut` local
// written under a divergent branch would otherwise need masked memory
// traffic, whereas a promoted one is just a block argument, blended at the
// join like any other varying value.
//
// Only allocations whose pointer never escapes are promoted: every use has to
// be the pointer operand of a Load or a Store, or the threading of that
// pointer through a block argument. A local aggregate indexed by a GEP, or
// one whose address is handed to a call, stays in memory and is left for the
// widening/gather-scatter path.
//
// `entry` is normally the function's entry block, which covers any ParFor
// bodies inside it too. An allocation is never promoted across a ParFor
// boundary: the body ends at a Yield, so a store there reaches no use the
// rename walk can see. Returns the number of allocations promoted.
size_t promote_allocas(Function &func, const std::string &entry);

} // namespace ssa
} // namespace ir
} // namespace bonsai
