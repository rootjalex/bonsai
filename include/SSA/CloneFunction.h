#pragma once

#include "SSA/SSA.h"

#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// A deep copy of `func`, sharing nothing with it: every block, instruction
// and value is new, and references between them point inside the copy.
//
// Vectorization uses this to specialize a callee for the gang that calls it,
// leaving the original alone for the scalar call sites that still want it.
std::shared_ptr<Function> clone_function(const Function &func);

// Rewrites `func` to have a single Return.
//
// Each existing Return becomes a jump to one new exit block, which takes the
// returned value as an argument (a void function's exit takes none). Partial
// linearization needs this: it folds the branches of a region into one path,
// which only makes sense if that path has a single end. It is also what makes
// an early return work per lane -- once the returns are edges rather than
// exits, the lanes that took one are simply masked off for the rest.
//
// Returns the name of the exit block.
std::string unify_returns(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
