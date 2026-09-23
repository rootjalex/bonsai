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

// A copy of the blocks `region` of `func`, renamed with `suffix`, keyed by
// the original block's name. Instructions get fresh names, and so does every
// argument that carried one of the old names on, so that the copy and the
// original can live in one function -- unless `keep_names`, for a copy that
// becomes a function of its own (stage(), SSA/Stage.cpp), where the
// program's names for its values have to survive for a schedule to point
// at them (`hits.specialize(material)`). A `let`'s Set keeps its name either
// way, and a name the region refers to but does not define -- a value from
// before it, threaded in -- keeps its name, since that is what its
// definition is called. Jumps to blocks outside the region are left
// pointing where they were, for the caller to redirect; the copies are not
// added to the function, and predecessors are not set -- the caller does
// both. Defined in SSA/Defer.cpp, which copies a producer's continuation
// with it; specialize() copies a loop's body per variant
// (SSA/Specialize.cpp).
std::map<std::string, std::shared_ptr<Block>>
clone_region(Function &func,
             const std::vector<std::shared_ptr<Block>> &region,
             const std::string &suffix, bool keep_names = false);

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

// The same for a parfor's body: rewrites the region rooted at `entry` to
// have a single Yield, every existing one becoming a jump to a new block
// that yields (a yield carries nothing, so the block takes no argument).
// The body of a loop written with several `continue`s, or built with several
// ends -- a split's tail beside its body, a drain's finished entries beside
// its saved ones -- has several exits, and linearization wants one. A nested
// parfor's yields are its own and are left alone. Returns the name of the
// one yielding block, or the empty string when the region has none.
std::string unify_yields(Function &func, const std::string &entry);

} // namespace ssa
} // namespace ir
} // namespace bonsai
