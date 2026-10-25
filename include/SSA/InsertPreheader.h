#pragma once

#include "SSA/SSA.h"

#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Gives a loop header a dedicated preheader, by moving everything the header
// does into a fresh block that becomes the loop header instead. `back_edges`
// names the blocks whose jumps to `header` close the loop; those are
// redirected to the new header, while everything entering from outside still
// arrives at the block that was there before. Returns the new header's name.
//
// Two things need this. A loop cannot grow arguments while its header is also
// the function's entry -- the entry's arguments are the parameters -- which is
// exactly the shape loopify() leaves behind, and which the live and exit masks
// of a uniformized divergent loop need (see SSA/UniformizeLoops.h). And a
// header that is the entry makes the values carried around the loop share
// names with the parameters they start out as, which are two different values:
// the parameter does not change, and the carried value is a fresh one every
// iteration. The header's arguments are therefore renamed, and the parameters
// left alone, since the parameter names are what the rest of the compiler
// refers to a function's arguments by.
std::string insert_preheader(Function &func, const std::string &header,
                             const std::set<std::string> &back_edges);

// Renames one argument, and every reference to it, throughout the region
// reachable from `region`.
//
// A name is a definition in this SSA form, so a rename has to reach the whole
// region the definition is visible in. For a block that dominates everything
// that can see the name -- a loop header, or the block a function's body was
// moved into -- that region is everything below it.
void rename_argument(Function &func, const std::string &region,
                     const std::string &from, const std::string &to);

// Replaces one argument by `value` throughout the region reachable from
// `region`, and drops the argument from every block of the region that
// declares it and the operand for it from every jump within the region that
// passes it on. What unrolling a loop does with its index (SSA/Unroll.h):
// each copy of the body reads the constant its iteration is, and no block
// of the copy takes the index. The region's entry loses the argument too;
// the jump into it from outside the region is the caller's to make without
// one.
void substitute_argument(Function &func, const std::string &region,
                         const std::string &name,
                         const std::shared_ptr<Value> &value);

} // namespace ssa
} // namespace ir
} // namespace bonsai
