#pragma once

#include "SSA/SSA.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// What the loop rewrites (collapse, reorder, split) ask about a parfor's
// values and emit for its indices.

// Does `v` change from one step of the loop to the next? An instruction does
// if it reads the index, or reads anything that does. `region` is the set of
// block names of the loop's body; an instruction owned outside it was worked
// out before the loop began.
bool varies_with(const std::shared_ptr<Value> &v, const std::string &index,
                 const std::set<std::string> &region);

// Is `v` the same on every step of the loop whose index is `index`?
//
// Of the body block's arguments only the index varies. A parfor body has no
// back edge -- it ends at a yield -- so the header's jump is the only way in,
// and everything the header passes it was worked out before the loop began.
// Rejecting all of them would refuse the ordinary nested `map` over
// `array[array[f32, n], m]`, where the inner bound `n` is a parameter that
// simply happens to reach the body as an argument.
bool invariant_in(const std::shared_ptr<Value> &v, const std::string &index,
                  const std::set<std::string> &region);

// The integer a constant value holds, if it is one.
std::optional<int64_t> as_int(const std::shared_ptr<Value> &v);

// A constant of the index type `type` -- an unsigned index makes an unsigned
// constant, which is the arm its type is read back out of (see split()).
std::shared_ptr<Value> index_constant(const Type &type, int64_t v);

// One arithmetic instruction appended to `block`, or the answer if it is
// already known.
//
// The index arithmetic the rewrites make is mostly identities -- a loop from
// zero, by one, contributes `- 0`, `* 1` and `/ 1` at every step -- and
// nothing downstream removes them: opt::Simplify runs before the SSA
// conversion, so it never sees what a rewrite builds afterwards. Folding here
// is what keeps a collapsed loop over a plain rectangle down to the two
// divisions it actually needs.
std::shared_ptr<Value> arith(Block &block, const Type &type,
                             Instruction::Op op,
                             const std::shared_ptr<Value> &lhs,
                             const std::shared_ptr<Value> &rhs);

// ceil((end - begin) / stride), as instructions appended to `block`.
std::shared_ptr<Value> trip_count(Block &block, const Type &type,
                                  const std::shared_ptr<Value> &begin,
                                  const std::shared_ptr<Value> &end,
                                  const std::shared_ptr<Value> &stride);

// (index - begin) / stride, as instructions appended to `block`: the step
// number of the iteration whose index is `index`, from zero.
std::shared_ptr<Value> normalized_index(Block &block, const Type &type,
                                        const std::shared_ptr<Value> &index,
                                        const std::shared_ptr<Value> &begin,
                                        const std::shared_ptr<Value> &stride);

} // namespace ssa
} // namespace ir
} // namespace bonsai
