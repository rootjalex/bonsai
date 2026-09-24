#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

// Peephole simplification of a function's instructions, and removal of the
// instructions nothing reads once it is done.
//
// The rewrites that build the graph make instructions by rule, not by
// inspection. A sorting network compares the keys it is handed whatever they
// are, and the keys a front-to-back traversal is handed are `cast(c)` and
// `cast(!c)` for one bool `c`: the network's whole comparison is `!c`, but
// nothing that built it could see that. The statement-level simplifier ran
// before the conversion and never sees what a rewrite makes, and the backends
// do not know these identities either -- LLVM leaves `uitofp(c) < select(c,
// 0.0, 1.0)` as a convert, a select and a compare. So a few identities are
// applied here, where the graph is, and what they leave unread is removed.
//
// The rules:
//   * a constant condition, operand or comparison folds;
//   * `!!x` is `x`; `x & x` and `x | x` are `x`;
//   * a select between equal arms is the arm;
//   * `cast(a) < cast(b)`, for bools `a` and `b` cast to one numeric type, is
//     `!a & b` -- the only way a number made from a bool is below another is
//     false below true;
//   * two instructions that are the same operation on the same operands are
//     one instruction.
void simplify(Function &func);

// Whether an instruction only computes a value, so that one nothing reads
// can go. Storage, effects and the fetch-and-add are kept; so is `rand`,
// which steps the generator's state whether or not its draw is read. A load
// counts as pure: it makes nothing happen, though where it may be moved to
// is a question of what is stored in between (see SSA/ReorderLoops.h).
bool pure(const Instruction &in);

} // namespace ssa
} // namespace ir
} // namespace bonsai
