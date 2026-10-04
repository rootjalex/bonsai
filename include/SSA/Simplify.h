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
//   * in `a & b` the value `a` is true wherever `b` matters, and `b` true
//     wherever `a` does (false under `|`); in `select(c, t, f)` the value
//     `c` is true inside `t` and false inside `f`. An occurrence of the one
//     inside the other, reached through pure value computations -- lanewise
//     ones, for a mask -- is the constant, and what it decided folds:
//     `hit & (select(hit, near, inf) <= best)` is `hit & (near <= best)`;
//   * two comparisons of one kind under `&` or `|` sharing a side are one
//     against a min or a max: `(a <= b) & (a <= c)` is `a <= min(b, c)`.
//     Over integers only -- std::min passes a NaN in its second argument
//     over, so over floats the two differ unless `c` is known not to be a
//     NaN, which nothing says;
//   * an operation over broadcasts alone is the broadcast of the scalar
//     operation, `min(bc(s), bc(t))` is `bc(min(s, t))`; and over integers
//     a broadcast joining a chain of mins or maxes that holds one joins it,
//     `min(min(x, bc(s)), bc(t))` is `min(x, bc(min(s, t)))`;
//   * two instructions that are the same operation on the same operands are
//     one instruction.
void simplify(Function &func);

// Whether an instruction only computes a value, so that one nothing reads
// can go. Storage, effects and the fetch-and-add are kept; so is `rand`,
// which steps the generator's state whether or not its draw is read. A load
// counts as pure: it makes nothing happen, though where it may be moved to
// is a question of what is stored in between (see SSA/ReorderLoops.h).
bool pure(const Instruction &in);

// Removes every pure instruction nothing reads, to a fixed point. What
// simplify() ends with once it has changed something; a rewrite that leaves
// values unread by itself -- loopify replacing a run of calls, whose lanes
// were read for the calls alone -- calls it for what it orphaned.
void remove_dead(Function &func);

// Whether an instruction reads memory: a load through a pointer, or an
// element read from an array -- which this form writes as extract_idx of the
// array handle (or of the struct a dynamic array is lowered to), the same
// instruction that takes a lane of a vector value, which reads no memory. A
// load_field reads a field of a struct *value*, a register; a field read
// through a pointer is a field.ptr and a load. What a read of memory gives
// depends on the stores before it, so it may be recomputed elsewhere only
// where the same stores have happened (see SSA/ReorderLoops.h).
bool reads_memory(const Instruction &in);

} // namespace ssa
} // namespace ir
} // namespace bonsai
