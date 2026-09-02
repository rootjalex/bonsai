#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

// Fuse a floating-point multiply into the add that consumes it, so that
// `a * b + c` rounds once instead of twice.
//
// This is `--ffp-contract`, and it exists because pbrt's answers are the fused
// ones. pbrt is built with gcc and no `-ffp-contract` flag, which means gcc's
// default of `-ffp-contract=fast`; apps/pbrt is a transcription of pbrt and
// cannot agree with it bit for bit while its arithmetic rounds twice where
// pbrt's rounds once. Fusing is also the *more* accurate of the two, so this is
// not a fast-maths switch: nothing here reassociates, drops a sign, or assumes
// anything about NaN. One rounding is removed and nothing else changes.
//
// It runs on the graph rather than before it or inside a backend, and both
// halves of that matter. Before the graph there is no def-use, so there is no
// way to ask the question that decides whether a fusion is free -- is this
// product used anywhere else? -- and the passes that rewrite arithmetic have
// not run yet, so it would be contracting a form that is not the final one.
// Inside a backend it would reach one backend: apps/pbrt generates C++, and
// clang's default `-ffp-contract=on` only fuses within a single expression,
// which the graph has already split into named temporaries. Here, one pass
// serves the direct path to LLVM and the path back through statements to C++,
// and they get the same arithmetic by construction rather than by two
// implementations agreeing.
//
// What it does not do is match gcc's *choice*. `(a*b + c*d)` has two legal
// contractions and gcc does not pick consistently -- `Sqr(x)+Sqr(y)+Sqr(z)`
// fuses the first product and `b0*n0+b1*n1+b2*n2` fuses the second. This fuses
// the add's first operand when both are products, which is a rule where gcc has
// none, so it will agree with gcc often and not always.
void contract_fp(Function &f);

} // namespace ssa
} // namespace ir
} // namespace bonsai
