#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs function inlining.
//
// One level at a time, callers first, with a merge between levels. A round
// replaces every call to an inlinable function with a copy of the callee's
// body as it stood when the round began -- the calls inside that body stay
// calls -- and is followed by simplification and common subexpression
// elimination over the result; rounds continue until one copies nothing,
// which is after as many as the call graph is deep.
//
// The merge between levels is the point of the order. A call is one value,
// and two calls with the same arguments are one to CSE; a body copied in is
// a run of statements, and once it has mutable locals of its own, two copies
// of it are two computations for good. Copying the callers first puts the
// callees' calls side by side in one function, where CSE merges them before
// the next round copies the one that is left. `intersects` and `distmin`
// over an AABB each call `aabb_span`: after one round the traversal holds
// two calls to it, after the merge one, and after the next round one slab
// test. Callee-first inlining -- LLVM's order -- would have copied
// `aabb_span` into each of the two before either reached the traversal, and
// nothing downstream merges two copies.
//
// A function on a cycle of the call graph is never inlined; neither is a
// kernel, a vectorized gang, one marked `[[noinline]]`, one that takes an
// argument by reference, or one whose body is not a plain run of statements
// with returns that nest (see Inline.cpp). A body of more than
// kMaxInlinedStatements is inlined only when the program asks with
// `[[inline]]`.
class Inline : public lower::Pass {
  public:
    const std::string name() const override { return "inline"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
