#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Removes a conditional branch whose outcome an earlier branch on the same
// condition has already decided, by replicating the code between the two.
//
//     if (c) { A } else { B }          if (c) { A; R[c = true] }
//     R, which tests c again     ==>   else   { B; R[c = false] }
//
// Inside each copy of R the second test is a constant and folds to the arm
// taken, and anything R computed in one arm of that test now sits beside the
// same computation in the matching arm of the first -- where common
// subexpression elimination, which does not reach across a branch, can make
// the two one. That is the point of it here. A tree query asks
// `intersects(r, g)`, `alpha_accepts(...)` and `distmin(r, g)` of a
// candidate in turn, and on a `Shape` each is a `match` over the same tag:
// three dispatches, and behind two of them the same ray-triangle test. With
// the dispatch pulled up over what follows it, a candidate that is a triangle
// runs one triangle test, which is what pbrt's `Shape::Intersect` does once
// and everything after it reads.
//
// The transform is *jump threading* -- LLVM's and GCC's name for the pass
// that threads a branch through code whose incoming edge decides it -- and
// its replication form is Mueller and Whalley's:
//
//   Frank Mueller and David B. Whalley. "Avoiding Conditional Branches by
//   Code Replication." PLDI 1995.
//
//   Rastislav Bodík, Rajiv Gupta and Mary Lou Soffa. "Interprocedural
//   Conditional Branch Elimination." PLDI 1997, for the path-sensitive
//   generalisation and the analysis of which branches are removable.
//
// Applied to a dispatch on a type tag, which is what a `match` lowers to, it
// is the *splitting* of the SELF compiler: the code after a type test is
// split per type so that later dispatches on the same receiver are static.
//
//   Craig Chambers and David Ungar. "Iterative Type Analysis and Extended
//   Message Splitting: Optimizing Dynamically-Typed Object-Oriented
//   Programs." PLDI 1990.
//
// What is threaded: a branch on a pure condition -- no call in it, and no
// free variable of it that anything in the function can assign -- followed in
// its sequence by statements that test the same condition, contain no loop,
// and are not too many to copy. A loop is left alone because a schedule
// names loops and two copies of one would answer to the same name. The
// variables the copied statements bind are renamed in each copy. A condition
// known inside an arm also folds the tests nested in that arm, which is the
// same fact without the copy.
class JumpThreading : public lower::Pass {
  public:
    const std::string name() const override { return "jump-threading"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
