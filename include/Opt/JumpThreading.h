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
// What the copies decide is left to the simplifier: a condition known in an
// arm folds every test of it in that arm (Opt/Simplify.h), with no copy, and
// this pass only makes the copies that put a later test inside an arm.
//
// What is threaded: a branch on a condition the simplifier could learn -- a
// pure value over names the function cannot assign -- followed in its
// sequence by statements that test the same condition and contain no loop
// (a schedule names loops, and two copies of one would answer to the same
// name). When an arm is itself such a branch and the run tests its condition
// too, the copies go into that branch's arms in turn: a `match` over a tag
// lowers to a chain of these, and the run is copied once per case rather
// than once per test. The names the copied statements bind are renamed in
// each copy.
//
// Two rules keep the replication linear, in time and in space:
//
//  * A copy is final. The statements copied into an arm are simplified under
//    what the arm knows and never threaded again; only the arm they were
//    appended to is. So a statement is copied by at most one threading, and
//    two independent conditions tested in a row do not multiply -- the
//    exponential that code replication is capable of (Mueller and Whalley
//    discuss it) cannot happen.
//
//  * A threading may add at most kMaxGrowth statements: the statements of
//    the run that survive in each copy once its arm's facts are applied,
//    summed over the copies, less the run as it stood. The arms of the
//    decided tests are distributed, not duplicated; what is duplicated is
//    the code between them, and that is what is counted. GCC bounds the
//    same thing at 15 statements (its `max-jump-thread-duplication-stmts`),
//    LLVM at 6 instructions (`jump-threading-threshold`).
//
// A function therefore grows by at most kMaxGrowth per branch it contains,
// and the pass visits each statement a bounded number of times: once to
// look for a run to thread, once per candidate branch in its sequence to
// measure it, and once to copy it.
class JumpThreading : public lower::Pass {
  public:
    const std::string name() const override { return "jump-threading"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
