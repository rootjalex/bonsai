#pragma once

#include "CompilerOptions.h"
#include "IR/Analysis.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <set>
#include <string>

namespace bonsai {
namespace opt {

// Performs peephole optimizations and the likes.
// e.g., A + 0 => A
//
// And what a branch decides. Inside `if (c) { A } else { B }` the condition
// is true throughout A and false throughout B, and after `if (c) { ...;
// return }` it is false for the rest of the sequence; every branch or select
// on the same condition in those places folds to the arm taken. This is
// the branch-condition half of what LLVM's CorrelatedValuePropagation and
// GVN's propagateEquality do, and of GCC's assertion-based VRP:
//
//   Mark N. Wegman and F. Kenneth Zadeck. "Constant Propagation with
//   Conditional Branches." TOPLAS 13(2), 1991 -- the conditional constant
//   propagation this is the structured-code form of.
//
// A condition is learned only when it is a pure value over names the
// function cannot assign, so that it means the same thing wherever it is
// read; and a select on a known condition folds only if the arm not taken
// has no side effect, since a select evaluates both.
//
// This part is a strict reduction: nothing is copied, and the work is one
// walk of the function with a lookup per branch against the facts in scope.
class Simplify : public lower::Pass {
  public:
    const std::string name() const override { return "simplify"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;

    static ir::Expr simplify(ir::Expr);
    static ir::Stmt simplify(ir::Stmt);

    // What a simplification may take as known about the function the
    // statement is in.
    struct Knowledge {
        // The names the function can assign (ir::assignable_names). A
        // condition over any of them is never learned.
        std::set<std::string> assignable;
        // The functions whose calls have side effects
        // (ir::find_side_effects), for whether a select's arm can be dropped.
        std::set<std::string> effectful;
        // Conditions decided where the statement begins.
        ir::Facts known;
    };
    static ir::Stmt simplify(ir::Stmt, const Knowledge &);
};

} // namespace opt
} // namespace bonsai
