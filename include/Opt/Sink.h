#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Assignment sinking: partial dead code elimination for this structured IR.
//
// A `let` of a pure value moves to its latest place -- past the following
// statements that neither use it nor write anything it reads, and into the
// arms of the branch that uses it when only some arms do (duplicated under
// a fresh name per using arm when more than one does). It never moves into
// a loop, never past a write to anything the value reads, and never across
// a statement it does not understand, so no path computes the value more
// often than before and the paths through the arms that never read it stop
// computing it at all -- which is exactly the profitability criterion of
// the literature this implements: Knoop, Rüthing and Steffen, "Lazy Code
// Motion" (PLDI 1992) and "Partial Dead Code Elimination" (PLDI 1994),
// restricted to sinking on structured control flow.
//
// What it is for, concretely: a fully inlined program computes values where
// inlining left them, not where they are needed. A ray query's any-hit
// program interpolated the hit's point before the switch on the texture
// kind whose uv-mapped arms never read a point; an instanced candidate's
// untransform loaded its matrix above the early accept that reads nothing.
// LLVM will not sink loads past branches; this IR knows its reads are of
// invariant pools and can.
class Sink : public lower::Pass {
  public:
    const std::string name() const override { return "sink"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
