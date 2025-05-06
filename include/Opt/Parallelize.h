#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Replaces a parallel ForAll loop with a CallStmt("launch", func, n)
// with a context struct and the number of iterations.
// Intended to be used for `dispatch_apply_f` code generation.
// Inserts new functions into the func map (parallel closures).

// parforall i in [start:end:stride]
//   body
// =>
// func closure(ctx : ptr[Context], j : u32) {
//   let i = start + stride * j in
//   body
// }
// CallStmt("launch", closure, (end - start + (stride - 1)) / stride)
class Parallelize : public lower::Pass {
  public:
    constexpr std::string name() const override { return "parallelize"; }

    // TODO: needs access to schedule, should be full-program lowering.
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
