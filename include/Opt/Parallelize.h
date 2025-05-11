#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Replaces a parallel ForAll loop with a Launch(func, n, {ctx})
// Where `ctx` is a context struct and `n` is the number of iterations.
// Intended to be used for `dispatch_apply_f` code generation.
// Inserts new functions into the func map (parallel closures).

// parforall i in [start:end:stride]
//   body
// =>
// func closure(ctx : ptr[Context], j : u32) {
//   let i = start + stride * j in
//   body
// }
// Launch(closure, (end - start + (stride - 1)) / stride, {ctx})
ir::Stmt parallelize_forall(const std::string &loop_idx, ir::Stmt body,
                            ir::FuncMap &funcs, ir::TypeMap &types);
} // namespace opt
} // namespace bonsai
