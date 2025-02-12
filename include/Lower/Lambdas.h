#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

// Lowers a lambda expression into a function. After this pass is complete, uses
// of lambda expressions will be replaced with calls of functions. This does
// *not* remove dead lambda expressions; we leave that to a dead code
// elimination pass. For example, the following program:
//      func foo() -> void {
//        L1 = |x: i32| x - 1;
//        y: i32 = L1(0);
//      }
//
//  will lower to:
//
//     func ?lambda(x: i32) -> i32 { return x - 1; }
//     func foo() -> void {
//       y: i32 = ?lambda(0);
//     }
//
ir::Program lower_lambda(const ir::Program &program);

} // namespace lower
} // namespace bonsai
