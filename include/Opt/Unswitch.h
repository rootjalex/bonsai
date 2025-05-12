#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Implements if statement lifting. This should run after LICM.
// e.g.
// if (a) {
//   if (b) { foo() }
// } else {
//   if (b) { bar() }
// }
// Becomes:
// if (b) {
//   if (a) { foo() }
//   else { bar() }
// }
// This will also perform classic loop unswitching.
// https://en.wikipedia.org/wiki/Loop_unswitching
class Unswitch : public lower::Pass {
  public:
    const std::string name() const override { return "unswitch"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
