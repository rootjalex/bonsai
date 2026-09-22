#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Gives an exported function whose C ABI differs from the program's own
// calling convention an internal twin, `_unexported_<name>`, that the
// program's calls go to, and shapes the exported entry for C.
//
// Two things make the ABIs differ. A struct return, which an ABI-conforming
// C++ compiler expects through an out-parameter:
//
//   S func [[export]] foo();
//   v = foo();
//
//  =>
//
//   void func [[export]] foo($r : mut S)
//   $r := build<S>();
//   foo($r);
//   v = $r
//
// And an array parameter, which a driver hands over as a buffer descriptor
// (runtime/bonsai_buffer.h) while the program's calls pass a pointer; the
// entry keeps its signature here and its prologue unwraps the descriptors
// at code generation (CodeGen_LLVM::SSALowering::run).
class ReturnToOutParameter : public Pass {
  public:
    const std::string name() const override { return "rtop"; }

    ir::FuncMap run(ir::FuncMap functions,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
