#pragma once

#include "IR/Target.h"

#include <string>

namespace bonsai {
namespace codegen {

// The machine a compile is for (ir::Target), from `--triple` and `--mcpu`.
// With neither named it is this machine -- its triple, its CPU and every
// feature it has, which is what `-march=native` gives a C compiler and what
// the code bonsai is measured against is built with. With either named it
// is that machine, and the features are the ones LLVM knows the CPU to have,
// so that generated code is the same wherever the compile runs. Resolved
// once, before lowering, so that the rewrites can read it; the backend's
// target machine is then built from the same triple, CPU and feature string
// (CodeGen_LLVM::make_target_machine).
ir::Target resolve_target(const std::string &triple, const std::string &cpu);

} // namespace codegen
} // namespace bonsai
