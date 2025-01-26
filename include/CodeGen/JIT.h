#pragma once

#include "CodeGen_LLVM.h"
#include "IR/Program.h"

namespace bonsai {
namespace codegen {

void jit(const ir::Program &program, CodeGen_LLVM *gen);

} // namespace codegen
} // namespace bonsai
