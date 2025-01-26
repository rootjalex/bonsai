#pragma once

#include "CodeGen_LLVM.h"
#include "IR/Program.h"

namespace bonsai {
namespace codegen {

void to_asm(const std::string &filename, const ir::Program &program,
            CodeGen_LLVM *gen);

} // namespace codegen
} // namespace bonsai
