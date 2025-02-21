#pragma once

#include "CodeGen_LLVM.h"
#include "IR/Program.h"

namespace bonsai {
namespace codegen {

// Converts program to ASM via the LLVM pipeline. If `filename` is empty, then
// prints to standard I/O.
void to_asm(std::string_view filename, const ir::Program &program,
            CodeGen_LLVM *gen);

} // namespace codegen
} // namespace bonsai
