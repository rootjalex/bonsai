#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"

#include <string>

namespace bonsai {
namespace codegen {

// Converts program to ASM via the LLVM pipeline. If `filename` is empty, then
// prints to standard I/O.
void to_asm(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen
} // namespace bonsai
