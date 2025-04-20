#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"

namespace bonsai {
namespace codegen {

void jit(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen
} // namespace bonsai
