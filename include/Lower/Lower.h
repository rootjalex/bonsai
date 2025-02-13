#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"

namespace bonsai {
namespace lower {

void lower(ir::Program &program, const CompilerOptions &options);

} // namespace lower
} // namespace bonsai
