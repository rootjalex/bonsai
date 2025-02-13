#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/PassManager.h"

namespace bonsai {
namespace lower {

// Lowers a Bonsai program with the provided compiler options.
void lower(ir::Program &program, const CompilerOptions &options);

PassManager register_passes();

} // namespace lower
} // namespace bonsai
