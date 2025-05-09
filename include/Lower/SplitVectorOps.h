#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Split actions on vectors into actions on scalars.
// Only really useful for the Packetization pass,
// this is required before packetizing.
// All vector arguments are split into scalars.
// This pass does not work on struct and accesses.
// Run split_struct_ops() before running this.
std::shared_ptr<ir::Function> split_vector_ops(const ir::Function &func);

} // namespace lower
} // namespace bonsai
