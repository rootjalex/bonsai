#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Split actions on structs into actions on scalars/vectors.
// Only really useful for the Packetization pass,
// this is required before packetizing.
// All struct arguments are split into their constituent pieces.
// Must be run before split_vector_ops()
// Only the return type can stay a struct, every other struct operation
// is removed.
std::shared_ptr<ir::Function> split_struct_ops(const ir::Function &func);

} // namespace lower
} // namespace bonsai
