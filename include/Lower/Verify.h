#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

// Performs basic checks to ensure the program is well formed.
void verify(const ir::Program &program);

} // namespace lower
} // namespace bonsai
