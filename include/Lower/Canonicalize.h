#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

ir::Program canonicalize(const ir::Program &program);

} // namespace lower
} // namespace bonsai
