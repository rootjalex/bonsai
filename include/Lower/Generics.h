#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

ir::Program lower_generics(const ir::Program &program);

} // namespace lower
} // namespace bonsai
