
#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

std::vector<std::string> func_topological_order(const ir::Program &program,
                                                const bool undef_calls);

} // namespace lower
} // namespace bonsai
