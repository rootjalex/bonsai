#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace opt {

// Performs dead code elimination on `program`.
ir::Program dce(const ir::Program &program);

} // namespace opt
} // namespace bonsai
