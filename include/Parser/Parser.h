#pragma once

#include <string>

#include "IR/Program.h"

namespace bonsai {
namespace parser {

ir::Program parse(const std::string &filename);

} // namespace parser
} // namespace bonsai
