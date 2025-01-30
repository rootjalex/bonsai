#pragma once

#include <sstream>
#include <string>

#include "Token.h"

namespace bonsai {
namespace parser {

TokenStream lex(const std::string &filename);

} // namespace parser
} // namespace bonsai
