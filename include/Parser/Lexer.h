#pragma once

#include <sstream>
#include <string>

#include "Token.h"

namespace bonsai {
namespace parser {

TokenStream lex(const std::string &filename);

static void invalidCaseSyle() {}

} // namespace parser
} // namespace bonsai
