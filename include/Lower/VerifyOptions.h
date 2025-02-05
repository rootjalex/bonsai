#pragma once

#include "IR/Program.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

// Statically verifies that options are always legally accessed. Due to its
// static nature, this results in incompleteness but enables the compiler to
// optimize `option` types. For example,
//
//      i: option<i32> = foo();
//      if i { use(i); } // LEGAL
//      use(i);          // ILLEGAL
void verify_options(const ir::Program &program);

} // namespace lower
} // namespace bonsai
