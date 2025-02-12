#pragma once

#include "IR/Program.h"
#include "Utils.h"
#include <string>

namespace bonsai {
namespace lower {

// TODO(cgyurgyik): Need some kind of pass registry so that command line can
// take in a `-p <pass-name>` argument.
class Pass {
  public:
    // Returns the name of this pass.
    virtual constexpr std::string_view name() = 0;

    // Runs this pass on program, and returns an error upon failure.
    virtual Error run(ir::Program &program) = 0;
};

} // namespace lower
} // namespace bonsai