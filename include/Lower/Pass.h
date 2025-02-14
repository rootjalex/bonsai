#pragma once

#include "IR/Program.h"

#include <string>

namespace bonsai {
namespace lower {

struct Pass {
    // Returns the name of this pass.
    virtual constexpr std::string name() const = 0;

    // Runs this pass on `program`.
    virtual void run(ir::Program &program) const = 0;

    virtual ~Pass() = default;
};

} // namespace lower
} // namespace bonsai
