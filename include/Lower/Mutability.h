#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Arguments passed as mutable args to function calls get rewritten to pointers.
class Mutability : public Pass {
  public:
    constexpr std::string name() const override { return "lower-mutable"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace lower
} // namespace bonsai
