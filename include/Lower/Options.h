#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers an `option` type to a form more amenable for backend code generation.
class LowerOption : public Pass {
  public:
    constexpr std::string name() const override { return "lower-option"; }

    // Lowers all option types
    ir::TypeMap run(ir::TypeMap types) const override;
    // Asserts no externs have options
    ir::ExternList run(ir::ExternList externs) const override;
    // Rewrites functions to use new lowered option code.
    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace lower
} // namespace bonsai
