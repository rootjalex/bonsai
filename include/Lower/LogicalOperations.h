#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers logical operations to their if-else form, e.g.,
//
// if (a && b) { foo(); }
// ->
// if (a) { if (b) { foo(); } }
class LowerLogicalOperations : public Pass {
  public:
    constexpr std::string name() const override {
        return "lower-logical-operation";
    }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
