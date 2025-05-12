#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers all GeomOps by searching for a typed implementation in program.funcs
class LowerGeometrics : public Pass {
  public:
    const std::string name() const override { return "lower-geometrics"; }

    // Rewrites GeomOps to correct Call nodes.
    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
