#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

// Builds the SSA form of a single lowered function.
std::shared_ptr<ssa::Function> build(const std::shared_ptr<ir::Function> &func);

class ConvertToSSA : public lower::Pass {
  public:
    const std::string name() const override { return "convert-to-ssa"; }

    // Reads schedule transforms (e.g. `vectorize`) off `program.schedules`
    // and applies the SSA-level rewrites from SSA/Rewrite.h to the relevant
    // functions.
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
