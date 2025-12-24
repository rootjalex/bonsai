#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace ir {
namespace ssa {

class ConvertToSSA : public lower::Pass {
  public:
    const std::string name() const override { return "convert-to-ssa"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
