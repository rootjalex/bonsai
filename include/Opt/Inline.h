#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs function inlining.
class Inline : public lower::Pass {
  public:
    constexpr std::string name() const override { return "inline"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
