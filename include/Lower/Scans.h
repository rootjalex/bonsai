#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Turns Scan into recloop
class LowerScans : public Pass {
  public:
    const std::string name() const override { return "lower-scans"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
