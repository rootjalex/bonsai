#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

class LowerDynamicArrays : public Pass {
  public:
    const std::string name() const override { return "lower-dynamic-arrays"; }

    // Requires full-program analysis (needs access to types + funcs).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
