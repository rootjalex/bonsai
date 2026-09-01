#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Some defers scheduling requires a concretized layout.
class LowerPostLayoutDefers : public Pass {
  public:
    const std::string name() const override {
        return "lower-post-layout-defers";
    }

    // Requires full-program (needs access to schedule).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
