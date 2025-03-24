#pragma once

#include "IR/Layout.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// TODO: implement
class LowerLayouts : public Pass {
  public:
    constexpr std::string name() const override { return "lower-layouts"; }

    // Requires full-program analysis (updates type list).
    ir::Program run(ir::Program program) const override;
};

} // namespace lower
} // namespace bonsai
