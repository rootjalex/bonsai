#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

class LowerArrays : public Pass {
  public:
    constexpr std::string name() const override { return "lower-arrays"; }

    // Requires full-program analysis (needs access to schedule).
    ir::Program run(ir::Program program) const override;
};

} // namespace lower
} // namespace bonsai
