#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

class VerifyLayouts : public Pass {
  public:
    constexpr std::string name() const override { return "verify-layouts"; }

    ir::ScheduleMap run(ir::ScheduleMap schedule) const override;
};

} // namespace lower
} // namespace bonsai
