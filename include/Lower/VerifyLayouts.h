#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

class VerifyLayouts : public Pass {
  public:
    const std::string name() const override { return "verify-layouts"; }

    ir::ScheduleMap run(ir::ScheduleMap schedule,
                        const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
