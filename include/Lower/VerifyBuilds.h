#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

class VerifyBuilds : public Pass {
  public:
    const std::string name() const override { return "verify-builds"; }

    ir::ScheduleMap run(ir::ScheduleMap schedule,
                        const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
