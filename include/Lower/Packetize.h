#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

class Packetize : public Pass {
  public:
    constexpr std::string name() const override { return "packetize"; }

    // Requires full-program analysis (needs access to schedule to know what to
    // packetize).
    // Also adds new packetized versions of functions.
    ir::Program run(ir::Program program) const override;
};

} // namespace lower
} // namespace bonsai
