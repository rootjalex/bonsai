#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

class ReturnToOutParameter : public Pass {
  public:
    constexpr std::string name() const override { return "r2op"; }

    ir::FuncMap run(ir::FuncMap functions) const override;
};

} // namespace lower
} // namespace bonsai
