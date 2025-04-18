#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

class LowerForEachs : public Pass {
  public:
    constexpr std::string name() const override { return "lower-foreachs"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace lower
} // namespace bonsai
