#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Turns RecLoop into true recursion
class LowerRecLoops : public Pass {
  public:
    constexpr std::string name() const override { return "lower-recloops"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace lower
} // namespace bonsai
