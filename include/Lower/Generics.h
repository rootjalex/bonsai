#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers generics to their respective typed variant.
class LowerGeneric : public Pass {
  public:
    constexpr std::string name() const override { return "lower-generic"; }

    ir::FuncMap run(ir::FuncMap &funcs) const override;
};

} // namespace lower
} // namespace bonsai
