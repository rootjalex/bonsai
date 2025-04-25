#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

class Rename : public Pass {
  public:
    constexpr std::string name() const override { return "rename"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace lower
} // namespace bonsai
