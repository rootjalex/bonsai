#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs intra-function common subexpression elimination (CSE).
class CSE : public lower::Pass {
  public:
    constexpr std::string name() const override { return "cse"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace opt
} // namespace bonsai
