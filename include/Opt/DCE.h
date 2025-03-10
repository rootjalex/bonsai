#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs dead code elimination on `program`.
// Runs "to convergence" in the sense that it
// removes dead code backwards, and does not
// need to be reapplied iteratively.
class DCE : public lower::Pass {
  public:
    constexpr std::string name() const override { return "dce"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace opt
} // namespace bonsai
