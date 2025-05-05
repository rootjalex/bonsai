#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <set>
#include <string>

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

ir::Stmt dce(ir::Stmt, const std::set<std::string> &mutable_func_args,
             const std::set<std::string> &se_functions);

} // namespace opt
} // namespace bonsai
