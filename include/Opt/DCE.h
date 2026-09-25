#pragma once

#include "CompilerOptions.h"
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
    const std::string name() const override { return "dce"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

// `context` names the function the statement came from, for the messages an
// internal assertion prints; it changes nothing else. `kept` names lets that
// are read from outside the body -- by a schedule directive,
// ir::Function::schedule_uses -- and stay whether or not the body reads
// them.
ir::Stmt dce(ir::Stmt, const std::set<std::string> &mutable_func_args,
             const std::set<std::string> &se_functions,
             const std::string &context = "",
             const std::set<std::string> &kept = {});

} // namespace opt
} // namespace bonsai
