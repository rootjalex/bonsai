#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs dead code elimination on `program`.
class DCE : public lower::Pass {
  public:
    constexpr std::string name() const override { return "dce"; }

    void run(ir::Program &program) const override { program = dce(program); }

  private:
    ir::Program dce(const ir::Program &program) const;
};

} // namespace opt
} // namespace bonsai
