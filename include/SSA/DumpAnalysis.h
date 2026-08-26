#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace ir {
namespace ssa {

// Builds the SSA form of every function and prints the results of the control
// flow analyses in SSA/Analysis.h, without running any rewrite or code
// generation.
//
// This exists so those analyses can be tested directly -- the properties they
// have to satisfy (dominance, post-dominance, control dependence, loop
// nesting, and the compactness of the block index required by partial
// linearization) are not observable from generated code alone.
class DumpSSAAnalysis : public lower::Pass {
  public:
    const std::string name() const override { return "ssa-dump-analysis"; }

    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
