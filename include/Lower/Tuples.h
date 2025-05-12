#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Tuples are lowered to structs, because LLVM does not support unnamed structs.
// This also lowers Extracts on tuples to field accesses, and builds of tuples
// to builds of the corresponding structs.
// Note that this requires that all Extracts on tuples are
// compile-time-constants.
class LowerTuples : public Pass {
  public:
    const std::string name() const override { return "lower-tuples"; }

    // Requires full-program analysis (needs access to types + funcs).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
