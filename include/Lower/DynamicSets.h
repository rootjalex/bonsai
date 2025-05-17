#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Any sets at this point in lowering are assumed to be dynamically sized. These
// are lowered to dynamically sized arrays.
class LowerDynamicSets : public Pass {
  public:
    const std::string name() const override { return "lower-dynamic-sets"; }

    ir::Program run(ir::Program, const CompilerOptions &) const override;
};

} // namespace lower
} // namespace bonsai
