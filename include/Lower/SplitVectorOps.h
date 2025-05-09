#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Split actions on vectors into actions on scalars.
// Only really useful for the Packetization pass,
// this is required before packetizing.
class SplitVectorOps : public Pass {
  public:
    constexpr std::string name() const override { return "split-vector-ops"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
