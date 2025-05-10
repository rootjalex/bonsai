#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Adds random_state arg to all funcs that can reach a call to random()
// This is a target-specific mutable state parameter (e.g. thread-specific in
// CUDA).
class LowerRandom : public Pass {
  public:
    constexpr std::string name() const override { return "lower-random"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

static const std::string rng_state_name = "_rng_state";

} // namespace lower
} // namespace bonsai
