#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Inserts externs as (const) arguments to all functions that reference them.
// Externs are added in the order they were parsed, to the end of a function.
// This is particularly relevant for exported functions
// TODO(ajr): should we just get rid of externs and be explicit with function
// args?
class LowerExterns : public Pass {
  public:
    constexpr std::string name() const override { return "lower-externs"; }

    // Requires full-program analysis (needs access to externs).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
