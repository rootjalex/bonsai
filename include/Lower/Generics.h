#pragma once

#include "IR/Program.h"
#include "Lower/Passes.h"

namespace bonsai {
namespace lower {

// Lowers generics to their respective typed variant.
class LowerGeneric : Pass {
  public:
    constexpr std::string_view name() override { return "lower-generic"; }

    void run(ir::Program &program) override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program);
};

} // namespace lower
} // namespace bonsai
