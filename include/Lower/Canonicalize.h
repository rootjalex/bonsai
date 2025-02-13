#pragma once

#include "IR/Program.h"
#include "Lower/Passes.h"

namespace bonsai {
namespace lower {

class Canonicalize : Pass {
  public:
    constexpr std::string_view name() override { return "canonicalize"; }

    void run(ir::Program &program) override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program);
};

} // namespace lower
} // namespace bonsai
