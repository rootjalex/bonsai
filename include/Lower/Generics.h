#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Lowers generics to their respective typed variant.
class LowerGeneric : public Pass {
  public:
    constexpr std::string name() const override { return "lower-generic"; }

    void run(ir::Program &program) const override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program) const;
};

} // namespace lower
} // namespace bonsai
