#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Verifies that options are always legally accessed. Due to its static nature,
// this results in incompleteness, i.e., there may exist valid option
// dereferences that are marked invalid, but enables the compiler to heavily
// optimize `option` types. For example,
//
//      i: option[i32] = foo();
//      if i { use(*i); } // LEGAL
//      use(*i);          // ILLEGAL
class VerifyOptions : public Pass {
  public:
    constexpr std::string name() const override { return "verify-option"; }

    void run(ir::Program &program) const override { program = lower(program); }

  private:
    ir::Program lower(const ir::Program &program) const;
};

} // namespace lower
} // namespace bonsai
