#pragma once

#include "CompilerOptions.h"
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
    const std::string name() const override { return "verify-option"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
