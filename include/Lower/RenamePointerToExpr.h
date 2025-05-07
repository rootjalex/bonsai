#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Gives a temporary variable for pointers to expressions (or r-values, in C++
// speak). This is undefined behavior for some backends, e.g., CUDA. For
// example,
//
//   foo(&build<Sphere>(...));
//
//  =>
//
//   x = build<Sphere>(...);
//   foo(&x);
class RenamePointerToExpr : public Pass {
  public:
    constexpr std::string name() const override { return "rpte"; }

    ir::FuncMap run(ir::FuncMap functions,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
