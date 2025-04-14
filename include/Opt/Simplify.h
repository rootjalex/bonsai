#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs peephole optimizations and the likes.
// e.g., A + 0 => A
class Simplify : public lower::Pass {
  public:
    constexpr std::string name() const override { return "simplify"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;

    static ir::Expr simplify(ir::Expr);
    static ir::Stmt simplify(ir::Stmt);
};

} // namespace opt
} // namespace bonsai
