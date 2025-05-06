#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs fusion on set operations, e.g.
// filter(|x| : foo(x), filter(|y| : bar(y), data))
// -> filter(|y| : foo(y) && bar(y), data)
// and
// map(|x| : foo(x), map(|y| : bar(y), data))
// -> map(|y| : foo(bar(y)), data)
class Fusion : public lower::Pass {
  public:
    constexpr std::string name() const override { return "fusion"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;

    static ir::Stmt fuse_within_stmt(const ir::Stmt &stmt,
                                     const ir::FuncMap &funcs);
};

} // namespace opt
} // namespace bonsai
