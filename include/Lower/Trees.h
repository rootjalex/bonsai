#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

ir::Stmt build_base_scan(const std::string &name, const ir::BVH_t *bvh_t);

// TODO: nice description
class LowerTrees : public Pass {
  public:
    const std::string name() const override { return "lower-trees"; }

    // Requires full-program analysis (needs access to schedule).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
