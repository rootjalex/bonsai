#pragma once

// A function bound to a hardware unit: `texture_filter.bind(TextureUnit,
// |t : ImageTexture, st : vec2f, dstdx : vec2f, dstdy : vec2f| ...)`.
//
// The program keeps the function as the algorithm -- pbrt's MIPMap filter,
// exact, running anywhere -- and the schedule says that on this hardware the
// function is what the lambda computes instead: the lambda's parameters are
// the function's, in order and type, its value the function's result, and
// inside it is the intrinsic the unit is reached by (`tex_sample_grad_2d`,
// which only the PTX backend lowers). This pass makes the function's body
// `return <the lambda's value>`, the lambda's parameter names replaced by
// the function's. First in the pipeline, so that every pass after -- the
// inliner above all -- sees the function as the unit runs it. The bind stays
// in the schedule as the record of what was done; the SSA conversion, which
// binds loops, passes over it (see ir::Bind::lambda).
#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

class LowerHardwareBinds : public Pass {
  public:
    const std::string name() const override { return "lower-hardware-binds"; }
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
