#pragma once

// What a ray query on the RT cores reads, carried to the trace.
//
// A query bound to RTCore becomes a trace and a set of programs the hardware
// runs at the hits (Lower/Trees.cpp): the programs read the tree's storage,
// the pools its elements live in, the textures an alpha test samples. The
// programs are called by nobody in the program, so nothing threads what they
// read to where the trace is made -- yet the trace is where they run, and
// the launch of the raygen program around it is what hands the hardware its
// data (CodeGen_OptiX reads a program's parameters out of the launch's
// parameters, by name). So this pass gives every `rt_trace` the programs'
// reads as further operands -- the programs' parameters other than the
// context, which is what they read once LowerExterns has threaded every
// extern to them through everything they call: the tree, and the pools a
// `tagged_index` layout made of an element's arms, which did not exist when
// the trace was made -- so that LowerExterns, running once more after,
// threads them up through the trace's callers to the loop that launches, as
// it threads every extern.
//
// After the LowerExterns that follows LowerADTs, the last pass to name new
// storage, and before another LowerExterns.

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

class LowerRTCoreReads : public Pass {
  public:
    const std::string name() const override { return "lower-rtcore-reads"; }
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
