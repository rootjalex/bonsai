#pragma once

#include "SSA/AnalyzeDivergence.h"
#include "SSA/SSA.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// Replaces every varying value of vector type in the region with one value
// per component, so that what is left for widening is scalars only.
//
// A lane holding a `vector[f32, 3]` cannot simply be widened: a gang of eight
// would need a vector of vectors, which no SIMD target has. The way round it
// is the layout ispc calls structure of arrays (section 5.2): keep the three
// components apart, and widen each of them into its own gang-wide vector. So
// `sub(a, b)` on per-lane 3-vectors becomes three subtractions, reading a
// component of a per-lane vector becomes reading the value that component was
// split into, and reading an array of them becomes one strided access per
// component.
//
// Uniform vectors are left alone. They are the same for every lane, so they
// stay a single vector value, and where one meets a split value its
// components are read out individually.
//
// What the split did to the region's interfaces.
struct SplitResult {
    // The names the varying entry-block arguments were split into, since
    // splitting a function's parameters changes its signature: a varying
    // `vector[f32, 3]` parameter becomes three `f32` parameters.
    std::set<std::string> parameters;

    // For each block ending in a call, how many values each of the callee's
    // original arguments now takes. A callee is specialized against its own
    // parameters, so whoever does that has to expand them the same way the
    // arguments here were expanded.
    std::map<std::string, std::vector<uint32_t>> call_shapes;
};
// `already_wide` names instructions that are gang-wide vectors rather than
// per-lane ones -- the lane indices, which are built as a vector to begin
// with. Splitting one of those would be splitting the gang itself.
SplitResult
split_aggregates(Function &func, const std::string &entry,
                 const Divergence &divergence,
                 const std::set<const Instruction *> &already_wide = {});

} // namespace ssa
} // namespace ir
} // namespace bonsai
