#pragma once

// specialize(): a bound loop's body copied per variant of a parameter.
//
// `render.specialize(integrator)`: `integrator` is a parameter of variant
// type -- which integrator the scene runs, fixed for the whole render -- and
// every `match` on it inside the sample loop is a switch on a value the
// compiler could have known. This directive makes it known: the body of
// each of the function's outermost parallel loops is copied once per
// variant of the type, the copy's `integrator` is the given one with its
// tag set to that variant's number, and the loop is entered through a
// dispatch on the tag, so each copy is one loop -- one kernel under a GPU
// bind, one thread body under a CPU bind, one sequential loop unbound -- in
// which the tag is a constant and the backend folds every dispatch on it to
// the one arm and deletes the rest.
// What ptxas then allocates registers for, and what the instruction cache
// holds, is the integrator the scene runs and not all four. pbrt's GPU
// build is one integrator by construction (the wavefront volpath); this is
// how a program that keeps all of pbrt's integrators compares like with
// like.
//
// It is Halide's `specialize()` -- a copy of a stage's loop nest under a
// condition, the general nest kept for the rest -- with the condition a
// variant's tag rather than a boolean, which is what a variant type makes
// natural: the copies are exhaustive, so there is no general nest to keep.
// In the language's terms it changes how the program runs and nothing of
// what it computes, each copy computing exactly what the original would
// have with that tag.
//
// The copies are of the loop as scheduled up to this directive: binds are
// already on the loop (they are tags set at conversion), and a loopify or
// vectorize written before it is in the body copied. A directive written
// after it that names one of the function's loops finds several of that
// name, so `specialize()` goes last among a function's directives.
//
// Only `Inline` storage is specialized (a tag beside the payload), which is
// how a parameter's variant type is stored; a `tagged_index` handle carries
// its tag in the top bits of one word and is not handled yet.

#include "IR/Program.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <map>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

void specialize_loops(FuncMap &fmap, const std::string &fname,
                      const std::string &param,
                      const std::map<std::string, ir::Program::AdtStorage>
                          &storages);

} // namespace ssa
} // namespace ir
} // namespace bonsai
