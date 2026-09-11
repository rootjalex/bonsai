#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Puts a function that returns a set back where it is called.
//
// A set is not a value the machine holds: a query's set is traversed, and the
// traversal is built over the whole query at once. So a call to a function
// returning one, made from inside a query -- `flatten(|p| geometry(p),
// prims)`, with `geometry(p : Primitive) -> set[Geometric]` pbrt's dispatch
// from a primitive to what it holds, named once and used by every query and by
// the element's extent -- is a name for part of that query, and this replaces
// it with the function's expression. It runs before anything reads a set
// expression, and it reaches the extents as well as the functions, since an
// extent is where such a function is most worth having a name.
//
// A set-valued function that is called nowhere is left as it was found, and
// an exported one is kept whether or not it is called: it is a query of its
// own, materialised for its caller (`collisions() = filter(..)`). One that was
// put back everywhere it was called and is not exported has nothing left to
// be, and is dropped.
class LowerSetFunctions : public Pass {
  public:
    const std::string name() const override { return "lower-set-functions"; }
    // Needs the extents, which only the program has.
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
