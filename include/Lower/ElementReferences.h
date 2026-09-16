#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// Gives a reference to a stored element (ir::ElementRef_t, made by ir::RefTo)
// the representation its tree's layout affords: the indices that pick the
// element out of the storage the layout keeps it in. See the header comment
// in src/Lower/ElementReferences.cpp.
//
// Runs after LowerLayouts and LowerForEachs, once every element a traversal
// reaches is spelled as a read from the tree's storage, and before anything
// that needs to know what a reference is made of.
class LowerElementReferences : public Pass {
  public:
    const std::string name() const override {
        return "lower-element-references";
    }

    // Whole-program: the type of a reference is decided from every place one
    // is made, and is registered as a type when it needs a name.
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
