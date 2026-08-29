#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Turns a parfor a schedule bound to hardware into a launch of it.
//
// A `bind()` records what a loop is to run on and changes nothing else; this
// is where that becomes code. The loop's body moves into a function of its
// own, everything it reads from around it moves into a context struct passed
// alongside, and the loop itself becomes one call handing that function to
// whatever runs it:
//
//   parfor i in start:end:stride         func closure(ctx : Context*, j) {
//     body                        =>       let i = start + stride * j in body
//                                        }
//                                        ctx = { ...free variables... }
//                                        launch (end - start + stride - 1)/stride
//                                            closure(ctx)
//
// A parfor with no binding is left alone: it is emitted as an ordinary
// sequential loop, which is what "may run in any order" allows.
//
// This runs after the SSA conversion, so the bodies it moves have already been
// through the rest of lowering. The functions it invents are built in the
// shape that leaves them in -- pointers for what Lower/Mutability.cpp would
// have made pointers -- since nothing runs afterwards to do it for them.
class LowerBindings : public Pass {
  public:
    const std::string name() const override { return "lower-bindings"; }

    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
