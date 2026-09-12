#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Performs function inlining.
//
// A function whose body has mutable locals of its own is copied only when
// `with_locals` is set, or when the program asks with `[[inline]]`. Copied,
// its state becomes a set of variables at every call site and CSE sees through
// none of them, so two calls with the same arguments are two computations for
// good; as a call it is one value that CSE can merge. The pipeline therefore
// runs this twice: once without, so that the calls such a function's callers
// hid become visible and CSE can merge them, and once with, so that the one
// call that is left is copied in and the values it computes can be shared with
// the code around it and hoisted out of the loops it sits in.
class Inline : public lower::Pass {
  public:
    explicit Inline(bool with_locals = false) : with_locals(with_locals) {}

    // Two names, because the pass manager keeps one pass per name: a
    // pipeline that ran this twice under one name would run the same round
    // twice.
    const std::string name() const override {
        return with_locals ? "inline-locals" : "inline";
    }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;

  private:
    bool with_locals;
};

} // namespace opt
} // namespace bonsai
