#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Turns variant types into whatever their layout says they are stored as.
//
// After this there are no ADT_t types, no Construct and no MatchVariant: a
// value of one is an ordinary value of whatever Lower/ADTLayout.h chose, and a
// match is a test on it. Nothing downstream -- the SSA form, the backends --
// needs to know that variants exist at all, which is the same arrangement
// Option_t has with Lower/Options.cpp.
//
// The pass does not know how a variant is stored. It asks the layout to build
// one, to say which variant a value is, and to read a field, and does nothing
// else with the representation. A new layout is an implementation of
// ADTLayout and no change here.
class LowerADTs : public Pass {
  public:
    const std::string name() const override { return "lower-adts"; }

    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
