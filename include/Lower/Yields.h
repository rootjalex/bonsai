#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// For now, this takes functions with a single Yield and turns it into a Return.
// In the future, this should lower Yields/Scans into the appropriate code
// that inserts into output data structures (which can be difficult in the
// dynamic size cases).
class LowerYields : public Pass {
  public:
    constexpr std::string name() const override { return "lower-yields"; }

    ir::FuncMap run(ir::FuncMap funcs) const override;
};

} // namespace lower
} // namespace bonsai
