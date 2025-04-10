#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"
#include "Utils.h"

#include <string>

namespace bonsai {
namespace lower {

// Flattens n-dimensional structures to 1-dimensional structures, where n > 1.
class Flatten : public Pass {
  public:
    constexpr std::string name() const override { return "flatten"; }

    ir::FuncMap run(ir::FuncMap functions) const override;
};

} // namespace lower
} // namespace bonsai
