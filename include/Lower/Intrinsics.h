#pragma once

#include "IR/Expr.h"

namespace bonsai {
namespace lower {

ir::Expr cross_product(const ir::Expr &a, const ir::Expr &b);

ir::Expr dot_product(const ir::Expr &a, const ir::Expr &b);

// The length of `a`, as the square root of its dot product with itself. Built
// from the pieces rather than from the `norm` intrinsic, which is what this
// replaces.
ir::Expr norm(const ir::Expr &a);

ir::Expr argmax(const ir::Expr &a);

} // namespace lower
} // namespace bonsai
