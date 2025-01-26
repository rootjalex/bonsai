#pragma once

#include "IR/Expr.h"

namespace bonsai {
namespace lower {

ir::Expr cross_product(const ir::Expr &a, const ir::Expr &b);

ir::Expr argmax(const ir::Expr &a);

} // namespace lower
} // namespace bonsai
