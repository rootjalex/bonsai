#pragma once

// #include "Expr.h"
// #include "Stmt.h"
#include "Type.h"

namespace bonsai {
namespace ir {

bool equals(const Type &t0, const Type &t1);

struct TypeLessThan {
    bool operator()(const Type &t0, const Type &t1) const;
};

} // namespace ir
} // namespace bonsai
