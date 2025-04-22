#pragma once

// #include "Expr.h"
// #include "Stmt.h"
#include "Type.h"
#include "WriteLoc.h"

namespace bonsai {
namespace ir {

bool equals(const Type &t0, const Type &t1);

struct TypeLessThan {
    bool operator()(const Type &t0, const Type &t1) const;
};

bool equals(const Expr &e0, const Expr &e1);

struct ExprLessThan {
    bool operator()(const Expr &e0, const Expr &e1) const;
};

struct WriteLocLessThan {
    bool operator()(const WriteLoc &w0, const WriteLoc &w1) const;
};

} // namespace ir
} // namespace bonsai
