#pragma once

// #include "Expr.h"
// #include "Stmt.h"
#include "Type.h"
#include "WriteLoc.h"

#include <unordered_set>

namespace bonsai {
namespace ir {

bool equals(const Type &t0, const Type &t1);

struct TypeLessThan {
    bool operator()(const Type &t0, const Type &t1) const;
};

// TODO(cgyurgyik): There are many mathematical identities that can also apply
// here, e.g., x + y = y + x (for integral types, at least). Either we want to
// look for those, or more likely apply some sort of canonicalization.
bool equals(const Expr &e0, const Expr &e1);

struct ExprLessThan {
    bool operator()(const Expr &e0, const Expr &e1) const;
};

struct WriteLocLessThan {
    bool operator()(const WriteLoc &w0, const WriteLoc &w1) const;
};

// TODO(cgyurgyik): Provide a real hash function.
struct ExprHashImpl {
    std::size_t operator()(const ir::Expr &expr) const { return 0; }
};

struct ExprEqualImpl {
    bool operator()(const ir::Expr &a, const ir::Expr &b) const {
        return ir::equals(a, b);
    }
};

using CseSet = std::unordered_set<ir::Expr, ExprHashImpl, ExprEqualImpl>;

} // namespace ir
} // namespace bonsai
