#pragma once

#include "Expr.h"
#include "Layout.h"
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

// Orders expressions structurally: a total order, so that a map keyed on it
// iterates the same way every time. Comparing costs a walk of the two trees
// down to their first difference; where only lookup is needed and iteration
// order is not, ExprHash and ExprEquals below are much cheaper.
struct ExprLessThan {
    bool operator()(const Expr &e0, const Expr &e1) const;
};

// Structural hashes, consistent with equals(): two values equals() calls
// equal hash the same, and what the comparison ignores the hash ignores
// too. Computed once per node and cached on it (IRNode::structural_hash),
// so hashing a tree whose children have been hashed costs a few multiplies.
uint64_t hash(const Type &t);
uint64_t hash(const Expr &e);

// For std::unordered_map and std::unordered_set keyed on expressions.
struct ExprHash {
    size_t operator()(const Expr &e) const { return size_t(hash(e)); }
};
struct ExprEquals {
    bool operator()(const Expr &e0, const Expr &e1) const {
        return equals(e0, e1);
    }
};

bool equals(const Layout &l0, const Layout &l1);

struct LayoutLessThan {
    bool operator()(const Layout &l0, const Layout &l1) const;
};

struct WriteLocLessThan {
    bool operator()(const WriteLoc &w0, const WriteLoc &w1) const;
};

} // namespace ir
} // namespace bonsai
