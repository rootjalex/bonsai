#pragma once

#include <iostream>
#include <map>

#include "Function.h"
#include "Type.h"

namespace bonsai {
namespace ir {

Expr operator+(Expr a, Expr b);
Expr operator-(Expr a, Expr b);
Expr operator*(Expr a, Expr b);
Expr operator/(Expr a, Expr b);
Expr operator%(Expr a, Expr b);
Expr operator&&(Expr a, Expr b);
Expr operator||(Expr a, Expr b);
Expr operator&(Expr a, Expr b);
Expr operator|(Expr a, Expr b);
Expr operator^(Expr a, Expr b);
Expr operator==(Expr a, Expr b);
Expr operator!=(Expr a, Expr b);
Expr operator<=(Expr a, Expr b);
Expr operator>=(Expr a, Expr b);
Expr operator<(Expr a, Expr b);
Expr operator>(Expr a, Expr b);

Expr operator~(Expr a);
Expr operator-(Expr a);

Expr select(Expr c, Expr t, Expr f);

// Geometric
Expr distmax(Expr a, Expr b);
Expr distmin(Expr a, Expr b);
Expr intersects(Expr a, Expr b);
Expr contains(Expr a, Expr b);

// Sets
Expr filter(Expr predicate, Expr set);
Expr argmin(Expr metric, Expr set);
Expr argmax(Expr metric, Expr set);
Expr minimum(Expr metric, Expr set);
Expr maximum(Expr metric, Expr set);
Expr any(Expr predicate, Expr set);
Expr all(Expr predicate, Expr set);
Expr map(Expr func, Expr set);
Expr product(Expr a, Expr b);

// Aggregates
Expr avg(Expr a);
Expr count(Expr a);
Expr prod(Expr a);
Expr sum(Expr a);
Expr reduce(Expr identity, Expr combiner, Expr a);

// `|a, b| a <op> b`, the combiner of a reduce built from a binary operator.
Expr binary_lambda(BinOp::OpType op, Type t);

// Rewrite an aggregation into the map-then-reduce form of Figure 2. `count`,
// for instance, maps every element to 1 and sums with identity 0. Returns the
// expression unchanged when it is already a `reduce`.
Expr expand_aggregate(const AggOp *agg);

Expr abs(Expr a);
Expr max(Expr a, Expr b);
Expr min(Expr a, Expr b);
Expr round(Expr a);
Expr sqr(Expr a);
Expr sqrt(Expr a);
Expr norm(Expr a);
Expr dot(Expr a, Expr b);

// Reductions
Expr all(Expr a);
Expr any(Expr a);

Expr cast(Type t, Expr e);

} // namespace ir
} // namespace bonsai
