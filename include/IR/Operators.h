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
// TODO: the rest

// Geometric
Expr distmax(Expr a, Expr b);
Expr distmin(Expr a, Expr b);
Expr intersects(Expr a, Expr b);
Expr contains(Expr a, Expr b);

// Sets
Expr filter(Expr predicate, Expr set);
Expr argmin(Expr metric, Expr set);
Expr map(Expr func, Expr set);

Expr sqrt(Expr a);
Expr norm(Expr a);
Expr dot(Expr a, Expr b);

// Reductions
Expr all(Expr a);
Expr any(Expr a);

Expr cast(Type t, Expr e);

} // namespace ir
} // namespace bonsai
