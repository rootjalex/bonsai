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
Expr operator^(Expr a, Expr b);
Expr operator==(Expr a, Expr b);
Expr operator!=(Expr a, Expr b);
Expr operator<=(Expr a, Expr b);
Expr operator>=(Expr a, Expr b);
Expr operator<(Expr a, Expr b);
Expr operator>(Expr a, Expr b);
// TODO: the rest

Expr distmax(Expr a, Expr b);
Expr distmin(Expr a, Expr b);
Expr intersects(Expr a, Expr b);
Expr contains(Expr a, Expr b);

Expr filter(Expr predicate, Expr set);
Expr argmin(Expr metric, Expr set);

Expr sqrt(Expr a);
Expr norm(Expr a);
Expr dot(Expr a, Expr b);

} // namespace ir
} // namespace bonsai
