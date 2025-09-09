#include "IR/Operators.h"

#include "IR/Equality.h"

namespace bonsai {
namespace ir {

Expr operator+(Expr a, Expr b) {
    return BinOp::make(BinOp::Add, std::move(a), std::move(b));
}

Expr operator-(Expr a, Expr b) {
    return BinOp::make(BinOp::Sub, std::move(a), std::move(b));
}

Expr operator*(Expr a, Expr b) {
    return BinOp::make(BinOp::Mul, std::move(a), std::move(b));
}

Expr operator/(Expr a, Expr b) {
    return BinOp::make(BinOp::Div, std::move(a), std::move(b));
}

Expr operator%(Expr a, Expr b) {
    return BinOp::make(BinOp::Mod, std::move(a), std::move(b));
}

Expr operator&&(Expr a, Expr b) {
    return BinOp::make(BinOp::LAnd, std::move(a), std::move(b));
}

Expr operator||(Expr a, Expr b) {
    return BinOp::make(BinOp::LOr, std::move(a), std::move(b));
}

Expr operator&(Expr a, Expr b) {
    return BinOp::make(BinOp::BwAnd, std::move(a), std::move(b));
}

Expr operator|(Expr a, Expr b) {
    return BinOp::make(BinOp::BwOr, std::move(a), std::move(b));
}

Expr operator^(Expr a, Expr b) {
    return BinOp::make(BinOp::Xor, std::move(a), std::move(b));
}

Expr operator==(Expr a, Expr b) {
    return BinOp::make(BinOp::Eq, std::move(a), std::move(b));
}

Expr operator!=(Expr a, Expr b) {
    return BinOp::make(BinOp::Neq, std::move(a), std::move(b));
}

Expr operator<=(Expr a, Expr b) {
    return BinOp::make(BinOp::Le, std::move(a), std::move(b));
}

Expr operator>=(Expr a, Expr b) {
    return BinOp::make(BinOp::Le, std::move(b), std::move(a));
}

Expr operator<(Expr a, Expr b) {
    return BinOp::make(BinOp::Lt, std::move(a), std::move(b));
}

Expr operator>(Expr a, Expr b) {
    return BinOp::make(BinOp::Lt, std::move(b), std::move(a));
}

Expr operator~(Expr a) { return UnOp::make(UnOp::Not, std::move(a)); }

Expr operator-(Expr a) { return UnOp::make(UnOp::Neg, std::move(a)); }

Expr select(Expr c, Expr t, Expr f) {
    return Select::make(std::move(c), std::move(t), std::move(f));
}

Expr distmax(Expr a, Expr b) {
    return GeomOp::make(GeomOp::distmax, std::move(a), std::move(b));
}

Expr distmin(Expr a, Expr b) {
    return GeomOp::make(GeomOp::distmin, std::move(a), std::move(b));
}

Expr intersects(Expr a, Expr b) {
    return GeomOp::make(GeomOp::intersects, std::move(a), std::move(b));
}

Expr contains(Expr a, Expr b) {
    return GeomOp::make(GeomOp::contains, std::move(a), std::move(b));
}

Expr filter(Expr predicate, Expr set) {
    return SetOp::make(SetOp::filter, std::move(predicate), std::move(set));
}

Expr argmin(Expr metric, Expr set) {
    return SetOp::make(SetOp::argmin, std::move(metric), std::move(set));
}

Expr map(Expr func, Expr set) {
    return SetOp::make(SetOp::map, std::move(func), std::move(set));
}

Expr product(Expr a, Expr b) {
    return SetOp::make(SetOp::product, std::move(a), std::move(b));
}

Expr abs(Expr a) { return Intrinsic::make(Intrinsic::abs, {std::move(a)}); }

Expr max(Expr a, Expr b) {
    return Intrinsic::make(Intrinsic::max, {std::move(a), std::move(b)});
}

Expr min(Expr a, Expr b) {
    return Intrinsic::make(Intrinsic::min, {std::move(a), std::move(b)});
}

Expr round(Expr a) { return Intrinsic::make(Intrinsic::round, {std::move(a)}); }

Expr sqr(Expr a) { return Intrinsic::make(Intrinsic::sqr, {std::move(a)}); }

Expr sqrt(Expr a) { return Intrinsic::make(Intrinsic::sqrt, {std::move(a)}); }

Expr norm(Expr a) { return Intrinsic::make(Intrinsic::norm, {std::move(a)}); }

Expr dot(Expr a, Expr b) {
    return Intrinsic::make(Intrinsic::dot, {std::move(a), std::move(b)});
}

Expr all(Expr a) { return VectorReduce::make(VectorReduce::And, a); }

Expr any(Expr a) { return VectorReduce::make(VectorReduce::Or, a); }

Expr cast(Type t, Expr e) {
    if (e.type().defined() && equals(t, e.type())) {
        return e;
    }
    return Cast::make(std::move(t), std::move(e));
}

} // namespace ir
} // namespace bonsai
