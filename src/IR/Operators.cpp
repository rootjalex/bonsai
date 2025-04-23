#include "IR/Operators.h"

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

Expr operator&&(Expr a, Expr b) {
    return BinOp::make(BinOp::LAnd, std::move(a), std::move(b));
}

Expr operator||(Expr a, Expr b) {
    return BinOp::make(BinOp::LOr, std::move(a), std::move(b));
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

Expr distmax(Expr a, Expr b) {
    return ir::GeomOp::make(ir::GeomOp::distmax, std::move(a), std::move(b));
}

Expr distmin(Expr a, Expr b) {
    return ir::GeomOp::make(ir::GeomOp::distmin, std::move(a), std::move(b));
}

Expr intersects(Expr a, Expr b) {
    return ir::GeomOp::make(ir::GeomOp::intersects, std::move(a), std::move(b));
}

Expr contains(Expr a, Expr b) {
    return ir::GeomOp::make(ir::GeomOp::contains, std::move(a), std::move(b));
}

Expr filter(Expr predicate, Expr set) {
    return ir::SetOp::make(ir::SetOp::filter, std::move(predicate),
                           std::move(set));
}

Expr argmin(Expr metric, Expr set) {
    return ir::SetOp::make(ir::SetOp::argmin, std::move(metric),
                           std::move(set));
}

} // namespace ir
} // namespace bonsai
