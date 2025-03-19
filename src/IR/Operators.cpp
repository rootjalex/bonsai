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
    return BinOp::make(BinOp::And, std::move(a), std::move(b));
}

Expr operator||(Expr a, Expr b) {
    return BinOp::make(BinOp::Or, std::move(a), std::move(b));
}

Expr operator^(Expr a, Expr b) {
    return BinOp::make(BinOp::Xor, std::move(a), std::move(b));
}

Expr operator==(Expr a, Expr b) {
    return BinOp::make(BinOp::Eq, std::move(a), std::move(b));
}

Expr operator<=(Expr a, Expr b) {
    return BinOp::make(BinOp::Le, std::move(a), std::move(b));
}

Expr operator<(Expr a, Expr b) {
    return BinOp::make(BinOp::Lt, std::move(a), std::move(b));
}

Expr distance(Expr a, Expr b) {
    return ir::GeomOp::make(ir::GeomOp::distance, std::move(a), std::move(b));
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
