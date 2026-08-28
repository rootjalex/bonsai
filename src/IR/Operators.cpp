#include "IR/Operators.h"

#include "IR/Equality.h"
#include "IR/Printer.h"

#include "Utils.h"

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

Expr covers(Expr a, Expr b) {
    return GeomOp::make(GeomOp::covers, std::move(a), std::move(b));
}

Expr disjoint(Expr a, Expr b) {
    return GeomOp::make(GeomOp::disjoint, std::move(a), std::move(b));
}

// Named to avoid colliding with ir::equals, the structural comparison.
Expr equals_geom(Expr a, Expr b) {
    return GeomOp::make(GeomOp::equals, std::move(a), std::move(b));
}

Expr touches(Expr a, Expr b) {
    return GeomOp::make(GeomOp::touches, std::move(a), std::move(b));
}

Expr within(Expr a, Expr b) {
    return GeomOp::make(GeomOp::within, std::move(a), std::move(b));
}

Expr ordering(GeomOp::OpType op, Expr a, Expr b) {
    internal_assert(op == GeomOp::lex || op == GeomOp::ley ||
                    op == GeomOp::lez || op == GeomOp::ltx ||
                    op == GeomOp::lty || op == GeomOp::ltz)
        << "Not an ordering predicate: " << GeomOp::intrinsic_name(op);
    return GeomOp::make(op, std::move(a), std::move(b));
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

Expr argmax(Expr metric, Expr set) {
    return SetOp::make(SetOp::argmax, std::move(metric), std::move(set));
}

Expr minimum(Expr metric, Expr set) {
    return SetOp::make(SetOp::minimum, std::move(metric), std::move(set));
}

Expr maximum(Expr metric, Expr set) {
    return SetOp::make(SetOp::maximum, std::move(metric), std::move(set));
}

Expr any(Expr predicate, Expr set) {
    return SetOp::make(SetOp::any, std::move(predicate), std::move(set));
}

Expr all(Expr predicate, Expr set) {
    return SetOp::make(SetOp::all, std::move(predicate), std::move(set));
}

Expr product(Expr a, Expr b) {
    return SetOp::make(SetOp::product, std::move(a), std::move(b));
}

Expr avg(Expr a) { return AggOp::make(AggOp::avg, std::move(a)); }

Expr count(Expr a) { return AggOp::make(AggOp::count, std::move(a)); }

Expr prod(Expr a) { return AggOp::make(AggOp::prod, std::move(a)); }

Expr sum(Expr a) {
    internal_assert(a.type().defined()) << a;
    if (a.type().as<Set_t>()) {
        return AggOp::make(AggOp::sum, std::move(a));
    } else if (a.type().as<Vector_t>()) {
        return VectorReduce::make(VectorReduce::Add, std::move(a));
    }
    internal_error << a;
}

Expr reduce(Expr identity, Expr combiner, Expr a) {
    return AggOp::make(std::move(identity), std::move(combiner), std::move(a));
}

Expr binary_lambda(BinOp::OpType op, Type t) {
    Expr a = Var::make(t, "_a");
    Expr b = Var::make(t, "_b");
    Expr body = BinOp::make(op, std::move(a), std::move(b));
    return Lambda::make({{"_a", t}, {"_b", t}}, std::move(body));
}

Expr expand_aggregate(const AggOp *agg) {
    internal_assert(agg) << "expand_aggregate received null";
    if (agg->op == AggOp::reduce) {
        return agg;
    }

    const Type elem_t = agg->a.type().element_of();
    switch (agg->op) {
    case AggOp::count: {
        // count(S) = reduce(0, +, map(|x| 1, S))
        const Type t = agg->type;
        Expr one = Lambda::make({{"_x", elem_t}}, make_one(t));
        return reduce(make_zero(t), binary_lambda(BinOp::Add, t),
                      map(std::move(one), agg->a));
    }
    case AggOp::sum: {
        // sum(S) = reduce(0, +, S)
        return reduce(make_zero(elem_t), binary_lambda(BinOp::Add, elem_t),
                      agg->a);
    }
    case AggOp::prod: {
        // prod(S) = reduce(1, *, S)
        return reduce(make_one(elem_t), binary_lambda(BinOp::Mul, elem_t),
                      agg->a);
    }
    case AggOp::avg: {
        // avg needs a pair reduction (running sum and count) which the
        // combiner machinery does not build yet.
        internal_error << "[unimplemented] avg over a set: " << Expr(agg);
    }
    case AggOp::reduce: {
        break; // handled above
    }
    }
    internal_error << "Unknown aggregation: " << Expr(agg);
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
