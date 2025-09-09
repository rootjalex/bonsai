#include "Lower/PredicateAnalysis.h"

#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

bool Interval::is_single_point(const ir::Expr &a) const {
    return is_bounded() && min.same_as(a) && max.same_as(a);
}

bool Interval::is_single_point() const {
    return is_bounded() && min.same_as(max);
}

bool Interval::has_upper_bound() const { return max.defined(); }

bool Interval::has_lower_bound() const { return min.defined(); }

bool Interval::is_bounded() const {
    return has_lower_bound() && has_upper_bound();
}

void Interval::include(const ir::Expr &e) {
    // TODO: Do Halide-style simplifications if necessary
    if (e.defined()) {
        if (max.defined()) {
            max = ir::max(e, std::move(max));
        }
        if (min.defined()) {
            min = ir::min(e, std::move(min));
        }
    } else {
        min = e;
        max = e;
    }
}

namespace {

struct PredicateAnalysis : public ir::Visitor {
    Interval interval;
    const VolumeMap &bounds;
    const IntervalMap &intervals;

    PredicateAnalysis(const VolumeMap &bounds, const IntervalMap &intervals)
        : bounds(bounds), intervals(intervals) {}

    // empty -> non-varying
    // undefined -> varying, but no bounding volume
    // defined -> varying with bounding volume
    std::optional<ir::Expr> bound(const std::string &name) const {
        const auto iter = bounds.find(name);
        if (iter != bounds.cend()) {
            return iter->second;
        } else {
            return {};
        }
    }

    void set(ir::Expr expr) {
        interval.min = expr;
        interval.max = std::move(expr);
    }

    void set(ir::Expr low, ir::Expr high) {
        interval.min = std::move(low);
        interval.max = std::move(high);
    }

    Interval get(const ir::Expr &expr) {
        const auto &iter = intervals.find(expr);
        if (iter != intervals.cend()) {
            return iter->second;
        }
        // Otherwise recurse.
        expr.accept(this);
        Interval value = std::move(interval);
        interval.min = ir::Expr();
        interval.max = ir::Expr();
        return value;
    }

    void make_bool_bounds() {
        interval.min = ir::BoolImm::make(false);
        interval.max = ir::BoolImm::make(true);
    }

    void visit(const ir::IntImm *node) override { set(node); }

    void visit(const ir::UIntImm *node) override { set(node); }

    void visit(const ir::FloatImm *node) override { set(node); }

    void visit(const ir::BoolImm *node) override { set(node); }

    void visit(const ir::Infinity *node) override { set(node); }

    void visit(const ir::Var *node) override { set(node); }

    ir::Expr make_and(ir::Expr a, ir::Expr b) {
        if (is_const_one(a)) {
            return b;
        }
        if (is_const_one(b)) {
            return a;
        }
        if (is_const_zero(a)) {
            return a;
        }
        if (is_const_zero(b)) {
            return b;
        }
        return a && b;
    }

    // Handle Lt/Le
    void visit_compare(const ir::BinOp *node) {
        Interval a = get(node->a);
        if (!a.has_upper_bound() && !a.has_lower_bound()) {
            make_bool_bounds();
            return;
        }
        Interval b = get(node->b);
        if (!b.has_upper_bound() && !b.has_lower_bound()) {
            make_bool_bounds();
            return;
        }
        // Initially unbounded.
        make_bool_bounds();

        // a.max <(=) b.min implies a <(=) b, so a <(=) b is at least
        // as true as a.max <(=) b.min. This does not depend on a's
        // lower bound or b's upper bound.
        if (a.has_upper_bound() && b.has_lower_bound()) {
            interval.min = ir::BinOp::make(node->op, a.max, b.min);
        }

        // a <(=) b implies a.min <(=) b.max, so a <(=) b is at most
        // as true as a.min <(=) b.max. This does not depend on a's
        // upper bound or b's lower bound.
        if (a.has_lower_bound() && b.has_upper_bound()) {
            interval.max = ir::BinOp::make(node->op, a.min, b.max);
        }
        return;
    }

    void visit(const ir::BinOp *node) override {
        switch (node->op) {
        case ir::BinOp::Lt:
        case ir::BinOp::Le: {
            visit_compare(node);
            return;
        }
        case ir::BinOp::Eq: {
            Interval a = get(node->a);
            if (!a.has_upper_bound() && !a.has_lower_bound()) {
                make_bool_bounds();
                return;
            }
            Interval b = get(node->b);
            if (!b.has_upper_bound() && !b.has_lower_bound()) {
                make_bool_bounds();
                return;
            }

            if (a.is_single_point(node->a) && b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
                return;
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(a.min == b.min);
                return;
            }

            // Initially unbounded.
            make_bool_bounds();

            // A sufficient condition is that all bounds are equal.
            if (a.is_bounded() && b.is_bounded()) {
                interval.min =
                    (a.min == a.max) && (b.min == b.max) && (a.min == b.min);
            }

            // A necessary condition is that the ranges overlap.
            if (a.is_bounded() && b.is_bounded()) {
                interval.max = a.min <= b.max && b.min <= a.max;
            } else if (a.has_upper_bound() && b.has_lower_bound()) {
                // a.min <= b.max is implied if a.min = -inf or b.max = +inf.
                interval.max = b.min <= a.max;
            } else if (a.has_lower_bound() && b.has_upper_bound()) {
                // b.min <= a.max is implied if a.max = +inf or b.min = -inf.
                interval.max = a.min <= b.max;
            }
            return;
        }
        case ir::BinOp::LAnd: {
            if (!node->type.is_bool()) {
                break; // TODO: handle non-booleans
            }
            Interval a = get(node->a);
            Interval b = get(node->b);
            // And is monotonic increasing in both args
            if (a.is_single_point(node->a) && b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(a.min && b.min);
            } else {
                interval.min = make_and(a.min, b.min);
                interval.max = make_and(a.max, b.max);
            }
            return;
        }
        case ir::BinOp::Add: {
            Interval a = get(node->a);
            Interval b = get(node->b);
            if (a.is_single_point(node->a) && b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(a.min + b.min);
            } else {
                // TODO: for integers, need to handle overflow if defined.
                if (a.has_lower_bound() && b.has_lower_bound()) {
                    interval.min = a.min + b.min;
                }
                if (a.has_upper_bound() && b.has_upper_bound()) {
                    interval.max = a.max + b.max;
                }
            }
            return;
        }
        case ir::BinOp::Sub: {
            Interval a = get(node->a);
            Interval b = get(node->b);
            if (a.is_single_point(node->a) && b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(a.min - b.min);
            } else {
                // TODO: for integers, need to handle overflow if defined.
                if (a.has_lower_bound() && b.has_upper_bound()) {
                    interval.min = a.min - b.max;
                }
                if (a.has_upper_bound() && b.has_lower_bound()) {
                    interval.max = a.max - b.min;
                }
            }
            return;
        }
        case ir::BinOp::Mul: {
            Interval a = get(node->a);
            Interval b = get(node->b);

            // Move constants to the right
            if (a.is_single_point() && !b.is_single_point()) {
                std::swap(a, b);
            }

            if (a.is_single_point(node->a) && b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(a.min * b.min);
            } else if (b.is_single_point()) {
                ir::Expr e1 = a.has_lower_bound() ? a.min * b.min : a.min;
                ir::Expr e2 = a.has_upper_bound() ? a.max * b.min : a.max;
                if (is_const_zero(b.min)) {
                    interval = b;
                } else if (is_positive_const(b.min) || node->type.is_uint()) {
                    interval.min = std::move(e1);
                    interval.max = std::move(e2);
                } else if (is_negative_const(b.min)) {
                    interval.min = std::move(e2);
                    interval.max = std::move(e1);
                } else if (a.is_bounded()) {
                    // Sign of b is unknown
                    ir::Expr cmp =
                        b.min >= make_zero(b.min.type().element_of());
                    interval.min = select(cmp, e1, e2);
                    interval.max = select(cmp, e2, e1);
                }
                // else unbounded
            } else if (a.is_bounded() && b.is_bounded()) {
                // TODO: let exprs for linearity.
            }
            // TODO: for integers, need to handle overflow if defined.
            return;
        }
        default: {
            break;
        }
        }
        internal_error << "TODO: implement predicate analysis on BinOp: "
                       << ir::Expr(node);
    }

    RESTRICT_VISITOR(ir::UnOp);
    RESTRICT_VISITOR(ir::Select);
    RESTRICT_VISITOR(ir::Cast);
    RESTRICT_VISITOR(ir::Broadcast);
    RESTRICT_VISITOR(ir::VectorReduce);
    RESTRICT_VISITOR(ir::VectorShuffle);
    RESTRICT_VISITOR(ir::Ramp);
    RESTRICT_VISITOR(ir::Extract);
    RESTRICT_VISITOR(ir::Build);
    RESTRICT_VISITOR(ir::Access);
    RESTRICT_VISITOR(ir::Unwrap);

    void visit(const ir::Intrinsic *node) override {
        switch (node->op) {
        case ir::Intrinsic::abs: {
            internal_assert(node->args.size() == 1);
            Interval a = get(node->args[0]);
            if (a.is_bounded()) {
                if (ir::equals(a.min, a.max)) {
                    interval = Interval::single_point(ir::abs(a.min));
                } else {
                    interval.min =
                        cast_to(node->type,
                                ir::max(a.min, -ir::min(make_zero(a.min.type()),
                                                        a.max)));
                    a.min = abs(a.min);
                    a.max = abs(a.max);
                    interval.max = ir::max(a.min, a.max);
                }
            } else {
                if (a.has_lower_bound()) {
                    // If a is strictly positive, then abs(a) is strictly
                    // positive.
                    interval.min = cast_to(node->type,
                                           max(make_zero(a.min.type()), a.min));
                } else if (a.has_upper_bound()) {
                    // If a is strictly negative, then abs(a) is strictly
                    // positive.
                    interval.min = cast_to(
                        node->type, -min(make_zero(a.max.type()), a.max));
                } else {
                    interval.min = make_zero(node->type);
                }
                // If the argument is unbounded on one side, then the max is
                // unbounded.
                interval.max = ir::Expr();
            }
            return;
        }
        case ir::Intrinsic::round:
        case ir::Intrinsic::sqrt: {
            // For monotonic, pure, single-argument functions, we can
            // make two calls for the min and the max.
            internal_assert(node->args.size() == 1);
            Interval a = get(node->args[0]);
            if (a.has_lower_bound()) {
                interval.min =
                    ir::Intrinsic::make(node->op, {std::move(a.min)});
            }
            if (a.has_upper_bound()) {
                interval.max =
                    ir::Intrinsic::make(node->op, {std::move(a.max)});
            }
            return;
        }
        case ir::Intrinsic::sqr: {
            internal_assert(node->args.size() == 1);
            Interval a = get(node->args[0]);
            if (a.is_bounded()) {
                interval.min = select(a.min >= 0, a.min * a.min,
                                      select(a.max <= 0, a.max * a.max, 0));
                interval.max = max(a.min * a.min, a.max * a.max);
            }
            return;
        }
        default: {
            internal_error << "TODO: predicate analysis of expression: "
                           << ir::Expr(node);
        }
        }
    }

    RESTRICT_VISITOR(ir::Generator);
    RESTRICT_VISITOR(ir::Lambda);

    void visit(const ir::GeomOp *node) override {
        const ir::Var *a_var = node->a.as<ir::Var>();
        const ir::Var *b_var = node->b.as<ir::Var>();
        internal_assert(a_var && b_var)
            << "TODO: support non-variable geometric ops in predicate analysis:"
            << ir::Expr(node);

        const auto a_vol = bound(a_var->name);
        const auto b_vol = bound(b_var->name);

        internal_assert(!a_vol.has_value() || a_vol->defined())
            << "LHS of geom op is varying but has no bounding volume: "
            << ir::Expr(node);
        internal_assert(!b_vol.has_value() || b_vol->defined())
            << "RHS of geom op is varying but has no bounding volume: "
            << ir::Expr(node);

        const bool a_varying = a_vol.has_value();
        const bool b_varying = b_vol.has_value();

        if (!a_varying && !b_varying) {
            set(node);
            return;
        }

        make_bool_bounds();

        switch (node->op) {
        case ir::GeomOp::intersects: {
            ir::Expr a = a_varying ? *a_vol : node->a;
            ir::Expr b = b_varying ? *b_vol : node->b;

            interval.max = ir::intersects(a, b);

            // TODO: handle lower bound? doesn't work for rays...

            return;
        }
        case ir::GeomOp::distmin: {
            ir::Expr a = a_varying ? *a_vol : node->a;
            ir::Expr b = b_varying ? *b_vol : node->b;

            interval.min = ir::distmin(a, b);

            // TODO: handle upper bound?
            interval.max = ir::distmax(a, b);

            return;
        }
        case ir::GeomOp::contains: {
            if (!a_varying) {
                // If a contains b's volume, a definitely contains b
                interval.min = ir::contains(node->a, *b_vol);
                // if a intersects b's volume, a could contain b.
                interval.max = ir::intersects(node->a, *b_vol);
            } else if (!b_varying) {
                // If a's volume fully contains b, could be true
                // otherwise, can't be true, because there is some space b
                // exists that a does not.
                interval.max = ir::contains(*a_vol, node->b);
                // No way to prove this is always true if a is varying.
            } else {
                // Both varying! no way to prove always true.
                // But can only be true if the volumes intersect/overlap in some
                // way.
                interval.max = ir::intersects(*a_vol, *b_vol);
            }
            return;
        }
        default: {
            internal_error << "TODO: predicate analysis for: "
                           << ir::Expr(node);
        }
        }
    }

    RESTRICT_VISITOR(ir::SetOp);
    RESTRICT_VISITOR(ir::Call);
    RESTRICT_VISITOR(ir::Instantiate);
    RESTRICT_VISITOR(ir::PtrTo);
    RESTRICT_VISITOR(ir::Deref);
    RESTRICT_VISITOR(ir::AtomicAdd);
};

} // namespace

Interval predicate_analysis(const ir::Expr &expr, const VolumeMap &bounds,
                            const IntervalMap &intervals) {
    PredicateAnalysis analysis(bounds, intervals);
    expr.accept(&analysis);
    return analysis.interval;
}

} // namespace lower
} // namespace bonsai
