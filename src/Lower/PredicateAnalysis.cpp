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

// The ordering predicate that is the strict/non-strict counterpart of `op`.
// `a <=_D b` is false exactly when `b <_D a`, and vice versa.
ir::GeomOp::OpType flip_strictness(ir::GeomOp::OpType op) {
    switch (op) {
    case ir::GeomOp::lex:
        return ir::GeomOp::ltx;
    case ir::GeomOp::ley:
        return ir::GeomOp::lty;
    case ir::GeomOp::lez:
        return ir::GeomOp::ltz;
    case ir::GeomOp::ltx:
        return ir::GeomOp::lex;
    case ir::GeomOp::lty:
        return ir::GeomOp::ley;
    case ir::GeomOp::ltz:
        return ir::GeomOp::lez;
    default:
        internal_error << "Not an ordering predicate: "
                       << ir::GeomOp::intrinsic_name(op);
    }
}

bool is_ordering(ir::GeomOp::OpType op) {
    switch (op) {
    case ir::GeomOp::lex:
    case ir::GeomOp::ley:
    case ir::GeomOp::lez:
    case ir::GeomOp::ltx:
    case ir::GeomOp::lty:
    case ir::GeomOp::ltz:
        return true;
    default:
        return false;
    }
}

// Algorithm 5: Geometric Upper Bounds. Following the paper's notation, `u` is
// a uniform operand, taken as it is, and `v` a varying one bounded by a volume
// V. `va` and `vb` are those volumes, and are undefined for uniform operands.
// An undefined result is the algorithm's `_ |-> true`: no useful necessary
// condition, so the predicate may always hold.
ir::Expr geom_upper_bound(ir::GeomOp::OpType op, const ir::Expr &a,
                          const ir::Expr &b, const ir::Expr &va,
                          const ir::Expr &vb) {
    const bool a_varying = va.defined();
    const bool b_varying = vb.defined();
    // Each operand, replaced by its bounding volume where it varies.
    const ir::Expr &ea = a_varying ? va : a;
    const ir::Expr &eb = b_varying ? vb : b;

    switch (op) {
    case ir::GeomOp::contains: {
        // Lines 5-7. Anything contained in the operand is also inside its
        // volume, so at the very least the two must overlap.
        if (a_varying && !b_varying) {
            // Line 6 gives intersects(V_v, u). This is the tighter form, and
            // still necessary: u lies inside v, which lies inside its own
            // volume. It is also what the paper itself uses for the
            // corresponding covers case on line 9.
            return ir::contains(ea, eb);
        }
        return ir::intersects(ea, eb);
    }
    case ir::GeomOp::covers: {
        // Lines 8-10. Everything a varying operand covers is inside its
        // volume, which line 9 states tightly; the other two cases keep only
        // that the operands must overlap.
        if (a_varying && !b_varying) {
            return ir::covers(ea, eb);
        }
        return ir::intersects(ea, eb);
    }
    case ir::GeomOp::touches:      // Lines 22-24.
    case ir::GeomOp::intersects: { // Lines 19-21.
        return ir::intersects(ea, eb);
    }
    case ir::GeomOp::disjoint: {
        // Lines 11-12. Disjointness is refuted only when the uniform operand
        // swallows the whole volume, which forces an overlap. The two cases
        // are the same relation spelled with the volume in the position its
        // operand held. With both operands varying nothing can be refuted.
        if (a_varying && b_varying) {
            return ir::Expr();
        }
        return a_varying ? ~ir::within(va, b) : ~ir::contains(a, vb);
    }
    case ir::GeomOp::within: {
        // Lines 13-15. `within(a, b)` is `a` inside `b`.
        if (a_varying && b_varying) {
            return ir::intersects(ea, eb);
        }
        if (b_varying) {
            // A uniform a inside a varying b must fit inside b's volume.
            return ir::within(a, vb);
        }
        return ir::intersects(va, b);
    }
    case ir::GeomOp::equals: {
        // Lines 16-18. Equality forces the uniform operand to fit inside the
        // varying one's volume, whichever side it is on. The two single-
        // varying cases are the same relation spelled two ways, each keeping
        // the volume in the position its operand held:
        // `within(u, V_v)` and `contains(V_v, u)`.
        if (a_varying && b_varying) {
            return ir::intersects(ea, eb);
        }
        return a_varying ? ir::contains(va, b) : ir::within(a, vb);
    }
    default: {
        // Ordering predicates. Section 6.3 notes only that these are
        // monotonic and bounded by rewriting varying arguments to their
        // bounding volumes. Monotonicity gives the usual comparison rule:
        // `a <=_D b` is necessary only if `b <_D a` fails on the volumes.
        internal_assert(is_ordering(op))
            << "No upper bound rule for: " << ir::GeomOp::intrinsic_name(op);
        return ~ir::ordering(flip_strictness(op), eb, ea);
    }
    }
}

// Algorithm 6, lines 1-13: Geometric Lower Bounds. An undefined result is the
// algorithm's `_ |-> false`: no sufficient condition, so the predicate can
// never be proven to hold over a whole subtree.
ir::Expr geom_lower_bound(ir::GeomOp::OpType op, const ir::Expr &a,
                          const ir::Expr &b, const ir::Expr &va,
                          const ir::Expr &vb) {
    const bool a_varying = va.defined();
    const bool b_varying = vb.defined();
    const ir::Expr &ea = a_varying ? va : a;
    const ir::Expr &eb = b_varying ? vb : b;

    switch (op) {
    case ir::GeomOp::contains: {
        // Line 5. A uniform a that swallows b's whole volume contains every
        // b in it. Nothing can be proven when a itself varies.
        return (!a_varying && b_varying) ? ir::contains(a, vb) : ir::Expr();
    }
    case ir::GeomOp::covers: {
        // Line 6.
        return (!a_varying && b_varying) ? ir::covers(a, vb) : ir::Expr();
    }
    case ir::GeomOp::disjoint: {
        // Lines 7-9. Disjointness is the one predicate that can be proven
        // with both operands varying: disjoint volumes have disjoint
        // contents.
        return ir::disjoint(ea, eb);
    }
    case ir::GeomOp::intersects: {
        // Lines 10-11. An operand whose entire volume sits inside the other
        // necessarily meets it.
        if (a_varying && b_varying) {
            return ir::Expr();
        }
        return b_varying ? ir::contains(a, vb) : ir::within(va, b);
    }
    case ir::GeomOp::within: {
        // Line 12. `a` is inside `b` whenever a's whole volume is.
        return (a_varying && !b_varying) ? ir::within(va, b) : ir::Expr();
    }
    case ir::GeomOp::equals:
    case ir::GeomOp::touches: {
        // Line 13. A bounding volume can never prove an exact boundary
        // relationship.
        return ir::Expr();
    }
    default: {
        // Ordering predicates, as above: the relation holds for every object
        // in the volumes exactly when it holds between the volumes.
        internal_assert(is_ordering(op))
            << "No lower bound rule for: " << ir::GeomOp::intrinsic_name(op);
        return ir::ordering(op, ea, eb);
    }
    }
}

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

    void visit(const ir::Extrema *node) override { set(node); }

    void visit(const ir::Var *node) override { set(node); }

    ir::Expr make_and(ir::Expr a, ir::Expr b) {
        if (!a.defined() || !b.defined()) {
            return ir::Expr();
        }
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

    ir::Expr make_or(ir::Expr a, ir::Expr b) {
        // A disjunction with an unbounded side is only as good as the other
        // side when that side settles the answer, which the constant folds
        // below cover; otherwise nothing can be said.
        if (is_const_one(a)) {
            return a;
        }
        if (is_const_one(b)) {
            return b;
        }
        if (!a.defined() || !b.defined()) {
            return ir::Expr();
        }
        if (is_const_zero(a)) {
            return b;
        }
        if (is_const_zero(b)) {
            return a;
        }
        return a || b;
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
        // Section 6.2: boolean and and or are monotonically increasing in
        // both arguments, so each bound is that operator applied to the
        // corresponding bounds. `&` and `|` on booleans mean the same thing
        // as `&&` and `||` here, and are what most predicates are written
        // with.
        case ir::BinOp::LAnd:
        case ir::BinOp::BwAnd:
        case ir::BinOp::LOr:
        case ir::BinOp::BwOr: {
            if (!node->type.is_bool()) {
                break; // TODO: handle non-booleans
            }
            const bool is_and =
                node->op == ir::BinOp::LAnd || node->op == ir::BinOp::BwAnd;
            Interval a = get(node->a);
            Interval b = get(node->b);
            if (a.is_single_point(node->a) && b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(is_and ? (a.min && b.min)
                                                         : (a.min || b.min));
            } else if (is_and) {
                interval.min = make_and(a.min, b.min);
                interval.max = make_and(a.max, b.max);
            } else {
                interval.min = make_or(a.min, b.min);
                interval.max = make_or(a.max, b.max);
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
                    ir::Expr cmp = b.min >= make_zero(b.min.type());
                    interval.min = select(cmp, e1, e2);
                    interval.max = select(cmp, e2, e1);
                }
                // else unbounded
            } else if (a.is_bounded() && b.is_bounded()) {
                // TODO: let exprs for linearity.

                ir::Expr low_high = a.min * b.max;
                ir::Expr low_low = a.min * b.min;
                ir::Expr high_low = a.max * b.min;
                ir::Expr high_high = a.max * b.max;

                // TODO: could do tons of casework, be stupid for now.

                interval.min =
                    min(min(low_low, low_high), min(high_low, high_high));
                interval.max =
                    max(max(low_low, low_high), max(high_low, high_high));
            }
            // TODO: for integers, need to handle overflow if defined.
            return;
        }
        case ir::BinOp::Div: {
            Interval a = get(node->a);
            Interval b = get(node->b);

            internal_assert(node->type.is_float())
                << "TODO: handle non-float division in predicate analysis: "
                << ir::Expr(node);

            // Do nothing with unbounded intervals
            if (!a.is_bounded() || !b.is_bounded()) {
                return;
            } else if (a.is_single_point(node->a) &&
                       b.is_single_point(node->b)) {
                interval = Interval::single_point(node);
                return;
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(a.min / b.min);
                return;
            }

            // Both are fully bounded, at least one is a true interval,
            // and the type is floating point.

            ir::Expr inf = ir::Extrema::make(node->type, ir::Extrema::inf);
            ir::Expr zero = make_zero(node->type);

            ir::Expr denom_positive = b.min > zero;
            ir::Expr denom_negative = b.max < zero;
            ir::Expr denom_contains_zero = b.min < zero && b.max > zero;
            // If ^ is true, bounds are infinite.

            ir::Expr num_nonneg = a.min >= zero;
            ir::Expr num_nonpos = a.max <= zero;
            ir::Expr num_spans = ~num_nonneg && ~num_nonpos;

            ir::Expr low_high = a.min / b.max;
            ir::Expr low_low = a.min / b.min;
            ir::Expr high_low = a.max / b.min;
            ir::Expr high_high = a.max / b.max;

            // TODO: could do tons of casework, be stupid for now.

            interval.min =
                min(min(low_low, low_high), min(high_low, high_high));
            interval.max =
                max(max(low_low, low_high), max(high_low, high_high));

            interval.min = select(denom_contains_zero, -inf, interval.min);
            interval.max = select(denom_contains_zero, inf, interval.max);
            return;
        }
        default: {
            break;
        }
        }
        internal_error << "TODO: implement predicate analysis on BinOp: "
                       << ir::Expr(node);
    }

    void visit(const ir::UnOp *node) override {
        switch (node->op) {
        case ir::UnOp::Not: {
            // Section 6.2: negation is monotonically decreasing, so it is
            // upper bounded by the negation of the lower bound and vice
            // versa.
            internal_assert(node->type.is_bool())
                << "TODO: predicate analysis of bitwise negation: "
                << ir::Expr(node);
            Interval a = get(node->a);
            if (a.is_single_point(node->a)) {
                interval = Interval::single_point(node);
                return;
            }
            if (a.min.defined()) {
                interval.max = ~a.min;
            }
            if (a.max.defined()) {
                interval.min = ~a.max;
            }
            return;
        }
        case ir::UnOp::Neg: {
            // Negation reverses the order of the bounds.
            Interval a = get(node->a);
            if (a.is_single_point(node->a)) {
                interval = Interval::single_point(node);
                return;
            }
            if (a.min.defined()) {
                interval.max = -a.min;
            }
            if (a.max.defined()) {
                interval.min = -a.max;
            }
            return;
        }
        }
    }

    void visit(const ir::Select *node) override {
        Interval c = get(node->cond);
        Interval t = get(node->tvalue);
        Interval f = get(node->fvalue);

        if (c.is_single_point()) {
            interval.min = select(c.min, t.min, f.min);
            interval.max = select(c.min, t.max, f.max);
        } else {
            interval.min =
                min(select(c.min, t.min, f.min), select(c.max, t.min, f.min));
            interval.max =
                max(select(c.min, t.max, f.max), select(c.max, t.max, f.max));
        }
    }
    RESTRICT_VISITOR(ir::Cast);
    RESTRICT_VISITOR(ir::Broadcast);
    RESTRICT_VISITOR(ir::VectorReduce);
    RESTRICT_VISITOR(ir::VectorShuffle);
    RESTRICT_VISITOR(ir::Ramp);
    RESTRICT_VISITOR(ir::Extract);
    RESTRICT_VISITOR(ir::Build);

    void visit(const ir::Access *node) override {
        Interval a = get(node->value);
        internal_assert(a.is_single_point(node->value))
            << "TODO: interval analysis of access on varying value: " << a.min
            << ", " << a.max << " of " << ir::Expr(node);
        interval = Interval::single_point(node);
    }

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
        case ir::Intrinsic::acos: {
            // acos is decreasing where it is defined, so the bounds cross
            // over: the smallest angle comes from the largest cosine. Outside
            // [-1, 1] it is a NaN, which is an unusable bound rather than a
            // wrong one -- the same answer log gives for a negative bound.
            internal_assert(node->args.size() == 1);
            Interval a = get(node->args[0]);
            if (a.has_upper_bound()) {
                interval.min =
                    ir::Intrinsic::make(node->op, {std::move(a.max)});
            }
            if (a.has_lower_bound()) {
                interval.max =
                    ir::Intrinsic::make(node->op, {std::move(a.min)});
            }
            return;
        }
        case ir::Intrinsic::round:
        // exp, log and atanh are increasing wherever they are defined, so the
        // same reasoning holds: the bounds of the result are the function of
        // the bounds. log of a non-positive lower bound gives -inf or a NaN,
        // which is an unusable bound rather than a wrong one, and so does atanh
        // of a bound outside (-1, 1). cosh is deliberately not here: it is even
        // rather than monotone, so an interval straddling zero has its minimum
        // in the middle rather than at an endpoint.
        case ir::Intrinsic::atanh:
        case ir::Intrinsic::exp:
        case ir::Intrinsic::log:
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
                interval.min = select(a.min >= 0, sqr(a.min),
                                      select(a.max <= 0, sqr(a.max), 0));
                interval.max = max(sqr(a.min), sqr(a.max));
            }
            return;
        }
        case ir::Intrinsic::max: {
            internal_assert(node->args.size() == 2);
            Interval a = get(node->args[0]);
            Interval b = get(node->args[1]);
            if (a.is_single_point(node->args[0]) &&
                b.is_single_point(node->args[1])) {
                interval = Interval::single_point(node);
            } else if (a.is_single_point() && b.is_single_point()) {
                interval = Interval::single_point(max(a.min, b.min));
            } else {
                if (a.has_lower_bound() && b.has_lower_bound()) {
                    interval.min = max(a.min, b.min);
                }
                if (a.has_upper_bound() && b.has_upper_bound()) {
                    interval.max = max(a.max, b.max);
                }
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

        // A varying operand with no bounding volume simply cannot be bounded.
        // That is a legitimate outcome, not an error: the caller sees an
        // undefined interval and emits an unguarded recursion.
        const bool a_unbounded = a_vol.has_value() && !a_vol->defined();
        const bool b_unbounded = b_vol.has_value() && !b_vol->defined();
        if (a_unbounded || b_unbounded) {
            interval = Interval{};
            return;
        }

        // Neither operand varies, so the expression is its own value.
        if (!a_vol.has_value() && !b_vol.has_value()) {
            set(node);
            return;
        }

        const ir::Expr va = a_vol.has_value() ? *a_vol : ir::Expr();
        const ir::Expr vb = b_vol.has_value() ? *b_vol : ir::Expr();

        if (is_geometric_metric(ir::GeomOp::intrinsic_name(node->op))) {
            // Algorithm 6, lines 15-25. Both distances are bounded below by
            // distmin and above by distmax over the bounding volumes.
            const ir::Expr a = va.defined() ? va : node->a;
            const ir::Expr b = vb.defined() ? vb : node->b;
            interval.min = ir::distmin(a, b);
            interval.max = ir::distmax(a, b);
            return;
        }

        make_bool_bounds();
        if (ir::Expr upper =
                geom_upper_bound(node->op, node->a, node->b, va, vb);
            upper.defined()) {
            interval.max = std::move(upper);
        }
        if (ir::Expr lower =
                geom_lower_bound(node->op, node->a, node->b, va, vb);
            lower.defined()) {
            interval.min = std::move(lower);
        }
    }

    RESTRICT_VISITOR(ir::SetOp);
    RESTRICT_VISITOR(ir::AggOp);
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
    // Go through get() rather than accepting directly, so that a tagged
    // interval on the expression itself is honoured. This matters for metrics
    // like `|p| p.id`, where the whole expression is the annotated field.
    return analysis.get(expr);
}

} // namespace lower
} // namespace bonsai
