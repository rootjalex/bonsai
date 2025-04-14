#include "Lower/PredicateAnalysis.h"

#include "Error.h"

#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

namespace bonsai {
namespace lower {

namespace {

struct PredicateAnalysis : public ir::Visitor {
    Interval interval;
    const VolumeMap &bounds;

    PredicateAnalysis(const VolumeMap &bounds) : bounds(bounds) {}

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
        expr.accept(this);
        return std::move(interval);
    }

    void visit(const ir::IntImm *node) override { set(node); }

    void visit(const ir::UIntImm *node) override { set(node); }

    void visit(const ir::FloatImm *node) override { set(node); }

    void visit(const ir::BoolImm *node) override { set(node); }

    void visit(const ir::Infinity *node) override { set(node); }

    void visit(const ir::Var *node) override { set(node); }

    void visit(const ir::BinOp *node) override {
        internal_error << "TODO: implement predicate analysis on BinOp: "
                       << ir::Expr(node);
    }

    void visit(const ir::UnOp *node) override {
        internal_error << "TODO: implement predicate analysis on UnOp: "
                       << ir::Expr(node);
    }

    void visit(const ir::Select *node) override {
        internal_error << "TODO: implement predicate analysis on Select: "
                       << ir::Expr(node);
    }

    void visit(const ir::Cast *node) override {
        internal_error << "TODO: implement predicate analysis on Cast: "
                       << ir::Expr(node);
    }

    void visit(const ir::Broadcast *node) override {
        internal_error << "TODO: implement predicate analysis on Broadcast: "
                       << ir::Expr(node);
    }

    void visit(const ir::VectorReduce *node) override {
        internal_error << "TODO: implement predicate analysis on VectorReduce: "
                       << ir::Expr(node);
    }

    void visit(const ir::VectorShuffle *node) override {
        internal_error
            << "TODO: implement predicate analysis on VectorShuffle: "
            << ir::Expr(node);
    }

    void visit(const ir::Ramp *node) override {
        internal_error << "TODO: implement predicate analysis on Ramp: "
                       << ir::Expr(node);
    }

    void visit(const ir::Extract *node) override {
        internal_error << "TODO: implement predicate analysis on Extract: "
                       << ir::Expr(node);
    }

    void visit(const ir::Build *node) override {
        internal_error << "TODO: implement predicate analysis on Build: "
                       << ir::Expr(node);
    }

    void visit(const ir::Access *node) override {
        internal_error << "TODO: implement predicate analysis on Access: "
                       << ir::Expr(node);
    }

    void visit(const ir::Unwrap *node) override {
        internal_error << "TODO: implement predicate analysis on Unwrap: "
                       << ir::Expr(node);
    }

    void visit(const ir::Intrinsic *node) override {
        internal_error << "TODO: implement predicate analysis on Intrinsic: "
                       << ir::Expr(node);
    }

    void visit(const ir::Lambda *node) override {
        internal_error << "TODO: implement predicate analysis on Lambda: "
                       << ir::Expr(node);
    }

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

        switch (node->op) {
        case ir::GeomOp::intersects: {
            ir::Expr a = a_varying ? *a_vol : node->a;
            ir::Expr b = b_varying ? *b_vol : node->b;

            interval.max = ir::intersects(a, b);

            // TODO: handle lower bound? doesn't work for rays...
            interval.min = ir::Expr();

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
                interval.min = ir::Expr();
            } else {
                // Both varying! no way to prove always true.
                interval.min = ir::Expr();
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

    void visit(const ir::SetOp *node) override {
        internal_error << "TODO: implement predicate analysis on SetOp: "
                       << ir::Expr(node);
    }

    void visit(const ir::Call *node) override {
        internal_error << "TODO: implement predicate analysis on Call: "
                       << ir::Expr(node);
    }

    void visit(const ir::Instantiate *node) override {
        internal_error << "TODO: implement predicate analysis on Instantiate: "
                       << ir::Expr(node);
    }
};

} // namespace

Interval predicate_analysis(const ir::Expr &expr, const VolumeMap &bounds) {
    PredicateAnalysis analysis(bounds);
    expr.accept(&analysis);
    return analysis.interval;
}

} // namespace lower
} // namespace bonsai
