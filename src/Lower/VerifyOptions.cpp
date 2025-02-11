#include "Lower/VerifyOptions.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Scope.h"
#include "Utils.h"

#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

using WriteLocSet = std::set<ir::WriteLoc, ir::WriteLocLessThan>;

// List of locations that are options
struct OptionSets {
    // Positive locations are safe to deref,
    // negative are explicitly not safe to deref.
    WriteLocSet positive, negative;
};

// Convert an expression, e.g. `a.field0.field1` into a `WriteLoc`.
ir::WriteLoc read_to_writeloc(const ir::Expr &expr) {
    if (expr.is<ir::Var>()) {
        const ir::Var *var = expr.as<ir::Var>();
        return ir::WriteLoc(var->name, var->type);
    } else if (expr.is<ir::Access>()) {
        const ir::Access *acc = expr.as<ir::Access>();
        ir::WriteLoc rec = read_to_writeloc(acc->value);
        rec.add_struct_access(acc->field);
        return rec;
    }
    // TODO(ajr): handle Index as well.
    internal_error << "Cannot convert to WriteLoc: " << expr;
    return ir::WriteLoc();
}

OptionSets get_option_sets(const ir::Expr &expr) {
    internal_assert(expr.type().is_bool()) << expr;
    if (const ir::Cast *cast = expr.as<ir::Cast>()) {
        if (cast->value.type().is<ir::Option_t>()) {
            ir::WriteLoc singleton = read_to_writeloc(cast->value);
            return OptionSets{.positive = {std::move(singleton)}};
        }
        return {}; // Don't peak through bool casts.
    }

    if (const ir::BinOp *node = expr.as<ir::BinOp>()) {
        switch (node->op) {
        // sets(a && b) = union(sets(a), sets(b))
        case ir::BinOp::And: {
            OptionSets a = get_option_sets(node->a);
            OptionSets b = get_option_sets(node->b);
            a.positive.insert(std::make_move_iterator(b.positive.begin()),
                              std::make_move_iterator(b.positive.end()));
            a.negative.insert(std::make_move_iterator(b.negative.begin()),
                              std::make_move_iterator(b.negative.end()));
            return a;
        }
        default:
            // TODO(rootjalex): Is there something we can do with a || b?
            // Consider that !(a || b) -> !a && !b = {.negative={a, b}}, which
            // we do not handle.
            return {};
        }
    }

    if (const ir::UnOp *node = expr.as<ir::UnOp>()) {
        switch (node->op) {
        case ir::UnOp::Not: {
            OptionSets a = get_option_sets(node->a);
            return OptionSets{.positive = std::move(a.negative),
                              .negative = std::move(a.positive)};
        }
        default:
            return {};
        }
    }

    return {};
}

// TODO(cgyurgyik): This is very simple static analysis. Other static analysis
// can be performed here to validate options, e.g.,
//      i: option[i32] = 42;
//      use(*i); // Legal, but will result in error.
class OptionVisitor : public ir::Visitor {
  public:
  private:
    // Tracks which options can be safely dereferenced in the current stack
    // frame.
    std::vector<OptionSets> frames;
    WriteLocSet always_safe;

    bool is_safe_to_deref(const ir::Expr &expr) const {
        ir::WriteLoc loc = read_to_writeloc(expr);
        for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
            if (it->positive.count(loc)) {
                return true;
            }
        }
        return always_safe.count(loc);
    }

    void push_frame() { frames.emplace_back(); }

    void push_frame(OptionSets options) {
        frames.emplace_back(std::move(options));
    }

    void replace_frame(OptionSets options) { frames.back() = options; }

    void pop_frame() { frames.pop_back(); }

    void make_always_safe(WriteLocSet safe) {
        always_safe.insert(safe.begin(), safe.end());
    }

    void visit(const ir::IfElse *node) override {
        // positives in the condition are safe to access in the then_body.
        OptionSets options = get_option_sets(node->cond);
        push_frame(options);
        node->then_body.accept(this);

        OptionSets swapped = {.positive = std::move(options.negative),
                              .negative = std::move(options.positive)};

        // negatives in the condition are safe to access in the else_body
        if (node->else_body.defined()) {
            replace_frame(swapped);
            node->else_body.accept(this);
        }
        pop_frame();

        // If then_body always returns, negatives are now always safe.
        // If else_body always returns, positives are now always safe.
        const bool then_returns = ir::always_returns(node->then_body);
        const bool else_returns =
            node->else_body.defined() && ir::always_returns(node->else_body);

        if (then_returns && else_returns) {
            return;
        } else if (then_returns) {
            make_always_safe(std::move(swapped.positive));
        } else if (else_returns) {
            make_always_safe(std::move(swapped.negative));
        }
    }

    void visit(const ir::Cast *node) override {
        ir::Visitor::visit(node);
        if (const ir::Option_t *as_opt =
                node->value.type().as<ir::Option_t>()) {
            if (node->type.is_bool()) {
                return;
            }
            internal_assert(ir::equals(as_opt->etype, node->type))
                << "Dereference of option[" << as_opt->etype << " into "
                << node->type << " is invalid.";
            internal_assert(is_safe_to_deref(node->value))
                << "illegal dereference of `" << node->value << ": "
                << node->type << "`";
        }
    }
};

} // namespace

void verify_options(const ir::Program &program) {
    for (const auto &[_, f] : program.funcs) {
        internal_assert(f->body.defined());
        OptionVisitor visitor;
        f->body.accept(&visitor);
    }
}

} // namespace lower
} // namespace bonsai
