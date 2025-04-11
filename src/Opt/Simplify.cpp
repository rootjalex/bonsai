#include "Opt/Simplify.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Utils.h"

#include "Lower/TopologicalOrder.h"

#include "Error.h"

#include <bit>
#include <concepts>
#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

// Bit casts `a` and `b` to type T, then applies `f`.
template <typename T, typename F>
T apply(F f, uint64_t a, uint64_t b) {
    return f(std::bit_cast<T>(a), std::bit_cast<T>(b));
}

// Attempts to constant fold the binary operations. Returns an undefined
// expression upon failure. A type parameter is optionally passed when
// interpreting a vector's broadcasted value.
//
// TODO(cgyurgyik): Support vectors and constant-sized array immediates.
template <typename F>
ir::Expr constant_fold(F f, ir::Expr a, ir::Expr b,
                       std::optional<ir::Type> type = {}) {
    internal_assert(ir::equals(a.type(), b.type()))
        << "a: " << a.type() << ", " << "b: " << b.type();
    if (!type.has_value()) {
        type = a.type();
    }
    std::optional<uint64_t> c_a = get_constant_value(a);
    std::optional<uint64_t> c_b = get_constant_value(b);
    if (!(c_a.has_value() && c_b.has_value())) {
        return ir::Expr();
    }
    if (type->is_scalar()) {
        if (type->is_int()) {
            return ir::IntImm::make(*type, apply<int64_t>(f, *c_a, *c_b));
        }
        if (type->is_uint()) {
            return ir::UIntImm::make(*type, apply<uint64_t>(f, *c_a, *c_b));
        }
        if (type->is_float()) {
            return ir::FloatImm::make(*type, apply<double>(f, *c_a, *c_b));
        }
    }
    if (const auto *vtype = type->as<ir::Vector_t>()) {
        ir::Expr result = constant_fold(f, a, b, vtype->etype);
        return !result.defined()
                   ? ir::Expr()
                   : ir::Broadcast::make(vtype->lanes, std::move(result));
    }
    return ir::Expr();
}

// Creates a new binary operation node with `a` and `b` if they've changed,
// otherwise returns the original `node`.
ir::Expr make(const ir::BinOp *node, ir::Expr a, ir::Expr b) {
    if (a.same_as(node->a) && b.same_as(node->b)) {
        return node;
    }
    return ir::BinOp::make(node->op, std::move(a), std::move(b));
}

struct Simplifier : ir::Mutator {
    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = mutate(node->a), b = mutate(node->b);
        internal_assert(ir::equals(a.type(), b.type()))
            << "a: " << a.type() << ", " << "b: " << b.type();

        ir::Type type = a.type();
        switch (node->op) {
        case ir::BinOp::OpType::Add: {
            if (ir::Expr e = constant_fold(std::plus<>{}, a, b); e.defined()) {
                return e;
            }
            if (is_const_zero(a)) {
                // 0 + b = b
                return b;
            }
            if (is_const_zero(b)) {
                // a + 0 = a
                return a;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Mul: {
            if (ir::Expr e = constant_fold(std::multiplies<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a) || is_const_zero(b)) {
                // x * 0 = 0
                return make_zero(std::move(type));
            }
            if (is_const_one(a)) {
                // x * 1 = x
                return b;
            }
            if (is_const_one(b)) {
                // 1 * x = x
                return a;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Div: {
            internal_assert(!is_const_zero(b)) << ir::Expr(node);
            if (ir::Expr e = constant_fold(std::divides<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_one(b)) {
                // a / 1 = a
                return a;
            }
            if (a.same_as(b)) {
                // a / a = 1
                return make_one(std::move(type));
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Sub: {
            if (ir::Expr e = constant_fold(std::minus<>{}, a, b); e.defined()) {
                return e;
            }
            if (is_const_zero(b)) {
                // a - 0 = 0
                return a;
            }
            // TODO(cgyurgyik): This checks for pointer equality, we want to
            // also check for semantic equality.
            if (a.same_as(b)) {
                // a - a = 0
                return make_zero(std::move(type));
            }
            if (is_const_zero(a)) {
                // 0 - a = -a
                return ir::UnOp::make(ir::UnOp::Neg, a);
            }
            return make(node, std::move(a), std::move(b));
        }
        default:
            return make(node, std::move(a), std::move(b));
        }
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        if (is_const(value) && node->type.is_scalar()) {
            return constant_cast(node->type, std::move(value));
        }
        if (equals(value.type(), node->type)) {
            // T v = ...; cast[[T]](v) = v
            return value;
        }
        if (value.same_as(node->value)) {
            return node;
        }
        return ir::Cast::make(node->type, std::move(value));
    }
};

} // namespace

/* static */ ir::Expr Simplify::simplify(ir::Expr e) {
    return Simplifier().mutate(std::move(e));
}

/* static */ ir::Stmt Simplify::simplify(ir::Stmt s) {
    return Simplifier().mutate(std::move(s));
}

ir::FuncMap Simplify::run(ir::FuncMap funcs) const {
    Simplifier lower;
    for (auto &[name, func] : funcs) {
        func->body = lower.mutate(func->body);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
