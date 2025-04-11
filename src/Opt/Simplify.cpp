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
// expression upon failure.
template <typename F>
ir::Expr simplify(F f, ir::Expr a, ir::Expr b) {
    if (ir::equals(a.type(), b.type())) {
        return ir::Expr();
    }
    ir::Type type = a.type();
    if (type.is_scalar()) {
        return ir::Expr();
    }
    std::optional<uint64_t> c_a = get_constant_value(a);
    std::optional<uint64_t> c_b = get_constant_value(b);
    if (c_a.has_value() && c_b.has_value()) {
        if (type.is_int()) {
            return ir::IntImm::make(std::move(type),
                                    apply<int64_t>(f, *c_a, *c_b));
        }
        if (type.is_uint()) {
            return ir::UIntImm::make(std::move(type),
                                     apply<uint64_t>(f, *c_a, *c_b));
        }
        if (type.is_float()) {
            return ir::FloatImm::make(std::move(type),
                                      apply<double>(f, *c_a, *c_b));
        }
    }
    return ir::Expr();
}

// TODO(cgyurgyik): better name for this? I'd rather not duplicate this code in
// each case.
ir::Expr v(const ir::BinOp *node, ir::Expr a, ir::Expr b) {
    if (a.same_as(node->a) && b.same_as(node->b)) {
        return node;
    }
    return ir::BinOp::make(node->op, std::move(a), std::move(b));
}

struct Simplifier : ir::Mutator {
    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = mutate(node->a), b = mutate(node->b);
        if (!ir::equals(a.type(), b.type())) {
            // Conservatively return if these do not share the same type.
            return v(node, std::move(a), std::move(b));
        }
        ir::Type type = a.type();
        switch (node->op) {
        case ir::BinOp::OpType::Add: {
            if (ir::Expr e = simplify(std::plus<>{}, a, b); e.defined()) {
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
            return v(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Mul: {
            if (ir::Expr e = simplify(std::multiplies<>{}, a, b); e.defined()) {
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
            return v(node, std::move(a), std::move(b));
        }
        default:
            return v(node, std::move(a), std::move(b));
        }
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        if (is_const(value) && node->type.is_scalar()) {
            return constant_cast(node->type, std::move(value));
        }
        if (value.same_as(node->value)) {
            return node;
        }
        return ir::Cast::make(node->type, std::move(value));
    }
};

} // namespace

/* static */ ir::Expr Simplify::simplify(ir::Expr e) {
    Simplifier lower;
    return lower.mutate(std::move(e));
}

/* static */ ir::Stmt Simplify::simplify(ir::Stmt s) {
    Simplifier lower;
    return lower.mutate(std::move(s));
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
