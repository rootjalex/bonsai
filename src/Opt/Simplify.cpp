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

struct Simplifier : ir::Mutator {
    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = node->a, b = node->b;
        if (!ir::equals(a.type(), b.type())) {
            // Conservatively return if these do not share the same type.
            return node;
        }
        std::optional<uint64_t> c_a = get_constant_value(a);
        std::optional<uint64_t> c_b = get_constant_value(b);
        switch (node->op) {
        case ir::BinOp::OpType::Add: {
            if (ir::Expr e = simplify(std::plus<>{}, a, b); e.defined()) {
                return e;
            }
            if (c_a.has_value() && *c_a == 0) {
                // 0 + b = b
                return b;
            }
            if (c_b.has_value() && *c_b == 0) {
                // a + 0 = a
                return a;
            }
            return node;
        }
        case ir::BinOp::OpType::Mul: {
            if (ir::Expr e = simplify(std::multiplies<>{}, a, b); e.defined()) {
                return e;
            }
            if ((c_a.has_value() && *c_a == 0) ||
                (c_b.has_value() && *c_b == 0)) {
                // x * 0 = 0
                return make_zero(a.type());
            }
            return node;
        }
        default:
            return node;
        }
    }

    ir::Expr visit(const ir::Cast *node) override {
        if (is_const(node->value) && node->type.is_scalar()) {
            return constant_cast(node->type, node->value);
        }
        return node;
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
