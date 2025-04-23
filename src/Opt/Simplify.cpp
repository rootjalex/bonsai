#include "Opt/Simplify.h"

#include "Error.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <bit>
#include <string>
#include <unordered_map>

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
template <typename F>
ir::Expr constant_fold(F f, ir::Expr a, ir::Expr b,
                       std::optional<ir::Type> type = {}) {
    if (!(a.defined() && b.defined())) {
        return ir::Expr();
    }
    internal_assert(ir::equals(a.type(), b.type()))
        << "a: " << a.type() << ", " << "b: " << b.type();
    if (!type.has_value()) {
        type = a.type();
    }
    // Vector case.
    if (const auto *vector_type = type->as<ir::Vector_t>()) {
        std::vector<ir::Expr> values;
        ir::Type element_of = vector_type->etype;
        for (int i = 0, e = vector_type->lanes; i < e; ++i) {
            ir::Expr result = constant_fold(f,
                                            /*a=*/get_value_at(a, i),
                                            /*b=*/get_value_at(b, i),
                                            /*type=*/element_of);
            if (!result.defined()) {
                return ir::Expr();
            }
            values.push_back(std::move(result));
        }
        return ir::VecImm::make(std::move(values));
    }

    // Scalar case.
    internal_assert(type->is_scalar()) << *type;
    std::optional<uint64_t> c_a = get_constant_value(a),
                            c_b = get_constant_value(b);
    if (!(c_a.has_value() && c_b.has_value())) {
        return ir::Expr();
    }
    if (type->is_int()) {
        return ir::IntImm::make(*type, apply<int64_t>(f, *c_a, *c_b));
    }
    if (type->is_uint()) {
        return ir::UIntImm::make(*type, apply<uint64_t>(f, *c_a, *c_b));
    }
    if (type->is_float()) {
        return ir::FloatImm::make(*type, apply<double>(f, *c_a, *c_b));
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
    ir::Expr visit(const ir::Var *node) override {
        auto it = name_to_immediate.find(node->name);
        if (it == name_to_immediate.end()) {
            return node;
        }
        return it->second;
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = mutate(node->value);
        if (is_const(value)) {
            name_to_immediate[node->loc.base] = value;
        }
        if (value.same_as(node->value)) {
            return node;
        }
        return ir::LetStmt::make(node->loc, std::move(value));
    }

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
        case ir::BinOp::OpType::And: {
            if (ir::Expr e = constant_fold(std::logical_and<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a) || is_const_zero(b)) {
                // false && x = false
                return a;
            }
            if (is_const_one(a)) {
                // true && x = x
                return b;
            }
            if (is_const_one(b)) {
                // x && true = x
                return a;
            }
            if (ir::equals(a, b)) {
                // x && x = x
                return a;
            }
            return make(node, std::move(a), std::move(b));
        }

        case ir::BinOp::OpType::Or: {
            if (ir::Expr e = constant_fold(std::logical_or<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_one(a) || is_const_one(b)) {
                // true || x = true
                return a;
            }
            if (is_const_zero(a)) {
                // false || x = x
                return b;
            }
            if (is_const_zero(b)) {
                // x || false = x
                return a;
            }
            if (ir::equals(a, b)) {
                // x || x = x
                return a;
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

    ir::Expr visit(const ir::Build *node) override {
        bool changed = false, is_all_constants = true;
        std::vector<ir::Expr> values;
        for (int32_t i = 0, e = node->values.size(); i < e; ++i) {
            ir::Expr v = mutate(node->values[i]);
            changed |= v.same_as(node->values[i]);
            is_all_constants &= is_const(v);
            values.push_back(std::move(v));
        }
        if (node->type.is_vector() && is_all_constants && !values.empty()) {
            // x: i32 = 1; v: Build<i32x2>(x, (i32)2) => [1, 2]
            return ir::VecImm::make(std::move(values));
        }
        return changed ? ir::Build::make(node->type, std::move(values)) : node;
    }

    ir::Expr visit(const ir::Extract *node) override {
        ir::Expr v = mutate(node->vec), i = mutate(node->idx);
        if (const auto *broadcast = v.as<ir::Broadcast>()) {
            return broadcast->value;
        }
        std::optional<uint64_t> index = get_constant_value(i);
        if (v.is<ir::VecImm, ir::Build>()) {
            if (index.has_value()) {
                if (std::optional<uint64_t> c = get_constant_value(v, index)) {
                    return make_const(v.type().element_of(), *c);
                }
            }
        }
        if (v.same_as(node->vec) && i.same_as(node->idx)) {
            return node;
        }
        return ir::Extract::make(std::move(v), std::move(i));
    }

  private:
    // Mapping from a variable name to its immediate value. This assumes
    // variable shadowing is illegal; if this were to change, we'd need to
    // introduce a frame stack.
    std::unordered_map<std::string, ir::Expr> name_to_immediate;
};

} // namespace

/* static */ ir::Expr Simplify::simplify(ir::Expr e) {
    return Simplifier().mutate(std::move(e));
}

/* static */ ir::Stmt Simplify::simplify(ir::Stmt s) {
    return Simplifier().mutate(std::move(s));
}

ir::FuncMap Simplify::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        func->body = Simplify::simplify(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
