#include "Opt/Simplify.h"

#include "Error.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <bit>
#include <functional>
#include <string>
#include <unordered_map>

namespace bonsai {
namespace opt {

namespace {

uint64_t log2(uint64_t value) {
    internal_assert(value > 0) << value;
    return std::bit_width(value) - 1;
}
// Bit casts `a` and `b` to type T, then applies `f`.
template <typename T, typename F>
T apply(F f, uint64_t a, uint64_t b) {
    return f(std::bit_cast<T>(a), std::bit_cast<T>(b));
}

template <typename T, typename C>
const T *is_op(ir::Expr e, C code) {
    if (const auto *v = e.as<T>()) {
        if (v->op == code) {
            return v;
        }
    }
    return nullptr;
}

// Attempts to constant fold the binary operations. Returns an undefined
// expression upon failure. A type parameter is optionally passed when
// interpreting a vector's broadcasted value.
// TODO(bonsai/issues/120): Add constant-fold floating point support.
// TODO(bonsai/issues/119): Support overflow arithmetic as Halide does:
// https://github.com/halide/Halide/blob/main/src/IRMatch.h#L919
template <typename F>
ir::Expr constant_fold_integral(F f, ir::Expr a, ir::Expr b,
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
            ir::Expr result = constant_fold_integral(f,
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
ir::Expr make(const ir::UnOp *node, ir::Expr a) {
    if (a.same_as(node->a)) {
        return node;
    }
    return ir::UnOp::make(node->op, std::move(a));
}

struct Simplifier : ir::Mutator {
    ir::Expr visit(const ir::Var *node) override {
        auto it = name_to_immediate.find(node->name);
        if (it == name_to_immediate.end()) {
            return node;
        }
        return it->second;
    }

    ir::Expr visit(const ir::UnOp *node) override {
        ir::Expr a = mutate(node->a);
        const ir::Type type = a.type();
        const ir::Expr zero = make_zero(type);
        switch (node->op) {
        case ir::UnOp::OpType::Neg:
            if (ir::Expr e = constant_fold_integral(std::minus<>{}, zero, a);
                e.defined()) {
                // -x <=> 0 - x
                return e;
            }
            if (auto *op = is_op<ir::UnOp>(a, ir::UnOp::OpType::Neg)) {
                // -(-x) = x
                return op->a;
            }
            return make(node, std::move(a));
        case ir::UnOp::OpType::Not:
            if (const std::optional<int64_t> v = get_constant_value(a)) {
                if (*v == 0) {
                    // !false = true
                    return make_const(type, 1);
                }
                if (*v == 1) {
                    // !true = false
                    return make_const(type, 0);
                }
            }
            if (auto *op = is_op<ir::UnOp>(a, ir::UnOp::OpType::Not)) {
                // !(!x) = x
                return op->a;
            }
            return make(node, std::move(a));
        }
    }

    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = mutate(node->a), b = mutate(node->b);
        internal_assert(ir::equals(a.type(), b.type()))
            << "a: " << a.type() << ", " << "b: " << b.type();

        const ir::Type type = a.type();
        const ir::Expr zero = make_zero(type), one = make_one(type);
        switch (node->op) {
        case ir::BinOp::OpType::Add: {
            if (ir::Expr e = constant_fold_integral(std::plus<>{}, a, b);
                e.defined()) {
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
            if (ir::Expr e = constant_fold_integral(std::multiplies<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a) || is_const_zero(b)) {
                // x * 0 = 0
                return zero;
            }
            if (is_const_one(a)) {
                // x * 1 = x
                return b;
            }
            if (is_const_one(b)) {
                // 1 * x = x
                return a;
            }

            if (type.is_int_or_uint()) {
                std::optional<int64_t> c_a = get_constant_value(a);
                if (c_a.has_value() && is_power_of_two(*c_a)) {
                    // n * x -> x << log2(n), where n is a power of 2.
                    return ir::BinOp::make(ir::BinOp::OpType::Shl, b,
                                           log2(*c_a));
                }
                std::optional<int64_t> c_b = get_constant_value(b);
                if (c_b.has_value() && is_power_of_two(*c_b)) {
                    // x * n -> x << log2(n), where n is a power of 2.
                    return ir::BinOp::make(ir::BinOp::OpType::Shl, a,
                                           log2(*c_b));
                }
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Div: {
            internal_assert(!is_const_zero(b)) << ir::Expr(node);
            if (ir::Expr e = constant_fold_integral(std::divides<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_one(b)) {
                // a / 1 = a
                return a;
            }
            if (a.same_as(b)) {
                // a / a = 1
                return one;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Sub: {
            if (ir::Expr e = constant_fold_integral(std::minus<>{}, a, b);
                e.defined()) {
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
                return zero;
            }
            if (is_const_zero(a)) {
                // 0 - a = -a
                return -a;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Mod: {
            if (ir::Expr e = constant_fold_integral(std::modulus<>{}, a, b);
                e.defined()) {
                return e;
            }
            std::optional<uint64_t> c_b = get_constant_value(b);
            if (c_b.has_value() && is_power_of_two(*c_b)) {
                // x % 2^n -> x & (2^n - 1)
                return a & make_const(type, *c_b - 1);
            }
        }
        case ir::BinOp::OpType::LAnd: {
            if (ir::Expr e = constant_fold_integral(std::logical_and<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a) || is_const_zero(b)) {
                // false && x = false
                return make_zero(type);
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

        case ir::BinOp::OpType::LOr: {
            if (ir::Expr e = constant_fold_integral(std::logical_or<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_one(a) || is_const_one(b)) {
                // true || x = true
                return make_one(type);
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
        case ir::BinOp::OpType::Xor: {
            if (ir::Expr e = constant_fold_integral(std::bit_xor<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a)) {
                // 0 ^ x = x
                return b;
            }
            if (is_const_zero(b)) {
                // x ^ 0 = x
                return a;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::BwAnd: {
            if (ir::Expr e = constant_fold_integral(std::bit_and<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a) || is_const_zero(b)) {
                // x & 0 = 0
                return zero;
            }
            if (is_const_all_ones(a)) {
                // ~0 & x  = x
                return b;
            }
            if (is_const_all_ones(b)) {
                // x & ~0  = x
                return a;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::BwOr: {
            if (ir::Expr e = constant_fold_integral(std::bit_or<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a)) {
                // 0 | x = x
                return b;
            }
            if (is_const_zero(b)) {
                // x | 0 = x
                return a;
            }
            if (is_const_all_ones(a) || is_const_all_ones(b)) {
                // ~0 | x = ~0
                return make_all_ones(type);
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
        } else if (const auto *map = as_map(v)) {
            internal_assert(map->b.type().is<ir::Array_t>());
            return mutate(call(map->a, ir::Extract::make(map->b, i)));
        } else if (const auto *gen = v.as<ir::Generator>()) {
            if (gen->op == ir::Generator::iter) {
                return i; // just the index.
            }
            // TODO(ajr): handle range() simplification
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

    ir::Expr visit(const ir::Access *node) override {
        ir::Expr value = mutate(node->value);

        if (const ir::Build *build = value.as<ir::Build>()) {
            const ir::Struct_t *struct_t =
                node->value.type().as<ir::Struct_t>();
            internal_assert(struct_t);
            const size_t idx = find_struct_index(node->field, struct_t->fields);
            internal_assert(idx < build->values.size());
            return build->values[idx];
        }

        if (value.same_as(node->value)) {
            return node;
        }
        return ir::Access::make(node->field, std::move(value));
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

    ir::Stmt visit(const ir::Sequence *node) override {
        bool changed = false;
        std::vector<ir::Stmt> stmts;
        stmts.reserve(node->stmts.size());

        auto flatten = [&](const ir::Stmt &stmt) {
            ir::Stmt mut = mutate(stmt);
            changed = changed || !mut.same_as(stmt);
            if (!mut.defined()) {
                changed = true;
                return;
            } else if (const ir::Sequence *seq = mut.as<ir::Sequence>()) {
                stmts.insert(stmts.end(), seq->stmts.begin(), seq->stmts.end());
                changed = true;
            } else {
                stmts.emplace_back(std::move(mut));
            }
        };

        for (const auto &stmt : node->stmts) {
            flatten(stmt);
        }

        if (!changed) {
            return node;
        }
        if (stmts.empty()) {
            return ir::Stmt();
        }
        return ir::Sequence::make(std::move(stmts));
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = mutate(node->cond);
        ir::Stmt then_body = mutate(node->then_body);
        ir::Stmt else_body = mutate(node->else_body);

        if (auto x = get_constant_value(cond); x.has_value()) {
            if (*x == 0) {
                return else_body;
            } else {
                return then_body;
            }
        }

        if (!then_body.defined()) {
            if (!else_body.defined()) {
                // No-op
                return then_body;
            }
            return ir::IfElse::make(~cond, std::move(else_body));
        }

        if (cond.same_as(node->cond) && then_body.same_as(node->then_body) &&
            else_body.same_as(node->else_body)) {
            return node;
        }
        return ir::IfElse::make(std::move(cond), std::move(then_body),
                                std::move(else_body));
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
