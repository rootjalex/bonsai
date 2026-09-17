#include "Opt/Simplify.h"

#include "Error.h"
#include "IR/Analysis.h"
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
#include <optional>
#include <string>
#include <type_traits>
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

// The bits an immediate holds for a fold's result: a bool as 0 or 1, an
// integer as its two's complement, a float as the double that holds it.
template <typename T>
uint64_t as_bits(T v) {
    if constexpr (std::is_same_v<T, bool>) {
        return v ? 1 : 0;
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::bit_cast<uint64_t>(double(v));
    } else {
        return static_cast<uint64_t>(v);
    }
}

// Applies `f` to two scalar constants in *their* type, and returns the bits
// of the result, or nothing if either is not a constant.
//
// The operands' type rather than the result's, because the two differ for a
// comparison -- its operands are numbers and its result is a bool -- and it
// is the operands' type that says what the bits mean. A float is folded in
// its own precision, so that the fold answers what the operation would have
// at run time: `0.1f * 3.0f` is not the double product rounded. A half has no
// host type to compute it in and is left alone.
template <typename F>
std::optional<uint64_t> fold_scalar(F f, const ir::Expr &a, const ir::Expr &b) {
    const ir::Type type = a.type();
    if (type.is_float()) {
        if constexpr (std::is_invocable_v<F, double, double>) {
            const auto *fa = a.as<ir::FloatImm>();
            const auto *fb = b.as<ir::FloatImm>();
            if (fa == nullptr || fb == nullptr) {
                return std::nullopt;
            }
            if (type.bits() == 32) {
                return as_bits(f(float(fa->value), float(fb->value)));
            }
            if (type.bits() == 64) {
                return as_bits(f(fa->value, fb->value));
            }
        }
        return std::nullopt;
    }
    std::optional<uint64_t> c_a = get_constant_value(a),
                            c_b = get_constant_value(b);
    if (!(c_a.has_value() && c_b.has_value())) {
        return std::nullopt;
    }
    if (type.is_int()) {
        return as_bits(apply<int64_t>(f, *c_a, *c_b));
    }
    return as_bits(apply<uint64_t>(f, *c_a, *c_b));
}

// Attempts to constant fold the binary operations. Returns an undefined
// expression upon failure. A type parameter is optionally passed when
// interpreting a vector's broadcasted value, and when the result's type is
// not the operands' -- a comparison's.
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

    // Scalar case. The operation is applied in the operands' type and the
    // result made in `type`. Applying it in the result's type compared the
    // operands' bit patterns instead, which put -1 above every positive
    // integer and every negative float above every positive one.
    internal_assert(type->is_scalar()) << *type;
    const std::optional<uint64_t> bits = fold_scalar(f, a, b);
    if (!bits.has_value()) {
        return ir::Expr();
    }
    if (type->is_float()) {
        return ir::FloatImm::make(*type, std::bit_cast<double>(*bits));
    }
    if (type->is_int()) {
        return ir::IntImm::make(*type, std::bit_cast<int64_t>(*bits));
    }
    if (type->is_uint()) {
        return ir::UIntImm::make(*type, *bits);
    }
    if (type->is_bool()) {
        return ir::BoolImm::make(*bits != 0);
    }

    return ir::Expr();
}

// `v[i] < c`, for a vector `v` and a scalar constant `c`, as `(v < c)[i]`:
// the same lane of the whole vector's comparison. Or nothing, when the
// comparison is not of that shape.
//
// The compare then names the vector, which is the form the same test takes
// wherever it is written over the vector rather than a lane of it -- pbrt's
// slab test asks `1/d < 0` of all three axes and its traversal order asks it
// of one -- so CSE can make the two one value, and hoisting the vector out of
// a loop carries the compare out with it and leaves only the lane read
// behind. On its own the vector compare costs what the scalar one did: the
// backends' lowering scalarizes a single-use lane of one straight back.
ir::Expr compare_lane(ir::BinOp::OpType op, const ir::Expr &a,
                      const ir::Expr &b) {
    const auto lane_of = [](const ir::Expr &e) -> const ir::Extract * {
        const auto *extract = e.as<ir::Extract>();
        if (extract == nullptr || !extract->vec.type().is<ir::Vector_t>() ||
            is_const(extract->vec)) {
            return nullptr;
        }
        return extract;
    };
    const auto splat = [](const ir::Expr &c, const ir::Type &vector) {
        return ir::Broadcast::make(vector.lanes(), c);
    };
    if (const ir::Extract *lane = lane_of(a);
        lane != nullptr && is_const(b) && b.type().is_scalar()) {
        return ir::Extract::make(
            ir::BinOp::make(op, lane->vec, splat(b, lane->vec.type())),
            lane->idx);
    }
    if (const ir::Extract *lane = lane_of(b);
        lane != nullptr && is_const(a) && a.type().is_scalar()) {
        return ir::Extract::make(
            ir::BinOp::make(op, splat(a, lane->vec.type()), lane->vec),
            lane->idx);
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
    Simplifier() = default;
    explicit Simplifier(const Simplify::Knowledge &knowledge)
        : knowledge(&knowledge), facts(knowledge.known) {}

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
        internal_error << "TODO simplify: " << ir::Expr(node);
    }

    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = mutate(node->a), b = mutate(node->b);
        internal_assert(ir::equals(a.type(), b.type()))
            << "a: " << a.type() << ", " << "b: " << b.type() << "\n"
            << ir::Expr(node);

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
            //
            // Integers only. `a - a` is zero for every integer and for every
            // *finite* float, and not for an infinity or a NaN, where it is a
            // NaN -- which is exactly what makes `(x - x) == 0` the standard
            // way to ask whether a float is finite. Folding it away answers
            // "yes, always", silently, and the guard it was written to be
            // disappears.
            //
            // pbrt reaches this: its RGB-to-spectrum table answers a pure black
            // texel with minus infinity, meaning "reflects nothing at any
            // wavelength", and the sigmoid that reads it has a branch for
            // exactly that. With the branch folded out, half an environment map
            // came back NaN.
            if (!type.is_float() && (a.same_as(b) || ir::equals(a, b))) {
                // a - a = 0
                return zero;
            }
            if (is_const_zero(a)) {
                // 0 - a = -a
                return -b;
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Mod: {
            if (ir::Expr e = constant_fold_integral(std::modulus<>{}, a, b);
                e.defined()) {
                return e;
            }
            if (is_const_zero(a)) {
                // 0 % N == 0
                return a;
            }
            if (is_const_one(b)) {
                // N % 1 == 0
                return make_zero(type);
            }
            std::optional<uint64_t> c_b = get_constant_value(b);
            if (c_b.has_value() && is_power_of_two(*c_b)) {
                // x % 2^n -> x & (2^n - 1)
                return a & make_const(type, *c_b - 1);
            }
            return make(node, std::move(a), std::move(b));
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
            if (a.same_as(b) || ir::equals(a, b)) {
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
            if (a.same_as(b) || ir::equals(a, b)) {
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
        case ir::BinOp::OpType::Eq: {
            if (ir::Expr e =
                    constant_fold_integral(std::equal_to<>{}, a, b, node->type);
                e.defined()) {
                return e;
            }
            if (a.same_as(b) || ir::equals(a, b)) {
                // x == x
                return make_one(node->type);
            }
            return make(node, std::move(a), std::move(b));
        }
        case ir::BinOp::OpType::Lt: {
            if (ir::Expr e =
                    constant_fold_integral(std::less<>{}, a, b, node->type);
                e.defined()) {
                return e;
            }
            if (const ir::Select *a_sel = a.as<ir::Select>()) {
                if (const ir::Select *b_sel = b.as<ir::Select>()) {
                    // This assumes a cost model of select being more expensive
                    // than less than. I think that's valid.
                    // select(a, x, y) < select(a, w, z)
                    //  -> select(a, x < w, y < z)
                    if (equals(a_sel->cond, b_sel->cond)) {
                        ir::Expr repl = ir::Select::make(
                            a_sel->cond, a_sel->tvalue < b_sel->tvalue,
                            a_sel->fvalue < b_sel->fvalue);
                        return mutate(repl);
                    }
                }
            }
            if (ir::Expr lane = compare_lane(node->op, a, b); lane.defined()) {
                return mutate(lane);
            }
            return make(node, std::move(a), std::move(b));
        }
        default:
            return make(node, std::move(a), std::move(b));
        }
    }

    ir::Expr visit(const ir::Select *node) override {
        ir::Expr cond = mutate(node->cond);
        if (std::optional<bool> value = ir::decided(facts, cond)) {
            // A select evaluates both of its arms, so the one not taken can
            // go only if nothing happens in it: a draw from a sampler there
            // still advanced the sampler.
            const ir::Expr &dropped = *value ? node->fvalue : node->tvalue;
            if (!effectful(dropped)) {
                return mutate(*value ? node->tvalue : node->fvalue);
            }
        }
        ir::Expr tvalue = mutate(node->tvalue), fvalue = mutate(node->fvalue);
        if (is_const_one(cond)) {
            // select(true, a, b) = a
            return tvalue;
        }
        if (is_const_zero(cond)) {
            // select(false, a, b) = b
            return fvalue;
        }
        if (is_const_zero(tvalue) && is_const_one(fvalue)) {
            // select(a, 0, 1) = cast<type>(!a) -- a true condition takes the
            // *true* arm, which is the zero.
            return cast(tvalue.type(), ~cond);
        }
        if (is_const_one(tvalue) && is_const_zero(fvalue)) {
            // select(a, 1, 0) = cast<type>(a)
            return cast(tvalue.type(), cond);
        }
        if (equals(tvalue, fvalue)) {
            // select(x, a, a) = a
            return tvalue;
        }
        if (cond.same_as(node->cond) && tvalue.same_as(node->tvalue) &&
            fvalue.same_as(node->fvalue)) {
            return node;
        }
        return ir::Select::make(std::move(cond), std::move(tvalue),
                                std::move(fvalue));
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        // Only a converting cast, which is the one that means "this number as
        // that type". Folding a reinterpret this way would compute the number
        // instead of keeping the bits: reinterpret<u32>(1.0f) is 1065353216,
        // and constant_cast would answer 1. The backends emit a bitcast, which
        // LLVM folds for a constant anyway, so nothing is lost by leaving it.
        if (node->mode == ir::Cast::Mode::Convert && is_const(value) &&
            node->type.is_scalar()) {
            return constant_cast(node->type, std::move(value));
        }
        if (equals(value.type(), node->type)) {
            // T v = ...; cast[[T]](v) = v
            return value;
        }
        if (value.same_as(node->value)) {
            return node;
        }
        return ir::Cast::make(node->type, std::move(value), node->mode);
    }

    ir::Expr visit(const ir::Build *node) override {
        bool changed = false, is_all_constants = true;
        std::vector<ir::Expr> values;
        for (int32_t i = 0, e = node->values.size(); i < e; ++i) {
            ir::Expr v = mutate(node->values[i]);
            changed |= !v.same_as(node->values[i]);
            is_all_constants &= is_const(v);
            values.push_back(std::move(v));
        }
        // The immediate keeps the build's type: packed storage (see
        // Vector_t::packed) stays the storage the field holding it says it
        // is, rather than becoming a vector proper.
        if (node->type.is_vector() && is_all_constants && !values.empty()) {
            // x: i32 = 1; v: Build<i32x2>(x, (i32)2) => [1, 2]
            return ir::VecImm::make(node->type, std::move(values));
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
        } else if (const auto *build = v.as<ir::Build>()) {
            if (build->type.is<ir::Tuple_t>()) {
                std::optional<uint64_t> index = get_constant_value(i);
                internal_assert(index.has_value())
                    << "Simplifier found non-constant index into tuple: " << i
                    << " of " << v;
                internal_assert(*index < build->values.size())
                    << "Simplifier found out-of-range index into tuple: " << i
                    << " of " << v;
                return build->values[*index];
            }
        }

        std::optional<uint64_t> index = get_constant_value(i);
        // Only a vector holds its elements as constants that can be folded
        // out this way; a Build of an array or a struct is just as constant
        // but its elements are Exprs, and folding one is the job of the
        // aggregate cases above.
        if (v.is<ir::VecImm, ir::Build>() && v.type().is_vector() &&
            index.has_value() && is_const(v)) {
            // The element itself, rather than its value read back as an
            // integer and made into a constant again. That round trip is only
            // right for integer elements: `get_constant_value` hands back the
            // bit pattern, so a lane holding 1.0f came out as the integer its
            // double spells, and every comparison against it was wrong.
            //
            // `is_const` is what makes dropping the other lanes safe.
            if (ir::Expr element = get_value_at(v, int64_t(*index));
                element.defined()) {
                return element;
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

        // What a statement decides for the ones after it: past
        // `if (c) { ...; return }` the rest of the sequence runs only with
        // `c` false. Learned as the statements go by, forgotten at the end.
        const size_t facts_before = facts.size();
        auto flatten = [&](const ir::Stmt &stmt) {
            ir::Stmt mut = mutate(stmt);
            changed = changed || !mut.same_as(stmt);
            if (!mut.defined()) {
                changed = true;
                return;
            }
            learn_after(mut);
            if (const ir::Sequence *seq = mut.as<ir::Sequence>()) {
                stmts.insert(stmts.end(), seq->stmts.begin(), seq->stmts.end());
                changed = true;
            } else {
                stmts.emplace_back(std::move(mut));
            }
        };

        for (const auto &stmt : node->stmts) {
            flatten(stmt);
        }
        facts.resize(facts_before);

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
        if (std::optional<bool> value = ir::decided(facts, cond)) {
            // Decided by a branch this one sits inside, or by one before it
            // whose other arm returned.
            return mutate(*value ? node->then_body : node->else_body);
        }
        const bool learn = learnable(cond);
        ir::Stmt then_body = with_fact(cond, true, learn, node->then_body);
        ir::Stmt else_body = with_fact(cond, false, learn, node->else_body);

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

    ir::Stmt visit(const ir::SwitchStmt *node) override {
        ir::Expr value = mutate(node->value);
        const size_t last = node->arms.size() - 1;
        if (auto x = get_constant_value<int64_t>(value); x.has_value()) {
            // The arm the value picks; the last for anything past the others
            // (see ir::SwitchStmt).
            const size_t k =
                *x < 0 ? last : std::min(static_cast<size_t>(*x), last);
            return mutate(node->arms[k]);
        }

        // Inside arm k the value is k. The last arm only knows it is none of
        // the others, which is nothing to learn from.
        auto key = [&](size_t k) -> ir::Expr {
            return value.type().is_uint()
                       ? ir::UIntImm::make(value.type(), k)
                       : ir::IntImm::make(value.type(), static_cast<int64_t>(k));
        };
        std::vector<ir::Stmt> arms;
        arms.reserve(node->arms.size());
        bool same = value.same_as(node->value);
        bool any = false;
        for (size_t k = 0; k < node->arms.size(); k++) {
            if (k == last) {
                arms.push_back(mutate(node->arms[k]));
            } else {
                const ir::Expr is_k =
                    ir::BinOp::make(ir::BinOp::OpType::Eq, value, key(k));
                arms.push_back(
                    with_fact(is_k, true, learnable(is_k), node->arms[k]));
            }
            same = same && arms.back().same_as(node->arms[k]);
            any = any || arms.back().defined();
        }
        if (!any) {
            // No-op
            return ir::Stmt();
        }
        if (same) {
            return node;
        }
        return ir::SwitchStmt::make(std::move(value), std::move(arms));
    }

    ir::Stmt visit(const ir::Store *node) override {
        ir::Expr value = node->value;
        ir::WriteLoc loc = node->loc;
        if (!value.defined() || !loc.accesses.empty()) {
            return ir::Mutator::visit(node);
        }
        value = mutate(std::move(value));
        ir::Expr v = ir::Var::make(loc.type, loc.base);
        if (ir::equals(v, value)) {
            // *a = *a;
            return ir::Stmt();
        }
        if (value.same_as(node->value)) {
            return node;
        }
        return ir::Store::make(loc, std::move(value));
    }

  private:
    // Mapping from a variable name to its immediate value. This assumes
    // variable shadowing is illegal; if this were to change, we'd need to
    // introduce a frame stack.
    std::unordered_map<std::string, ir::Expr> name_to_immediate;

    // What may be learned about the function, or nothing, when the
    // simplification is of a fragment with no function around it.
    const Simplify::Knowledge *knowledge = nullptr;
    // The conditions decided where the mutation is, most recent last.
    ir::Facts facts;

    // Whether `cond` means the same thing throughout the code it guards: a
    // pure value, over names the function cannot assign.
    bool learnable(const ir::Expr &cond) const {
        if (knowledge == nullptr || !ir::is_pure_value(cond)) {
            return false;
        }
        for (const ir::TypedVar &v : ir::gather_free_vars(cond)) {
            if (knowledge->assignable.count(v.name)) {
                return false;
            }
        }
        return true;
    }

    bool effectful(const ir::Expr &e) const {
        return knowledge == nullptr ||
               ir::has_side_effects(e, knowledge->effectful);
    }

    ir::Stmt with_fact(const ir::Expr &cond, bool value, bool learn,
                       const ir::Stmt &arm) {
        if (!arm.defined() || !learn) {
            return mutate(arm);
        }
        facts.emplace_back(cond, value);
        ir::Stmt out = mutate(arm);
        facts.pop_back();
        return out;
    }

    // What `stmt`, just mutated, decides for the statements after it in its
    // sequence: a branch with one arm that always returns leaves the other
    // arm's condition holding for everything that follows.
    void learn_after(const ir::Stmt &stmt) {
        const ir::IfElse *branch = stmt.as<ir::IfElse>();
        if (const auto *seq = stmt.as<ir::Sequence>();
            seq != nullptr && !seq->stmts.empty()) {
            branch = seq->stmts.back().as<ir::IfElse>();
        }
        if (branch == nullptr || !learnable(branch->cond)) {
            return;
        }
        const bool then_returns = ir::always_returns(branch->then_body);
        const bool else_returns = branch->else_body.defined() &&
                                  ir::always_returns(branch->else_body);
        if (then_returns && !else_returns) {
            facts.emplace_back(branch->cond, false);
        } else if (else_returns && !then_returns) {
            facts.emplace_back(branch->cond, true);
        }
    }
};

} // namespace

/* static */ ir::Expr Simplify::simplify(ir::Expr e) {
    return Simplifier().mutate(std::move(e));
}

/* static */ ir::Stmt Simplify::simplify(ir::Stmt s) {
    return Simplifier().mutate(std::move(s));
}

/* static */ ir::Stmt Simplify::simplify(ir::Stmt s,
                                         const Knowledge &knowledge) {
    return Simplifier(knowledge).mutate(std::move(s));
}

ir::FuncMap Simplify::run(ir::FuncMap funcs,
                          const CompilerOptions &options) const {
    // Templated functions are not simplified, and not analysed either: a
    // call in one names its callee by an instantiation, not a function.
    ir::FuncMap concrete;
    for (const auto &[name, func] : funcs) {
        if (func->interfaces.empty()) {
            concrete[name] = func;
        }
    }
    const std::set<std::string> effectful = ir::find_side_effects(concrete);
    for (auto &[name, func] : concrete) {
        const Knowledge knowledge{ir::assignable_names(*func), effectful, {}};
        func->body = Simplifier(knowledge).mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
