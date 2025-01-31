#include "Lower/Intrinsics.h"

#include "IR/Operators.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

ir::Expr cross_product(const ir::Expr &a, const ir::Expr &b) {
    static const ir::Type u32 = ir::UInt_t::make(32);
    static const ir::Expr zero = make_const(u32, 0);
    static const ir::Expr one = make_const(u32, 1);
    static const ir::Expr two = make_const(u32, 2);
    ir::Expr a0 = ir::Extract::make(a, zero);
    ir::Expr a1 = ir::Extract::make(a, one);
    ir::Expr a2 = ir::Extract::make(a, two);
    ir::Expr b0 = ir::Extract::make(b, zero);
    ir::Expr b1 = ir::Extract::make(b, one);
    ir::Expr b2 = ir::Extract::make(b, two);
    ir::Expr s0 = a1 * b2 - a2 * b1;
    ir::Expr s1 = a2 * b0 - a0 * b2;
    ir::Expr s2 = a0 * b1 - a1 * b0;
    return ir::Build::make(a.type(), {s0, s1, s2});
}

ir::Expr argmax(const ir::Expr &a) {
    internal_assert(a.type().element_of().is_scalar())
        << "TODO: implement argmax lowering for 2D: " << a;
    ir::Expr max_value = ir::VectorReduce::make(ir::VectorReduce::Max, a);
    if (a.type().lanes() == 3) {
        ir::Type u32 = ir::UInt_t::make(32);
        ir::Expr zero = make_const(u32, 0);
        ir::Expr one = make_const(u32, 1);
        ir::Expr two = make_const(u32, 2);
        ir::Expr a0 = ir::Extract::make(a, zero);
        ir::Expr a1 = ir::Extract::make(a, one);
        ir::Expr a2 = ir::Extract::make(a, two);
        return ir::Select::make(max_value == a0, zero,
                                ir::Select::make(max_value == a1, one, two));
    } else {
        internal_error << "TODO: implement large argmax lowering: " << a;
        // From Andrew: min_reduce(ramp(0, 1, 8) & v ==
        // broadcast(max_reduce(v)))
        return ir::Expr();
    }
}

} // namespace lower
} // namespace bonsai
