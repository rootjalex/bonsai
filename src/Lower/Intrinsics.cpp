#include "Lower/Intrinsics.h"

#include "IR/Operators.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

ir::Expr cross_product(const ir::Expr &a, const ir::Expr &b) {
    static const ir::Type u32 = ir::UInt_t::make(32);
    ir::Expr a0 = ir::Extract::make(a, 0);
    ir::Expr a1 = ir::Extract::make(a, 1);
    ir::Expr a2 = ir::Extract::make(a, 2);
    ir::Expr b0 = ir::Extract::make(b, 0);
    ir::Expr b1 = ir::Extract::make(b, 1);
    ir::Expr b2 = ir::Extract::make(b, 2);
    ir::Expr s0 = a1 * b2 - a2 * b1;
    ir::Expr s1 = a2 * b0 - a0 * b2;
    ir::Expr s2 = a0 * b1 - a1 * b0;
    return ir::Build::make(a.type(), {s0, s1, s2});
}

ir::Expr argmax(const ir::Expr &a) {
    internal_assert(a.type().element_of().is_scalar())
        << "TODO: implement argmax lowering for 2D: " << a;
    ir::Expr best = ir::VectorReduce::make(ir::VectorReduce::Max, a);
    if (a.type().lanes() == 3) {
        ir::Type u32 = ir::UInt_t::make(32);
        ir::Expr _0 = make_const(u32, 0);
        ir::Expr _1 = make_const(u32, 1);
        ir::Expr _2 = make_const(u32, 2);
        ir::Expr a0 = ir::Extract::make(a, 0);
        ir::Expr a1 = ir::Extract::make(a, 1);
        ir::Expr a2 = ir::Extract::make(a, 2);
        return ir::Select::make(best == a0, _0,
                                ir::Select::make(best == a1, _1, _2));
    } else {
        internal_error << "TODO: implement large argmax lowering: " << a;
        // From Andrew: min_reduce(ramp(0, 1, 8) & v ==
        // broadcast(max_reduce(v)))
    }
}

} // namespace lower
} // namespace bonsai
