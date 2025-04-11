#include "Opt/Simplify.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Utils.h"

#include "Lower/TopologicalOrder.h"

#include "Error.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

struct Simplifier : ir::Mutator {
    ir::Expr visit(const ir::BinOp *node) override {
        ir::Expr a = node->a, b = node->b;
        switch (node->op) {
        // case ir::BinOp::OpType::Add: {
        //     if (is_const(a) && is_const(b) && ir::equals(a.type(), b.type())
        //     &&
        //         a.type().bits() <= 64) {
        //         if (a.type().is_int()) {
        //             int64_t c_a = get_constant_value<int64_t>(a);
        //             int64_t c_b = get_constant_value<int64_t>(b);
        //             return ir::IntImm::make(a.type(), c_a + c_b);
        //         }
        //         if (a.type().is_uint()) {
        //             uint64_t c_a = get_constant_value<uint64_t>(a);
        //             uint64_t c_b = get_constant_value<uint64_t>(b);
        //             return ir::UIntImm::make(a.type(), c_a + c_b);
        //         }
        //     }
        //     if (is_const(a) && get_constant_value(a) == 0 &&
        //         ir::equals(a.type(), b.type())) {
        //         return b;
        //     }
        //     if (is_const(b) && get_constant_value(b) == 0 &&
        //         ir::equals(a.type(), b.type())) {
        //         return a;
        //     }
        //     return node;
        // }
        // case ir::BinOp::OpType::Mul: {
        //     if (is_const(a) && is_const(b) && ir::equals(a.type(), b.type())
        //     &&
        //         a.type().bits() <= 64) {
        //         if (a.type().is_int()) {
        //             int64_t c_a = get_constant_value<int64_t>(a);
        //             int64_t c_b = get_constant_value<int64_t>(b);
        //             return ir::IntImm::make(a.type(), c_a * c_b);
        //         }
        //         if (a.type().is_uint()) {
        //             uint64_t c_a = get_constant_value<uint64_t>(a);
        //             uint64_t c_b = get_constant_value<uint64_t>(b);
        //             return ir::UIntImm::make(a.type(), c_a * c_b);
        //         }
        //     }
        //     if (is_const(a) && get_constant_value(a) == 0 &&
        //         ir::equals(a.type(), b.type())) {
        //         return a;
        //     }
        //     if (is_const(b) && get_constant_value(b) == 0 &&
        //         ir::equals(a.type(), b.type())) {
        //         return b;
        //     }
        //     return node;
        // }
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

ir::FuncMap Simplify::run(ir::FuncMap funcs) const {
    Simplifier lower;
    for (auto &[name, func] : funcs) {
        func->body = lower.mutate(func->body);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
