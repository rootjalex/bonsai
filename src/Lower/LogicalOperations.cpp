#include "Lower/LogicalOperations.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <variant>

namespace bonsai {
namespace lower {

namespace {

class LowerImpl : public ir::Mutator {

    ir::Stmt visit(const ir::IfElse *node) override {
        if (const auto *logical_operation = node->cond.as<ir::BinOp>()) {
            if (logical_operation->op == ir::BinOp::OpType::LAnd) {
                return lower_and(node);
            }
        }
        return ir::IfElse::make(/*cond=*/mutate(node->cond),
                                /*then_body=*/mutate(node->then_body),
                                /*else_body=*/mutate(node->else_body));
    }

  private:
    ir::Stmt lower_and(ir::Stmt statement) {
        const auto *if_else = statement.as<ir::IfElse>();
        internal_assert(if_else) << statement;
        const auto *cond = if_else->cond.as<ir::BinOp>();
        internal_assert(cond) << if_else->cond;

        if (!if_else->else_body.defined()) {
            // if (a && b) { <body> }  ->  if (a) { if (b) { <body> } }
            ir::Stmt then_body =
                mutate(ir::IfElse::make(cond->b, if_else->then_body));
            return mutate(ir::IfElse::make(cond->a, std::move(then_body)));
        }
        return statement;
    }
};

} // namespace

ir::FuncMap LowerLogicalOperations::run(ir::FuncMap funcs) const {
    for (auto &[_, func] : funcs) {
        LowerImpl lower;
        func->body = lower.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
