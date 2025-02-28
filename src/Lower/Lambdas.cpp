#include "Lower/Lambdas.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

// Stores data necessary to safely perform a lambda-to-function conversion.
struct Metadata {
    // The name of this lambda, if it was stored to a local
    // variable, and the empty string otherwise.
    std::string name = "";
    // The function that will replace it.
    std::shared_ptr<ir::Function> function = nullptr;
};

class ConvertLambdaToStruct : public ir::Mutator {
  public:
    ConvertLambdaToStruct(
        const std::unordered_set<const ir::Lambda *> &blacklisted_lambdas)
        : blacklisted_lambdas(blacklisted_lambdas) {};

  private:
    // Lambdas that should not be visited.
    const std::unordered_set<const ir::Lambda *> &blacklisted_lambdas;

    int64_t counter = 0;
    std::string generate_name() {
        std::string name = "?lambda";
        name += std::to_string(counter++);
        return name;
    }

    ir::Expr visit(const ir::Lambda *lambda) override {
        if (blacklisted_lambdas.contains(lambda))
            return lambda;

        // DO NOT SUBMIT: add implicitly captured variables.
        ir::Struct_t::Map fields = {};
        ir::Type type =
            ir::Struct_t::make(generate_name(), fields, lambda->type);

        ir::Expr value = lambda->value;
        if (const ir::Lambda *inner = value.as<ir::Lambda>()) {
            value = visit(inner);
        }
        ir::Struct_t::DefMap values = {};
        auto call = ir::Build::Call{
            .args = lambda->args,
            .value = lambda->value,
        };
        ir::Expr b = ir::Build::make(type, values, call);
        return b; // return ir::Access::make(CALL, );
    }
};

// Visitor class to retrieve a list of lambdas that should *not* be converted.
class Blacklist : public ir::Visitor {
  public:
    const std::unordered_set<const ir::Lambda *> &get() {
        return blacklisted_lambdas;
    }

  private:
    std::unordered_set<const ir::Lambda *> blacklisted_lambdas;

    void visit(const ir::SetOp *node) override {
        switch (node->op) {
        case ir::SetOp::OpType::product:
            return;
        case ir::SetOp::OpType::argmin:
        case ir::SetOp::OpType::map:
        case ir::SetOp::OpType::filter: {
            const ir::Lambda *op = node->a.as<ir::Lambda>();
            internal_assert(op) << "first operand of a ir::SetOp should have "
                                   "type ir::Lambda, received: "
                                << node->a;
            blacklisted_lambdas.insert(op);
            if (const auto *b = node->b.as<ir::SetOp>()) {
                visit(b);
            }
        }
        }
    }
};

// Performs the orchestration for lambda lowering.
ir::Program lower_program(const ir::Program &old_program) {
    // Some lambdas, e.g., those used in set queries, will not be converted.
    Blacklist blacklisted_lambdas;
    for (const auto &[_, f] : old_program.funcs) {
        if (const ir::Stmt &body = f->body; body.defined()) {
            body.accept(&blacklisted_lambdas);
        }
    }

    ConvertLambdaToStruct clts(blacklisted_lambdas.get());
    ir::Program new_program;
    new_program.externs = old_program.externs;
    new_program.types = old_program.types;
    for (const auto &[f, func] : old_program.funcs) {
        ir::Stmt body = clts.mutate(std::move(func->body));
        new_program.funcs[f] = std::make_shared<ir::Function>(
            func->name, func->args, func->ret_type, body, func->interfaces);
    }

    return new_program;
}

} // namespace

ir::Program LowerLambda::lower(const ir::Program &program) const {
    return lower_program(program);
}

} // namespace lower
} // namespace bonsai
