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

// Performs necessary mutations to convert lambdas to functions.
struct ConvertLambdaToFunction : public ir::Mutator {
    ConvertLambdaToFunction(
        std::unordered_map<const ir::Lambda *, Metadata> &lambda_metadata,
        const std::unordered_set<const ir::Lambda *> &blacklisted_lambdas)
        : lambda_metadata(lambda_metadata),
          blacklisted_lambdas(blacklisted_lambdas), counter(0) {};

  private:
    // A mapping from a lambda to metadata required for said lambda.
    std::unordered_map<const ir::Lambda *, Metadata> &lambda_metadata;

    // Returns a unique name for the function replacing this lambda.
    // TODO(cgyurgyik): We need some program-level name hygiene guarantees.
    std::string generate_name() {
        std::string name = "_lambda";
        name += std::to_string(counter++);
        return name;
    }

    // A set of lambdas that should not be converted.
    const std::unordered_set<const ir::Lambda *> &blacklisted_lambdas;
    int64_t counter; // A counter for name hygiene.

    ir::Stmt visit(const ir::LetStmt *let) override {
        ir::WriteLoc lhs = let->loc;
        internal_assert(lhs.accesses.empty()) << "unimplemented";

        ir::Expr rhs = let->value;
        if (const ir::Call *call = rhs.as<ir::Call>()) {
            return ir::Mutator::visit(let);
        }
        const ir::Lambda *lambda = rhs.as<ir::Lambda>();
        if (lambda == nullptr) {
            return let;
        }

        // Visit this lambda.
        rhs = visit(lambda);
        lambda = rhs.as<ir::Lambda>();
        internal_assert(lambda_metadata.contains(lambda));

        auto it = lambda_metadata.find(lambda);
        Metadata &m = it->second;
        m.name = lhs.base;
        return let;
    }

    ir::Expr visit(const ir::Lambda *lambda) override {
        if (blacklisted_lambdas.contains(lambda))
            return lambda;

        // Convert lambda arguments to function arguments.
        const std::vector<ir::TypedVar> &before = lambda->args;
        std::vector<ir::Function::Argument> arguments;
        std::transform(before.begin(), before.end(),
                       std::back_inserter(arguments),
                       [](const ir::TypedVar &a) {
                           return ir::Function::Argument(a.name, a.type);
                       });

        ir::Type type = lambda->value.type();
        auto [it, succeeded] = lambda_metadata.try_emplace(lambda, Metadata{});
        if (!succeeded) {
            // This lambda has already been visited, and a function created.
            return lambda;
        }
        Metadata &m = it->second;
        m.function = std::make_shared<ir::Function>(
            generate_name(), arguments,
            /*return_type=*/type,
            /*body=*/ir::Return::make(lambda->value),
            /*interfaces=*/ir::Function::InterfaceList{},
            std::vector<ir::Function::Attribute>{});

        return lambda;
    }

    ir::Expr visit(const ir::Call *call) override {
        const ir::Var *v = call->func.as<ir::Var>();
        if (v == nullptr)
            return call;

        // Note: we assume there will be a small constant number of lambdas.
        auto it = std::find_if(lambda_metadata.begin(), lambda_metadata.end(),
                               [&](const auto &kv) {
                                   const Metadata &m = kv.second;
                                   return m.name == v->name;
                               });
        if (it == lambda_metadata.end()) {
            return call; // This is a call to a non-lambda.
        }
        std::shared_ptr<ir::Function> &f = it->second.function;

        ir::Type type = f->call_type();
        return ir::Call::make(ir::Var::make(type, f->name), call->args);
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
    // A mapping from lambda to metadata required for safe replacement.
    std::unordered_map<const ir::Lambda *, Metadata> lambda_metadata;

    // Some lambdas, e.g., those used in set queries, will not be converted.
    Blacklist blacklisted_lambdas;
    for (const auto &[_, f] : old_program.funcs) {
        if (const ir::Stmt &body = f->body; body.defined()) {
            body.accept(&blacklisted_lambdas);
        }
    }

    ConvertLambdaToFunction cltf(lambda_metadata, blacklisted_lambdas.get());
    ir::Program new_program;
    new_program.externs = old_program.externs;
    new_program.types = old_program.types;
    for (const auto &[f, func] : old_program.funcs) {
        ir::Stmt body = cltf.mutate(func->body);
        new_program.funcs[f] = func->replace_body(std::move(body));
    }

    // Guarantee deterministic ordering for insertion.
    std::vector<std::shared_ptr<ir::Function>> functions;
    functions.reserve(lambda_metadata.size());
    std::transform(lambda_metadata.begin(), lambda_metadata.end(),
                   std::back_inserter(functions),
                   [](const auto &kv) { return kv.second.function; });
    std::sort(
        functions.begin(), functions.end(),
        [](const auto &m1, const auto &m2) { return m1->name < m2->name; });

    // Add newly created functions to replace the lambda expressions.
    for (std::shared_ptr<ir::Function> f : functions) {
        new_program.funcs[f->name] = f;
    }
    return new_program;
}

} // namespace

ir::Program LowerLambdas::run(ir::Program program) const {
    ir::Program new_program = lower_program(program);
    new_program.schedules = std::move(program.schedules);
    return new_program;
}

} // namespace lower
} // namespace bonsai
