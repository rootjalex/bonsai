#include "Lower/Lambdas.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <unordered_map>
#include <unordered_set>

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
        std::unordered_map<const ir::Lambda *, Metadata> &map,
        const std::unordered_set<const ir::Lambda *> &blacklist)
        : map(map), blacklist(blacklist), counter(0) {};

  private:
    // A mapping from a lambda to metadata required for said lambda.
    std::unordered_map<const ir::Lambda *, Metadata> &map;
    // A set of lambdas that should not be converted.
    const std::unordered_set<const ir::Lambda *> &blacklist;
    int64_t counter; // A counter for name hygiene.

    ir::Stmt visit(const ir::LetStmt *let) override {
        ir::WriteLoc lhs = let->loc;
        internal_assert(lhs.accesses.empty()) << "unimplemented";

        ir::Expr rhs = let->value;
        if (const ir::Call *call = rhs.as<ir::Call>()) {
            return ir::LetStmt::make(lhs, visit(call));
        }
        const ir::Lambda *lambda = rhs.as<ir::Lambda>();
        if (lambda == nullptr) {
            return let;
        }

        // Visit this lambda.
        rhs = visit(lambda);
        lambda = rhs.as<ir::Lambda>();
        internal_assert(map.contains(lambda));

        auto it = map.find(lambda);
        Metadata &m = it->second;
        m.name = lhs.base;
        return let;
    }

    ir::Expr visit(const ir::Lambda *lambda) override {
        if (blacklist.contains(lambda))
            return lambda;

        // Convert lambda arguments to function arguments.
        const std::vector<ir::Lambda::Argument> &before = lambda->args;
        std::vector<ir::Function::Argument> arguments;
        std::transform(before.begin(), before.end(),
                       std::back_inserter(arguments),
                       [](const ir::Lambda::Argument &a) {
                           return ir::Function::Argument(a.name, a.type);
                       });

        // TODO(cgyurgyik): We need some program-level name hygiene guarantees.
        std::string name = "?lambda";
        name += std::to_string(counter++);
        ir::Type type = lambda->value.type();

        auto [it, _] = map.try_emplace(lambda, Metadata{});
        Metadata &m = it->second;
        m.function = std::make_shared<ir::Function>(
            name, arguments,
            /*return_type=*/type,
            /*body=*/ir::Return::make(lambda->value),
            /*interfaces=*/ir::Function::InterfaceList{});

        return lambda;
    }

    ir::Expr visit(const ir::Call *call) override {
        const ir::Var *v = call->func.as<ir::Var>();
        if (v == nullptr)
            return call;

        // Note: we assume there will be a small constant number of lambdas.
        auto it = std::find_if(map.begin(), map.end(), [&](const auto &kv) {
            const Metadata &m = kv.second;
            return m.name == v->name;
        });
        if (it == map.end())
            return call; // This is a call to a function.
        std::shared_ptr<ir::Function> &f = it->second.function;

        ir::Type type = ir::Function_t::make(f->ret_type, f->argument_types());
        return ir::Call::make(ir::Var::make(type, f->name), call->args);
    }
};

// Visitor class to retrieve a list of lambdas that should *not* be converted.
class Blacklist : public ir::Visitor {
  public:
    const std::unordered_set<const ir::Lambda *> &get() { return blacklist; }

  private:
    std::unordered_set<const ir::Lambda *> blacklist;

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
            blacklist.insert(op);
        }
        }
    }
};

// Performs the orchestration for lambda lowering.
ir::Program lower(const ir::Program &old_program) {
    // A mapping from lambda to metadata required for safe replacement.
    std::unordered_map<const ir::Lambda *, Metadata> map;

    // Some lambdas, e.g., those used in set queries, will not be converted.
    Blacklist blacklist;
    for (const auto &[_, f] : old_program.funcs) {
        if (const ir::Stmt &body = f->body; body.defined())
            body.accept(&blacklist);
    }
    if (const ir::Stmt &main = old_program.main_body; main.defined())
        main.accept(&blacklist);

    ConvertLambdaToFunction cltf(map, blacklist.get());
    ir::Program new_program;
    new_program.externs = old_program.externs;
    new_program.types = old_program.types;
    for (const auto &[f, func] : old_program.funcs) {
        ir::Stmt body = cltf.mutate(std::move(func->body));
        new_program.funcs[f] = std::make_shared<ir::Function>(
            func->name, func->args, func->ret_type, body, func->interfaces);
    }
    new_program.main_body = cltf.mutate(old_program.main_body);

    // Guarantee deterministic ordering for insertion.
    std::vector<std::shared_ptr<ir::Function>> functions;
    functions.reserve(map.size());
    std::transform(map.begin(), map.end(), std::back_inserter(functions),
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

ir::Program lower_lambda(const ir::Program &program) { return lower(program); }

} // namespace lower
} // namespace bonsai
