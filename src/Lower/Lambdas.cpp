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

// Explicitly adds additional arguments to the lambda for implicitly captured
// variables. For example,
//
//     x: i32 = 42;
//     L = |y: i32| y + x + 1;
//     L(1);
//
//   =>
//
//     x: i32 = 42;
//     L = |y: i32, $x: i32| y + $x + 1;
//     L(1, x);
struct LambdaImplicitCapture : public ir::Mutator {
  public:
    LambdaImplicitCapture() {}

  private:
    // Character used when adding explicit arguments to lambdas for their
    // previously implicit capture.
    static constexpr std::string_view LAMBDA_PREFIX = "$";

    // Demarcates when we are visiting the body of a lambda. This is necessary
    // to ensure we are only updating variable names located inside the body.
    bool visiting_lambda = false;

    // Explicitly captured arguments within the scope of this lambda.
    std::vector<std::string> explicit_variables;

    // Implicitly captured arguments for this lambda.
    std::vector<const ir::Var *> implicit_variables;

    // A map from lambda variable name to lambda pointer. This is required to
    // update the respective calls.
    std::unordered_map<std::string, const ir::Lambda *> name_to_lambda;

    // Prefixes `v` with the LAMBDA_PREFIX symbol.
    std::string generate_name(std::string_view v) {
        std::string name(LAMBDA_PREFIX);
        name += v;
        return name;
    }

    ir::Stmt visit(const ir::LetStmt *let) override {
        ir::WriteLoc loc = let->loc;
        internal_assert(loc.accesses.empty()) << "unimplemented";

        ir::Expr rhs = let->value;
        if (const ir::Call *call = rhs.as<ir::Call>()) {
            return ir::LetStmt::make(loc, visit(call));
        }
        if (const ir::Lambda *lambda = rhs.as<ir::Lambda>()) {
            ir::Expr updated_lambda = visit(lambda);
            name_to_lambda[loc.base] = updated_lambda.as<ir::Lambda>();
            return ir::LetStmt::make(loc, updated_lambda);
        }
        return let;
    }

    ir::Expr visit(const ir::Var *var) override {
        if (!visiting_lambda) {
            return var;
        }

        if (auto it = std::find(explicit_variables.begin(),
                                explicit_variables.end(), var->name);
            it != explicit_variables.end()) {
            // This variable has already been explicitly captured.
            return var;
        }

        std::string new_name = generate_name(var->name);
        ir::Expr new_variable = ir::Var::make(var->type, std::move(new_name));
        implicit_variables.push_back(new_variable.as<ir::Var>());
        return new_variable;
    }

    ir::Expr visit(const ir::Lambda *lambda) override {
        const unsigned ev_size = explicit_variables.size(),
                       iv_size = implicit_variables.size();

        std::vector<ir::Lambda::Argument> args = lambda->args;
        const unsigned before = args.size();
        // 1. Update the explicit variables before visiting the lambda body.
        std::transform(args.begin(), args.end(),
                       std::back_inserter(explicit_variables),
                       [](const ir::Lambda::Argument &a) { return a.name; });

        // 2. Visit the lambda body, capturing the implicit variables.
        visiting_lambda = true;
        ir::Expr value = this->mutate(lambda->value);
        visiting_lambda = false;

        // 3. Update the lambda arguments with any implicit variables.
        std::transform(implicit_variables.begin(), implicit_variables.end(),
                       std::back_inserter(args), [](const ir::Var *v) {
                           return ir::Lambda::Argument{
                               .name = v->name,
                               .type = v->type,
                           };
                       });

        // 4. Finally, pop off the explicit/implict variables for this lambda.
        while (explicit_variables.size() > ev_size) {
            explicit_variables.pop_back();
        }
        while (implicit_variables.size() > iv_size) {
            implicit_variables.pop_back();
        }

        if (const unsigned after = args.size(); before == after) {
            // No additional arguments were added.
            return lambda;
        }
        return ir::Lambda::make(args, value);
    }

    ir::Expr visit(const ir::Call *call) override {
        const ir::Var *v = call->func.as<ir::Var>();
        if (v == nullptr)
            return call;

        auto it = name_to_lambda.find(v->name);
        if (it == name_to_lambda.end()) {
            return call; // This is a call to a non-lambda.
        }

        const ir::Lambda *lambda = it->second;
        const std::vector<ir::Lambda::Argument> &largs = lambda->args;
        std::vector<ir::Expr> cargs = call->args;
        if (cargs.size() == largs.size()) {
            return call; // No implicit arguments were added.
        }

        const ir::Function_t *vtype = v->type.as<ir::Function_t>();
        std::vector<ir::Type> ctypes = vtype->arg_types;
        // Update the lambda arguments and type to include the (previously)
        // implicit arguments.
        for (unsigned i = cargs.size(), e = largs.size(); i < e; ++i) {
            const ir::Lambda::Argument &arg = largs[i];
            // We are copying the implicit arguments from the new lambda
            // signature, which will be prefixed with a special character (for
            // name hygiene purposes). However, the values being passed in
            // should retain their original name, so we remove the prefix.
            std::string name = arg.name;
            internal_assert(name.starts_with(LAMBDA_PREFIX))
                << "implicit arguments should be prefixed with `"
                << LAMBDA_PREFIX << "`, received: " << name
                << " for lambda: " << ir::Expr(lambda);
            cargs.push_back(ir::Var::make(arg.type, name.substr(1)));
            ctypes.push_back(arg.type);
        }

        ir::Expr new_variable = ir::Var::make(
            ir::Function_t::make(vtype->ret_type, std::move(ctypes)), v->name);
        return ir::Call::make(std::move(new_variable), std::move(cargs));
    }
};

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
        std::string name = "?lambda";
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
        const std::vector<ir::Lambda::Argument> &before = lambda->args;
        std::vector<ir::Function::Argument> arguments;
        std::transform(before.begin(), before.end(),
                       std::back_inserter(arguments),
                       [](const ir::Lambda::Argument &a) {
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
            /*interfaces=*/ir::Function::InterfaceList{});

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

        ir::Type type = ir::Function_t::make(f->ret_type, f->argument_types());
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
    ir::Program new_program;
    new_program.externs = old_program.externs;
    new_program.types = old_program.types;

    // Ensure lambdas explicitly capture their implicit arguments.
    for (const auto &[f, func] : old_program.funcs) {
        ir::Stmt body = LambdaImplicitCapture().mutate(std::move(func->body));
        new_program.funcs[f] = std::make_shared<ir::Function>(
            func->name, func->args, func->ret_type, std::move(body),
            func->interfaces);
    }

    // A mapping from lambda to metadata required for safe replacement.
    std::unordered_map<const ir::Lambda *, Metadata> lambda_metadata;

    // Some lambdas, e.g., those used in set queries, will not be converted.
    Blacklist blacklisted_lambdas;
    for (const auto &[_, f] : new_program.funcs) {
        if (const ir::Stmt &body = f->body; body.defined()) {
            body.accept(&blacklisted_lambdas);
        }
    }

    ConvertLambdaToFunction cltf(lambda_metadata, blacklisted_lambdas.get());
    for (const auto &[f, func] : new_program.funcs) {
        ir::Stmt body = cltf.mutate(std::move(func->body));
        new_program.funcs[f] = std::make_shared<ir::Function>(
            func->name, func->args, func->ret_type, std::move(body),
            func->interfaces);
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

ir::Program LowerLambda::lower(const ir::Program &program) const {
    return lower_program(program);
}

} // namespace lower
} // namespace bonsai
