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

// Canonicalizes a lambda to a struct. For example,
//     x: i32 = 42;
//     L2 = |y: i32| y + x + 1;
//     L2(1);
// ->
//   x: i32 = 42;
//   let L2 = build<?lambda0>(x)(y : i32){ y + x + 1 }
//   L2(1);
class ConvertLambdaToStruct : public ir::Mutator {
  public:
    ConvertLambdaToStruct(
        const std::unordered_set<const ir::Lambda *> &blacklisted_lambdas)
        : blacklisted_lambdas(blacklisted_lambdas) {};

    // Clean up data structures between function visits.
    void clear() {
        visiting_lambda_body = false;
        frames.clear();
        implicit_variables.clear();
        name_to_lambda.clear();
    }

  private:
    // Lambdas that should not be visited.
    const std::unordered_set<const ir::Lambda *> &blacklisted_lambdas;

    int64_t counter = 0;
    std::string generate_name() {
        std::string name = "?lambda";
        name += std::to_string(counter++);
        return name;
    }

    // Demarcates when we are visiting the body of a lambda. This is necessary
    // to ensure we are only updating variable names located inside the body.
    bool visiting_lambda_body = false;

    // Implicitly captured arguments for this lambda.
    std::vector<const ir::Var *> implicit_variables;

    // A map from lambda variable name to lambda pointer. This is required to
    // update the respective calls.
    std::unordered_map<std::string, const ir::Lambda *> name_to_lambda;

    // Represents a scoped environment within the Bonsai program.
    struct Frame {
        // Explicitly captured arguments within the scope of this lambda.
        std::vector<std::string> explicit_variables;
    };
    std::vector<Frame> frames;

    void new_frame() { frames.push_back(Frame{}); }

    std::vector<std::string> &explicit_variables() {
        return frames.back().explicit_variables;
    }

    // Returns whether this is an explicit variable in the current frame.
    bool is_explicit_variable(const ir::Var *variable) {
        const std::vector<std::string> &ev = explicit_variables();
        return std::find(ev.begin(), ev.end(), variable->name) != ev.end();
    }
    void pop_frame() { frames.pop_back(); }

    ir::Expr visit(const ir::Var *var) override {
        if (!visiting_lambda_body) {
            return var;
        }

        if (is_explicit_variable(var)) {
            return var; // This variable has already been explicitly captured.
        }

        implicit_variables.push_back(var);
        return var;
    }

    ir::Expr visit(const ir::Lambda *lambda) override {
        if (blacklisted_lambdas.contains(lambda))
            return lambda;

        new_frame();
        std::vector<ir::Argument> args = lambda->args;
        std::transform(args.begin(), args.end(),
                       std::back_inserter(explicit_variables()),
                       [](const ir::Argument &a) { return a.name; });

        visiting_lambda_body = true;
        ir::Expr value = mutate(lambda->value);
        visiting_lambda_body = false;

        ir::Struct_t::Map fields;
        ir::Struct_t::DefMap values;
        for (const ir::Var *v : implicit_variables) {
            if (is_explicit_variable(v)) {
                continue;
            }
            fields.push_back({v->name, v->type});
            values[v->name] = v;
        }

        // Propagate implicitly captured variables to parent lambdas.
        std::vector<const ir::Var *> new_implicit_variables;
        std::copy_if(
            implicit_variables.begin(), implicit_variables.end(),
            std::back_inserter(new_implicit_variables), [&](const ir::Var *v) {
                const std::vector<std::string> &ev = explicit_variables();
                return std::find(ev.begin(), ev.end(), v->name) == ev.end();
            });
        implicit_variables = std::move(new_implicit_variables);
        pop_frame();

        ir::Type type =
            ir::Struct_t::make(generate_name(), fields, lambda->type);
        auto call = ir::Build::Call{
            .args = lambda->args,
            .value = value,
        };
        return ir::Build::make(type, values, call);
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
        clts.clear();
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
