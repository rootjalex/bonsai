#include "Lower/Lambdas.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

struct FindImplicitCaptures : public ir::Visitor {
    const std::set<std::string> &lambda_args;

    ir::Struct_t::Map implicits;
    std::vector<ir::Expr> build_args;
    std::set<std::string> seen;

    FindImplicitCaptures(const std::set<std::string> &lambda_args) : lambda_args(lambda_args) {}

    // TODO: could optimize further by capturing accesses, e.g.
    // x : struct = { . . . }
    // l = |i : i32| i + x.a; // should (maybe?) capture x.a only.

    void visit(const ir::Var *node) override {
        if (lambda_args.contains(node->name)) return;

        auto [_, inserted] = seen.insert(node->name);
        if (inserted) {
            implicits.emplace_back(node->name, node->type);
            build_args.emplace_back(node);
        }
    }
};

struct ConvertLambdas : public ir::Mutator {
    std::vector<std::pair<std::string, ir::Type>> ordered_types;

    size_t counter = 0;

    std::string generate_name() {
        return "?lambda" + std::to_string(counter++);
    }

    ir::Expr visit(const ir::Lambda *node) override {
        static const std::string call_name = "call";

        // Recursively mutate (handles nested lambdas).
        ir::Expr value = mutate(node->value);

        // Build callable Function.
        std::vector<ir::Function::Argument> f_args;
        std::transform(node->args.begin(), node->args.end(),
            std::inserter(f_args, f_args.end()),
            [](const auto &arg) { return ir::Function::Argument(arg.name, arg.type); });
        ir::Type f_ret_type = value.type();
        internal_assert(f_ret_type.defined())
            << "Lambda lowering called before type inference ran: "
            << ir::Expr(node);
        ir::Stmt f_body = ir::Return::make(std::move(value));
        ir::Function::InterfaceList interfaces; // TODO: will this ever be used?
        std::shared_ptr<ir::Function> f = std::make_shared<ir::Function>(call_name, std::move(f_args), std::move(f_ret_type), std::move(f_body), interfaces);
        ir::Struct_t::MethodMap methods = {{call_name, std::move(f)}};

        // Find implicitly-captured variables.
        std::set<std::string> args;
        std::transform(node->args.begin(), node->args.end(),
                   std::inserter(args, args.end()),
                   [](const auto &arg) { return arg.name; });
        FindImplicitCaptures finder(args);
        node->value.accept(&finder);

        // Build a struct with implicitly captured vars as fields,
        // and a single callable method.
        std::string struct_name = generate_name();
        ir::Type type = ir::Struct_t::make(struct_name, std::move(finder.implicits), std::move(methods));
        ordered_types.emplace_back(std::move(struct_name), type);
        ir::Expr build = ir::Build::make(std::move(type), std::move(finder.build_args));
        return ir::Access::make(call_name, std::move(build));
    }

    ir::Expr visit(const ir::SetOp *node) override {
        // internal_error << "SetOps should be lowered before lambdas: " << ir::Expr(node);
        return node;
    }
};

ir::Program lower_program(const ir::Program &old_program) {
    ir::Program new_program;
    new_program.externs = old_program.externs;
    new_program.types = old_program.types;

    ConvertLambdas converter;
    for (const auto &[f, func] : old_program.funcs) {
        ir::Stmt body = converter.mutate(std::move(func->body));
        new_program.funcs[f] = std::make_shared<ir::Function>(
            func->name, func->args, func->ret_type, body, func->interfaces);
    }

    for (auto &[name, type] : converter.ordered_types) {
        // Don't std::move(name) because of error message printing.
        const auto [_, inserted] = new_program.types.try_emplace(name, std::move(type));
        internal_assert(inserted) << "Lambda struct name already exists in types: " << name;
    }

    return new_program;
}

} // namespace

ir::Program LowerLambda::run(ir::Program program) const {
    return lower_program(program);
}

} // namespace lower
} // namespace bonsai
