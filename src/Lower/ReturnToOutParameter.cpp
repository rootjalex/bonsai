#include "Lower/ReturnToOutParameter.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {
// A counter for uniquely naming the newly created return values.
static int32_t counter = 0;
// Name of the new return value parameter.
static constexpr std::string_view PARAMETER_NAME = "_ret";

class RtOP : public ir::Mutator {
  public:
    RtOP(const ir::Function &current, const ir::FuncMap &functions)
        : current(current), functions(functions) {}

  private:
    const ir::Function &current;
    const ir::FuncMap &functions;

    // If this is an exported call nested within another exported call, we
    // handle the updated call statement here.
    std::optional<ir::Stmt> get_nested_export_call(ir::Expr node,
                                                   const ir::WriteLoc &loc) {
        const auto *call = node.as<ir::Call>();
        if (call == nullptr) {
            return {};
        }
        const auto *v = call->func.as<ir::Var>();
        internal_assert(v) << call->func;
        auto it = functions.find(v->name);
        if (it == functions.end()) {
            return {};
        }
        const ir::Function &f = *it->second;
        if (!f.is_export || f.args.empty()) {
            return {};
        }
        if (!f.args[0].name.starts_with(PARAMETER_NAME)) {
            // We assume this is the only place that will use such a name.
            return {};
        }
        ir::Expr updated = ir::Var::make(f.call_type(), f.name);
        std::vector<ir::Expr> arguments = {ir::Var::make(loc.type, loc.base)};
        arguments.insert(arguments.end(), call->args.begin(), call->args.end());
        return ir::CallStmt::make(std::move(updated), std::move(arguments));
    }

    ir::Stmt visit(const ir::Return *node) override {
        ir::Expr value = node->value;
        if (!value.defined()) {
            return node;
        }
        if (!current.is_export) {
            return ir::Mutator::visit(node);
        }
        const auto &arguments = current.args;
        if (!arguments.front().mutating) {
            return ir::Mutator::visit(node);
        }
        std::string identifier = arguments.front().name;
        ir::WriteLoc location(identifier, value.type());
        if (std::optional<ir::Stmt> call =
                get_nested_export_call(value, location)) {
            return ir::Sequence::make({*call, ir::Return::make()});
        }
        return ir::Sequence::make({
            ir::Assign::make(location, std::move(value), /*mutating=*/true),
            ir::Return::make(),
        });
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        const auto *call = node->value.as<ir::Call>();
        if (call == nullptr) {
            return ir::Mutator::visit(node);
        }
        ir::Expr func = call->func;
        const auto *f = func.as<ir::Var>();
        internal_assert(f) << func;
        std::string function_name = f->name;
        auto it = functions.find(function_name);
        internal_assert(it != functions.end()) << function_name;
        const auto &function = it->second;
        if (!function->is_export) {
            return ir::Mutator::visit(node);
        }
        const auto &arguments = function->args;
        if (!arguments.front().mutating) {
            return ir::Mutator::visit(node);
        }
        const ir::Type argument_type = arguments.front().type;
        auto function_variable =
            ir::Var::make(function->call_type(), function_name);

        ir::WriteLoc location(node->loc.base, argument_type);
        std::vector<ir::Expr> args = {
            ir::Var::make(argument_type, node->loc.base)};
        args.insert(args.end(), call->args.begin(), call->args.end());

        return ir::Sequence::make({
            ir::Assign::make(location, ir::Build::make(argument_type),
                             /*mutating=*/false),
            ir::CallStmt::make(std::move(function_variable), std::move(args)),
        });
    }

    ir::Expr visit(const ir::Call *node) override { return node; }
};
} // namespace

ir::FuncMap ReturnToOutParameter::run(ir::FuncMap functions) const {
    ir::FuncMap new_functions;

    // First, update function argument and type signatures.
    for (const auto &[name, function] : functions) {
        if (!function->is_export) {
            new_functions[name] = std::move(function);
            continue;
        }
        ir::Type return_type = function->ret_type;
        const auto *struct_type = return_type.as<ir::Struct_t>();
        if (struct_type == nullptr) {
            new_functions[name] = std::move(function);
            continue;
        }

        // Update function arguments with additional mutable argument that
        // signifies the returned value.
        const auto &function_arguments = function->args;
        std::string argument_name =
            std::string(PARAMETER_NAME) + std::to_string(counter++);
        std::vector<ir::Function::Argument> arguments = {
            ir::Function::Argument(
                /*name=*/argument_name,
                /*type=*/return_type,
                /*default_value=*/ir::Expr(),
                /*mutating=*/true),
        };
        arguments.insert(arguments.end(), function_arguments.begin(),
                         function_arguments.end());
        new_functions[name] = std::make_shared<ir::Function>(
            name, arguments, ir::Void_t::make(), function->body,
            function->interfaces, function->is_export);
    }

    // Next, update function bodies.
    for (auto &[name, func] : new_functions) {
        func->body = RtOP(*func, new_functions).mutate(func->body);
    }
    return new_functions;
}

} // namespace lower
} // namespace bonsai
