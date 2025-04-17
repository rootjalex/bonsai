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
// A counter
static int32_t counter = 0;

// TODO(cgyurgyik): This does not work for anything but the simple case.
class RtOP : public ir::Mutator {
  public:
    RtOP(ir::Function &current, ir::FuncMap &functions)
        : current(current), functions(functions) {}

  private:
    ir::Function &current;
    ir::FuncMap &functions;

    ir::Stmt visit(const ir::Return *node) override {
        ir::Expr value = node->value;
        if (!current.is_export) {
            return ir::Mutator::visit(node);
        }
        const auto &arguments = current.args;
        if (!arguments.front().mutating) {
            return ir::Mutator::visit(node);
        }
        std::string identifier = arguments.front().name;
        ir::WriteLoc location(std::move(identifier), value.type());
        return ir::Assign::make(location, std::move(value), /*mutating=*/true);
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
        auto &function = functions[function_name];
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
        std::string id = "$r" + std::to_string(counter++);
        ir::WriteLoc location(id, argument_type);
        std::vector<ir::Expr> args = {ir::Var::make(argument_type, id)};
        args.insert(args.end(), call->args.begin(), call->args.end());

        return ir::Sequence::make({
            ir::Assign::make(location, ir::Build::make(argument_type),
                             // TODO(cgyurgyik): Set to true.
                             /*mutating=*/false),
            ir::VoidCall::make(std::move(function_variable), std::move(args)),
        });
    }

    ir::Expr visit(const ir::Call *node) override { return node; }
};
} // namespace
ir::FuncMap ReturnToOutParameter::run(ir::FuncMap functions) const {
    ir::FuncMap new_functions;

    // First, update function argument and type signatures.
    std::vector<std::string> topological_order =
        func_topological_order(functions, /*undef_calls=*/false);
    for (const std::string &name : topological_order) {
        auto &function = functions[name];
        if (!function->is_export) {
            new_functions[name] = std::move(function);
            continue;
        }
        internal_assert(!is_recursive(*function))
            << "[unimplemented recursive [[export]] function: " << *function;
        ir::Type return_type = function->ret_type;
        const auto *struct_type = return_type.as<ir::Struct_t>();
        if (struct_type == nullptr) {
            new_functions[name] = std::move(function);
            continue;
        }
        // Update function arguments with additional mutable argument that
        // signifies the returned value.
        const auto &function_arguments = function->args;
        std::string argument_name = "$r" + std::to_string(counter++);
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
        ir::Stmt body = RtOP(*func, new_functions).mutate(func->body);
        func = func->replace_body(std::move(body));
    }

    return new_functions;
}

} // namespace lower
} // namespace bonsai
