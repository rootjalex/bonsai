#include "Lower/ReturnToOutParameter.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
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

std::string get_unexported_name(const std::string &name) {
    return "_unexported_" + name;
}

class RtOP : public ir::Mutator {
  public:
    RtOP(const ir::Function &current) : current(current) {}

  private:
    const ir::Function &current;

    ir::Stmt visit(const ir::Return *node) override {
        ir::Expr value = node->value;
        if (!value.defined()) {
            return node;
        }
        const auto &arguments = current.args;
        internal_assert(arguments.front().mutating);
        std::string identifier = arguments.front().name;
        internal_assert(ir::equals(arguments.front().type, value.type()));
        ir::WriteLoc location(identifier, value.type());
        return ir::Sequence::make({
            ir::Assign::make(location, std::move(value), /*mutating=*/true),
            ir::Return::make(),
        });
    }
};

struct ReplaceExportedCalls : public ir::Mutator {
    const std::set<std::string> &exported_funcs;

    ReplaceExportedCalls(const std::set<std::string> &exported_funcs)
        : exported_funcs(exported_funcs) {}

    ir::Expr visit(const ir::Var *node) override {
        if (node->type.is_func() && exported_funcs.contains(node->name)) {
            return ir::Var::make(node->type, get_unexported_name(node->name));
        }
        return node;
    }
};

} // namespace

ir::FuncMap ReturnToOutParameter::run(ir::FuncMap functions,
                                      const CompilerOptions &options) const {
    ir::FuncMap new_functions;

    std::set<std::string> exported_funcs;

    // First, update function argument and type signatures.
    for (const auto &[name, function] : functions) {
        if (!function->is_exported()) {
            new_functions[name] = std::move(function);
            continue;
        }
        ir::Type return_type = function->ret_type;
        const auto *struct_type = return_type.as<ir::Struct_t>();
        if (struct_type == nullptr) {
            new_functions[name] = std::move(function);
            continue;
        }

        exported_funcs.insert(name);

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
            function->interfaces, function->attributes);
        // Keep the original function around for other functions that call it.
        std::string unexported = get_unexported_name(name);
        new_functions[unexported] = std::make_shared<ir::Function>(
            unexported, function->args, function->ret_type, function->body,
            function->interfaces, std::vector<ir::Function::Attribute>{});
    }

    // Next, update function bodies.
    for (auto &[name, func] : new_functions) {
        if (func->is_exported() && func->ret_type.is<ir::Void_t>()) {
            func->body = RtOP(*func).mutate(std::move(func->body));
        }
        func->body =
            ReplaceExportedCalls(exported_funcs).mutate(std::move(func->body));
    }
    return new_functions;
}

} // namespace lower
} // namespace bonsai
