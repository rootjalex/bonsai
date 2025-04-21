#include "Opt/Inline.h"

#include "Error.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <unordered_map>

namespace bonsai {
namespace opt {

namespace {

class Inliner : public ir::Mutator {
  public:
    Inliner(const ir::FuncMap &functions,
            const std::unordered_map<std::string, ir::Expr> &function_to_expr)
        : functions(functions), function_to_expr(function_to_expr) {}

    ir::Expr visit(const ir::Call *node) override {
        const ir::Var *v = node->func.as<ir::Var>();
        if (v == nullptr) {
            return node;
        }
        const std::string &function_name = v->name;
        auto it = function_to_expr.find(function_name);
        if (it == function_to_expr.end()) {
            return node;
        }
        auto f = functions.find(function_name);
        internal_assert(f != functions.end());
        std::vector<std::string> argument_names;
        const std::vector<ir::Function::Argument> &args = f->second->args;
        std::transform(args.begin(), args.end(),
                       std::back_inserter(argument_names),
                       [](const auto &a) { return a.name; });
        // Replace function arguments with call arguments.
        std::map<std::string, ir::Expr> repls;
        internal_assert(argument_names.size() == node->args.size());
        for (int i = 0, e = argument_names.size(); i < e; ++i) {
            repls[argument_names[i]] = node->args[i];
        }
        return replace(repls, it->second);
    }

  private:
    const ir::FuncMap &functions;
    const std::unordered_map<std::string, ir::Expr> &function_to_expr;
};

} // namespace

ir::FuncMap Inline::run(ir::FuncMap funcs) const {
    // If a function simply returns a value, replace the call with said value.
    // We refrain from performing more complex inlining for now to avoid code
    // size blowup.
    std::unordered_map<std::string, ir::Expr> function_to_expr;
    for (const auto &[name, func] : funcs) {
        if (const auto *body = func->body.as<ir::Return>()) {
            internal_assert(body->value.defined());
            function_to_expr[name] = body->value;
        }
    }

    // We assume the inliner will not change the number of arguments in a
    // function, and thus it is ok to only instantiate it once per program. If
    // the `Inliner` class were to break this assumption, this would need to
    // change.
    Inliner inliner(funcs, function_to_expr);
    for (auto &[name, func] : funcs) {
        func->body = inliner.mutate(std::move(func->body));
    }

    return funcs;
}

} // namespace opt
} // namespace bonsai
