#include "Lower/Externs.h"

#include "Lower/TopologicalOrder.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

using VarList = std::vector<ir::TypedVar>;

struct InsertExternsIntoCalls : public ir::Mutator {
    const std::map<std::string, VarList> &funcs_with_externs;
    const ir::FuncMap &funcs;

    InsertExternsIntoCalls(
        const std::map<std::string, VarList> &funcs_with_externs,
        const ir::FuncMap &funcs)
        : funcs_with_externs(funcs_with_externs), funcs(funcs) {}

    ir::Expr visit(const ir::Call *node) override {
        // Recurse into arguments to the call.
        ir::Expr rec = ir::Mutator::visit(node);
        node = rec.as<ir::Call>();
        internal_assert(node);
        // Check if this call is to a declared function
        const ir::Var *func = node->func.as<ir::Var>();
        if (func == nullptr) {
            return node;
        }
        // If the function is not labelled with externs, early out
        const auto &iter = funcs_with_externs.find(func->name);
        if (iter == funcs_with_externs.cend()) {
            return node;
        }
        // Now need to explicitly add externs to the function call.
        // First, make the new call type of the Var.
        const auto &fiter = funcs.find(func->name);
        internal_assert(fiter != funcs.cend());
        ir::Type call_type = fiter->second->call_type();
        ir::Expr callee = ir::Var::make(std::move(call_type), func->name);
        // Now build the new arguments list.
        std::vector<ir::Expr> args = node->args;
        args.insert(args.end(), iter->second.begin(), iter->second.end());
        return ir::Call::make(std::move(callee), std::move(args));
    }
};

} // namespace

ir::Program LowerExterns::run(ir::Program program,
                              const CompilerOptions &options) const {
    if (program.externs.empty()) {
        return program;
    }

    // Iterate in topological order, because callees that require explicit
    // extern arguments propagate that requirement to the caller.

    const std::vector<std::string> topo_order =
        lower::func_topological_order(program.funcs, /*undef_calls=*/false);

    std::map<std::string, VarList> funcs_with_externs;

    for (const std::string &f : topo_order) {
        auto &func = program.funcs[f];
        func->body = InsertExternsIntoCalls(funcs_with_externs, program.funcs)
                         .mutate(func->body);

        // Find free_vars AKA externs in the new body.
        const VarList free_vars = ir::gather_free_vars(*func);
        if (free_vars.empty()) {
            continue;
        }

        std::vector<ir::Function::Argument> new_args(free_vars.size());
        size_t counter = 0;
        // Insert externs in extern parsed order.
        for (const auto &ext : program.externs) {
            // Find free_var with matching name as ext, insert into new_args if
            // types match, error if types are !equal()
            const auto it = std::find_if(
                free_vars.cbegin(), free_vars.cend(),
                [&](const auto &var) { return var.name == ext.name; });
            if (it == free_vars.cend()) {
                continue;
            }
            internal_assert(ir::equals(ext.type, (*it).type))
                << "Lowering of extern found mistmatched type reference: "
                << ext.type << " vs. " << (*it).type;

            new_args[counter].name = ext.name;
            new_args[counter].type = ext.type;
            new_args[counter].mutating = false;
            counter++;
        }
        internal_assert(counter == free_vars.size())
            << "Free vars: " << free_vars.size() << " but added: " << counter
            << " args to: " << *func;
        // append new arguments to function call, and store this dependency for
        // calls to this func.
        func->args.insert(func->args.end(),
                          std::make_move_iterator(new_args.begin()),
                          std::make_move_iterator(new_args.end()));

        funcs_with_externs[f] = free_vars;

        // Handle recursive case.
        std::map<std::string, VarList> singleton;
        singleton[f] = std::move(free_vars);

        func->body =
            InsertExternsIntoCalls(singleton, program.funcs).mutate(func->body);
    }

    // TODO(ajr): would be ideal to clear here, but this breaks layout lowering.
    // program.externs.clear();

    return program;
}

} // namespace lower
} // namespace bonsai
