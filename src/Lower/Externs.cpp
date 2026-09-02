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

    // The callee and the arguments a call needs, or nothing if it needs none.
    //
    // Shared by the two ways this IR spells a call, which differ only in
    // whether the result is used: a Call is an expression and a CallStmt is a
    // statement. Only the first used to be rewritten here, and what that cost
    // was a whole class of call being left with too few arguments -- see
    // Lower/RecLoops.cpp, which builds the self-call of a loopified recursion
    // as a CallStmt.
    std::optional<std::pair<ir::Expr, std::vector<ir::Expr>>>
    with_externs(const ir::Expr &func, const std::vector<ir::Expr> &args) const {
        const ir::Var *name = func.as<ir::Var>();
        if (name == nullptr) {
            return std::nullopt;
        }
        const auto iter = funcs_with_externs.find(name->name);
        if (iter == funcs_with_externs.cend()) {
            return std::nullopt;
        }
        const auto fiter = funcs.find(name->name);
        internal_assert(fiter != funcs.cend());
        std::vector<ir::Expr> whole = args;
        whole.insert(whole.end(), iter->second.begin(), iter->second.end());
        // The callee's type has to be remade too: it now takes more.
        return std::make_pair(
            ir::Var::make(fiter->second->call_type(), name->name),
            std::move(whole));
    }

    ir::Expr visit(const ir::Call *node) override {
        // Recurse into arguments to the call.
        ir::Expr rec = ir::Mutator::visit(node);
        node = rec.as<ir::Call>();
        internal_assert(node);
        auto whole = with_externs(node->func, node->args);
        if (!whole.has_value()) {
            return node;
        }
        return ir::Call::make(std::move(whole->first), std::move(whole->second));
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        ir::Stmt rec = ir::Mutator::visit(node);
        node = rec.as<ir::CallStmt>();
        internal_assert(node);
        auto whole = with_externs(node->func, node->args);
        if (!whole.has_value()) {
            return node;
        }
        return ir::CallStmt::make(std::move(whole->first),
                                  std::move(whole->second));
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
    // Which of a function's extern parameters it writes to, directly or
    // through something it calls. Filled in topological order, so a callee's
    // answer is known by the time a caller needs it.
    std::map<std::string, std::set<std::string>> writes_externs;

    for (const std::string &f : topo_order) {
        auto &func = program.funcs[f];
        func->body = InsertExternsIntoCalls(funcs_with_externs, program.funcs)
                         .mutate(func->body);

        // Find free_vars AKA externs in the new body.
        const VarList free_vars = ir::gather_free_vars(*func);
        if (free_vars.empty()) {
            continue;
        }
        // An extern this function writes to arrives as a mutating parameter.
        // Storing through one that says it does not mutate leaves the write
        // with nowhere to land: what a non-mutating parameter names is a copy,
        // and the passes downstream that turn a write into a store through a
        // pointer go looking for a definition of the name and find none. In
        // C++ it is the difference between `Sph*` and `const Sph*`.
        //
        // Writing to one includes handing it to something that writes to it.
        // A function that only forwards an extern still has to take it
        // mutably, or the call it forwards to will not type-check.
        std::set<std::string> written = ir::mutated_variables(func->body);
        for (const std::string &callee : ir::called_functions(func->body)) {
            const auto found = writes_externs.find(callee);
            if (found != writes_externs.end()) {
                written.insert(found->second.begin(), found->second.end());
            }
        }
        writes_externs[f] = written;

        std::vector<ir::Function::Argument> new_args(free_vars.size());
        // The same externs in the same order, to hand to the callers.
        //
        // These have to be one order, and it cannot be the order
        // gather_free_vars returns: the parameters below are appended in the
        // order the externs were declared, so a caller passing them in
        // discovery order lines the arguments up wrongly. With a single extern
        // the two orders agree and nothing shows; with several they do not,
        // and what surfaces is a type mismatch at a call whose arguments were
        // never written down by hand.
        VarList ordered(free_vars.size());
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
            new_args[counter].mutating = written.contains(ext.name);
            ordered[counter] = *it;
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

        funcs_with_externs[f] = ordered;

        // Handle recursive case.
        //
        // `ordered`, not `free_vars`: the parameters were appended in the order
        // the externs were declared, so a self-call passing them in the order
        // they were discovered lines the arguments up wrongly. The same reason
        // the comment above gives, and the same thing it warns is invisible
        // with a single extern.
        std::map<std::string, VarList> singleton;
        singleton[f] = std::move(ordered);

        func->body =
            InsertExternsIntoCalls(singleton, program.funcs).mutate(func->body);
    }

    // TODO(ajr): would be ideal to clear here, but this breaks layout lowering.
    // program.externs.clear();

    return program;
}

} // namespace lower
} // namespace bonsai
