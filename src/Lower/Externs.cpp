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
#include <sstream>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

using VarList = std::vector<ir::TypedVar>;

// A function's extern parameters -- the tail of its parameter list -- before
// and after this run gave it more: `before` as it had them, `after` as it has
// them now, in the order the externs were declared. A call's arguments follow
// the parameters, so a call passes its old tail's values where `after` puts
// their names and a new extern's own name where `after` names one it did not
// have.
struct ExternTail {
    std::vector<std::string> before;
    VarList after;
};

struct InsertExternsIntoCalls : public ir::Mutator {
    const std::map<std::string, ExternTail> &funcs_with_externs;
    const ir::FuncMap &funcs;

    InsertExternsIntoCalls(
        const std::map<std::string, ExternTail> &funcs_with_externs,
        const ir::FuncMap &funcs)
        : funcs_with_externs(funcs_with_externs), funcs(funcs) {}

    // The callee and the arguments a call needs, or nothing if it needs none.
    //
    // Shared by the three ways this IR spells a call: a Call is an expression,
    // a CallStmt is a statement, and a MultiRecurse is a run of statements
    // sharing a callee. Only the first used to be rewritten here, and what that
    // cost was a whole class of call being left with too few arguments -- see
    // Lower/RecLoops.cpp, which builds the self-call of a loopified recursion
    // as a CallStmt, and now as a MultiRecurse.
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
        const ExternTail &tail = iter->second;
        // The callee's parameters are its own and then `after`; the call's
        // arguments are its own and then the values of `before`.
        internal_assert(fiter->second->args.size() >= tail.after.size());
        const size_t own = fiter->second->args.size() - tail.after.size();
        internal_assert(args.size() == own + tail.before.size())
            << "a call of `" << name->name << "` passes " << args.size()
            << " arguments where the function took " << own << " of its own and "
            << tail.before.size() << " externs";
        std::vector<ir::Expr> whole(args.begin(), args.begin() + own);
        for (const ir::TypedVar &var : tail.after) {
            const auto had = std::find(tail.before.begin(), tail.before.end(),
                                       var.name);
            whole.push_back(had != tail.before.end()
                                ? args[own + size_t(had - tail.before.begin())]
                                : ir::Var::make(var.type, var.name));
        }
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

    ir::Stmt visit(const ir::MultiRecurse *node) override {
        ir::Stmt rec = ir::Mutator::visit(node);
        node = rec.as<ir::MultiRecurse>();
        internal_assert(node);
        // The externs land after the arguments the call already had, so the
        // varying positions -- which index the leading arguments -- still name
        // the same things.
        auto whole = with_externs(node->func, node->args);
        if (!whole.has_value()) {
            return node;
        }
        return ir::MultiRecurse::make(
            std::move(whole->first), std::move(whole->second), node->varying_at,
            node->varying, node->keys);
    }
};

} // namespace

ir::Program LowerExterns::run(ir::Program program,
                              const CompilerOptions &options) const {
    if (program.externs.empty()) {
        return program;
    }

    // Callees first, because a callee that takes explicit extern arguments
    // passes that requirement to its callers -- and a strongly connected
    // component of the call graph at a time, because a mutual recursion (the
    // material kernel handing the next ray back to the trace kernel) has no
    // callee-first order inside it. Each member of a component reaches every
    // other, so the component's externs are the union of its members', every
    // member takes them all, and the calls among the members pass them on. A
    // function that calls itself is the component of one, the same way.

    std::map<std::string, ExternTail> funcs_with_externs;
    // Which of a function's extern parameters it writes to, directly or
    // through something it calls. Filled callee-first, so a callee's answer
    // is known by the time a caller needs it.
    std::map<std::string, std::set<std::string>> writes_externs;
    // Where each extern was declared: the order the parameters go in.
    std::map<std::string, size_t> declared_at;
    for (size_t i = 0; i < program.externs.size(); i++) {
        declared_at[program.externs[i].name] = i;
    }
    // Whether a parameter is an extern the function took in an earlier run
    // of this pass: an extern's name and type. (A run adds the externs a
    // pass since the last run made free -- the pools a layout made of an
    // element's arms, the reads of a query's hit programs -- and they are
    // as often declared before the ones the function already has as after.)
    const auto is_extern = [&](const ir::Function::Argument &arg) {
        const auto at = declared_at.find(arg.name);
        return at != declared_at.end() &&
               ir::equals(program.externs[at->second].type, arg.type);
    };

    for (const std::vector<std::string> &component :
         lower::func_scc_order(program.funcs)) {
        // Calls out of the component pass what their callees take now.
        for (const std::string &f : component) {
            auto &func = program.funcs[f];
            func->body =
                InsertExternsIntoCalls(funcs_with_externs, program.funcs)
                    .mutate(func->body);
        }

        // Find free_vars AKA externs in the new bodies: over the component,
        // each name once.
        VarList free_vars;
        // An extern this function writes to arrives as a mutating parameter.
        // Storing through one that says it does not mutate leaves the write
        // with nowhere to land: what a non-mutating parameter names is a copy,
        // and the passes downstream that turn a write into a store through a
        // pointer go looking for a definition of the name and find none. In
        // C++ it is the difference between `Sph*` and `const Sph*`.
        //
        // Writing to one includes handing it to something that writes to it.
        // A function that only forwards an extern still has to take it
        // mutably, or the call it forwards to will not type-check. Over a
        // component that is one answer for every member, since each can reach
        // the write through the others.
        std::set<std::string> written;
        for (const std::string &f : component) {
            auto &func = program.funcs[f];
            for (const auto &var : ir::gather_free_vars(*func)) {
                const bool seen = std::any_of(
                    free_vars.cbegin(), free_vars.cend(),
                    [&](const auto &v) { return v.name == var.name; });
                if (!seen) {
                    free_vars.push_back(var);
                }
            }
            const std::set<std::string> own = ir::mutated_variables(func->body);
            written.insert(own.begin(), own.end());
            for (const std::string &callee : ir::called_functions(func->body)) {
                const auto found = writes_externs.find(callee);
                if (found != writes_externs.end()) {
                    written.insert(found->second.begin(), found->second.end());
                }
            }
        }
        if (free_vars.empty()) {
            continue;
        }
        for (const std::string &f : component) {
            writes_externs[f] = written;
        }

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
        // Everything free in a function at this point should be an extern, so
        // when one is not, the useful thing to say is which -- the function
        // printed below is long, and a name that got loose in it is not easy
        // to spot by reading.
        if (counter != free_vars.size()) {
            std::ostringstream missing;
            for (const auto &var : free_vars) {
                const bool declared =
                    std::any_of(program.externs.cbegin(),
                                program.externs.cend(), [&](const auto &ext) {
                                    return ext.name == var.name;
                                });
                if (!declared) {
                    missing << " " << var.name << " : " << var.type;
                }
            }
            internal_error << "Free vars: " << free_vars.size()
                           << " but added: " << counter
                           << " args. Not declared as externs:" << missing.str()
                           << "\nin: " << *program.funcs[component.front()];
        }
        // Every member takes the component's externs as parameters, and the
        // calls of the members -- from outside, later, and among themselves,
        // a function's of itself included, now -- pass them along.
        //
        // In declaration order, over the externs it had and the new ones
        // together: the extern parameters are the tail of the list, and a
        // function that took some in an earlier run of this pass has the new
        // ones merged into that tail rather than appended after it, so that
        // the order is the declaration's however many runs it took to find
        // them all. An exported function's caller -- a driver written by
        // hand against the generated header -- passes the externs in the
        // order they were declared, and that order must not depend on which
        // pass made an extern free. The calls follow (ExternTail).
        std::map<std::string, ExternTail> within;
        for (const std::string &f : component) {
            auto &func = program.funcs[f];
            size_t own = func->args.size();
            while (own > 0 && is_extern(func->args[own - 1])) {
                own--;
            }
            ExternTail tail;
            std::vector<ir::Function::Argument> merged(
                func->args.begin() + own, func->args.end());
            for (const ir::Function::Argument &arg : merged) {
                tail.before.push_back(arg.name);
                internal_assert(std::none_of(new_args.begin(), new_args.end(),
                                             [&](const auto &a) {
                                                 return a.name == arg.name;
                                             }))
                    << f << " already takes the extern `" << arg.name
                    << "` it was found reading freely";
            }
            for (ir::Function::Argument &arg : merged) {
                // The component may write now to what it only read before.
                arg.mutating = arg.mutating || written.contains(arg.name);
            }
            merged.insert(merged.end(), new_args.begin(), new_args.end());
            std::sort(merged.begin(), merged.end(),
                      [&](const auto &a, const auto &b) {
                          return declared_at.at(a.name) < declared_at.at(b.name);
                      });
            func->args.erase(func->args.begin() + own, func->args.end());
            func->args.insert(func->args.end(), merged.begin(), merged.end());
            for (const ir::Function::Argument &arg : merged) {
                tail.after.emplace_back(arg.name, arg.type);
            }
            funcs_with_externs[f] = tail;
            within[f] = tail;
        }
        for (const std::string &f : component) {
            auto &func = program.funcs[f];
            func->body =
                InsertExternsIntoCalls(within, program.funcs).mutate(func->body);
        }
    }

    // TODO(ajr): would be ideal to clear here, but this breaks layout lowering.
    // program.externs.clear();

    return program;
}

} // namespace lower
} // namespace bonsai
