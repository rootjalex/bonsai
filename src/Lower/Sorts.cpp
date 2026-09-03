#include "Lower/Sorts.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

using namespace ir;

std::map<std::string, Type> get_names_in_scope(const Function &func) {
    std::map<std::string, Type> args;
    for (const auto &arg : func.args) {
        args.try_emplace(arg.name, arg.type);
    }
    return args;
}

Stmt apply_sort(const Location &loc, const Expr &cost_func, Stmt stmt,
                FuncMap &funcs, std::map<std::string, Type> names_in_scope) {
    struct ApplySortImpl : public Mutator {
        const Location &loc;
        const Expr &cost_func;
        FuncMap &funcs;
        bool found_match = false;
        bool found_from = false;

        std::map<std::string, Type> names_in_scope;
        Expr current_match_arg;

        ApplySortImpl(const Location &loc, const Expr &cost_func,
                      FuncMap &funcs,
                      std::map<std::string, Type> names_in_scope)
            : loc(loc), cost_func(cost_func), funcs(funcs),
              names_in_scope(std::move(names_in_scope)) {}

        Stmt visit(const LetStmt *node) override {
            names_in_scope.try_emplace(node->loc.base, node->loc.base_type);
            return Mutator::visit(node);
        }

        Stmt visit(const Allocate *node) override {
            names_in_scope.try_emplace(node->loc.base, node->loc.base_type);
            return Mutator::visit(node);
        }

        Expr sort_cost(size_t i) const {
            const Lambda *lambda = cost_func.as<Lambda>();
            internal_assert(lambda) << cost_func;
            internal_assert(!lambda->args.empty()) << cost_func;
            const std::string &idx = lambda->args[0].name;
            Expr value = make_const(lambda->args[0].type, i);

            std::map<std::string, Expr> temp_repls;

            for (size_t i = 1; i < lambda->args.size(); i++) {
                if (const auto &iter =
                        names_in_scope.find(lambda->args[i].name);
                    iter != names_in_scope.cend()) {
                    internal_assert(equals(iter->second, lambda->args[i].type))
                        << "Failure in sort() lowering, argument: "
                        << lambda->args[i].name
                        << " of sort lambda: " << cost_func
                        << " does not match the type in scope: "
                        << iter->second;
                    continue;
                } else {
                    Expr expr =
                        Access::make(lambda->args[i].name, current_match_arg);
                    internal_assert(equals(expr.type(), lambda->args[i].type))
                        << "Failure in sort() lowering, argument: "
                        << lambda->args[i].name
                        << " of sort lambda: " << cost_func
                        << " does not match the type in scope: " << expr.type();
                    temp_repls[lambda->args[i].name] = std::move(expr);
                }
            }
            temp_repls[idx] = std::move(value);
            return opt::Simplify::simplify(replace(temp_repls, lambda->value));
        }

        // TODO(ajr): There should be a way to target only a single YieldFrom...
        Stmt visit(const YieldFrom *node) override {
            if (!found_match) {
                return node;
            }
            internal_assert(!found_from)
                << "Found duplicate YieldFrom when lowering sort(): "
                << Stmt(node);
            std::vector<Expr> exprs = break_tuple(node->value);
            if (exprs.size() < 2) {
                return node; // one branch is already in order
            }
            found_from = true;

            // All this pass does is work out what each branch is worth. The
            // reordering is not here, and deliberately: doing it at this level
            // means selecting between whole subtree *values*, because the
            // layout has not run yet and a branch is still a tree reference
            // rather than an index into one. That is what the previous version
            // did -- a bitonic network of selects over those references, built
            // right here -- and it cost more than the better traversal order
            // saved.
            //
            // The keys ride along on the `from` instead. They are lowered like
            // any other expression, so a key naming the node's split axis
            // becomes the same field load anything else would, and the
            // permutation happens on SSA once a branch is a number.
            std::vector<Expr> keys(exprs.size());
            for (size_t i = 0; i < exprs.size(); i++) {
                keys[i] = sort_cost(i);
            }
            return YieldFrom::make(node->value, std::move(keys));
        }

        Stmt visit(const Match *node) override {
            const Var *var = node->loc.as<Var>();
            internal_assert(var) << Stmt(node);

            if (var->name != loc.names[0]) {
                return Mutator::visit(node);
            }

            internal_assert(!found_match)
                << "Found duplicate traversal when lowering sort(): "
                << Stmt(node);
            found_match = true;
            bool found = false;
            const size_t n = node->arms.size();
            Match::Arms new_arms(n);
            for (size_t i = 0; i < n; i++) {
                Stmt stmt = node->arms[i].second;
                if (node->arms[i].first.name() == loc.names[1]) {
                    current_match_arg = Unwrap::make(i, node->loc);
                    stmt = mutate(stmt);
                    found = true;
                    current_match_arg = Expr();
                }
                new_arms[i] = {node->arms[i].first, std::move(stmt)};
            }

            internal_assert(found)
                << "Failed to find match arm: " << loc.names[1]
                << " in match:\n"
                << Stmt(node);

            return Match::make(node->loc, std::move(new_arms));
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                std::string name = var->name;
                // TODO(ajr): hope to God it's impossible to have
                // self-recursion in these.
                if (name.starts_with("_traverse_tree")) {
                    internal_assert(funcs.contains(name));
                    auto temp = std::move(names_in_scope);
                    names_in_scope = get_names_in_scope(*funcs[name]);
                    funcs[name]->body = mutate(funcs[name]->body);
                    names_in_scope = std::move(temp);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    // TODO(ajr): would be 1 if this is applied to a queue.
    internal_assert(loc.names.size() == 2);
    ApplySortImpl mutator(loc, cost_func, funcs, std::move(names_in_scope));
    Stmt change = mutator.mutate(std::move(stmt));
    internal_assert(mutator.found_match && mutator.found_from)
        << "Failed to lower sort(): " << stmt;
    return change;
}

Program LowerSorts::run(Program program, const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }

    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    TransformMap &transforms = program.schedules[Target::Host].func_transforms;

    if (transforms.empty()) {
        return program;
    }

    for (const auto &[name, ts] : transforms) {
        auto fiter = program.funcs.find(name);
        internal_assert(fiter != program.funcs.end());

        auto &func = fiter->second;

        Stmt body = std::move(func->body);

        size_t counter = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            const auto &t = ts[i];
            if (std::holds_alternative<Sort>(t)) {
                if (counter != i) {
                    internal_error
                        << "Bonsai expects sort() to be applied before "
                           "any other scheduling primitives: "
                        << name;
                }
                counter++;
                const Sort &sort = std::get<Sort>(t);
                body = apply_sort(sort.loc, sort.lambda, std::move(body),
                                  program.funcs, get_names_in_scope(*func));
            }
        }
        func->body = std::move(body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
