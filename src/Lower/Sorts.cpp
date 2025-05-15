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

// https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
uint32_t next_pow2(uint32_t n) {
    if (n == 0) {
        return 1;
    }
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

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
            found_from = true;
            // TODO(ajr): maybe we want a sort() IRNode that can be
            // device-specific?
            std::vector<Expr> exprs = break_tuple(node->value);
            std::vector<Expr> costs(exprs.size());
            for (size_t i = 0; i < exprs.size(); i++) {
                costs[i] = sort_cost(i);
            }
            const size_t n = next_pow2(exprs.size());

            std::vector<Stmt> temps;

            size_t bool_counter = 0;
            auto make_temp_bool = [&temps, &bool_counter](Expr expr) {
                std::string name = "_sort_cmp" + std::to_string(bool_counter++);
                temps.push_back(LetStmt::make(WriteLoc(name, Bool_t::make()),
                                              std::move(expr)));
                return Var::make(Bool_t::make(), std::move(name));
            };

            size_t cost_counter = 0;
            auto make_temp_cost = [&temps, &cost_counter](Expr expr) {
                std::string name =
                    "_sort_cost" + std::to_string(cost_counter++);
                Type t = expr.type();
                temps.push_back(
                    LetStmt::make(WriteLoc(name, t), std::move(expr)));
                return Var::make(std::move(t), std::move(name));
            };

            size_t val_counter = 0;
            auto make_temp_val = [&temps, &val_counter](Expr expr) {
                std::string name = "_sort_tmp" + std::to_string(val_counter++);
                Type t = expr.type();
                temps.push_back(
                    LetStmt::make(WriteLoc(name, t), std::move(expr)));
                return Var::make(std::move(t), std::move(name));
            };

            // Use bitonic sorting network.
            // https://en.wikipedia.org/wiki/Bitonic_sorter
            for (size_t k = 2; k <= n; k *= 2) {
                for (size_t j = k / 2; j > 0; j /= 2) {
                    for (size_t i = 0; i < n; i++) {
                        const size_t l = i ^ j;
                        if (l > i && std::max(i, l) < exprs.size()) {
                            Expr compare_cost = ((i & k) == 0)
                                                    ? (costs[i] < costs[l])
                                                    : (costs[i] > costs[l]);

                            compare_cost =
                                make_temp_bool(std::move(compare_cost));
                            Expr cost0 = costs[i], cost1 = costs[l];
                            Expr expr0 = exprs[i], expr1 = exprs[l];
                            costs[i] = select(compare_cost, cost0, cost1);
                            costs[i] = make_temp_cost(std::move(costs[i]));
                            exprs[i] = select(compare_cost, expr0, expr1);
                            exprs[i] = make_temp_val(std::move(exprs[i]));
                            costs[l] = select(compare_cost, cost1, cost0);
                            costs[l] = make_temp_cost(std::move(costs[l]));
                            exprs[l] = select(compare_cost, expr1, expr0);
                            exprs[l] = make_temp_val(std::move(exprs[l]));
                        }
                    }
                }
            }
            Expr value = make_tuple(exprs);
            Stmt yield = YieldFrom::make(std::move(value));
            internal_assert(!temps.empty());
            temps.emplace_back(std::move(yield));
            return Sequence::make(std::move(temps));
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
