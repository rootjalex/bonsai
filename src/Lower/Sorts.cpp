#include "Opt/Parallelize.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
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

Stmt apply_sort(const Location &loc, const Expr &cost_func, Stmt stmt,
                FuncMap &funcs) {
    struct ApplySortImpl : public Mutator {
        const Location &loc;
        const Expr &cost_func;
        FuncMap &funcs;
        bool found = false;

        ApplySortImpl(const Location &loc, const Expr &cost_func,
                      FuncMap &funcs)
            : loc(loc), cost_func(cost_func), funcs(funcs) {}

        Expr sort_cost(size_t i) const {
            const Lambda *lambda = cost_func.as<Lambda>();
            internal_assert(lambda) << cost_func;
            internal_assert(!lambda->args.empty()) << cost_func;
            const std::string &idx = lambda->args[0].name;
            Expr value = make_const(lambda->args[0].type, i);
            // TODO(ajr): somewhere, we need to have asserted that the lambda
            // args are in scope. I am not quite sure where this can easily be
            // done.
            return opt::Simplify::simplify(replace(idx, value, lambda->value));
        }

        // TODO(ajr): There should be a way to target only a single YieldFrom...
        Stmt visit(const YieldFrom *node) override {
            internal_assert(!found)
                << "Found duplicate YieldFrom when lowering sort(): "
                << Stmt(node);
            found = true;
            // TODO(ajr): maybe we want a sort() IRNode that can be
            // device-specific?
            std::vector<Expr> exprs = break_tuple(node->value);
            std::vector<Expr> costs(exprs.size());
            for (size_t i = 0; i < exprs.size(); i++) {
                costs[i] = sort_cost(i);
            }
            // TODO(ajr): Figure out how to generate a sorting network.
            internal_assert(exprs.size() == 2);
            const size_t n = exprs.size();
            internal_assert((n & (n - 1)) == 0) << "Expected power of 2: " << n;

            // Use bitonic sorting network.
            // https://en.wikipedia.org/wiki/Bitonic_sorter
            for (size_t k = 2; k <= n; k *= 2) {
                for (size_t j = k / 2; j > 0; j /= 2) {
                    for (size_t i = 0; i < n; i++) {
                        const size_t l = i ^ j;
                        if (l > i) {
                            Expr compare_cost = ((i & k) == 0)
                                                    ? (costs[i] < costs[l])
                                                    : (costs[i] > costs[l]);
                            Expr cost0 = costs[i], cost1 = costs[l];
                            Expr expr0 = exprs[i], expr1 = exprs[l];
                            costs[i] = select(compare_cost, cost0, cost1);
                            exprs[i] = select(compare_cost, expr0, expr1);
                            costs[l] = select(compare_cost, cost1, cost0);
                            exprs[l] = select(compare_cost, expr1, expr0);
                        }
                    }
                }
            }
            Expr value = make_tuple(exprs);
            return YieldFrom::make(std::move(value));
        }

        // TODO(ajr): Matches have already been lowered, annoyingly...
        /*
        Stmt visit(const Match *node) override {
            const Var *var = node->loc.as<Var>();
            internal_assert(var) << Stmt(node);

            if (var->name != loc.names[0]) {
                return Mutator::visit(node);
            }

            internal_assert(!found)
                << "Found duplicate traversal when lowering sort(): "
                << Stmt(node);
            const size_t n = node->arms.size();
            Match::Arms new_arms(n);
            for (size_t i = 0; i < n; i++) {
                Stmt stmt = node->arms[i].second;
                if (node->arms[i].first.name() == loc.names[1]) {
                    stmt = lower_sort(cost_func, std::move(stmt));
                    found = true;
                }
                new_arms[i] = {node->arms[i].first, std::move(stmt)};
            }

            internal_assert(found)
                << "Failed to find match arm: " << loc.names[1]
                << " in match:\n"
                << Stmt(node);

            return Match::make(node->loc, std::move(new_arms));
        }
        */

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                std::string name = var->name;
                // TODO(ajr): hope to God it's impossible to have
                // self-recursion in these.
                if (name.starts_with("_traverse_tree")) {
                    std::cout << "recursing into: " << funcs[name]->body;
                    funcs[name]->body = mutate(funcs[name]->body);
                    std::cout << "made: " << funcs[name]->body;
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    // TODO(ajr): would be 1 if this is applied to a queue.
    internal_assert(loc.names.size() == 2);
    ApplySortImpl mutator(loc, cost_func, funcs);
    Stmt change = mutator.mutate(std::move(stmt));
    internal_assert(mutator.found) << "Failed to lower sort(): " << stmt;
    return change;
}

} // namespace lower
} // namespace bonsai
