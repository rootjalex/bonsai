#include "Lower/Random.h"

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

using namespace ir;

static const Expr rng_state_var =
    Var::make(Rand_State_t::make(), rng_state_name);

bool calls_rand(const Stmt &stmt,
                const std::set<std::string> &funcs_call_rand) {
    // Purposefully does not look through Launch()!
    // RNG state is thread-specific.
    struct CallsRandFinder : public Visitor {
        bool found = false;
        const std::set<std::string> &funcs_call_rand;

        CallsRandFinder(const std::set<std::string> &funcs_call_rand)
            : funcs_call_rand(funcs_call_rand) {}

        void visit(const Intrinsic *node) override {
            found = found || (node->op == Intrinsic::rand);
            if (found) {
                return;
            }
            Visitor::visit(node);
        }

        void visit(const Call *node) override {
            if (found) {
                return;
            }
            if (const Var *var = node->func.as<Var>()) {
                if (funcs_call_rand.contains(var->name)) {
                    found = true;
                    return;
                }
            }
            Visitor::visit(node);
        }

        void visit(const CallStmt *node) override {
            if (found) {
                return;
            }
            if (const Var *var = node->func.as<Var>()) {
                if (funcs_call_rand.contains(var->name)) {
                    found = true;
                    return;
                }
            }
            Visitor::visit(node);
        }
    };
    CallsRandFinder finder(funcs_call_rand);
    stmt.accept(&finder);
    return finder.found;
}

Stmt insert_rand_state(const Stmt &stmt,
                       const std::set<std::string> &funcs_call_rand) {
    // Purposefully does not look through Launch()!
    // RNG state is thread-specific.
    struct CallsRandFinder : public Mutator {
        bool found = false;
        const std::set<std::string> &funcs_call_rand;

        CallsRandFinder(const std::set<std::string> &funcs_call_rand)
            : funcs_call_rand(funcs_call_rand) {}

        std::pair<std::vector<Expr>, bool>
        visit_list(const std::vector<Expr> &args) {
            bool not_changed = true;
            const size_t n = args.size();
            std::vector<Expr> new_args(n);
            for (size_t i = 0; i < n; i++) {
                new_args[i] = mutate(args[i]);
                not_changed = not_changed && new_args[i].same_as(args[i]);
            }
            return {std::move(new_args), not_changed};
        }

        struct CallSig {
            Expr func;
            std::vector<Expr> args;
            bool not_changed;
        };

        CallSig handle(const Expr &func, const std::vector<Expr> args) {
            auto [new_args, not_changed] = visit_list(args);
            if (const Var *var = func.as<Var>()) {
                if (funcs_call_rand.contains(var->name)) {
                    new_args.push_back(rng_state_var);
                    const Function_t *func_t = var->type.as<Function_t>();
                    internal_assert(func_t);
                    std::vector<Function_t::ArgSig> arg_types =
                        func_t->arg_types;
                    arg_types.push_back({Rand_State_t::make(),
                                         /*is_mutable=*/true});
                    Type call_type = Function_t::make(func_t->ret_type,
                                                      std::move(arg_types));
                    Expr new_func = Var::make(std::move(call_type), var->name);
                    return {new_func, new_args, true};
                }
            }
            return {Expr(), new_args, not_changed};
        }

        Expr visit(const Call *node) override {
            auto [func, args, not_changed] = handle(node->func, node->args);
            if (func.defined()) {
                return Call::make(std::move(func), std::move(args));
            } else if (not_changed) {
                return node;
            }
            return Call::make(node->func, std::move(args));
        }

        Stmt visit(const CallStmt *node) override {
            auto [func, args, not_changed] = handle(node->func, node->args);
            if (func.defined()) {
                return CallStmt::make(std::move(func), std::move(args));
            } else if (not_changed) {
                return node;
            }
            return CallStmt::make(node->func, std::move(args));
        }
    };
    CallsRandFinder finder(funcs_call_rand);
    return finder.mutate(stmt);
}

} // namespace

FuncMap LowerRandom::run(FuncMap funcs, const CompilerOptions &options) const {

    const std::vector<std::string> topo_order =
        lower::func_topological_order(funcs, /*undef_calls=*/false);

    std::set<std::string> call_rand;

    // First find the set of all random calls.
    // Technically needs to be done to convergence for mutual recursion.
    static const size_t max_allowed_iters = 5;
    size_t iter_count = 0;
    size_t old_size = 0;

    do {
        old_size = call_rand.size();

        for (const auto &[name, func] : funcs) {
            if (call_rand.contains(name)) {
                continue;
            } else if (calls_rand(func->body, call_rand)) {
                call_rand.insert(name);
            }
        }

        if (iter_count++ > max_allowed_iters) {
            internal_error << "May have found pathological mutual recursion in "
                              "Random lowering.";
        }
    } while (call_rand.size() > old_size);

    for (const auto &fname : call_rand) {
        funcs[fname]->body = insert_rand_state(funcs[fname]->body, call_rand);
        // Parallel functions must set up their own random state.
        if (!(fname == "main" || funcs[fname]->is_kernel() ||
              funcs[fname]->is_exported())) {
            funcs[fname]->args.emplace_back(rng_state_name,
                                            Rand_State_t::make(),
                                            /*default_value=*/Expr(),
                                            /*mutating=*/true);
        } else {
            funcs[fname]->attributes.push_back(Function::Attribute::setup_rng);
        }
    }

    return funcs;
}

} // namespace lower
} // namespace bonsai
