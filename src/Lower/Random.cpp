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

        void visit(const MultiRecurse *node) override {
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

// Determines whether a rand() call occurs anywhere in the call graph. This is
// necessary to propagate past functions that may not use it, even when their
// children use it.
bool calls_rand(const Function &f, const std::set<std::string> &funcs_call_rand,
                const lower::CallGraph &call_graph, const FuncMap &funcs) {
    if (calls_rand(f.body, funcs_call_rand)) {
        return true;
    }
    auto cit = call_graph.find(f.name);
    internal_assert(cit != call_graph.end()) << f.name;
    for (const std::string &call : cit->second) {
        auto it = funcs.find(call);
        internal_assert(it != funcs.end()) << call;
        if (calls_rand(*it->second, funcs_call_rand, call_graph, funcs)) {
            return true;
        }
    }
    return false;
}

// Passes the random state to every call of a function in `funcs_call_rand`,
// except one in `sets_up_own`: a function that makes its own state -- main,
// a kernel, an exported function, which may be entered from outside with no
// state to hand it -- takes no state parameter, so a call to it from inside
// the module (its own recursion, say) must not pass one.
Stmt insert_rand_state(const Stmt &stmt,
                       const std::set<std::string> &funcs_call_rand,
                       const std::set<std::string> &sets_up_own) {
    // Purposefully does not look through Launch()!
    // RNG state is thread-specific.
    struct CallsRandFinder : public Mutator {
        bool found = false;
        const std::set<std::string> &funcs_call_rand;
        const std::set<std::string> &sets_up_own;

        CallsRandFinder(const std::set<std::string> &funcs_call_rand,
                        const std::set<std::string> &sets_up_own)
            : funcs_call_rand(funcs_call_rand), sets_up_own(sets_up_own) {}

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
                if (funcs_call_rand.contains(var->name) &&
                    !sets_up_own.contains(var->name)) {
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

        Stmt visit(const MultiRecurse *node) override {
            // The state goes on the end of the uniform arguments, so the
            // varying positions -- which are leading -- keep their indices.
            auto [func, args, not_changed] = handle(node->func, node->args);
            std::vector<std::vector<Expr>> varying;
            varying.reserve(node->varying.size());
            for (const auto &vs : node->varying) {
                auto [mutated, same] = visit_list(vs);
                not_changed = not_changed && same;
                varying.push_back(std::move(mutated));
            }
            auto [keys, keys_same] = visit_list(node->keys);
            not_changed = not_changed && keys_same;
            if (func.defined()) {
                return MultiRecurse::make(std::move(func), std::move(args),
                                          node->varying_at, std::move(varying),
                                          std::move(keys));
            } else if (not_changed) {
                return node;
            }
            return MultiRecurse::make(node->func, std::move(args),
                                      node->varying_at, std::move(varying),
                                      std::move(keys));
        }
    };
    CallsRandFinder finder(funcs_call_rand, sets_up_own);
    return finder.mutate(stmt);
}

} // namespace

FuncMap LowerRandom::run(FuncMap funcs, const CompilerOptions &options) const {
    std::set<std::string> call_rand;
    static const size_t max_allowed_iters = 5;
    size_t iter_count = 0, old_size = 0, new_size = 0;
    // First find the set of all random calls.
    // Technically needs to be done to convergence for mutual recursion.

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
        new_size = call_rand.size();
    } while (new_size != old_size);
    // Parallel functions, and those entered from outside, must set up their
    // own random state: they take no state parameter, and a call to one from
    // inside the module -- an exported function's recursion -- passes none.
    std::set<std::string> sets_up_own;
    for (const auto &fname : call_rand) {
        if (fname == "main" || funcs[fname]->is_kernel() ||
            funcs[fname]->is_exported()) {
            sets_up_own.insert(fname);
        }
    }
    for (const auto &fname : call_rand) {
        funcs[fname]->body =
            insert_rand_state(funcs[fname]->body, call_rand, sets_up_own);
        if (!sets_up_own.contains(fname)) {
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
