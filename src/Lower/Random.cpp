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

// Splits functions that exist on both host and device and use rand() on CUDA.
// There are two separate paths for these, since cuRAND is only allowed on host.
void split_functions_with_rand(FuncMap &funcs) {
    std::set<std::string> devices = find_device_functions(funcs),
                          hosts = find_host_functions(funcs);
    std::vector<std::string> intersection;
    std::set_intersection(devices.begin(), devices.end(), hosts.begin(),
                          hosts.end(), std::back_inserter(intersection));
    for (const std::string &name : intersection) {
        const auto &func = funcs[name];
        if (!calls_rand(func->body, {})) {
            continue;
        }
        // Insert two copies, one for host and one for device.
        std::string new_device_name = "d_" + name;
        std::string new_host_name = "h_" + name;
        funcs[new_device_name] = std::make_shared<Function>(
            new_device_name, func->args, func->ret_type, func->body,
            func->interfaces, func->attributes);
        funcs[new_host_name] = std::make_shared<Function>(
            new_host_name, func->args, func->ret_type, func->body,
            func->interfaces, func->attributes);

        // Update the bodies of functions that use them.
        for (const std::string &device_name : devices) {
            internal_assert(funcs.contains(device_name)) << device_name;
            auto &device_func = funcs[device_name];
            std::map<std::string, std::string> replacements = {
                {name, new_device_name},
            };
            device_func->body = replace(replacements, device_func->body);
        }
        for (const std::string &host_name : hosts) {
            internal_assert(funcs.contains(host_name)) << host_name;
            auto &host_func = funcs[host_name];
            std::map<std::string, std::string> replacements = {
                {name, new_host_name},
            };
            host_func->body = replace(replacements, host_func->body);
        }
        funcs.erase(name);
    }
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
    std::set<std::string> call_rand;
    switch (options.target) {
    case BackendTarget::CUDA: {
        split_functions_with_rand(funcs);
        // In CUDA, instances of rand() are lowered to cuRAND when running on
        // the device. Since cuRAND is only available on device, we assume RNG
        // state will be set up in the kernel, and then propagated to nested
        // functions from there. Any host functions will simply use a CPU
        // built-in random function.
        lower::CallGraph call_graph = lower::build_call_graph(funcs);
        // TODO(cgyurgyik): Extend this to support propagation through mutual
        // recursion.
        std::vector<std::string> topological_order =
            func_topological_order(funcs);
        std::reverse(topological_order.begin(), topological_order.end());
        for (int i = 0, e = topological_order.size(); i < e; ++i) {
            const std::string &name = topological_order[i];
            auto &func = *funcs[name];
            if (!func.is_kernel() && !call_rand.contains(name)) {
                continue;
            }
            if (func.is_kernel()) {
                // The kernel only sets up the RNG.
                func.attributes.push_back(Function::Attribute::setup_rng);
            }
            auto it = call_graph.find(name);
            internal_assert(it != call_graph.end()) << name;
            for (const std::string &nested : it->second) {
                internal_assert(funcs.contains(nested)) << nested;
                const Function &nested_function = *funcs[nested];
                if (calls_rand(nested_function.body, call_rand)) {
                    // We must propagate it from the kernel level.
                    call_rand.insert(nested);
                }
            }
        }
        // Fix up the bodies of kernel functions.
        for (const auto &[name, func] : funcs) {
            if (!func->is_kernel()) {
                continue;
            }
            func->body = insert_rand_state(std::move(func->body), call_rand);
        }
        // Insert the random state.
        for (const std::string &name : call_rand) {
            auto &func = funcs[name];
            func->body = insert_rand_state(func->body, call_rand);
            func->args.emplace_back(rng_state_name, Rand_State_t::make(),
                                    /*default_value=*/Expr(),
                                    /*mutating=*/true);
        }
        return funcs;
    }
    default: {
        const std::vector<std::string> topo_order =
            lower::func_topological_order(funcs);
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
                internal_error
                    << "May have found pathological mutual recursion in "
                       "Random lowering.";
            }
            new_size = call_rand.size();
        } while (new_size != old_size);
        for (const auto &fname : call_rand) {
            funcs[fname]->body =
                insert_rand_state(funcs[fname]->body, call_rand);
            // Parallel functions must set up their own random state.
            if (!(fname == "main" || funcs[fname]->is_kernel() ||
                  funcs[fname]->is_exported())) {
                funcs[fname]->args.emplace_back(rng_state_name,
                                                Rand_State_t::make(),
                                                /*default_value=*/Expr(),
                                                /*mutating=*/true);
            } else {
                funcs[fname]->attributes.push_back(
                    Function::Attribute::setup_rng);
            }
        }

        return funcs;
    }
    }
}

} // namespace lower
} // namespace bonsai
