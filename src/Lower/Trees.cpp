#include "Lower/Trees.h"

#include "Lower/PredicateAnalysis.h"

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

struct RewriteTree : public ir::Mutator {
    using RewriteFunc = std::function<ir::Stmt(ir::Expr)>;
    const RewriteFunc &rewrite_yield;
    const RewriteFunc &rewrite_scan;
    const RewriteFunc &rewrite_yieldfrom;

    RewriteTree(const RewriteFunc &yield, const RewriteFunc &scan, const RewriteFunc &from)
        : rewrite_yield(yield), rewrite_scan(scan), rewrite_yieldfrom(from) {}

    ir::Stmt visit(const ir::Yield *node) override {
        return rewrite_yield(node->value);
    }

    ir::Stmt visit(const ir::Scan *node) override {
        return rewrite_scan(node->value);
    }

    ir::Stmt visit(const ir::YieldFrom *node) override {
        return rewrite_yieldfrom(node->value);
    }
};

// returns has_data, has_children
std::pair<std::vector<ir::BVH_t::Param>, std::vector<ir::BVH_t::Param>> analyze_node(const ir::BVH_t::Node &node, const ir::Type &prim_t) {
    std::vector<ir::BVH_t::Param> data, children;
    for (const auto &param : node.params) {
        if (ir::equals(prim_t, param.type)) {
            data.push_back(param);
        } else if (param.type.is<ir::Ptr_t>()) {
            children.push_back(param);
        }
    }
    return {data, children};
}

ir::Stmt build_filter(ir::Stmt body, ir::Expr predicate) {
    struct RewriteFilter : public ir::Mutator {
        ir::Expr predicate;

        RewriteFilter(ir::Expr pred) : predicate(std::move(pred)) {}

        std::vector<ir::Expr> volumes;

        ir::Stmt visit(const ir::Match *node) override {
            const ir::Var *var = node->loc.as<ir::Var>();
            internal_assert(var) << "TODO: handle Match on non-Var";

            const size_t n = node->arms.size();
            ir::Match::Arms new_arms(n);
            for (size_t i = 0; i < n; i++) {
                // static_assert(false);
                // Insert volume as Var::make
                if (node->arms[i].first.volume.has_value()) {
                    // TODO: this needs to have special lowering later!
                    ir::Expr vol = ir::Var::make(node->arms[i].first.volume->struct_type, var->name + ".volume");
                    volumes.emplace_back(std::move(vol));
                } else {
                    volumes.emplace_back(); // undef volume
                }
                ir::Stmt stmt = mutate(node->arms[i].second);
                volumes.pop_back();
                new_arms[i] = {node->arms[i].first, std::move(stmt)};
            }

            return ir::Match::make(node->loc, std::move(new_arms));
        }

        ir::Stmt visit(const ir::Yield *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda) << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);
            internal_assert(ir::equals(lambda->args[0].type, node->value.type()));
            ir::Expr cond = replace(lambda->args[0].name, node->value, lambda->value);

            // if (predicate) yield data
            ir::Stmt body = ir::IfElse::make(std::move(cond), node);

            // If any volumes are defined, use predicate analysis.
            if (std::any_of(volumes.cbegin(), volumes.cend(), [](const auto &vol) { return vol.defined(); })) {
                VolumeMap vols;
                const size_t n = volumes.size();
                for (size_t i = 0; i < n; i++) {
                    // Even if a volume is undefined, needs to be added so
                    // predicate analysis knows it's non-varying.
                    vols[lambda->args[i].name] = volumes[i];
                }
                Interval bounds = predicate_analysis(lambda->value, vols);
                if (bounds.max.defined()) {
                    // Maybe true.
                    body = ir::IfElse::make(std::move(bounds.max), std::move(body));
                }
                if (bounds.min.defined()) {
                    // Always true.
                    body = ir::IfElse::make(std::move(bounds.min), node, std::move(body));
                }
            }
            return body;
        }

        ir::Stmt visit(const ir::Scan *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda) << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);

            if (std::any_of(volumes.cbegin(), volumes.cend(), [](const auto &vol) { return vol.defined(); })) {
                VolumeMap vols;
                const size_t n = volumes.size();
                for (size_t i = 0; i < n; i++) {
                    // Even if a volume is undefined, needs to be added so
                    // predicate analysis knows it's non-varying.
                    vols[lambda->args[i].name] = volumes[i];
                }
                Interval bounds = predicate_analysis(lambda->value, vols);
                internal_assert(bounds.max.defined())
                    << "Cannot accelerate predicate: " << predicate << " on: " << ir::Stmt(node);

                // Make a recursive call
                ir::Stmt body = ir::YieldFrom::make(ir::filter(predicate, node->value));
                // Add the maybe case -> recursive call
                body = ir::IfElse::make(std::move(bounds.max), std::move(body));

                // Check for always case
                if (bounds.min.defined()) {
                    body = ir::IfElse::make(std::move(bounds.min), node, std::move(body));
                }
                return body;
            } else {
                internal_error << "Cannot lower filter on scan (no volumes): " << predicate << " on " << ir::Stmt(node);
            }
        }

        ir::Stmt visit(const ir::YieldFrom *node) override {
            internal_error << "TODO: " << ir::Stmt(node);
        }
    };

    return RewriteFilter(std::move(predicate)).mutate(body);
}

ir::Stmt build_traversal(const ir::Expr &expr, const ir::TypeMap &tree_types) {
    // TODO: not necessarily always a Var, could be e.g. an Access.
    if (auto as_var = expr.as<ir::Var>()) {
        internal_assert(as_var->type.is<ir::Set_t>())
            << "Cannot build traversal for non-set: " << expr;
        const auto &iter = tree_types.find(as_var->name);
        internal_assert(iter != tree_types.cend())
            << "Lowering of: " << expr << " does not have associated BVH type.";
        const ir::Type &tree = iter->second;
        const ir::BVH_t *bvh = tree.as<ir::BVH_t>();
        internal_assert(bvh);
        
        const size_t n_nodes = bvh->nodes.size();
        ir::Match::Arms arms(n_nodes);
        for (size_t i = 0; i < n_nodes; i++) {
            const auto [data, children] = analyze_node(bvh->nodes[i], as_var->type.element_of());
            
            std::vector<ir::Stmt> stmts(data.size() + children.size());
            // TODO: visit order should be scheduable?
            for (size_t i = 0; i < data.size(); i++) {
                stmts[i] = ir::Yield::make(ir::Var::make(data[i].type, data[i].name));
            }
            for (size_t j = 0; j < children.size(); j++) {
                // Type is recursively a tree!
                stmts[data.size() + j] = ir::Scan::make(ir::Var::make(tree, children[j].name));
            }

            arms[i].first = bvh->nodes[i];
            if (stmts.size() == 1) {
                // Special case.
                arms[i].second = stmts[0];
            } else {
                arms[i].second = ir::Sequence::make(std::move(stmts));
            }
        }
        ir::Expr var = ir::Var::make(tree, as_var->name);
        return ir::Match::make(std::move(var), std::move(arms));
    }

    const ir::SetOp *as_set = expr.as<ir::SetOp>();
    if (as_set == nullptr) {
        internal_error << "[unimplemented] Unknown traversal pattern: " << expr;
    }

    switch (as_set->op) {
        case ir::SetOp::filter: {
            ir::Stmt body = build_traversal(as_set->b, tree_types);
            return build_filter(body, as_set->a);
        }
        default: {
            internal_error << "TODO: " << expr;
        }
    }
}

struct LowerBVH : public ir::Mutator {
    const ir::TypeMap &tree_types;
    ir::FuncMap new_funcs;

    LowerBVH(const ir::TypeMap &tree_types) : tree_types(tree_types) {}

    // For unique func names
    size_t counter = 0;

    std::string new_func_name() {
        return "?traverse" + std::to_string(counter++);
    }

    // Returns a call to the func.
    // Inserts the built func into new_funcs
    ir::Expr build_func(const ir::Expr &expr) {
        const std::string func = new_func_name();
        const auto free_vars = ir::gather_free_vars(expr);

        bool found = false;
        for (const auto &var : free_vars) {
            if (tree_types.contains(var->name)) {
                found = true;
                break;
            }
        }
        internal_assert(found) << "Lowering of: " << expr << " does not contain any tree types.";

        // TODO: build func
        ir::Stmt body = build_traversal(expr, tree_types);

        internal_assert(body.defined());

        std::vector<ir::Function::Argument> func_args;
        std::transform(free_vars.cbegin(), free_vars.cend(),
                       std::back_inserter(func_args),
                       [&](const auto &var) {
                            const auto &iter = this->tree_types.find(var->name);
                            if (iter != this->tree_types.cend()) {
                                return ir::Function::Argument(var->name, iter->second);
                            }
                            return ir::Function::Argument(var->name, var->type);
                       });

        // When should this type be concretized into e.g. a list?
        ir::Type ret_type = expr.type();
        auto f = std::make_shared<ir::Function>(func, std::move(func_args), std::move(ret_type), std::move(body), ir::Function::InterfaceList{});
        ir::Type call_type = f->call_type();
        new_funcs[func] = std::move(f);

        // TODO: this allocates unnecessarily, 
        std::vector<ir::Expr> call_args;
        std::transform(free_vars.begin(), free_vars.end(),
                       std::back_inserter(call_args),
                       [&](auto &var) -> ir::Expr {
                            const auto &iter = this->tree_types.find(var->name);
                            if (iter != this->tree_types.cend()) {
                                return ir::Var::make(iter->second, var->name);
                            }
                           return var;
                       });

        return ir::Call::make(ir::Var::make(std::move(call_type), func), call_args);
    }

    ir::Expr visit(const ir::SetOp *op) override {
        return build_func(op);
    }
};

} // namespace

ir::Program LowerTrees::run(ir::Program program) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    // Pop tree schedule, no longer necessary.
    ir::TypeMap tree_types = std::move(program.schedules[ir::Target::Host].tree_types);

    LowerBVH converter(tree_types);

    // Remap externs.
    for (auto &[name, type] : program.externs) {
        const auto &iter = tree_types.find(name);
        if (iter != tree_types.cend()) {
            type = iter->second;
        }
    }

    for (auto &[_, f] : program.funcs) {
        f->body = converter.mutate(f->body);
    }

    for (auto &[name, f] : converter.new_funcs) {
        auto [_, inserted] = program.funcs.try_emplace(std::move(name), std::move(f));
        internal_assert(inserted);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
