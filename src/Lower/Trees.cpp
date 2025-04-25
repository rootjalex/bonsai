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

ir::Stmt build_traversal(const ir::Expr &expr, const ir::TypeMap &tree_types,
                         const IntervalMap &intervals);

static size_t counter = 0;

std::string unique_iter_name() { return "_iter" + std::to_string(counter++); }

// returns has_data, has_children
std::pair<std::vector<ir::TypedVar>, std::vector<ir::TypedVar>>
analyze_node(const ir::BVH_t::Node &node, const ir::Type &prim_t) {
    std::vector<ir::TypedVar> data, children;
    for (const auto &param : node.fields()) {
        if (ir::equals(prim_t, param.type) ||
            (param.type.is<ir::Array_t>() &&
             ir::equals(prim_t, param.type.as<ir::Array_t>()->etype))) {
            data.push_back(param);
        } else if (param.type.is<ir::Ref_t>()) { // TODO: and is ref to
                                                 // current tree type?
            children.push_back(param);
        }
    }

    return {data, children};
}

struct Rewriter : public ir::Mutator {
    std::vector<ir::Expr> volumes;

    ir::Stmt visit(const ir::Match *node) final override {
        const ir::Var *var = node->loc.as<ir::Var>();
        internal_assert(var) << "TODO: handle Match on non-Var";

        const size_t n = node->arms.size();
        ir::Match::Arms new_arms(n);
        for (size_t i = 0; i < n; i++) {
            ir::Expr tree = ir::Unwrap::make(i, node->loc);
            if (node->arms[i].first.volume.has_value()) {
                const size_t n_args =
                    node->arms[i].first.volume->initializers.size();
                std::vector<ir::Expr> args(n_args);
                for (size_t j = 0; j < n_args; j++) {
                    const auto &name =
                        node->arms[i].first.volume->initializers[j];
                    args[j] = ir::Access::make(name, tree);
                }
                ir::Expr vol = ir::Build::make(
                    node->arms[i].first.volume->struct_type, args);
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

    VolumeMap make_volume_map(const std::vector<ir::TypedVar> &args) const {
        VolumeMap vols;
        const size_t n = volumes.size();
        internal_assert(n == args.size())
            << "Making volume map with incorrect number of arguments: "
            << args.size() << " vs. " << n;
        for (size_t i = 0; i < n; i++) {
            // Even if a volume is undefined, needs to be added so
            // predicate analysis knows it's non-varying.
            vols[args[i].name] = volumes[i];
        }
        return vols;
    }

    using ir::Mutator::visit;
};

ir::Stmt build_filter(ir::Stmt body, ir::Expr predicate,
                      const IntervalMap &intervals) {
    struct RewriteFilter : public Rewriter {
        ir::Expr predicate;
        const IntervalMap &intervals;

        RewriteFilter(ir::Expr pred, const IntervalMap &intervals)
            : predicate(std::move(pred)), intervals(intervals) {}

        using ir::Mutator::visit;

        ir::Stmt visit(const ir::Yield *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() == lambda->args.size());
            const size_t n_args = lambda->args.size();

            std::map<std::string, ir::Expr> repls;

            if (n_args == 1) {
                internal_assert(
                    ir::equals(lambda->args[0].type, node->value.type()));
                repls[lambda->args[0].name] = node->value;
            } else {
                internal_assert(node->value.type().is<ir::Tuple_t>());
                for (size_t i = 0; i < n_args; i++) {
                    // TODO: this needs to simplify or have CSE for it to be
                    // efficient!
                    ir::Expr value = ir::Extract::make(node->value, i);
                    internal_assert(
                        ir::equals(value.type(), lambda->args[i].type));
                    repls[lambda->args[i].name] = std::move(value);
                }
            }

            ir::Expr cond = replace(repls, lambda->value);

            // if (predicate) yield data
            ir::Stmt body = ir::IfElse::make(std::move(cond), node);

            VolumeMap vols = make_volume_map(lambda->args);

            Interval bounds =
                predicate_analysis(lambda->value, vols, intervals);
            if (bounds.max.defined()) {
                // Maybe true.
                body = ir::IfElse::make(std::move(bounds.max), std::move(body));
            }
            if (bounds.min.defined() && !is_const_zero(bounds.min)) {
                // Always true.
                body = ir::IfElse::make(std::move(bounds.min), node,
                                        std::move(body));
            }

            return body;
        }

        ir::Stmt visit(const ir::Scan *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() == lambda->args.size());

            VolumeMap vols = make_volume_map(lambda->args);

            Interval bounds =
                predicate_analysis(lambda->value, vols, intervals);
            internal_assert(bounds.max.defined())
                << "Cannot accelerate predicate: " << predicate
                << " on: " << ir::Stmt(node);

            // Make a recursive call
            // TODO: this should be wrapped in a filter, for cases with
            // simplified predicates. This is required for proper predicate
            // analysis of conjunctions/disjunctions. ir::Stmt body =
            // ir::YieldFrom::make(ir::filter(predicate, node->value));
            ir::Stmt body = ir::YieldFrom::make(node->value);
            // Add the maybe case -> recursive call
            body = ir::IfElse::make(std::move(bounds.max), std::move(body));

            // Check for always case
            if (bounds.min.defined() && !is_const_zero(bounds.min)) {
                body = ir::IfElse::make(std::move(bounds.min), node,
                                        std::move(body));
            }
            return body;
        }

        ir::Stmt visit(const ir::YieldFrom *node) override {
            // TODO: this should be wrapped in a filter, for cases with
            // simplified predicates. This is required for proper predicate
            // analysis of conjunctions/disjunctions. return
            // ir::YieldFrom::make(ir::filter(predicate, node->value));
            return node;
        }
    };

    return RewriteFilter(std::move(predicate), intervals).mutate(body);
}

ir::Expr try_fuse_filter(const ir::Lambda *metric, ir::Expr best,
                         ir::Expr maybe_filter) {
    if (const ir::SetOp *as_set = maybe_filter.as<ir::SetOp>()) {
        if (as_set->op == ir::SetOp::filter) {
            // Can fuse!
            const ir::Lambda *predicate = as_set->a.as<ir::Lambda>();
            internal_assert(predicate); // TODO: support non-lambdas

            // Look if any lambda names don't match up, e.g.
            // argmin(|t| : ..., filter(|r| ... ))
            std::map<std::string, ir::Expr> repls;
            for (size_t i = 0; i < metric->args.size(); i++) {
                if (metric->args[i].name != predicate->args[i].name) {
                    repls[metric->args[i].name] = ir::Var::make(
                        predicate->args[i].type, predicate->args[i].name);
                }
                internal_assert(
                    equals(metric->args[i].type, predicate->args[i].type))
                    << "Mismatched types in argmin-filter fusion: "
                    << metric->args[i].type
                    << " != " << predicate->args[i].type;
            }

            // Check for convenient case of same naming / types.
            ir::Expr new_cond =
                repls.empty() ? (predicate->value && (metric->value < best))
                              : predicate->value &&
                                    (replace(repls, metric->value) < best);
            // Construct fused filter.
            ir::Expr new_lambda =
                ir::Lambda::make(predicate->args, std::move(new_cond));
            return filter(std::move(new_lambda), as_set->b);
        }
    }

    // Not a nested filter, so just wrap in a filter and return
    ir::Expr new_cond = (metric->value < best);
    ir::Expr new_lambda = ir::Lambda::make(metric->args, std::move(new_cond));
    return filter(std::move(new_lambda), std::move(maybe_filter));
}

ir::Stmt build_argmin(ir::Expr metric, ir::Expr inner,
                      const ir::TypeMap &tree_types,
                      const IntervalMap &intervals) {
    struct RewriteArgmin : public Rewriter {
        ir::Expr metric;
        ir::WriteLoc loc;
        ir::Type tuple_t;

        RewriteArgmin(ir::Expr met, ir::WriteLoc l, ir::Type t)
            : metric(std::move(met)), loc(std::move(l)), tuple_t(std::move(t)) {
        }

        size_t counter = 0;

        std::string make_temp_name() {
            return loc.base + "_temp" + std::to_string(counter++);
        }

        using ir::Mutator::visit;

        ir::Stmt visit(const ir::Yield *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volumes.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);
            internal_assert(
                ir::equals(lambda->args[0].type, node->value.type()));
            ir::Expr value =
                replace(lambda->args[0].name, node->value, lambda->value);

            std::vector<ir::Expr> values = {std::move(value), node->value};
            ir::Expr update = ir::Build::make(tuple_t, std::move(values));
            return ir::Accumulate::make(loc, ir::Accumulate::Argmin,
                                        std::move(update));
        }

        ir::Stmt visit(const ir::Scan *node) override { return node; }

        ir::Stmt visit(const ir::YieldFrom *node) override { return node; }
    };

    const ir::Lambda *lambda = metric.as<ir::Lambda>();
    internal_assert(lambda) << "Metric is not a lambda: " << metric;
    ir::Type metric_t = lambda->value.type();

    ir::Type ret_type = inner.type().element_of();
    ir::Type tuple_t = ir::Tuple_t::make({metric_t, ret_type});

    static size_t counter = 0;
    std::string name = "_best" + std::to_string(counter++);
    ir::WriteLoc loc(name, tuple_t);

    ir::Expr inf = ir::Infinity::make(std::move(metric_t));
    static const std::vector<ir::Expr> empty_list = {};
    ir::Expr empty = ir::Build::make(ret_type, empty_list);
    std::vector<ir::Expr> values = {std::move(inf), std::move(empty)};
    ir::Expr init = ir::Build::make(tuple_t, std::move(values));

    // Alocate
    ir::Stmt header =
        ir::Assign::make(loc, std::move(init), /*mutating=*/false);

    // Make return
    ir::Expr ret_var = ir::Var::make(tuple_t, std::move(name));
    ir::Expr best_ref = ir::Extract::make(ret_var, 1);
    // TODO: should this be a Return?
    // TODO: should this be an If (best[0] != inf) yield best[1] else {} ?
    ir::Stmt footer = ir::Yield::make(std::move(best_ref));
    ir::Expr best_metric = ir::Extract::make(std::move(ret_var), 0);

    // No lower bound (can always get better)
    // Upper bound is the current value (must be at least that good).
    IntervalMap local_intervals = intervals;
    local_intervals[best_metric] = Interval{ir::Expr(), best_metric};

    // Try to build fused filter inside.
    ir::Expr fused_filter = try_fuse_filter(lambda, best_metric, inner);
    ir::Stmt body = build_traversal(fused_filter, tree_types, local_intervals);

    body = RewriteArgmin(std::move(metric), std::move(loc), std::move(tuple_t))
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

ir::Stmt build_product(ir::Stmt a_body, ir::Stmt b_body, ir::Type ret_type) {
    struct RewriteProduct : public Rewriter {
        ir::Stmt b_body;
        ir::Type ret_type;

        RewriteProduct(ir::Stmt b_body, ir::Type ret_type)
            : b_body(std::move(b_body)), ret_type(std::move(ret_type)) {}

        using ir::Mutator::visit;

        ir::Stmt a_body;

        ir::Expr make_tuple(ir::Expr a, ir::Expr b) {
            ir::Type tuple_t = ir::Tuple_t::make({a.type(), b.type()});
            std::vector<ir::Expr> values = {std::move(a), std::move(b)};
            return ir::Build::make(std::move(tuple_t), std::move(values));
        }

        ir::Stmt visit(const ir::Yield *node) override {
            if (!a_body.defined()) {
                a_body = node;
                ir::Stmt ret = mutate(b_body);
                a_body = ir::Stmt();
                return ret;
            } else {
                if (const ir::Yield *yield = a_body.as<ir::Yield>()) {
                    return ir::Yield::make(
                        make_tuple(yield->value, node->value));
                } else if (const ir::Scan *scan = a_body.as<ir::Scan>()) {
                    return ir::Scan::make(make_tuple(scan->value, node->value));
                } else if (const ir::YieldFrom *from =
                               a_body.as<ir::YieldFrom>()) {
                    internal_error
                        << "TODO: lower Yield + YieldFrom properly: " << a_body
                        << " and " << ir::Stmt(node);
                } else {
                    internal_error << "Failure in lowering product: " << a_body
                                   << " and " << ir::Stmt(node);
                }
            }
        }

        ir::Stmt visit(const ir::Scan *node) override {
            if (!a_body.defined()) {
                a_body = node;
                ir::Stmt ret = mutate(b_body);
                a_body = ir::Stmt();
                return ret;
            } else {
                if (const ir::Yield *yield = a_body.as<ir::Yield>()) {
                    return ir::Scan::make(
                        make_tuple(yield->value, node->value));
                } else if (const ir::Scan *scan = a_body.as<ir::Scan>()) {
                    return ir::Scan::make(make_tuple(scan->value, node->value));
                } else if (const ir::YieldFrom *from =
                               a_body.as<ir::YieldFrom>()) {
                    internal_error
                        << "TODO: lower Scan + YieldFrom properly: " << a_body
                        << " and " << ir::Stmt(node);
                } else {
                    internal_error << "Failure in lowering product: " << a_body
                                   << " and " << ir::Stmt(node);
                }
            }
        }

        ir::Stmt visit(const ir::YieldFrom *node) override {
            // TODO: this should be wrapped in a product, for cases with
            // simplified predicates. This is required for proper predicate
            // analysis of conjunctions/disjunctions.
            internal_error << "Failure in lowering product: " << a_body
                           << " and " << ir::Stmt(node);
            // return node;
        }
    };

    // TODO: is ordering scheduable?
    return RewriteProduct(std::move(b_body), std::move(ret_type))
        .mutate(a_body);
}

ir::Stmt build_traversal(const ir::Expr &expr, const ir::TypeMap &tree_types,
                         const IntervalMap &intervals) {
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

        ir::Expr bvh_expr = ir::Var::make(tree, as_var->name);

        const size_t n_nodes = bvh->nodes.size();
        ir::Match::Arms arms(n_nodes);
        for (size_t i = 0; i < n_nodes; i++) {
            ir::Expr node = ir::Unwrap::make(i, bvh_expr);
            const auto [data, children] =
                analyze_node(bvh->nodes[i], as_var->type.element_of());

            std::vector<ir::Stmt> stmts(data.size() + children.size());
            // TODO: visit order should be scheduable?
            for (size_t i = 0; i < data.size(); i++) {
                ir::Expr access = ir::Access::make(data[i].name, node);
                if (data[i].type.is_iterable()) {
                    // forall d in data: yield d
                    std::string name = unique_iter_name();
                    ir::Stmt body = ir::Yield::make(
                        ir::Var::make(data[i].type.element_of(), name));
                    stmts[i] = ir::ForEach::make(
                        std::move(name), std::move(access), std::move(body));
                } else {
                    // yield d
                    stmts[i] = ir::Yield::make(std::move(access));
                }
            }
            for (size_t j = 0; j < children.size(); j++) {
                // Type is recursively a tree.
                stmts[data.size() + j] =
                    ir::Scan::make(ir::Access::make(children[j].name, node));
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
        ir::Stmt body = build_traversal(as_set->b, tree_types, intervals);
        return build_filter(body, as_set->a, intervals);
    }
    case ir::SetOp::argmin: {
        // Argmin is a bit more complicated, because of filter fusion.
        return build_argmin(as_set->a, as_set->b, tree_types, intervals);
    }
    case ir::SetOp::product: {
        ir::Stmt a_body = build_traversal(as_set->a, tree_types, intervals);
        ir::Stmt b_body = build_traversal(as_set->b, tree_types, intervals);
        return build_product(a_body, b_body, expr.type().element_of());
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
        return "_traverse_tree" + std::to_string(counter++);
    }

    // Returns a call to the func.
    // Inserts the built func into new_funcs
    ir::Expr build_func(const ir::Expr &expr) {
        const std::string func = new_func_name();
        const auto free_vars = ir::gather_free_vars(expr);

        bool found = false;
        for (const auto &var : free_vars) {
            if (tree_types.contains(var.name)) {
                found = true;
                break;
            }
        }
        internal_assert(found)
            << "Lowering of: " << expr << " does not contain any tree types.";

        // TODO(ajr): is there more we can do with intervals?
        // e.g. bounded interval hierarchies, kd-trees, tri.x < bound queries,
        // etc?
        IntervalMap intervals;
        ir::Stmt body = build_traversal(expr, tree_types, intervals);

        internal_assert(body.defined());

        std::vector<ir::Function::Argument> func_args;
        std::transform(free_vars.cbegin(), free_vars.cend(),
                       std::back_inserter(func_args), [&](const auto &var) {
                           const auto &iter = this->tree_types.find(var.name);
                           if (iter != this->tree_types.cend()) {
                               return ir::Function::Argument(var.name,
                                                             iter->second);
                           }
                           return ir::Function::Argument(var.name, var.type);
                       });

        // When should this type be concretized into e.g. a list?
        ir::Type ret_type = expr.type();
        auto f = std::make_shared<ir::Function>(
            func, std::move(func_args), std::move(ret_type), std::move(body),
            ir::Function::InterfaceList{}, std::vector<ir::Function::Attribute>{});
        ir::Type call_type = f->call_type();
        new_funcs[func] = std::move(f);

        // TODO: this allocates unnecessarily,
        std::vector<ir::Expr> call_args;
        std::transform(free_vars.begin(), free_vars.end(),
                       std::back_inserter(call_args),
                       [&](auto &var) -> ir::Expr {
                           const auto &iter = this->tree_types.find(var.name);
                           if (iter != this->tree_types.cend()) {
                               return ir::Var::make(iter->second, var.name);
                           }
                           return var;
                       });

        return ir::Call::make(ir::Var::make(std::move(call_type), func),
                              call_args);
    }

    ir::Expr visit(const ir::SetOp *op) override { return build_func(op); }
};

} // namespace

ir::Program LowerTrees::run(ir::Program program) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    // Pop tree schedule, no longer necessary.
    ir::TypeMap tree_types =
        std::move(program.schedules[ir::Target::Host].tree_types);

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
        auto [_, inserted] =
            program.funcs.try_emplace(std::move(name), std::move(f));
        internal_assert(inserted);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
