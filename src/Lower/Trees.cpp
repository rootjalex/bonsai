#include "Lower/Trees.h"

#include "Lower/PredicateAnalysis.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include "Opt/Simplify.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

// #define LOG_TREE_VISITS 1

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

struct RewriteYields : public ir::Mutator {
    std::function<ir::Stmt(const ir::Expr &)> f;
    RewriteYields(std::function<ir::Stmt(const ir::Expr &)> f)
        : f(std::move(f)) {}

    ir::Stmt visit(const ir::Yield *node) override { return f(node->value); }
};

ir::Expr make_tuple_pair(ir::Expr a, ir::Expr b) {
    ir::Type tuple_t = ir::Tuple_t::make({a.type(), b.type()});
    std::vector<ir::Expr> values = {std::move(a), std::move(b)};
    return ir::Build::make(std::move(tuple_t), std::move(values));
}

ir::Stmt lower_iterate(const ir::Expr &expr) {
    if (const ir::SetOp *setop = expr.as<ir::SetOp>()) {
        if (setop->op == ir::SetOp::product) {
            ir::Stmt left = lower_iterate(setop->a);
            return RewriteYields([&](const ir::Expr &a) {
                       ir::Stmt right = lower_iterate(setop->b);
                       return RewriteYields([&](const ir::Expr &b) {
                                  return ir::Yield::make(make_tuple_pair(a, b));
                              })
                           .mutate(right);
                   })
                .mutate(left);
        }
        internal_error << "TODO: lower_iterate for: " << expr;
    }
    std::string name = unique_iter_name();
    ir::Stmt body =
        ir::Yield::make(ir::Var::make(expr.type().element_of(), name));
    return ir::ForEach::make(std::move(name), expr, std::move(body));
}

struct Rewriter : public ir::Mutator {
    // The list of volumes for the currently match arms.
    std::vector<ir::Expr> volumes;
    // The list of nodes for the current matches.
    std::vector<ir::Expr> locs;
    // Any child-named volumes
    std::map<std::string, std::map<std::string, ir::Expr>> child_volumes;

    ir::Expr make_volume(const ir::Expr &tree,
                         const ir::BVH_t::Volume &volume) {
        const size_t n_args = volume.initializers.size();
        std::vector<ir::Expr> args(n_args);
        for (size_t j = 0; j < n_args; j++) {
            const auto &name = volume.initializers[j];
            args[j] = ir::Access::make(name, tree);
        }
        return ir::Build::make(volume.struct_type, args);
    }

    ir::Stmt visit(const ir::Match *node) final override {
        const ir::Var *var = node->loc.as<ir::Var>();
        internal_assert(var) << "TODO: handle Match on non-Var";
        locs.push_back(node->loc);

        const size_t n = node->arms.size();
        ir::Match::Arms new_arms(n);
        for (size_t i = 0; i < n; i++) {
            ir::Expr tree = ir::Unwrap::make(i, node->loc);
            if (node->arms[i].first.volume.has_value()) {
                ir::Expr vol = make_volume(tree, *node->arms[i].first.volume);
                volumes.emplace_back(std::move(vol));
            } else {
                volumes.emplace_back(); // undef volume
            }
            if (!node->arms[i].first.child_volumes.empty()) {
                std::map<std::string, ir::Expr> built_child_volumes;
                for (const auto &[name, volume] :
                     node->arms[i].first.child_volumes) {
                    built_child_volumes[name] = make_volume(tree, volume);
                }
                child_volumes[var->name] = std::move(built_child_volumes);
            }
            ir::Stmt stmt = mutate(node->arms[i].second);
            volumes.pop_back();
            new_arms[i] = {node->arms[i].first, std::move(stmt)};
        }
        locs.pop_back();

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

    VolumeMap make_volume_map(const std::vector<ir::TypedVar> &args,
                              const ir::Expr &value) const {
        auto get_volume = [&](const ir::Expr &e) {
            const ir::Access *access = e.as<ir::Access>();
            internal_assert(access) << e;
            const ir::Unwrap *unwrap = access->value.as<ir::Unwrap>();
            internal_assert(unwrap) << e;
            const ir::Var *var = unwrap->value.as<ir::Var>();
            internal_assert(var) << e;
            const auto &iter = child_volumes.find(var->name);
            internal_assert(iter != child_volumes.cend()) << e;
            const auto &citer = iter->second.find(access->field);
            internal_assert(citer != iter->second.cend()) << e;
            return citer->second;
        };
        VolumeMap vols;
        if (args.size() == 1) {
            vols[args[0].name] = get_volume(value);
        } else {
            const auto values = break_tuple(value);
            internal_assert(args.size() == values.size()) << value;
            for (size_t i = 0; i < args.size(); i++) {
                // TODO(ajr): this doesn't work with mismatching BVHs.
                vols[args[i].name] = get_volume(values[i]);
            }
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
                    value = opt::Simplify::simplify(value);
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

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
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

            ir::Stmt body;

            // if any of these values have a volume wrap, try to filter it out.
            // ajr: this is some of the hackiest shit I have ever done, but I
            // think it works...
            if (!child_volumes.empty()) {
                auto as = break_tuple(node->value);
                std::vector<ir::Expr> upper_bounds;
                for (const auto &a : as) {
                    VolumeMap cvols = make_volume_map(lambda->args, a);
                    Interval cbounds =
                        predicate_analysis(lambda->value, cvols, intervals);
                    internal_assert(is_const_zero(cbounds.min) &&
                                    cbounds.max.defined())
                        << cbounds.min << " and " << cbounds.max;
                    upper_bounds.push_back(std::move(cbounds.max));
                }
                internal_assert(as.size() == 2)
                    << "[unimplemented] generalized decision tree for child "
                       "volume optimization";

                ir::Expr both_good = upper_bounds[0] && upper_bounds[1];
                ir::Stmt singletons = ir::IfElse::make(
                    upper_bounds[0], ir::YieldFrom::make(make_tuple({as[0]})),
                    ir::IfElse::make(upper_bounds[1],
                                     ir::YieldFrom::make(make_tuple({as[1]}))));
                body = ir::IfElse::make(std::move(both_good),
                                        ir::YieldFrom::make(node->value),
                                        std::move(singletons));
            } else {
                body = ir::YieldFrom::make(node->value);
            }

            // Make a recursive call
            // TODO: this should be wrapped in a filter, for cases with
            // simplified predicates. This is required for proper predicate
            // analysis of conjunctions/disjunctions. ir::Stmt body =
            // ir::YieldFrom::make(ir::filter(predicate, node->value));
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

std::pair<ir::Expr, bool> try_fuse_filter(const ir::Lambda *metric,
                                          ir::Expr best,
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
            return {filter(std::move(new_lambda), as_set->b), true};
        }
    }

    // Not a nested filter, so just wrap in a filter and return
    ir::Expr new_cond = (metric->value < best);
    ir::Expr new_lambda = ir::Lambda::make(metric->args, std::move(new_cond));
    return {filter(std::move(new_lambda), std::move(maybe_filter)), false};
}

ir::Stmt build_argmin(ir::Expr metric, ir::Expr inner,
                      const ir::TypeMap &tree_types,
                      const IntervalMap &intervals, ir::Type expect_type) {
    struct RewriteArgmin : public Rewriter {
        ir::Expr metric;
        ir::WriteLoc loc;
        ir::Type tuple_t;
        const IntervalMap &intervals;
        const bool update_from_yfs;

        RewriteArgmin(ir::Expr met, ir::WriteLoc l, ir::Type t,
                      const IntervalMap &intervals, const bool update_from_yfs)
            : metric(std::move(met)), loc(std::move(l)), tuple_t(std::move(t)),
              intervals(intervals), update_from_yfs(update_from_yfs) {}

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

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        ir::Stmt visit(const ir::Scan *node) override { return node; }

        ir::Stmt visit(const ir::YieldFrom *node) override {
            if (!update_from_yfs) {
                return node;
            }
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volumes.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);

            VolumeMap vols = make_volume_map(lambda->args);

            Interval bounds =
                predicate_analysis(lambda->value, vols, intervals);
            internal_assert(bounds.max.defined())
                << "Cannot accelerate metric: " << lambda->value
                << " on: " << ir::Stmt(node);

            // Best must be at most max.
            ir::Expr value = bounds.max + std::numeric_limits<float>::epsilon();

            ir::Expr empty_expr =
                ir::Build::make(tuple_t.as<ir::Tuple_t>()->etypes[1]);
            std::vector<ir::Expr> values = {std::move(value), empty_expr};
            ir::Expr update = ir::Build::make(tuple_t, std::move(values));
            ir::Stmt do_update = ir::Accumulate::make(
                loc, ir::Accumulate::Argmin, std::move(update));
            return ir::Sequence::make({std::move(do_update), node});
        }
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
    std::vector<ir::Expr> values = {inf, std::move(empty)};
    ir::Expr init = ir::Build::make(tuple_t, std::move(values));

    // TODO(ajr): is stack memory ok here? it's not an array.
    ir::Stmt header =
        ir::Allocate::make(loc, std::move(init), ir::Allocate::Memory::Stack);

    // Make return
    ir::Expr ret_var = ir::Var::make(tuple_t, std::move(name));
    ir::Expr best_metric = ir::Extract::make(ret_var, 0);
    ir::Expr best_ref = ir::Extract::make(ret_var, 1);
    // TODO: should this be a Return?
    ir::Stmt footer;
    if (!ir::equals(ret_type, expect_type)) {
        // If (best[0] != inf) yield best[1] else {}
        ir::Expr result = ir::Select::make(
            best_metric != inf, ir::Build::make(expect_type, {best_ref}),
            ir::Build::make(expect_type));
        footer = ir::Yield::make(std::move(result));
    } else {
        footer = ir::Yield::make(std::move(best_ref));
    }

    // No lower bound (can always get better)
    // Upper bound is the current value (must be at least that good).
    IntervalMap local_intervals = intervals;
    local_intervals[best_metric] = Interval{ir::Expr(), best_metric};

    // Try to build fused filter inside.
    // TODO(ajr): scans need to be
    auto [fused_filter, fused] = try_fuse_filter(lambda, best_metric, inner);
    ir::Stmt body = build_traversal(fused_filter, tree_types, local_intervals);

    body = RewriteArgmin(std::move(metric), std::move(loc), std::move(tuple_t),
                         intervals, !fused)
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

ir::Stmt build_minimum(ir::Expr metric, ir::Expr inner,
                       const ir::TypeMap &tree_types,
                       const IntervalMap &intervals) {
    struct RewriteMinimum : public Rewriter {
        ir::Expr metric;
        ir::WriteLoc loc;
        const IntervalMap &intervals;
        const bool update_from_yfs;

        RewriteMinimum(ir::Expr met, ir::WriteLoc l,
                       const IntervalMap &intervals, const bool update_from_yfs)
            : metric(std::move(met)), loc(std::move(l)), intervals(intervals),
              update_from_yfs(update_from_yfs) {}

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
            return ir::Accumulate::make(loc, ir::Accumulate::Min,
                                        std::move(value));
        }

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        ir::Stmt visit(const ir::Scan *node) override {
            internal_error << ir::Stmt(node);
        }

        ir::Stmt visit(const ir::YieldFrom *node) override {
            if (!update_from_yfs) {
                return node;
            }
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volumes.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);

            ir::Expr upper_bound;
            ir::Stmt body = node;

            // if any of these values have a volume wrap, use it to tighten the
            // bound. ajr: this is some of the hackiest shit I have ever done,
            // but I think it works...
            if (!child_volumes.empty()) {
                auto as = break_tuple(node->value);
                std::vector<ir::Expr> upper_bounds;
                std::vector<ir::Expr> lower_bounds;
                for (const auto &a : as) {
                    VolumeMap cvols = make_volume_map(lambda->args, a);
                    Interval cbounds =
                        predicate_analysis(lambda->value, cvols, intervals);
                    internal_assert(cbounds.min.defined() &&
                                    cbounds.max.defined())
                        << cbounds.min << " and " << cbounds.max;
                    upper_bounds.push_back(std::move(cbounds.max));
                    lower_bounds.push_back(std::move(cbounds.min));
                }

                upper_bound = upper_bounds[0];
                for (size_t i = 1; i < as.size(); i++) {
                    upper_bound = max(std::move(upper_bound), upper_bounds[i]);
                }
                if (as.size() >= 2) {
                    internal_assert(as.size() == 2) << ir::Stmt(node); // TODO
                    // If a is strictly better than b, only include it
                    ir::Expr a_winner = upper_bounds[0] < lower_bounds[1];
                    ir::Expr b_winner = upper_bounds[1] < lower_bounds[0];
                    // If their a has a better lb *or* eq lb and better ub
                    ir::Expr a_closer = (lower_bounds[0] < lower_bounds[1]) |
                                        (lower_bounds[0] == lower_bounds[1]) &
                                            (upper_bounds[0] < upper_bounds[1]);
                    body = ir::IfElse::make(
                        a_winner, ir::YieldFrom::make(make_tuple({as[0]})),
                        ir::IfElse::make(
                            b_winner, ir::YieldFrom::make(make_tuple({as[1]})),
                            ir::YieldFrom::make(
                                make_tuple({select(a_closer, as[1], as[0]),
                                            select(a_closer, as[0], as[1])}))));
                }
            } else {
                VolumeMap vols = make_volume_map(lambda->args);

                Interval bounds =
                    predicate_analysis(lambda->value, vols, intervals);
                internal_assert(bounds.max.defined())
                    << "Cannot accelerate metric: " << lambda->value
                    << " on: " << ir::Stmt(node);

                // Best must be at most max.
                // Epsilon only needed if subtree isn't tight / can be empty.
                upper_bound = bounds.max;
            }

            ir::Stmt do_update = ir::Accumulate::make(loc, ir::Accumulate::Min,
                                                      std::move(upper_bound));
            return ir::Sequence::make({std::move(do_update), std::move(body)});
        }
    };

    const ir::Lambda *lambda = metric.as<ir::Lambda>();
    internal_assert(lambda) << "Metric is not a lambda: " << metric;
    ir::Type metric_t = lambda->value.type();

    static size_t counter = 0;
    std::string name = "_best" + std::to_string(counter++);
    ir::WriteLoc loc(name, metric_t);

    ir::Expr init = ir::Infinity::make(metric_t);

    // TODO(ajr): is stack memory ok here? it's not an array.
    ir::Stmt header =
        ir::Allocate::make(loc, std::move(init), ir::Allocate::Memory::Stack);

    // Make return
    ir::Expr ret_var = ir::Var::make(metric_t, std::move(name));
    // TODO: should this be a Return?
    ir::Stmt footer;
    footer = ir::Yield::make(ret_var);

    // No lower bound (can always get better)
    // Upper bound is the current value (must be at least that good).
    IntervalMap local_intervals = intervals;
    local_intervals[ret_var] = Interval{ir::Expr(), ret_var};

    // Try to build fused filter inside.
    // TODO(ajr): scans need to be
    auto [fused_filter, fused] = try_fuse_filter(lambda, ret_var, inner);
    ir::Stmt body = build_traversal(fused_filter, tree_types, local_intervals);

    body = RewriteMinimum(std::move(metric), std::move(loc), intervals, !fused)
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

        ir::Stmt visit(const ir::Yield *node) override {
            if (!a_body.defined()) {
                a_body = node;
                ir::Stmt ret = mutate(b_body);
                a_body = ir::Stmt();
                return ret;
            } else {
                if (const ir::Yield *yield = a_body.as<ir::Yield>()) {
                    return ir::Yield::make(
                        make_tuple_pair(yield->value, node->value));
                } else if (const ir::Iterate *iterate =
                               a_body.as<ir::Iterate>()) {
                    internal_error << "TODO: lower Yield x Iterate in product.";
                } else if (const ir::Scan *scan = a_body.as<ir::Scan>()) {
                    internal_assert(locs.size() == 2);
                    auto as = break_tuple(scan->value);
                    auto b = locs.back();
                    std::vector<ir::Expr> vals;
                    vals.reserve(as.size());
                    for (const auto &a : as) {
                        vals.push_back(make_tuple_pair(a, b));
                    }
                    return ir::Scan::make(make_tuple(std::move(vals)));
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

        ir::Stmt visit(const ir::Iterate *node) override {
            if (!a_body.defined()) {
                a_body = node;
                ir::Stmt ret = mutate(b_body);
                a_body = ir::Stmt();
                return ret;
            } else {
                if (const ir::Yield *yield = a_body.as<ir::Yield>()) {
                    ir::Stmt body = lower_iterate(node->value);
                    return mutate(body);
                } else if (const ir::Iterate *iterate =
                               a_body.as<ir::Iterate>()) {
                    return ir::Iterate::make(
                        product(iterate->value, node->value));
                } else if (const ir::Scan *scan = a_body.as<ir::Scan>()) {
                    internal_assert(locs.size() == 2);
                    auto as = break_tuple(scan->value);
                    auto b = locs.back();
                    std::vector<ir::Expr> vals;
                    vals.reserve(as.size());
                    for (const auto &a : as) {
                        vals.push_back(make_tuple_pair(a, b));
                    }
                    return ir::Scan::make(make_tuple(std::move(vals)));
                } else if (const ir::YieldFrom *from =
                               a_body.as<ir::YieldFrom>()) {
                    internal_error
                        << "TODO: lower Iterate + YieldFrom properly: "
                        << a_body << " and " << ir::Stmt(node);
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
                    internal_assert(locs.size() == 2);
                    auto bs = break_tuple(node->value);
                    auto a = locs.front();
                    std::vector<ir::Expr> vals;
                    vals.reserve(bs.size());
                    for (const auto &b : bs) {
                        vals.push_back(make_tuple_pair(a, b));
                    }
                    return ir::Scan::make(make_tuple(std::move(vals)));
                } else if (const ir::Iterate *iterate =
                               a_body.as<ir::Iterate>()) {
                    internal_assert(locs.size() == 2);
                    auto bs = break_tuple(node->value);
                    auto a = locs.front();
                    std::vector<ir::Expr> vals;
                    vals.reserve(bs.size());
                    for (const auto &b : bs) {
                        vals.push_back(make_tuple_pair(a, b));
                    }
                    return ir::Scan::make(make_tuple(std::move(vals)));
                } else if (const ir::Scan *scan = a_body.as<ir::Scan>()) {
                    // Cartesian product of nodes! TODO: doesn't have to be...
                    // Make this scheduable?
                    auto as = break_tuple(scan->value);
                    auto bs = break_tuple(node->value);
                    std::vector<ir::Expr> pairs;
                    for (const auto &av : as) {
                        for (const auto &bv : bs) {
                            pairs.push_back(make_tuple_pair(av, bv));
                        }
                    }
                    return ir::Scan::make(make_tuple(std::move(pairs)));
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

            std::vector<ir::Stmt> stmts(data.size() + !children.empty());
            // TODO: visit order should be scheduable?
            for (size_t i = 0; i < data.size(); i++) {
                ir::Expr access = ir::Access::make(data[i].name, node);
                if (data[i].type.is_iterable()) {
                    // forall d in data: yield d
                    stmts[i] = ir::Iterate::make(std::move(access));
                } else {
                    // yield d
                    stmts[i] = ir::Yield::make(std::move(access));
                }
            }
            if (!children.empty()) {
                std::vector<ir::Expr> cs;
                cs.reserve(children.size());
                for (const auto &c : children) {
                    cs.push_back(ir::Access::make(c.name, node));
                }
                stmts.back() = ir::Scan::make(make_tuple(cs));
            }

            arms[i].first = bvh->nodes[i];
            internal_assert(!stmts.empty());
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
    // Argmin/Minimum are a bit more complicated, because of filter fusion.
    case ir::SetOp::minimum: {
        return build_minimum(as_set->a, as_set->b, tree_types, intervals);
    }
    case ir::SetOp::argmin: {
        return build_argmin(as_set->a, as_set->b, tree_types, intervals,
                            expr.type());
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

#ifdef LOG_TREE_VISITS
ir::Stmt add_tree_visit_logger(ir::Stmt stmt) {
    struct AddLogger : public ir::Mutator {
        const ir::WriteLoc &from_log_loc, &yield_log_loc;
        const ir::Expr from_str = ir::StringImm::make("Nodes visited: "),
                       yield_str = ir::StringImm::make("Primitives tested: ");

        AddLogger(const ir::WriteLoc &from_log_loc,
                  const ir::WriteLoc &yield_log_loc)
            : from_log_loc(from_log_loc), yield_log_loc(yield_log_loc) {}

        ir::Stmt visit(const ir::YieldFrom *node) override {
            return ir::Sequence::make(
                {ir::Accumulate::make(
                     from_log_loc, ir::Accumulate::Add,
                     make_const(
                         from_log_loc.type,
                         node->value.type().as<ir::Tuple_t>()->etypes.size())),
                 node});
        }

        // Bleh, Yields in reductions have been rewritten to accumulates by
        // argmins.

        ir::Stmt visit(const ir::Accumulate *node) override {
            return ir::Sequence::make({
                ir::Accumulate::make(yield_log_loc, ir::Accumulate::Add,
                                     make_one(yield_log_loc.type)),
                ir::Print::make(
                    {ir::StringImm::make("before "),
                     ir::StringImm::make(node->loc.base),
                     ir::Var::make(node->loc.type, node->loc.base)}),
                node,
                ir::Print::make(
                    {ir::StringImm::make("updated "),
                     ir::StringImm::make(node->loc.base),
                     ir::Var::make(node->loc.type, node->loc.base)}),
            });
        }

        ir::Stmt visit(const ir::IfElse *node) override {
            const ir::BinOp *binop = node->cond.as<ir::BinOp>();
            if (!binop || binop->op != ir::BinOp::Lt ||
                node->then_body.is<ir::IfElse>()) {
                return ir::Mutator::visit(node);
            }
            return ir::Sequence::make(
                {ir::Print::make({binop->a, ir::StringImm::make(" less than "),
                                  binop->b, node->cond}),
                 ir::Mutator::visit(node)});
        }

        ir::Stmt visit(const ir::Yield *node) override {
            // This should print, assuming an argmin
            ir::Expr from_log =
                ir::Var::make(from_log_loc.type, from_log_loc.base);
            ir::Expr yield_log =
                ir::Var::make(yield_log_loc.type, yield_log_loc.base);
            return ir::Sequence::make(
                {ir::Print::make({from_str, from_log}),
                 ir::Print::make(
                     {yield_str,
                      yield_log - from_log / 2}), // hacky binary assumption.
                 node});
        }
    };

    static uint64_t counter = 0;
    static const ir::Type u32 = ir::UInt_t::make(32);

    const std::string from_log_name = "_from_log" + std::to_string(counter);
    ir::WriteLoc from_log_loc(from_log_name, u32);
    const std::string yield_log_name = "_yield_log" + std::to_string(counter);
    ir::WriteLoc yield_log_loc(yield_log_name, u32);
    counter++;

    std::vector<ir::Stmt> stmts{
        ir::Allocate::make(from_log_loc, make_zero(u32),
                           ir::Allocate::Memory::Stack),
        ir::Allocate::make(yield_log_loc, make_zero(u32),
                           ir::Allocate::Memory::Stack),
        AddLogger(from_log_loc, yield_log_loc).mutate(stmt),
    };
    return ir::Sequence::make(std::move(stmts));
}
#endif
// Wrap the first Match seen in a recursive loop on all trees seen in the body.
struct WrapMatchInRecLoop : public ir::Mutator {
    std::vector<ir::TypedVar> trees;

    WrapMatchInRecLoop(std::vector<ir::TypedVar> trees)
        : trees(std::move(trees)) {}

    ir::Stmt visit(const ir::Match *node) override {
        return ir::RecLoop::make(std::move(trees), node);
    }
};

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

        std::vector<ir::TypedVar> trees;
        std::vector<ir::Function::Argument> func_args;
        func_args.reserve(free_vars.size());
        for (const auto &var : free_vars) {
            if (const auto &iter = tree_types.find(var.name);
                iter != tree_types.cend()) {
                trees.push_back({var.name, iter->second});
                func_args.emplace_back(var.name, iter->second);
            } else {
                // TODO: mutability? only if the free vars are mutated in the
                // traversal somehow... That shouldn't happen, I think?
                func_args.emplace_back(var.name, var.type);
            }
        }
        // TODO(ajr): relax this, when we lower trees before arrays.
        internal_assert(!trees.empty())
            << "Lowering of: " << expr << " does not contain any tree types.";

        // TODO(ajr): is there more we can do with intervals?
        // e.g. bounded interval hierarchies, kd-trees, tri.x < bound
        // queries, etc?
        IntervalMap intervals;
        ir::Stmt body = build_traversal(expr, tree_types, intervals);
#ifdef LOG_TREE_VISITS
        body = add_tree_visit_logger(std::move(body));
#endif
        internal_assert(body.defined());
        // Now wrap in a recursive loop on any trees.
        body = WrapMatchInRecLoop(std::move(trees)).mutate(body);

        // When should this type be concretized into e.g. a list?
        ir::Type ret_type = expr.type();
        auto f = std::make_shared<ir::Function>(
            func, std::move(func_args), std::move(ret_type), std::move(body),
            ir::Function::InterfaceList{},
            std::vector<ir::Function::Attribute>{});
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

ir::Program LowerTrees::run(ir::Program program,
                            const CompilerOptions &options) const {
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
