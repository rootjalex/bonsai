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
#include <functional>
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
    // Search for nodes annotated as data.
    for (const auto &annot : node.annotations) {
        if (const auto *d = annot.as<ir::Annotation::Data>()) {
            // TODO: make this not n^2.
            for (const auto &param : node.fields()) {
                if (param.name == d->name) {
                    const bool is_prim = ir::equals(prim_t, param.type);
                    const bool is_array_prim =
                        (param.type.is<ir::Array_t>() &&
                         ir::equals(prim_t,
                                    param.type.as<ir::Array_t>()->etype));
                    const bool is_vector_prim =
                        (param.type.is<ir::Vector_t>() &&
                         ir::equals(prim_t,
                                    param.type.as<ir::Vector_t>()->etype));

                    internal_assert(is_prim || is_array_prim || is_vector_prim)
                        << param.type << " versus " << prim_t << "\n";
                    data.push_back(param);
                }
            }
        }
    }

    // Search for recursive references.
    for (const auto &param : node.fields()) {
        if (param.type.is<ir::Ref_t>()) { // TODO: and is ref to
                                          // current tree type?
            children.push_back(param);
        }
    }

    return {data, children};
}

// Key under which a node's precomputed aggregate is recorded, e.g. "count()"
// or "min(id)". Must agree between the annotation and the query operator
// looking the aggregate up.
std::string aggregate_key(const ir::Annotation::Aggregate &agg) {
    std::string key = to_string(agg.op) + "(";
    for (size_t i = 0; i < agg.args.size(); i++) {
        if (i != 0) {
            key += ", ";
        }
        key += agg.args[i];
    }
    key += ")";
    return key;
}

std::string aggregate_key(ir::Annotation::Aggregate::OpType op,
                          const std::vector<std::string> &args) {
    return aggregate_key(ir::Annotation::Aggregate{op, args, ""});
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
    // Volumes annotated on a node's children rather than on the node itself.
    // Keyed by the matched tree variable, then by child field name.
    std::map<std::string, std::map<std::string, ir::Expr>> child_volumes;
    // The list of tagged intervals. Holds scalar interval OR map of field
    // intervals.
    std::vector<
        std::variant<std::monostate, Interval, std::map<std::string, Interval>>>
        intervals;
    // Reduction augmentations for the current match arms: the field holding a
    // precomputed aggregate over the subtree, keyed by the aggregate it
    // stores.
    // TODO: key is a string concat of the aggregation request, not ideal...
    std::vector<std::map<std::string, ir::Expr>> aggregations;
    // The list of nodes for the current matches.
    std::vector<ir::Expr> locs;

    ir::Stmt visit(const ir::Match *node) final override {
        const ir::Var *var = node->loc.as<ir::Var>();
        internal_assert(var) << "TODO: handle Match on non-Var";
        locs.push_back(node->loc);

        const size_t n = node->arms.size();
        ir::Match::Arms new_arms(n);
        for (size_t i = 0; i < n; i++) {
            ir::Expr tree = ir::Unwrap::make(i, node->loc);

            const auto &bvh_node = node->arms[i].first;

            const auto make_interval =
                [&](const std::string &low,
                    const std::string &high) -> Interval {
                ir::Expr low_expr = ir::Access::make(low, tree);
                ir::Expr high_expr = ir::Access::make(high, tree);
                return Interval{std::move(low_expr), std::move(high_expr)};
            };

            const auto make_volume =
                [&](const ir::Annotation::Volume *volume) -> ir::Expr {
                const auto &inits = volume->initializers;
                const size_t n_args = inits.size();
                std::vector<ir::Expr> args(n_args);
                for (size_t j = 0; j < n_args; j++) {
                    const auto &name = inits[j];
                    args[j] = ir::Access::make(name, tree);
                }
                return ir::Build::make(volume->struct_type, args);
            };

            if (bvh_node.has_volume()) {
                volumes.emplace_back(make_volume(bvh_node.get_volume()));
            } else {
                volumes.emplace_back(); // undef volume
            }
            std::variant<std::monostate, Interval,
                         std::map<std::string, Interval>>
                interval;
            std::map<std::string, ir::Expr> aggregation;
            std::map<std::string, ir::Expr> built_child_volumes;
            for (const auto &annot : node->arms[i].first.annotations) {
                if (const auto *a_interval =
                        annot.as<ir::Annotation::Interval>()) {
                    Interval m_interval =
                        make_interval(a_interval->low, a_interval->high);
                    if (a_interval->scalar.empty()) {
                        internal_assert(
                            std::holds_alternative<std::monostate>(interval))
                            << "Multiple primitive interval annotations on "
                               "node: "
                            << ir::Stmt(node);
                        interval = m_interval;
                    } else {
                        if (std::holds_alternative<std::monostate>(interval)) {
                            interval = std::map<std::string, Interval>{
                                {a_interval->scalar, m_interval}};
                        } else {
                            auto *as_map =
                                std::get_if<std::map<std::string, Interval>>(
                                    &interval);
                            (*as_map)[a_interval->scalar] = m_interval;
                        }
                    }
                } else if (const auto *agg =
                               annot.as<ir::Annotation::Aggregate>()) {
                    aggregation[aggregate_key(*agg)] =
                        ir::Access::make(agg->value, tree);
                } else if (const auto *vol =
                               annot.as<ir::Annotation::Volume>()) {
                    if (vol->geometry.empty()) {
                        continue; // the node's own volume, handled above
                    }
                    // Tagged on a child.
                    built_child_volumes[vol->geometry] = make_volume(vol);
                }
            }
            intervals.emplace_back(std::move(interval));
            aggregations.emplace_back(std::move(aggregation));
            if (!built_child_volumes.empty()) {
                child_volumes[var->name] = built_child_volumes;
            }

            ir::Stmt stmt = mutate(node->arms[i].second);
            volumes.pop_back();
            intervals.pop_back();
            aggregations.pop_back();
            if (!built_child_volumes.empty()) {
                child_volumes.erase(var->name);
            }
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

    // Build a volume map from the child volumes attached to `value`, for the
    // case where a node annotates its children's bounds rather than its own.
    VolumeMap make_volume_map(const std::vector<ir::TypedVar> &args,
                              const ir::Expr &value) const {
        const auto get_volume = [&](const ir::Expr &e) {
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

    IntervalMap make_interval_map(const std::vector<ir::TypedVar> &args,
                                  const IntervalMap &existing) const {
        IntervalMap ints = existing;
        const size_t n = intervals.size();
        internal_assert(n == args.size())
            << "Making interval map with incorrect number of arguments: "
            << args.size() << " vs. " << n;
        for (size_t i = 0; i < n; i++) {
            if (const auto *interval = std::get_if<Interval>(&intervals[i])) {
                // name -> interval (set of scalars)
                ints[args[i]] = *interval;
            } else if (const auto *field_map =
                           std::get_if<std::map<std::string, Interval>>(
                               &intervals[i])) {
                ir::Expr var = args[i];
                for (const auto &field : *field_map) {
                    ir::Expr expr = ir::Access::make(field.first, var);
                    ints[expr] = field.second;
                }
            }
        }
        return ints;
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
            IntervalMap ints = make_interval_map(lambda->args, intervals);

            Interval bounds = predicate_analysis(lambda->value, vols, ints);
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
            IntervalMap ints = make_interval_map(lambda->args, intervals);

            Interval bounds = predicate_analysis(lambda->value, vols, ints);

            // Make a recursive call
            // TODO: this should be wrapped in a filter, for cases with
            // simplified predicates. This is required for proper predicate
            // analysis of conjunctions/disjunctions. ir::Stmt body =
            // ir::YieldFrom::make(ir::filter(predicate, node->value));
            ir::Stmt body = ir::YieldFrom::make(node->value);
            // Add the maybe case -> recursive call. A bound that is trivially
            // true prunes nothing, so skip the guard entirely.
            if (bounds.max.defined() && !is_const_one(bounds.max)) {
                body = ir::IfElse::make(std::move(bounds.max), std::move(body));
            }

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

// Fuse `metric(x) < best` into an existing filter if there is one, so that the
// value-based pruning condition participates in predicate analysis. The bool
// reports whether an existing filter was fused into; when false the caller
// must still update the accumulator from recursive calls itself.
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

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
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

    ir::Expr inf = ir::Extrema::make(std::move(metric_t), ir::Extrema::inf);
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
    auto [fused_filter, fused] = try_fuse_filter(lambda, best_metric, inner);
    ir::Stmt body = build_traversal(fused_filter, tree_types, local_intervals);

    body = RewriteArgmin(std::move(metric), std::move(loc), std::move(tuple_t))
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

// Algorithm 2 (LowerMin) of the paper. `min` is both associative and
// idempotent, so the traversal prunes in two complementary ways: value-based
// pruning, where a subtree whose metric cannot beat the running minimum is
// skipped, and inclusion, where a subtree that stores its own minimum updates
// the accumulator without being visited.
//
// The value-based half is expressed by fusing `M(x) < best` into the inner
// filter (try_fuse_filter) and letting predicate analysis bound it; the
// accumulator's interval is recorded as (-inf, best] so that analysis knows a
// candidate must beat the running value.
ir::Stmt build_minimum(ir::Expr metric, ir::Expr inner,
                       const ir::TypeMap &tree_types,
                       const IntervalMap &intervals, ir::Type expect_type) {
    struct RewriteMinimum : public Rewriter {
        ir::Expr metric;
        ir::WriteLoc loc;
        const IntervalMap &intervals;
        // True when try_fuse_filter did not fuse, so recursive calls are not
        // already guarded by the value-based condition and this rewrite must
        // tighten the accumulator itself.
        const bool update_from_yfs;

        RewriteMinimum(ir::Expr met, ir::WriteLoc l,
                       const IntervalMap &intervals, const bool update_from_yfs)
            : metric(std::move(met)), loc(std::move(l)), intervals(intervals),
              update_from_yfs(update_from_yfs) {}

        using ir::Mutator::visit;

        // yield x => upd a minb(a, M(x))
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

        // iter xs => upd a minb(a, min(M, xs))
        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        ir::Stmt visit(const ir::Scan *node) override {
            internal_error << "Cannot scan a subtree for a minimum without an "
                              "aggregate augmentation: "
                           << ir::Stmt(node);
        }

        // from tr => if maybe(min(M(tr)) < a): upd a minb(a, max(M, tr)); from tr
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
            Interval bounds = predicate_analysis(lambda->value, vols, intervals);
            internal_assert(bounds.max.defined())
                << "Cannot accelerate metric: " << lambda->value
                << " on: " << ir::Stmt(node);

            // The best value in this subtree is at most the metric's upper
            // bound over the subtree's volume, so the accumulator can be
            // tightened before recursing.
            ir::Stmt do_update =
                ir::Accumulate::make(loc, ir::Accumulate::Min, bounds.max);
            return ir::Sequence::make({std::move(do_update), node});
        }
    };

    const ir::Lambda *lambda = metric.as<ir::Lambda>();
    internal_assert(lambda) << "Metric is not a lambda: " << metric;
    ir::Type metric_t = lambda->value.type();

    static size_t counter = 0;
    std::string name = "_best" + std::to_string(counter++);
    ir::WriteLoc loc(name, metric_t);

    // WrapWithAccumulator(a, inf)
    ir::Expr init = ir::Extrema::make(metric_t, ir::Extrema::inf);
    // TODO(ajr): is stack memory ok here? it's not an array.
    ir::Stmt header =
        ir::Allocate::make(loc, std::move(init), ir::Allocate::Memory::Stack);

    ir::Expr ret_var = ir::Var::make(metric_t, std::move(name));
    // A minimum over a set that can be empty is optional: the accumulator is
    // still at its identity exactly when nothing was visited.
    ir::Stmt footer;
    if (!ir::equals(metric_t, expect_type)) {
        ir::Expr result = ir::Select::make(
            ret_var != ir::Extrema::make(metric_t, ir::Extrema::inf),
            ir::Build::make(expect_type, {ret_var}),
            ir::Build::make(expect_type));
        footer = ir::Yield::make(std::move(result));
    } else {
        footer = ir::Yield::make(ret_var);
    }

    // No lower bound (can always get better).
    // Upper bound is the current value (must be at least that good).
    IntervalMap local_intervals = intervals;
    local_intervals[ret_var] = Interval{ir::Expr(), ret_var};

    auto [fused_filter, fused] = try_fuse_filter(lambda, ret_var, inner);
    ir::Stmt body = build_traversal(fused_filter, tree_types, local_intervals);

    body = RewriteMinimum(std::move(metric), std::move(loc), intervals, !fused)
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

// Algorithm 1, lines 13-22 (associative reduction), specialized to `count`.
// A node that stores a count augmentation contributes it wholesale instead of
// being traversed; otherwise the scan itself becomes a counting scan.
ir::Stmt build_count(ir::Stmt body) {
    struct RewriteCount : public Rewriter {
        ir::WriteLoc loc;

        RewriteCount(ir::WriteLoc l) : loc(std::move(l)) {}

        using ir::Mutator::visit;

        // yield x => upd a (a + 1)
        ir::Stmt visit(const ir::Yield *node) override {
            return ir::Accumulate::make(loc, ir::Accumulate::Add,
                                        make_one(loc.base_type));
        }

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        // scan tr => if tr has count(tr) then upd a (a + tr.count) else scan<count> tr
        ir::Stmt visit(const ir::Scan *node) override {
            const std::string key = aggregate_key(
                ir::Annotation::Aggregate::count, /*args=*/{});
            // TODO: how does this work with joins...?
            const bool all_have_count =
                std::all_of(aggregations.begin(), aggregations.end(),
                            [&](const auto &agg) { return agg.contains(key); });
            if (!all_have_count) {
                return ir::Scan::make(ir::AggOp::count, node->value);
            }
            // Coiterating several trees counts their product.
            ir::Expr total = aggregations.front().at(key);
            for (size_t i = 1; i < aggregations.size(); i++) {
                total = total * aggregations[i].at(key);
            }
            return ir::Accumulate::make(loc, ir::Accumulate::Add,
                                        std::move(total));
        }

        // TODO: this should be a sum of the results!
        ir::Stmt visit(const ir::YieldFrom *node) override { return node; }
    };

    const ir::Type ret_type = ir::UInt_t::make(64); // TODO: adjustable?

    static size_t counter = 0;
    std::string name = "_count" + std::to_string(counter++);
    ir::WriteLoc loc(name, ret_type);

    // WrapWithAccumulator(a, 0)
    ir::Stmt header = ir::Allocate::make(loc, make_zero(ret_type),
                                         ir::Allocate::Memory::Stack);
    ir::Expr ret_var = ir::Var::make(ret_type, std::move(name));
    ir::Stmt footer = ir::Yield::make(std::move(ret_var));

    body = RewriteCount(std::move(loc)).mutate(body);

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
                    return ir::Scan::make({}, make_tuple(std::move(vals)));
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
                    return ir::Scan::make({}, make_tuple(std::move(vals)));
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
                    return ir::Scan::make({}, make_tuple(std::move(vals)));
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
                    return ir::Scan::make({}, make_tuple(std::move(vals)));
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
                    return ir::Scan::make({}, make_tuple(std::move(pairs)));
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

        return build_base_scan(as_var->name, bvh);
    }

    if (const ir::AggOp *as_agg = expr.as<ir::AggOp>()) {
        switch (as_agg->op) {
        case ir::AggOp::count: {
            ir::Stmt body = build_traversal(as_agg->a, tree_types, intervals);
            return build_count(body);
        }
        default: {
            internal_error << "TODO: " << expr << " tree fusion.";
        }
        }
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
        return build_argmin(as_set->a, as_set->b, tree_types, intervals,
                            expr.type());
    }
    case ir::SetOp::minimum: {
        return build_minimum(as_set->a, as_set->b, tree_types, intervals,
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
    ir::Expr visit(const ir::AggOp *op) override { return build_func(op); }
};

} // namespace

ir::Stmt build_base_scan(const std::string &name, const ir::BVH_t *bvh_t) {
    ir::Expr bvh_expr = ir::Var::make(bvh_t, name);

    const size_t n_nodes = bvh_t->nodes.size();
    ir::Match::Arms arms(n_nodes);
    for (size_t i = 0; i < n_nodes; i++) {
        ir::Expr node = ir::Unwrap::make(i, bvh_expr);
        const auto [data, children] =
            analyze_node(bvh_t->nodes[i], bvh_t->primitive);

        std::vector<ir::Stmt> stmts(data.size() + !children.empty());
        // TODO: visit order should be scheduable?
        for (size_t i = 0; i < data.size(); i++) {
            ir::Expr access = ir::Access::make(data[i].name, node);
            if (data[i].type.is_iterable()) {
                // forall d in data: yield d
                if (data[i].type.is<ir::Array_t>() &&
                    data[i].type.as<ir::Array_t>()->size.defined() &&
                    !is_const(data[i].type.as<ir::Array_t>()->size)) {
                    // Size must be a parameter of the type, need to change the
                    // size somehow...
                    internal_error
                        << "TODO: handle array size in tree params\n";
                }
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
            stmts.back() = ir::Scan::make({}, make_tuple(cs));
        }

        arms[i].first = bvh_t->nodes[i];
        internal_assert(!stmts.empty());
        if (stmts.size() == 1) {
            // Special case.
            arms[i].second = stmts[0];
        } else {
            arms[i].second = ir::Sequence::make(std::move(stmts));
        }
    }
    return ir::Match::make(std::move(bvh_expr), std::move(arms));
}

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
