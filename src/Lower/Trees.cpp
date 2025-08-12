#include "Lower/Trees.h"

#include "Lower/PredicateAnalysis.h"

#include "IR/Analysis.h"
#include "IR/Argument.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"

#include "Error.h"
#include "Log.h"

#include "Utils.h"

#include "Opt/Simplify.h"

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

struct VariantData {
    std::vector<ir::TypedVar> payload;

    struct ChildAccess {
        ir::TypedVar child;
        std::optional<uint32_t> index = {};
    };
    std::vector<ChildAccess> children;
};

VariantData analyze_node(const ir::BVH_t::Variant &variant,
                         const ir::Type &primitive_type,
                         const ir::Type &tree_type) {
    const ir::BVH_t *bvh = tree_type.as<ir::BVH_t>();
    internal_assert(bvh) << tree_type;
    ir::Type tree_reference = ir::Ref_t::make(bvh->name);
    std::vector<ir::TypedVar> payload;
    std::vector<VariantData::ChildAccess> children;
    for (const ir::TypedVar &parameter : variant.fields()) {
        ir::Type parameter_type = parameter.type;
        if (ir::equals(parameter_type, primitive_type) ||
            (parameter_type.is_iterable() &&
             ir::equals(primitive_type, parameter_type.element_of()))) {
            payload.push_back(parameter);
            continue;
        }
        if (ir::equals(parameter_type, tree_reference)) {
            children.emplace_back(VariantData::ChildAccess{.child = parameter});
            continue;
        }
        if (parameter_type.is_iterable() &&
            ir::equals(parameter_type.element_of(), tree_reference)) {
            std::optional<uint32_t> size =
                get_constant_value(parameter_type.size());
            internal_assert(size.has_value()) << parameter_type;
            for (int i = 0, e = *size; i < e; ++i) {
                children.push_back(VariantData::ChildAccess{
                    .child = parameter,
                    .index = i,
                });
            }
            continue;
        }
    }

    return VariantData{
        .payload = payload,
        .children = children,
    };
}

int64_t get_child_reference_count(const ir::BVH_t::Variant &variant,
                                  const ir::Type tree) {
    const ir::BVH_t *bvh = tree.as<ir::BVH_t>();
    internal_assert(bvh) << tree;
    ir::Type tree_reference = ir::Ref_t::make(bvh->name);

    int64_t count = 0;
    for (const auto &[_, type] : variant.fields()) {
        if (type.is<ir::Ref_t>() &&
            ir::equals(type.element_of(), tree_reference)) {
            ++count;
            continue;
        }
        if (type.is_iterable() && type.element_of().is<ir::Ref_t>() &&
            ir::equals(type.element_of(), tree_reference)) {
            std::optional<uint32_t> size = get_constant_value(type.size());
            internal_assert(size.has_value()) << type;
            count += *size;
            continue;
        }
    }
    return count;
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
    // The list of volumes for the current match arms.
    mutable std::vector<ir::Expr> volumes;
    // The list of nodes for the current matches.
    std::vector<ir::Expr> locs;

    ir::Stmt visit(const ir::Match *node) final override {
        const ir::Var *var = node->loc.as<ir::Var>();
        internal_assert(var) << "TODO: handle Match on non-Var";
        locs.push_back(node->loc);

        ir::Match::Arms new_arms;
        for (size_t i = 0, e = node->arms.size(); i < e; ++i) {
            ir::Expr tree = ir::Unwrap::make(i, node->loc);
            auto [variant, statement] = node->arms[i];
            if (std::optional<ir::BVH_t::Volume> volume = variant.volume;
                volume.has_value()) {
                const size_t n_args = variant.volume->initializers.size();
                switch (volume->bound_type) {
                case ir::BVH_t::Volume::BoundType::Enclosing: {
                    std::vector<ir::Expr> args;
                    args.reserve(n_args);
                    for (size_t j = 0; j < n_args; ++j) {
                        const std::string &name = volume->initializers[j];
                        args.push_back(ir::Access::make(name, tree));
                    }
                    volumes.emplace_back(
                        ir::Build::make(volume->struct_type, args));
                } break;
                case ir::BVH_t::Volume::BoundType::Childwise: {
                    std::optional<ir::BVH_t::Volume> volume = variant.volume;
                    internal_assert(volume.has_value())
                        << "[unexpected] variant with no volume for childwise "
                           "bounding: "
                        << variant;
                    int child_count =
                        get_child_reference_count(variant, node->loc.type());
                    for (int i = 0; i < child_count; ++i) {
                        std::vector<ir::Expr> args;
                        args.reserve(n_args);
                        const auto *struct_t =
                            volume->struct_type.as<ir::Struct_t>();
                        internal_assert(struct_t);
                        for (size_t j = 0; j < n_args; ++j) {
                            const std::string &name = volume->initializers[j];
                            ir::Expr arg = ir::Access::make(name, tree);
                            if (!ir::equals(arg.type(),
                                            struct_t->fields[j].type)) {
                                arg = ir::Extract::make(arg, ir::Expr(i));
                            }
                            args.push_back(arg);
                        }
                        volumes.emplace_back(
                            ir::Build::make(volume->struct_type, args));
                    }
                    std::reverse(volumes.begin(), volumes.end());
                }
                }
            } else {
                volumes.emplace_back(); // undefined volume
            }
            statement = mutate(statement);
            new_arms.push_back({
                std::move(variant),
                std::move(statement),
            });
        }
        locs.pop_back();

        return ir::Match::make(node->loc, std::move(new_arms));
    }

    VolumeMap make_volume_map(const std::vector<ir::TypedVar> &args) const {
        VolumeMap volume_map;
        for (size_t i = 0, n = args.size(); i < n; ++i) {
            // Even if a volume is undefined, needs to be added so
            // predicate analysis knows it's non-varying.
            volume_map[args[i].name] = volumes.back();
            volumes.pop_back();
        }
        return volume_map;
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
            internal_assert(volumes.size() == lambda->args.size())
                << volumes.size() << " vs " << lambda->args.size();
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
    if (const auto *as_var = expr.as<ir::Var>()) {
        internal_assert(as_var->type.is<ir::Set_t>())
            << "Cannot build traversal for non-set: " << expr;
        const auto &iter = tree_types.find(as_var->name);
        internal_assert(iter != tree_types.cend())
            << "Lowering of: " << expr << " does not have associated BVH type.";
        const ir::Type &tree = iter->second;
        const ir::BVH_t *bvh = tree.as<ir::BVH_t>();
        internal_assert(bvh);

        ir::Expr bvh_expr = ir::Var::make(tree, as_var->name);

        const size_t n_nodes = bvh->variants.size();
        ir::Match::Arms arms(n_nodes);
        for (size_t i = 0; i < n_nodes; i++) {
            ir::Expr node = ir::Unwrap::make(/*unwrap_index=*/i, bvh_expr);
            const auto [payload, children] =
                analyze_node(bvh->variants[i], as_var->type.element_of(), tree);

            std::vector<ir::Stmt> statements;
            // TODO: visit order should be scheduable?
            for (size_t i = 0; i < payload.size(); i++) {
                ir::Expr access = ir::Access::make(payload[i].name, node);
                statements.push_back(payload[i].type.is_iterable()
                                         // forall d in data: yield d
                                         ? ir::Iterate::make(std::move(access))
                                         // yield d
                                         : ir::Yield::make(std::move(access)));
            }
            if (!children.empty()) {
                const std::optional<ir::BVH_t::Volume> &volume =
                    bvh->variants[i].volume;
                internal_assert(volume.has_value())
                    << "[unexpected] no bounding volume: " << bvh->variants[i];
                switch (volume->bound_type) {
                case ir::BVH_t::Volume::BoundType::Enclosing: {
                    // Since the bounding volume encloses all children, we have
                    // a single scan encompassing all accesses.
                    std::vector<ir::Expr> accesses;
                    accesses.reserve(children.size());
                    for (const auto &[child, index] : children) {
                        ir::Expr access = ir::Access::make(child.name, node);
                        internal_assert(!index.has_value());
                        accesses.push_back(access);
                    }
                    statements.push_back(ir::Scan::make(make_tuple(accesses)));
                } break;
                case ir::BVH_t::Volume::BoundType::Childwise: {
                    for (const auto &[child, index] : children) {
                        ir::Expr access = ir::Access::make(child.name, node);
                        internal_assert(index.has_value());
                        access = ir::Extract::make(
                            access,
                            ir::UIntImm::make(ir::UInt_t::make(32), *index));
                        statements.push_back(ir::Scan::make(access));
                    }
                } break;
                }
            }

            arms[i].first = bvh->variants[i];

            internal_assert(!statements.empty());
            arms[i].second = ir::Sequence::make(std::move(statements));
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

// Wrap the first Match seen in a recursive loop on all trees seen in the body.
struct WrapMatchInRecLoop : public ir::Mutator {
    std::vector<ir::Argument> trees;

    WrapMatchInRecLoop(std::vector<ir::Argument> trees)
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

        std::vector<ir::Argument> trees;
        std::vector<ir::Argument> func_args;
        func_args.reserve(free_vars.size());
        for (const auto &var : free_vars) {
            if (const auto &iter = tree_types.find(var.name);
                iter != tree_types.cend()) {
                trees.push_back(ir::Argument(var.name, iter->second));
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

    LowerBVH lower(tree_types);

    // Remap externs.
    for (auto &[name, type] : program.externs) {
        const auto &iter = tree_types.find(name);
        if (iter != tree_types.cend()) {
            type = iter->second;
        }
    }

    for (auto &[_, f] : program.funcs) {
        f->body = lower.mutate(f->body);
    }

    for (auto &[name, f] : lower.new_funcs) {
        auto [_, inserted] =
            program.funcs.try_emplace(std::move(name), std::move(f));
        internal_assert(inserted);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
