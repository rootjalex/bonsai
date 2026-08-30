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
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

ir::Stmt build_traversal(const ir::Expr &expr, const ir::TypeMap &tree_types,
                         const IntervalMap &intervals);

static size_t counter = 0;

std::string unique_iter_name() { return "_iter" + std::to_string(counter++); }

// The fields of a variant that a traversal cares about.
struct VariantData {
    std::vector<ir::TypedVar> payload;

    struct ChildAccess {
        ir::TypedVar child;
        // Set when the child lives at an index of an array of children, as in
        // a wide BVH.
        std::optional<uint32_t> index = {};
    };
    std::vector<ChildAccess> children;
};

// Which fields of a variant hold primitives, and which hold child references.
// Payload is recognised by type: any field whose type is the tree's primitive,
// or an iterable of it. A `data` annotation, where one is present, restricts
// the search to the named field and is checked for consistency.
VariantData analyze_node(const ir::BVH_t::Variant &variant,
                         const ir::Type &primitive_type, const ir::BVH_t *bvh) {
    internal_assert(bvh);
    const ir::Type tree_reference = ir::Ref_t::make(bvh->name);

    // A `with data = x` annotation names the payload field explicitly.
    std::set<std::string> annotated;
    for (const auto &annot : variant.annotations) {
        if (const auto *d = annot.as<ir::Annotation::Data>()) {
            annotated.insert(d->name);
        }
    }

    std::vector<ir::TypedVar> payload;
    std::vector<VariantData::ChildAccess> children;
    for (const ir::TypedVar &parameter : variant.fields()) {
        const ir::Type parameter_type = parameter.type;
        const bool is_primitive =
            ir::equals(parameter_type, primitive_type) ||
            (parameter_type.is_iterable() &&
             ir::equals(primitive_type, parameter_type.element_of()));
        if (is_primitive) {
            // With annotations present, only the annotated fields count.
            if (annotated.empty() || annotated.contains(parameter.name)) {
                payload.push_back(parameter);
            }
            continue;
        }
        internal_assert(!annotated.contains(parameter.name))
            << "Field " << parameter.name << " is annotated as data but has "
            << "type " << parameter_type << ", not the tree's primitive "
            << primitive_type;
        if (ir::equals(parameter_type, tree_reference)) {
            children.emplace_back(VariantData::ChildAccess{.child = parameter});
            continue;
        }
        if (parameter_type.is_iterable() &&
            ir::equals(parameter_type.element_of(), tree_reference)) {
            // An array of children, as in a wide BVH.
            std::optional<uint32_t> size =
                get_constant_value(parameter_type.size());
            internal_assert(size.has_value()) << parameter_type;
            for (uint32_t i = 0, e = *size; i < e; ++i) {
                children.push_back(
                    VariantData::ChildAccess{.child = parameter, .index = i});
            }
            continue;
        }
    }

    return VariantData{.payload = std::move(payload),
                       .children = std::move(children)};
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

// Substitute a yielded element into a lambda over the set's elements. A
// multi-argument lambda comes from coiterating several sets, so the element is
// a tuple to be destructured componentwise.
ir::Expr apply_lambda(const ir::Expr &func, const ir::Expr &value) {
    const ir::Lambda *lambda = func.as<ir::Lambda>();
    internal_assert(lambda) << "Not a lambda: " << func;
    if (lambda->args.size() == 1) {
        internal_assert(ir::equals(lambda->args[0].type, value.type()))
            << lambda->args[0].type << " versus " << value.type();
        return replace(lambda->args[0].name, value, lambda->value);
    }
    internal_assert(value.type().is<ir::Tuple_t>()) << value;
    std::map<std::string, ir::Expr> repls;
    for (size_t i = 0; i < lambda->args.size(); i++) {
        // TODO: this needs to simplify or have CSE for it to be efficient!
        ir::Expr component = opt::Simplify::simplify(
            ir::Extract::make(value, static_cast<int>(i)));
        internal_assert(ir::equals(component.type(), lambda->args[i].type));
        repls[lambda->args[i].name] = std::move(component);
    }
    return replace(repls, lambda->value);
}

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
    // The list of volumes for the current match arms. Each arm contributes
    // one entry: a single undefined expression when the arm has no volume,
    // one volume when the bound is enclosing, and one per child when the
    // bound is childwise.
    struct VolumeMetadata {
        std::vector<ir::Expr> volumes;
        ir::Annotation::Volume::BoundType type;
    };
    mutable std::vector<VolumeMetadata> volume_metadata;
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
        internal_assert(var)
            << "[unimplemented] Match location of type: " << node->loc.type();
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

            update_volumes(bvh_node, tree, node->loc.type());
            std::variant<std::monostate, Interval,
                         std::map<std::string, Interval>>
                interval;
            std::map<std::string, ir::Expr> aggregation;
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
                }
            }
            intervals.emplace_back(std::move(interval));
            aggregations.emplace_back(std::move(aggregation));

            ir::Stmt stmt = mutate(node->arms[i].second);
            volume_metadata.pop_back();
            intervals.pop_back();
            aggregations.pop_back();
            new_arms[i] = {node->arms[i].first, std::move(stmt)};
        }
        locs.pop_back();

        return ir::Match::make(node->loc, std::move(new_arms));
    }

  protected:
    // Creates a mapping from lambda argument in a given geometric operation to
    // its respective bounding volume.
    VolumeMap make_volume_map(const std::vector<ir::TypedVar> &args) const {
        VolumeMap volume_map;
        const size_t n = volume_metadata.size(), m = args.size();
        internal_assert(n == m)
            << "volume map with incorrect number of arguments: " << m << " vs "
            << n;
        for (size_t i = 0; i < n; i++) {
            auto &[children, type] = volume_metadata[i];
            // Even if a volume is undefined, it needs to be added so
            // predicate analysis knows it's non-varying.
            internal_assert(!children.empty());
            const std::string &argument_name = args[i].name;
            volume_map[argument_name] = children.back();
            if (type == ir::Annotation::Volume::BoundType::Childwise) {
                // TODO(cgyurgyik): the issue here is we actually want to reuse
                // volumes when doing the cross product of two trees's
                // variants. A simple solution would just store a map from
                // an Unwrap -> Volume, e.g.,
                // `(triangles1 as AABBNode).children[0]` -> `AABB`
                //
                // ...but what about `triangles2` in the code below:
                // from (((triangles1 as AABBNode).children[0], triangles2),
                //                                              ^^^^^^^^^^
                // what is the AABB stored for this...?
                // code reference:
                // https://gist.github.com/cgyurgyik/eefcdfac0866ec7d5b6a7d7e694f6937
                children.pop_back();
            }
        }
        return volume_map;
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

  private:
    void update_volumes(const ir::BVH_t::Variant &variant,
                        const ir::Expr &unwrap, const ir::Type &tree_type) {
        const ir::Annotation::Volume *volume = variant.find_volume();
        if (volume == nullptr) {
            volume_metadata.push_back(VolumeMetadata{
                {ir::Expr()},
                ir::Annotation::Volume::BoundType::Enclosing,
            });
            return;
        }
        const size_t n_args = volume->initializers.size();
        switch (volume->bound_type()) {
        case ir::Annotation::Volume::BoundType::Enclosing: {
            std::vector<ir::Expr> args;
            args.reserve(n_args);
            for (size_t j = 0; j < n_args; ++j) {
                const std::string &name = volume->initializers[j];
                args.push_back(ir::Access::make(name, unwrap));
            }
            volume_metadata.push_back(VolumeMetadata{
                .volumes = {ir::Build::make(volume->struct_type, args)},
                .type = volume->bound_type(),
            });
            return;
        }
        case ir::Annotation::Volume::BoundType::Childwise: {
            int32_t child_count = get_child_reference_count(variant, tree_type);
            std::vector<ir::Expr> child_volumes;
            for (int i = 0; i < child_count; ++i) {
                std::vector<ir::Expr> args;
                args.reserve(n_args);
                const auto *struct_t = volume->struct_type.as<ir::Struct_t>();
                internal_assert(struct_t);
                for (size_t j = 0; j < n_args; ++j) {
                    const std::string &name = volume->initializers[j];
                    ir::Expr arg = ir::Access::make(name, unwrap);
                    if (!ir::equals(arg.type(), struct_t->fields[j].type)) {
                        arg = ir::Extract::make(arg, ir::Expr(i));
                    }
                    args.push_back(arg);
                }
                child_volumes.emplace_back(
                    ir::Build::make(volume->struct_type, args));
            }
            // Reverse it, since we'll be popping from the back.
            std::reverse(child_volumes.begin(), child_volumes.end());
            volume_metadata.push_back(VolumeMetadata{
                .volumes = std::move(child_volumes),
                .type = volume->bound_type(),
            });
            return;
        }
        }
    }
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
            internal_assert(!volume_metadata.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
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
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volume_metadata.size() == lambda->args.size());

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

// Which end of the metric an extremum operator seeks.
enum class Extremum { Min, Max };

const char *name_of(Extremum dir, bool arg) {
    if (arg) {
        return dir == Extremum::Min ? "argmin" : "argmax";
    }
    return dir == Extremum::Min ? "minimum" : "maximum";
}

// The identity of the extremum: the worst possible value, so that any element
// improves on it. Note that for signed integers this is -INT_MAX rather than
// INT_MIN, matching how Extrema::inf already approximates infinity there.
ir::Expr extremum_identity(const ir::Type &t, Extremum dir) {
    ir::Expr inf = ir::Extrema::make(t, ir::Extrema::inf);
    if (dir == Extremum::Min) {
        return inf;
    }
    return t.is_uint() ? make_zero(t) : -inf;
}

// `value` beats `best`.
ir::Expr improves_on(Extremum dir, ir::Expr value, ir::Expr best) {
    return dir == Extremum::Min ? (std::move(value) < std::move(best))
                                : (std::move(best) < std::move(value));
}

// What predicate analysis can say about the best value reachable in a
// subtree: for a minimum that is the metric's upper bound over the subtree,
// and for a maximum its lower bound.
ir::Expr reachable_bound(Extremum dir, const Interval &bounds) {
    return dir == Extremum::Min ? bounds.max : bounds.min;
}

// The bound that decides whether a subtree is worth visiting: a minimum has
// to descend when the subtree's smallest reachable value beats the running
// best, which is the metric's lower bound over the subtree.
ir::Expr promising_bound(Extremum dir, const Interval &bounds) {
    return dir == Extremum::Min ? bounds.min : bounds.max;
}

// The accumulator's own interval. A running minimum is an upper bound on any
// value that can still be accepted, and vice versa.
Interval accumulator_interval(Extremum dir, const ir::Expr &acc) {
    return dir == Extremum::Min ? Interval{ir::Expr(), acc}
                                : Interval{acc, ir::Expr()};
}

// Fuse "this element improves on the running best" into an existing filter if
// there is one, so that the value-based pruning condition participates in
// predicate analysis. The bool reports whether an existing filter was fused
// into; when false the caller must still update the accumulator from
// recursive calls itself.
std::pair<ir::Expr, bool> try_fuse_filter(Extremum dir,
                                          const ir::Lambda *metric,
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
                    << "Mismatched types in metric-filter fusion: "
                    << metric->args[i].type
                    << " != " << predicate->args[i].type;
            }

            // Check for convenient case of same naming / types.
            ir::Expr value =
                repls.empty() ? metric->value : replace(repls, metric->value);
            ir::Expr new_cond =
                predicate->value && improves_on(dir, std::move(value), best);
            // Construct fused filter.
            ir::Expr new_lambda =
                ir::Lambda::make(predicate->args, std::move(new_cond));
            return {filter(std::move(new_lambda), as_set->b), true};
        }
    }

    // Not a nested filter, so just wrap in a filter and return
    ir::Expr new_cond = improves_on(dir, metric->value, std::move(best));
    ir::Expr new_lambda = ir::Lambda::make(metric->args, std::move(new_cond));
    return {filter(std::move(new_lambda), std::move(maybe_filter)), false};
}

// Algorithm 3: argmin and argmax. These mirror Algorithm 2 but also track the
// element achieving the extremum, so the accumulator is a (metric, element)
// pair updated with an argmin/argmax accumulate.
ir::Stmt build_arg_extremum(Extremum dir, ir::Expr metric, ir::Expr inner,
                            const ir::TypeMap &tree_types,
                            const IntervalMap &intervals,
                            ir::Type expect_type) {
    struct RewriteArgExtremum : public Rewriter {
        Extremum dir;
        ir::Expr metric;
        ir::WriteLoc loc;
        ir::Type tuple_t;
        const IntervalMap &intervals;
        const bool update_from_yfs;

        RewriteArgExtremum(Extremum dir, ir::Expr met, ir::WriteLoc l,
                           ir::Type t, const IntervalMap &intervals,
                           const bool update_from_yfs)
            : dir(dir), metric(std::move(met)), loc(std::move(l)),
              tuple_t(std::move(t)), intervals(intervals),
              update_from_yfs(update_from_yfs) {}

        using ir::Mutator::visit;

        // yield x => upd a arg(a, (M(x), x))
        ir::Stmt visit(const ir::Yield *node) override {
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);
            internal_assert(
                ir::equals(lambda->args[0].type, node->value.type()));
            ir::Expr value =
                replace(lambda->args[0].name, node->value, lambda->value);

            std::vector<ir::Expr> values = {std::move(value), node->value};
            ir::Expr update = ir::Build::make(tuple_t, std::move(values));
            return ir::Accumulate::make(loc,
                                        dir == Extremum::Min
                                            ? ir::Accumulate::Argmin
                                            : ir::Accumulate::Argmax,
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
            // internal_assert(volume_metadata.size() == lambda->args.size());
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
    internal_assert(lambda)
        << "Metric of " << name_of(dir, true) << " is not a lambda: " << metric;
    ir::Type metric_t = lambda->value.type();

    ir::Type ret_type = inner.type().element_of();
    ir::Type tuple_t = ir::Tuple_t::make({metric_t, ret_type});

    static size_t counter = 0;
    std::string name = "_best" + std::to_string(counter++);
    ir::WriteLoc loc(name, tuple_t);

    // WrapWithAccumulator(a, (worst, null))
    ir::Expr identity = extremum_identity(metric_t, dir);
    static const std::vector<ir::Expr> empty_list = {};
    ir::Expr empty = ir::Build::make(ret_type, empty_list);
    std::vector<ir::Expr> values = {identity, std::move(empty)};
    ir::Expr init = ir::Build::make(tuple_t, std::move(values));

    // TODO(ajr): is stack memory ok here? it's not an array.
    ir::Stmt header =
        ir::Allocate::make(loc, std::move(init), ir::Allocate::Memory::Stack);

    ir::Expr ret_var = ir::Var::make(tuple_t, std::move(name));
    ir::Expr best_metric = ir::Extract::make(ret_var, 0);
    ir::Expr best_ref = ir::Extract::make(ret_var, 1);
    // TODO: should this be a Return?
    ir::Stmt footer;
    if (!ir::equals(ret_type, expect_type)) {
        // If nothing improved on the identity, the set was empty.
        ir::Expr result = ir::Select::make(
            best_metric != identity, ir::Build::make(expect_type, {best_ref}),
            ir::Build::make(expect_type));
        footer = ir::Yield::make(std::move(result));
    } else {
        footer = ir::Yield::make(best_ref);
    }

    IntervalMap local_intervals = intervals;
    local_intervals[best_metric] = accumulator_interval(dir, best_metric);

    // Try to build fused filter inside.
    auto [fused_filter, fused] =
        try_fuse_filter(dir, lambda, best_metric, inner);
    ir::Stmt body = build_traversal(fused_filter, tree_types, local_intervals);

    body = RewriteArgExtremum(dir, std::move(metric), std::move(loc),
                              std::move(tuple_t), intervals, !fused)
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

// Algorithm 2 (LowerMin), and its mirror for maxima. These reductions are both
// associative and idempotent, so the traversal prunes in two complementary
// ways: value-based pruning, where a subtree whose metric cannot beat the
// running extremum is skipped, and inclusion, where a subtree that stores its
// own extremum updates the accumulator without being visited.
//
// The value-based half is expressed by fusing "improves on best" into the
// inner filter (try_fuse_filter) and letting predicate analysis bound it; the
// accumulator's interval is recorded so that analysis knows a candidate must
// beat the running value.
ir::Stmt build_extremum(Extremum dir, ir::Expr metric, ir::Expr inner,
                        const ir::TypeMap &tree_types,
                        const IntervalMap &intervals, ir::Type expect_type) {
    struct RewriteExtremum : public Rewriter {
        Extremum dir;
        ir::Expr metric;
        ir::WriteLoc loc;
        const IntervalMap &intervals;
        // The augmentation storing this extremum over a subtree, if any.
        std::optional<std::string> key;
        // True when try_fuse_filter did not fuse, so recursive calls are not
        // already guarded by the value-based condition and this rewrite must
        // tighten the accumulator itself.
        const bool update_from_yfs;

        RewriteExtremum(Extremum dir, ir::Expr met, ir::WriteLoc l,
                        const IntervalMap &intervals,
                        std::optional<std::string> key,
                        const bool update_from_yfs)
            : dir(dir), metric(std::move(met)), loc(std::move(l)),
              intervals(intervals), key(std::move(key)),
              update_from_yfs(update_from_yfs) {}

        ir::Accumulate::OpType accumulate_op() const {
            return dir == Extremum::Min ? ir::Accumulate::Min
                                        : ir::Accumulate::Max;
        }

        using ir::Mutator::visit;

        // yield x => upd a minb(a, M(x))
        ir::Stmt visit(const ir::Yield *node) override {
            internal_assert(!volume_metadata.empty());
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volume_metadata.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);
            internal_assert(
                ir::equals(lambda->args[0].type, node->value.type()));
            ir::Expr value =
                replace(lambda->args[0].name, node->value, lambda->value);
            return ir::Accumulate::make(loc, accumulate_op(), std::move(value));
        }

        // iter xs => upd a minb(a, min(M, xs))
        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        // scan tr => if tr has min(M, tr) then upd a minb(a, min(M, tr))
        //            else if maybe(min(M(tr)) < a): upd a minb(a, max(M, tr));
        //                 from tr
        //
        // A whole subtree is included here, so a node that stores this
        // extremum over it settles the contribution exactly and does not need
        // to be visited at all. That is the inclusion case of Section 2.
        ir::Stmt visit(const ir::Scan *node) override {
            const bool stored =
                key.has_value() &&
                std::all_of(
                    aggregations.begin(), aggregations.end(),
                    [&](const auto &agg) { return agg.contains(*key); });
            if (stored) {
                std::vector<ir::Stmt> stmts;
                for (const auto &agg : aggregations) {
                    stmts.push_back(ir::Accumulate::make(loc, accumulate_op(),
                                                         agg.at(*key)));
                }
                return stmts.size() == 1 ? stmts.front()
                                         : ir::Sequence::make(std::move(stmts));
            }

            // Nothing stored, so the subtree has to be visited. Value-based
            // pruning still applies: skip it when it cannot beat the running
            // best, and tighten the accumulator with what it could reach.
            Interval bounds = subtree_bounds();
            std::vector<ir::Stmt> stmts;
            if (ir::Expr reachable = reachable_bound(dir, bounds);
                reachable.defined()) {
                stmts.push_back(ir::Accumulate::make(loc, accumulate_op(),
                                                     std::move(reachable)));
            }
            stmts.push_back(ir::YieldFrom::make(node->value));
            ir::Stmt body = stmts.size() == 1
                                ? std::move(stmts.front())
                                : ir::Sequence::make(std::move(stmts));
            if (ir::Expr promising = promising_bound(dir, bounds);
                promising.defined()) {
                body = ir::IfElse::make(
                    improves_on(dir, std::move(promising), loc.to_expr()),
                    std::move(body));
            }
            return body;
        }

        // The bounds of the metric over the subtree currently being matched.
        Interval subtree_bounds() const {
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volume_metadata.size() == lambda->args.size());
            VolumeMap vols = make_volume_map(lambda->args);
            IntervalMap ints = make_interval_map(lambda->args, intervals);
            return predicate_analysis(lambda->value, vols, ints);
        }

        // from tr => upd a minb(a, max(M, tr)); from tr
        ir::Stmt visit(const ir::YieldFrom *node) override {
            if (!update_from_yfs) {
                return node;
            }
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volume_metadata.size() == lambda->args.size());
            // TODO: handle tuple data, e.g. from product()
            internal_assert(lambda->args.size() == 1);

            ir::Expr bound = reachable_bound(dir, subtree_bounds());
            if (!bound.defined()) {
                // Nothing can be said about this subtree; just recurse.
                return node;
            }

            // The best value in this subtree is no better than the metric's
            // bound over the subtree's volume, so the accumulator can be
            // tightened before recursing.
            ir::Stmt do_update =
                ir::Accumulate::make(loc, accumulate_op(), std::move(bound));
            return ir::Sequence::make({std::move(do_update), node});
        }
    };

    const ir::Lambda *lambda = metric.as<ir::Lambda>();
    internal_assert(lambda) << "Metric of " << name_of(dir, false)
                            << " is not a lambda: " << metric;
    ir::Type metric_t = lambda->value.type();

    static size_t counter = 0;
    std::string name = "_best" + std::to_string(counter++);
    ir::WriteLoc loc(name, metric_t);

    // WrapWithAccumulator(a, worst)
    ir::Expr identity = extremum_identity(metric_t, dir);
    // TODO(ajr): is stack memory ok here? it's not an array.
    ir::Stmt header =
        ir::Allocate::make(loc, identity, ir::Allocate::Memory::Stack);

    ir::Expr ret_var = ir::Var::make(metric_t, std::move(name));
    // An extremum over a set that can be empty is optional: the accumulator is
    // still at its identity exactly when nothing was visited.
    ir::Stmt footer;
    if (!ir::equals(metric_t, expect_type)) {
        ir::Expr result = ir::Select::make(
            ret_var != identity, ir::Build::make(expect_type, {ret_var}),
            ir::Build::make(expect_type));
        footer = ir::Yield::make(std::move(result));
    } else {
        footer = ir::Yield::make(ret_var);
    }

    IntervalMap local_intervals = intervals;
    local_intervals[ret_var] = accumulator_interval(dir, ret_var);

    // An included subtree can be folded in wholesale when the node stores this
    // extremum over the field the metric reads.
    std::optional<std::string> key;
    if (const ir::Access *access = lambda->value.as<ir::Access>()) {
        if (const ir::Var *var = access->value.as<ir::Var>();
            var != nullptr && var->name == lambda->args[0].name) {
            key = aggregate_key(dir == Extremum::Min
                                    ? ir::Annotation::Aggregate::min
                                    : ir::Annotation::Aggregate::max,
                                {access->field});
        }
    }

    // Fusing the value-based condition into a filter is what lets predicate
    // analysis prune on it, and it also puts the node-level bound in front of
    // every arm, including the leaves. Synthesizing that filter consumes the
    // `scan` standing for a wholly included subtree, so give it up only when
    // this metric could actually read a stored extremum off a node -- that
    // is, when the metric names a field an augmentation might cover.
    auto [fused_filter, fused] = try_fuse_filter(dir, lambda, ret_var, inner);
    const bool keep_scan = !fused && key.has_value();
    ir::Stmt body = build_traversal(keep_scan ? inner : fused_filter,
                                    tree_types, local_intervals);

    body = RewriteExtremum(dir, std::move(metric), std::move(loc), intervals,
                           std::move(key), !fused)
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

// Algorithm 4: any and all. Both are idempotent reductions over the boolean
// lattice, so a subtree can be skipped as soon as the answer it could
// contribute is settled: for `any` once the predicate is proven always true
// (the result is true) or never true (the subtree cannot help), and dually for
// `all`.
ir::Stmt build_quantifier(bool is_any, ir::Expr predicate, ir::Expr inner,
                          const ir::TypeMap &tree_types,
                          const IntervalMap &intervals) {
    struct RewriteQuantifier : public Rewriter {
        bool is_any;
        ir::Expr predicate;
        ir::WriteLoc loc;
        const IntervalMap &intervals;

        RewriteQuantifier(bool is_any, ir::Expr p, ir::WriteLoc l,
                          const IntervalMap &intervals)
            : is_any(is_any), predicate(std::move(p)), loc(std::move(l)),
              intervals(intervals) {}

        using ir::Mutator::visit;

        // `a` is settled once it reaches the absorbing element of the lattice.
        ir::Expr still_undecided() const {
            ir::Expr acc = loc.to_expr();
            return is_any ? ~acc : acc;
        }

        // yield x => upd a (a | P(x))
        ir::Stmt visit(const ir::Yield *node) override {
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            ir::Expr p = apply_lambda(predicate, node->value);
            ir::Expr acc = loc.to_expr();
            ir::Expr combined = is_any ? (acc || p) : (acc && p);
            return ir::Store::make(loc, std::move(combined));
        }

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        // The bounds of the predicate over the subtree currently being matched.
        Interval subtree_bounds() const {
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volume_metadata.size() == lambda->args.size());
            VolumeMap vols = make_volume_map(lambda->args);
            IntervalMap ints = make_interval_map(lambda->args, intervals);
            return predicate_analysis(lambda->value, vols, ints);
        }

        // scan tr => if always(P, tr): a = <settled>
        //            elif <undecided> && maybe(P, tr): from tr
        ir::Stmt visit(const ir::Scan *node) override {
            Interval bounds = subtree_bounds();
            ir::Stmt recurse = ir::YieldFrom::make(node->value);

            // For `any`, a subtree the predicate can never hold on contributes
            // nothing; for `all` it settles the answer to false.
            ir::Stmt otherwise;
            if (!is_any) {
                otherwise = ir::Store::make(loc, ir::BoolImm::make(false));
            }

            if (bounds.max.defined() && !is_const_one(bounds.max)) {
                recurse =
                    ir::IfElse::make(still_undecided() && bounds.max,
                                     std::move(recurse), std::move(otherwise));
            } else if (otherwise.defined()) {
                recurse =
                    ir::IfElse::make(still_undecided(), std::move(recurse));
            } else {
                recurse =
                    ir::IfElse::make(still_undecided(), std::move(recurse));
            }

            if (bounds.min.defined() && !is_const_zero(bounds.min)) {
                // Proven on the whole subtree: `any` is decided, `all` learns
                // nothing new and can skip it.
                ir::Stmt settled =
                    is_any ? ir::Store::make(loc, ir::BoolImm::make(true))
                           : ir::Stmt();
                recurse =
                    settled.defined()
                        ? ir::IfElse::make(bounds.min, settled,
                                           std::move(recurse))
                        : ir::IfElse::make(~bounds.min, std::move(recurse));
            }
            return recurse;
        }

        // from tr => if <undecided> && maybe(P, tr): from tr
        ir::Stmt visit(const ir::YieldFrom *node) override {
            Interval bounds = subtree_bounds();
            ir::Expr cond = still_undecided();
            if (bounds.max.defined() && !is_const_one(bounds.max)) {
                cond = cond && bounds.max;
            }
            return ir::IfElse::make(std::move(cond), node);
        }
    };

    const ir::Type bool_t = ir::Bool_t::make();

    static size_t counter = 0;
    std::string name = "_holds" + std::to_string(counter++);
    ir::WriteLoc loc(name, bool_t);

    // WrapWithAccumulator(a, false) for any, (a, true) for all.
    ir::Stmt header = ir::Allocate::make(loc, ir::BoolImm::make(!is_any),
                                         ir::Allocate::Memory::Stack);
    ir::Expr ret_var = ir::Var::make(bool_t, std::move(name));
    ir::Stmt footer = ir::Yield::make(ret_var);

    ir::Stmt body = build_traversal(inner, tree_types, intervals);
    body = RewriteQuantifier(is_any, std::move(predicate), std::move(loc),
                             intervals)
               .mutate(body);

    return ir::Sequence::make(
        {std::move(header), std::move(body), std::move(footer)});
}

// Algorithm 1, lines 9-12. A map changes only what a traversal yields, so it
// rewrites into the yield, iter and scan constructs without affecting how
// recursion proceeds.
ir::Stmt build_map(ir::Stmt body, ir::Expr func) {
    struct RewriteMap : public Rewriter {
        ir::Expr func;

        RewriteMap(ir::Expr f) : func(std::move(f)) {}

        using ir::Mutator::visit;

        // yield x => yield F(x)
        ir::Stmt visit(const ir::Yield *node) override {
            return ir::Yield::make(apply_lambda(func, node->value));
        }

        // iter xs => iter map(F, xs)
        ir::Stmt visit(const ir::Iterate *node) override {
            return ir::Iterate::make(map(func, node->value));
        }

        // scan tr => scan F(tr)
        ir::Stmt visit(const ir::Scan *node) override {
            internal_assert(!node->func.defined())
                << "TODO: compose nested maps on a scan: " << ir::Stmt(node);
            return ir::Scan::make(node->op, node->loc, func, node->value);
        }

        // A recursive call already evaluates the mapped query.
        ir::Stmt visit(const ir::YieldFrom *node) override { return node; }
    };

    return RewriteMap(std::move(func)).mutate(body);
}

// The augmentation, if any, that stores this reduction's value over a
// subtree. A reduction is recognised by its combiner together with the map
// feeding it: summing a constant 1 is a `count()`, summing a field `f` is a
// `sum(f)`, and so on.
std::optional<std::string> aggregate_key_for(const ir::Expr &combiner,
                                             const ir::Expr &func) {
    const ir::Lambda *lambda = combiner.as<ir::Lambda>();
    if (lambda == nullptr || lambda->args.size() != 2) {
        return {};
    }
    const ir::BinOp *binop = lambda->value.as<ir::BinOp>();
    if (binop == nullptr) {
        return {};
    }
    // The combiner must be exactly `|a, b| a <op> b`.
    const ir::Var *a = binop->a.as<ir::Var>();
    const ir::Var *b = binop->b.as<ir::Var>();
    if (a == nullptr || b == nullptr || a->name != lambda->args[0].name ||
        b->name != lambda->args[1].name) {
        return {};
    }

    ir::Annotation::Aggregate::OpType op;
    switch (binop->op) {
    case ir::BinOp::Add:
        op = ir::Annotation::Aggregate::sum;
        break;
    case ir::BinOp::Mul:
        op = ir::Annotation::Aggregate::prod;
        break;
    default:
        return {};
    }

    if (!func.defined()) {
        // Reducing the elements themselves.
        return aggregate_key(op, {});
    }
    const ir::Lambda *map_fn = func.as<ir::Lambda>();
    if (map_fn == nullptr || map_fn->args.size() != 1) {
        return {};
    }
    // Summing a constant 1 over the subtree counts it.
    if (op == ir::Annotation::Aggregate::sum && is_const_one(map_fn->value)) {
        return aggregate_key(ir::Annotation::Aggregate::count, {});
    }
    // Reducing a single field.
    if (const ir::Access *access = map_fn->value.as<ir::Access>()) {
        if (const ir::Var *var = access->value.as<ir::Var>();
            var != nullptr && var->name == map_fn->args[0].name) {
            return aggregate_key(op, {access->field});
        }
    }
    return {};
}

// `upd a (a (+) v)`. Emits an Accumulate for the combiners the backends know
// how to update in place, and otherwise inlines the combiner into a store.
ir::Stmt make_update(const ir::WriteLoc &loc, const ir::Expr &combiner,
                     ir::Expr value) {
    const ir::Lambda *lambda = combiner.as<ir::Lambda>();
    internal_assert(lambda && lambda->args.size() == 2)
        << "Combiner is not a binary lambda: " << combiner;

    if (const ir::BinOp *binop = lambda->value.as<ir::BinOp>()) {
        const ir::Var *a = binop->a.as<ir::Var>();
        const ir::Var *b = binop->b.as<ir::Var>();
        if (a != nullptr && b != nullptr && a->name == lambda->args[0].name &&
            b->name == lambda->args[1].name) {
            switch (binop->op) {
            case ir::BinOp::Add:
                return ir::Accumulate::make(loc, ir::Accumulate::Add,
                                            std::move(value));
            case ir::BinOp::Mul:
                return ir::Accumulate::make(loc, ir::Accumulate::Mul,
                                            std::move(value));
            default:
                break;
            }
        }
    }

    std::map<std::string, ir::Expr> repls;
    repls[lambda->args[0].name] = loc.to_expr();
    repls[lambda->args[1].name] = std::move(value);
    return ir::Store::make(loc, replace(repls, lambda->value));
}

// Algorithm 1, lines 13-22. An associative reduction is computed into an
// accumulator: leaves update it with their own value, and a subtree that
// stores a precomputed aggregate updates it wholesale instead of being
// traversed.
ir::Stmt build_reduce(ir::Expr identity, ir::Expr combiner, ir::Expr inner,
                      const ir::TypeMap &tree_types,
                      const IntervalMap &intervals) {
    struct RewriteReduce : public Rewriter {
        ir::Expr combiner;
        ir::WriteLoc loc;
        // The augmentation this reduction can read off a node, if any.
        std::optional<std::string> key;
        // The reduction, and the map feeding it, for a fallback scan.
        std::optional<ir::AggOp::OpType> scan_op;
        ir::Expr func;

        RewriteReduce(ir::Expr c, ir::WriteLoc l,
                      std::optional<std::string> key,
                      std::optional<ir::AggOp::OpType> scan_op, ir::Expr func)
            : combiner(std::move(c)), loc(std::move(l)), key(std::move(key)),
              scan_op(std::move(scan_op)), func(std::move(func)) {}

        using ir::Mutator::visit;

        // yield x => upd a (a (+) x)
        ir::Stmt visit(const ir::Yield *node) override {
            return make_update(loc, combiner, node->value);
        }

        // iter xs => upd a (a (+) reduce(xs))
        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        // scan tr => if tr has C(tr) then upd a (a (+) tr.C) else scan<C> tr
        ir::Stmt visit(const ir::Scan *node) override {
            if (key.has_value()) {
                const bool all_stored = std::all_of(
                    aggregations.begin(), aggregations.end(),
                    [&](const auto &agg) { return agg.contains(*key); });
                if (all_stored) {
                    // Coiterating several trees reduces over their product.
                    ir::Expr total = aggregations.front().at(*key);
                    for (size_t i = 1; i < aggregations.size(); i++) {
                        total = total * aggregations[i].at(*key);
                    }
                    return make_update(loc, combiner, std::move(total));
                }
            }
            internal_assert(scan_op.has_value())
                << "Cannot scan a subtree for a reduction the runtime cannot "
                   "combine: "
                << ir::Stmt(node);
            return ir::Scan::make(
                scan_op, loc, func.defined() ? func : node->func, node->value);
        }

        // A recursive call updates the same accumulator.
        ir::Stmt visit(const ir::YieldFrom *node) override { return node; }
    };

    const ir::Type acc_t = identity.type();

    static size_t counter = 0;
    std::string name = "_acc" + std::to_string(counter++);
    ir::WriteLoc loc(name, acc_t);

    // WrapWithAccumulator(a, id)
    ir::Stmt header = ir::Allocate::make(loc, std::move(identity),
                                         ir::Allocate::Memory::Stack);
    ir::Expr ret_var = ir::Var::make(acc_t, std::move(name));
    ir::Stmt footer = ir::Yield::make(ret_var);

    // Peel an immediately enclosed map so the reduction can be matched against
    // the augmentations a node stores; the map itself is still lowered below.
    ir::Expr func;
    if (const ir::SetOp *as_map_op = as_map(inner)) {
        func = as_map_op->a;
    }
    std::optional<std::string> key = aggregate_key_for(combiner, func);

    std::optional<ir::AggOp::OpType> scan_op;
    if (const ir::Lambda *lambda = combiner.as<ir::Lambda>()) {
        if (const ir::BinOp *binop = lambda->value.as<ir::BinOp>()) {
            if (binop->op == ir::BinOp::Add) {
                scan_op = ir::AggOp::sum;
            } else if (binop->op == ir::BinOp::Mul) {
                scan_op = ir::AggOp::prod;
            }
        }
    }

    ir::Stmt body = build_traversal(inner, tree_types, intervals);
    body = RewriteReduce(std::move(combiner), std::move(loc), std::move(key),
                         std::move(scan_op), std::move(func))
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
                    return ir::Scan::make(ir::Expr(),
                                          make_tuple(std::move(vals)));
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
                    return ir::Scan::make(ir::Expr(),
                                          make_tuple(std::move(vals)));
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
                    return ir::Scan::make(ir::Expr(),
                                          make_tuple(std::move(vals)));
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
                    return ir::Scan::make(ir::Expr(),
                                          make_tuple(std::move(vals)));
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
                    return ir::Scan::make(ir::Expr(),
                                          make_tuple(std::move(pairs)));
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

        return build_base_scan(as_var->name, bvh);
    }

    if (const ir::AggOp *as_agg = expr.as<ir::AggOp>()) {
        if (as_agg->op != ir::AggOp::reduce) {
            // count, sum and prod are sugar for a map followed by a reduce.
            return build_traversal(expand_aggregate(as_agg), tree_types,
                                   intervals);
        }
        return build_reduce(as_agg->identity, as_agg->combiner, as_agg->a,
                            tree_types, intervals);
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
    case ir::SetOp::map: {
        ir::Stmt body = build_traversal(as_set->b, tree_types, intervals);
        return build_map(body, as_set->a);
    }
    case ir::SetOp::argmin:
    case ir::SetOp::argmax: {
        // These are a bit more complicated, because of filter fusion.
        const Extremum dir =
            as_set->op == ir::SetOp::argmin ? Extremum::Min : Extremum::Max;
        return build_arg_extremum(dir, as_set->a, as_set->b, tree_types,
                                  intervals, expr.type());
    }
    case ir::SetOp::minimum:
    case ir::SetOp::maximum: {
        const Extremum dir =
            as_set->op == ir::SetOp::minimum ? Extremum::Min : Extremum::Max;
        return build_extremum(dir, as_set->a, as_set->b, tree_types, intervals,
                              expr.type());
    }
    case ir::SetOp::any:
    case ir::SetOp::all: {
        return build_quantifier(as_set->op == ir::SetOp::any, as_set->a,
                                as_set->b, tree_types, intervals);
    }
    case ir::SetOp::product: {
        ir::Stmt a_body = build_traversal(as_set->a, tree_types, intervals);
        ir::Stmt b_body = build_traversal(as_set->b, tree_types, intervals);
        return build_product(a_body, b_body, expr.type().element_of());
    }
    default: {
        internal_error << "[unimplemented] build_traversal(" << expr << " : "
                       << expr.type() << ")";
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
    ir::Expr visit(const ir::AggOp *op) override { return build_func(op); }
};

} // namespace

ir::Stmt build_base_scan(const std::string &name, const ir::BVH_t *bvh_t) {
    ir::Expr bvh_expr = ir::Var::make(bvh_t, name);

    const size_t n_nodes = bvh_t->variants.size();
    ir::Match::Arms arms(n_nodes);
    for (size_t i = 0; i < n_nodes; i++) {
        ir::Expr node = ir::Unwrap::make(i, bvh_expr);
        const auto [data, children] =
            analyze_node(bvh_t->variants[i], bvh_t->primitive, bvh_t);

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
                ir::Expr access = ir::Access::make(c.child.name, node);
                // A wide BVH stores its children in an array; pick the lane.
                if (c.index.has_value()) {
                    access = ir::Extract::make(
                        access, ir::Expr(static_cast<int32_t>(*c.index)));
                }
                cs.push_back(std::move(access));
            }
            stmts.back() = ir::Scan::make(ir::Expr(), make_tuple(cs));
        }

        arms[i].first = bvh_t->variants[i];
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
    const ir::TypeMap &tree_types =
        program.schedules[ir::Target::Host].tree_types;

    for (const auto &[name, type] : tree_types) {
        if (const auto *bvh_t = type.as<ir::BVH_t>()) {
            for (const ir::BVH_t::Variant &variant : bvh_t->variants) {
                if (program.types.contains(variant.name())) {
                    continue;
                }
                program.types.insert(
                    {variant.name(),
                     ir::Struct_t::make(variant.name(), variant.fields())});
            }
        }
    }

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
