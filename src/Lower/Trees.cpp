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
                         const std::map<std::string, ir::Expr> &extents,
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
                        << "The data of tree node " << node.name() << " is "
                        << d->name << ", of type " << param.type
                        << ", but the tree is declared over " << prim_t
                        << ". A node's data has to be the tree's own primitive "
                        << "type, or an array or vector of it -- check what "
                        << "the tree[[...]] declaration says it holds.";
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
        if (setop->op == ir::SetOp::map) {
            // iter map(F, xs) => foreach x in xs: yield F(x).
            //
            // A collection at a leaf that has had a function mapped over it --
            // an instance's triangles, placed where the instance puts them.
            // The function rides along to each yield rather than being applied
            // to the collection, which would mean materializing it.
            ir::Stmt body = lower_iterate(setop->b);
            return RewriteYields([&](const ir::Expr &x) {
                       return ir::Yield::make(apply_lambda(setop->a, x));
                   })
                .mutate(body);
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
        // Keyed by how the tree was reached rather than by its name, because a
        // tree held in a field -- `i.blas` -- has no name. The key only has to
        // scope child volumes to the match they belong to, and the expression
        // that reached the tree identifies that exactly.
        const std::string loc_key = ir::to_string(node->loc);
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
                ir::Expr volume = make_volume(bvh_node.get_volume());
                // A tree's stored bounds bound what it was built over. If a
                // geometric function has been mapped over its elements, the
                // bound that still holds is the mapped one -- an instance's
                // triangles are inside its boxes, but the same triangles
                // *where the instance puts them* are not.
                if (node->volume_map.defined()) {
                    volume = apply_lambda(node->volume_map, volume);
                }
                volumes.emplace_back(std::move(volume));
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
                child_volumes[loc_key] = built_child_volumes;
            }

            ir::Stmt stmt = mutate(node->arms[i].second);
            volumes.pop_back();
            intervals.pop_back();
            aggregations.pop_back();
            if (!built_child_volumes.empty()) {
                child_volumes.erase(loc_key);
            }
            new_arms[i] = {node->arms[i].first, std::move(stmt)};
        }
        locs.pop_back();

        return ir::Match::make(node->loc, std::move(new_arms),
                               node->volume_map);
    }

    VolumeMap make_volume_map(const std::vector<ir::TypedVar> &args) const {
        VolumeMap vols;
        const size_t n = volumes.size();
        // A flattened query has one parameter per level, and at a node above
        // the innermost level there are fewer volumes than parameters: at a
        // top-level node the instances are bounded but their triangles have
        // not been reached yet. Those parameters take the innermost volume so
        // far, which bounds them because the element that reaches them
        // promised as much -- `with extent` says everything reachable through
        // an element lies within its extent, and the node's volume bounds the
        // extent. That promise is the entire reason a query over two levels
        // can prune at the first one.
        internal_assert(n <= args.size())
            << "Making volume map with incorrect number of arguments: "
            << args.size() << " vs. " << n;
        internal_assert(n > 0 || args.empty())
            << "Making volume map with no volumes for " << args.size()
            << " arguments.";
        for (size_t i = 0; i < args.size(); i++) {
            // Even if a volume is undefined, needs to be added so
            // predicate analysis knows it's non-varying.
            vols[args[i].name] = volumes[std::min(i, n - 1)];
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
            // Keyed by how the tree was reached; see the Match visitor.
            const auto &iter =
                child_volumes.find(ir::to_string(unwrap->value));
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
        // Fewer levels entered than the query has parameters; see
        // `make_volume_map`. A scalar interval says nothing about a parameter
        // one level down -- the promise `with extent` makes is geometric --
        // so the extra parameters simply get no interval, which leaves them
        // unbounded and prunes nothing rather than pruning wrongly.
        internal_assert(n <= args.size())
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
        // Set while rewriting the body of a leaf's element loop, where the
        // predicate's bound over the *node's* volume has already been emitted
        // outside the loop. See `visit(Iterate)`.
        bool bound_hoisted = false;

        RewriteFilter(ir::Expr pred, const IntervalMap &intervals)
            : predicate(std::move(pred)), intervals(intervals) {}

        using ir::Mutator::visit;

        // The predicate's bound over the volume of the node currently being
        // matched. It is a property of the node, so it is the same for every
        // element a leaf holds.
        Interval node_bounds() const {
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            if (lambda == nullptr || volumes.empty() ||
                volumes.size() != lambda->args.size()) {
                return Interval{};
            }
            VolumeMap vols = make_volume_map(lambda->args);
            IntervalMap ints = make_interval_map(lambda->args, intervals);
            return predicate_analysis(lambda->value, vols, ints);
        }

        ir::Stmt visit(const ir::Yield *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() <= lambda->args.size());
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
            if (bounds.max.defined() && !bound_hoisted) {
                // Maybe true. Skipped when this yield is the body of a leaf's
                // element loop, because `visit(Iterate)` has already emitted it
                // once outside the loop.
                body = ir::IfElse::make(std::move(bounds.max), std::move(body));
            }
            if (bounds.min.defined() && !is_const_zero(bounds.min)) {
                // Always true.
                body = ir::IfElse::make(std::move(bounds.min), node,
                                        std::move(body));
            }

            return body;
        }

        // A leaf's elements, and the one place the node-level bound belongs.
        //
        // The bound is a property of the node's *volume*, so it is the same for
        // every element the leaf holds -- and it used to be emitted inside the
        // loop, once per element. On a BVH that is the leaf's own box
        // re-intersected with the ray for each of the four-odd primitives in
        // it, to answer a question that cannot change between them.
        //
        // Hoisting is not merely code motion, because for a reduction the bound
        // contains the running accumulator: `distmin(r, leafbox) < best`, and
        // `best` tightens inside the loop. Evaluating it once at entry is still
        // exact, and in the safe direction -- `best` only ever decreases, so
        // the condition is at its weakest here and a leaf is never skipped that
        // should have been entered. What is given up is abandoning the rest of
        // a leaf part-way through, which for four primitives is not worth one
        // box intersection each.
        ir::Stmt visit(const ir::Iterate *node) override {
            Interval bounds = node_bounds();
            const bool hoist = bounds.max.defined();

            const bool outer = bound_hoisted;
            bound_hoisted = hoist;
            ir::Stmt loop =
                mutate(lower_iterate(node->value)); // a concrete loop.
            bound_hoisted = outer;

            if (hoist) {
                loop = ir::IfElse::make(std::move(bounds.max), std::move(loop));
            }
            return loop;
        }

        ir::Stmt visit(const ir::Scan *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() <= lambda->args.size());

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
                            const std::map<std::string, ir::Expr> &extents,
                            const IntervalMap &intervals,
                            ir::Type expect_type) {
    struct RewriteArgExtremum : public Rewriter {
        Extremum dir;
        ir::Expr metric;
        ir::WriteLoc loc;
        ir::Type tuple_t;

        RewriteArgExtremum(Extremum dir, ir::Expr met, ir::WriteLoc l,
                           ir::Type t)
            : dir(dir), metric(std::move(met)), loc(std::move(l)),
              tuple_t(std::move(t)) {}

        using ir::Mutator::visit;

        // yield x => upd a arg(a, (M(x), x))
        ir::Stmt visit(const ir::Yield *node) override {
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volumes.size() <= lambda->args.size());
            // Tuple data, from a product or a flatten: the metric names one
            // parameter per level and the yielded element is the tuple of
            // them, which `apply_lambda` already takes apart.
            ir::Expr value = apply_lambda(metric, node->value);

            std::vector<ir::Expr> values = {std::move(value), node->value};
            ir::Expr update = ir::Build::make(tuple_t, std::move(values));

            // A plain write, not an Accumulate::Arg{min,max}. Filter fusion
            // has already put the comparison against the current best into the
            // filter this yield sits inside, so anything reaching here is
            // better and the write is unconditional. Spelling it as an
            // accumulate would say the same thing twice, and leave every
            // backend to implement a compare-and-select over a tuple -- which
            // the SSA path has no operation for, so `-p ssa` could not lower a
            // tree query at all.
            return ir::Store::make(loc, std::move(update));
        }

        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(
                lower_iterate(node->value)); // lower into a concrete loop.
        }

        ir::Stmt visit(const ir::Scan *node) override { return node; }

        ir::Stmt visit(const ir::YieldFrom *node) override { return node; }
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
    ir::Stmt body =
        build_traversal(fused_filter, tree_types, extents, local_intervals);

    body = RewriteArgExtremum(dir, std::move(metric), std::move(loc),
                              std::move(tuple_t))
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
                        const std::map<std::string, ir::Expr> &extents,
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
            internal_assert(!volumes.empty());
            const ir::Lambda *lambda = metric.as<ir::Lambda>();
            internal_assert(lambda) << "Metric is not a lambda: " << metric;
            internal_assert(volumes.size() <= lambda->args.size());
            // Tuple data; see the note in RewriteArgExtremum.
            ir::Expr value = apply_lambda(metric, node->value);
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
            internal_assert(volumes.size() <= lambda->args.size());
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
            internal_assert(volumes.size() <= lambda->args.size());

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
                                    tree_types, extents, local_intervals);

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
                          const std::map<std::string, ir::Expr> &extents,
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

        // yield x => upd a (a | P(x)), under the same two guards `visit(Scan)`
        // puts on a subtree.
        //
        // The guards are not an optimization the backend could find. Without
        // them a quantifier evaluates its predicate on every element of every
        // leaf the traversal reaches: after the answer is already settled, and
        // -- the expensive one -- inside leaves whose own bounding volume the
        // predicate provably cannot hold anywhere in. A leaf is where the
        // elements are, so that is where the predicate is at its most
        // expensive: skipping one leaf of a BVH costs a box test and saves as
        // many triangle intersections as the leaf holds.
        //
        // `build_filter` puts the volume guard on its yields already, which is
        // why `argmin(f, filter(p, tree))` prunes leaves and a bare
        // `any(p, tree)` did not. The same reasoning applies to both and it
        // belongs in both.
        // Wraps a body in the two guards the *volume* justifies: skip where
        // the predicate cannot hold anywhere in it, settle where it must hold
        // everywhere in it. Both are properties of the node, not of any one
        // element, which is what makes them hoistable out of a leaf's loop.
        ir::Stmt guard_with_volume(ir::Stmt body, const Interval &bounds) const {
            // Where the predicate cannot hold over the volume: `any` learns
            // nothing and skips, `all` is settled false -- exactly as a
            // subtree that cannot hold it settles `all` false.
            ir::Stmt otherwise;
            if (!is_any) {
                otherwise = ir::Store::make(loc, ir::BoolImm::make(false));
            }
            if (bounds.max.defined() && !is_const_one(bounds.max)) {
                body = ir::IfElse::make(bounds.max, std::move(body),
                                        std::move(otherwise));
            }

            // And where it provably holds over the whole volume it holds for
            // everything in it, so `any` is decided and `all` learns nothing.
            if (bounds.min.defined() && !is_const_zero(bounds.min)) {
                ir::Stmt settled =
                    is_any ? ir::Store::make(loc, ir::BoolImm::make(true))
                           : ir::Stmt();
                body = settled.defined()
                           ? ir::IfElse::make(bounds.min, settled,
                                              std::move(body))
                           : ir::IfElse::make(~bounds.min, std::move(body));
            }

            // One test on the accumulator gates the whole node, rather than
            // being fused into the `maybe` arm and leaving `always` outside
            // it. Nothing under here can change a settled answer, and `always`
            // is itself a pair of geometric tests -- run, before this, to
            // re-decide a question that was already decided.
            return ir::IfElse::make(still_undecided(), std::move(body));
        }

        ir::Stmt visit(const ir::Yield *node) override {
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            ir::Expr p = apply_lambda(predicate, node->value);
            ir::Expr acc = loc.to_expr();
            ir::Expr combined = is_any ? (acc || p) : (acc && p);
            ir::Stmt test = ir::Store::make(loc, std::move(combined));

            if (bound_hoisted) {
                // Inside a leaf's element loop: `visit(Iterate)` emitted the
                // volume guards once before it, and what is left here is the
                // early exit -- the one condition that changes as the loop
                // runs, so the one that cannot be hoisted with them.
                return ir::IfElse::make(still_undecided(), std::move(test));
            }
            return guard_with_volume(std::move(test), subtree_bounds());
        }

        // A leaf's elements, and the one place the node-level bound belongs.
        //
        // The volume guards are the same for every element the leaf holds, and
        // were being emitted inside the loop, once per element: on a BVH that
        // is the leaf's own box tested against the ray four-odd times over --
        // `contains`, `intersects`, `distmin` and `distmax` apiece -- to answer
        // a question that cannot change between them. `build_filter` has
        // hoisted them for a while; a bare `any(p, tree)` never did.
        //
        // What stays inside is the accumulator's guard, which is the early
        // exit and has to be re-read as the loop runs.
        ir::Stmt visit(const ir::Iterate *node) override {
            Interval bounds = subtree_bounds();

            const bool outer = bound_hoisted;
            bound_hoisted = true;
            ir::Stmt loop =
                mutate(lower_iterate(node->value)); // a concrete loop.
            bound_hoisted = outer;

            return guard_with_volume(std::move(loop), bounds);
        }

        // Whether the enclosing `visit(Iterate)` has already emitted the
        // volume guards, so the yields inside its loop should not repeat them.
        bool bound_hoisted = false;

        // The bounds of the predicate over the subtree currently being matched.
        Interval subtree_bounds() const {
            const ir::Lambda *lambda = predicate.as<ir::Lambda>();
            internal_assert(lambda)
                << "Predicate is not a lambda: " << predicate;
            internal_assert(volumes.size() <= lambda->args.size());
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
                recurse = ir::IfElse::make(bounds.max, std::move(recurse),
                                           std::move(otherwise));
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

            // As in `guard_with_volume`: one test on the accumulator gates the
            // node, so a settled query pays a bool rather than the `always`
            // bound's geometry on the way out.
            return ir::IfElse::make(still_undecided(), std::move(recurse));
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

    ir::Stmt body = build_traversal(inner, tree_types, extents, intervals);
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
                      const std::map<std::string, ir::Expr> &extents,
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

    ir::Stmt body = build_traversal(inner, tree_types, extents, intervals);
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

// flatten(F, S): the union, over the elements of S, of the set F(x) reached
// from each one, paired with the element it was reached from.
//
// Where `product` coiterates two trees and takes the Cartesian product of
// their children, this descends: the outer traversal runs as it would alone,
// and every element it arrives at opens a traversal of its own tree. The pair
// is kept because the outer element is what makes the inner one mean anything
// -- an instance's transform is what puts its triangles in the world.
//
// Nothing here knows about transforms, or about frames. `F` is an ordinary
// lambda and the tree it names is reached the ordinary way; what makes the
// inner tree's contents comparable with the outer tree's bounds is the outer
// element's `with extent` promise, which predicate analysis reads and this
// does not.
// Defined below, beside the pass it wraps.
ir::Stmt set_nested_volume_maps(ir::Stmt body,
                                const std::map<std::string, ir::Expr> &extents);

ir::Stmt build_flatten(ir::Stmt outer, ir::Expr func,
                       const ir::TypeMap &tree_types,
                       const std::map<std::string, ir::Expr> &extents,
                       const IntervalMap &intervals) {
    // Pairs everything the inner traversal produces with the outer element it
    // was reached from.
    struct PairWith : public Rewriter {
        ir::Expr outer;

        PairWith(ir::Expr o) : outer(std::move(o)) {}

        using ir::Mutator::visit;

        ir::Stmt visit(const ir::Yield *node) override {
            return ir::Yield::make(make_tuple_pair(outer, node->value));
        }

        // A collection at an inner leaf: every element of it pairs with the
        // same outer element, so the iteration is opened up and each yield
        // paired in turn.
        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(lower_iterate(node->value));
        }
    };

    struct RewriteFlatten : public Rewriter {
        ir::Expr func;
        const ir::TypeMap &tree_types;
        const std::map<std::string, ir::Expr> &extents;
        const IntervalMap &intervals;

        RewriteFlatten(ir::Expr f, const ir::TypeMap &t,
                       const std::map<std::string, ir::Expr> &e,
                       const IntervalMap &i)
            : func(std::move(f)), tree_types(t), extents(e), intervals(i) {}

        using ir::Mutator::visit;

        // yield x => the whole traversal of F(x), each of its answers paired
        // with x.
        ir::Stmt visit(const ir::Yield *node) override {
            ir::Expr root = apply_lambda(func, node->value);
            ir::Stmt inner =
                build_traversal(root, tree_types, extents, intervals);
            // Before anything reads a bound: the tree just built is in the
            // element's frame, and its stored bounds only bound this query
            // once the element's map has carried them into the query's frame.
            inner = set_nested_volume_maps(std::move(inner), extents);
            return PairWith(node->value).mutate(inner);
        }

        // iter xs => foreach x in xs: the above.
        ir::Stmt visit(const ir::Iterate *node) override {
            return mutate(lower_iterate(node->value));
        }

        // A recursive call already evaluates the flattened query.
        ir::Stmt visit(const ir::YieldFrom *node) override { return node; }
    };

    return RewriteFlatten(std::move(func), tree_types, extents, intervals)
        .mutate(outer);
}

ir::Stmt build_traversal(const ir::Expr &expr, const ir::TypeMap &tree_types,
                         const std::map<std::string, ir::Expr> &extents,
                         const IntervalMap &intervals) {
    // A set held in a field of an element rather than bound to a name of its
    // own: `i.blas`, the acceleration structure an instance carries.
    //
    // The schedule names it by the path that reaches it -- `Instance.blas :
    // BLAS` -- because that is what such a set has: one specification per
    // field, and one node pool behind it, however many values of the element
    // there are. So the key here is built the same way, and the root of the
    // traversal is the access itself rather than a variable.
    if (auto as_access = expr.as<ir::Access>();
        as_access != nullptr && as_access->type.is<ir::Set_t>()) {
        const ir::Struct_t *element =
            as_access->value.type().as<ir::Struct_t>();
        internal_assert(element)
            << "Cannot build traversal for a set reached through a "
               "non-element: "
            << expr << " on " << as_access->value.type();

        const std::string key = element->name + "." + as_access->field;
        const auto &iter = tree_types.find(key);
        internal_assert(iter != tree_types.cend())
            << "Lowering of: " << expr
            << " does not have an associated BVH type. A set held in a field "
               "gets one the same way a top-level set does, by being named in "
               "the schedule: `"
            << key << " : <TreeType>;`";

        const ir::BVH_t *bvh = iter->second.as<ir::BVH_t>();
        internal_assert(bvh);

        // Retyped to the tree, so the traversal unwraps the node arms out of
        // the very expression that reached it.
        return build_base_scan(
            ir::Access::make(as_access->field, as_access->value, iter->second),
            bvh);
    }

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
        if (as_agg->op != ir::AggOp::reduce) {
            // count, sum and prod are sugar for a map followed by a reduce.
            return build_traversal(expand_aggregate(as_agg), tree_types,
                                   extents, intervals);
        }
        return build_reduce(as_agg->identity, as_agg->combiner, as_agg->a,
                            tree_types, extents, intervals);
    }

    const ir::SetOp *as_set = expr.as<ir::SetOp>();
    if (as_set == nullptr) {
        internal_error << "[unimplemented] Unknown traversal pattern: " << expr;
    }

    switch (as_set->op) {
    case ir::SetOp::filter: {
        ir::Stmt body = build_traversal(as_set->b, tree_types, extents, intervals);
        return build_filter(body, as_set->a, intervals);
    }
    case ir::SetOp::map: {
        ir::Stmt body = build_traversal(as_set->b, tree_types, extents, intervals);
        return build_map(body, as_set->a);
    }
    case ir::SetOp::argmin:
    case ir::SetOp::argmax: {
        // These are a bit more complicated, because of filter fusion.
        const Extremum dir =
            as_set->op == ir::SetOp::argmin ? Extremum::Min : Extremum::Max;
        return build_arg_extremum(dir, as_set->a, as_set->b, tree_types,
                                  extents, intervals, expr.type());
    }
    case ir::SetOp::minimum:
    case ir::SetOp::maximum: {
        const Extremum dir =
            as_set->op == ir::SetOp::minimum ? Extremum::Min : Extremum::Max;
        return build_extremum(dir, as_set->a, as_set->b, tree_types, extents,
                              intervals, expr.type());
    }
    case ir::SetOp::any:
    case ir::SetOp::all: {
        return build_quantifier(as_set->op == ir::SetOp::any, as_set->a,
                                as_set->b, tree_types, extents, intervals);
    }
    case ir::SetOp::flatten: {
        // Only the outer set is traversed here. The inner one is reached from
        // an element, so it cannot be built until there is an element to reach
        // it from -- which is at the outer traversal's yields.
        ir::Stmt outer = build_traversal(as_set->b, tree_types, extents, intervals);
        return build_flatten(outer, as_set->a, tree_types, extents, intervals);
    }
    case ir::SetOp::product: {
        ir::Stmt a_body = build_traversal(as_set->a, tree_types, extents, intervals);
        ir::Stmt b_body = build_traversal(as_set->b, tree_types, extents, intervals);
        return build_product(a_body, b_body, expr.type().element_of());
    }
    default: {
        internal_error << "TODO: " << expr;
    }
    }
}

// Wrap the first Match seen in a recursive loop on all trees seen in the body.
// Substitutes one expression for another throughout a statement. Needed for a
// tree reached by an expression rather than named: the recursion has to name
// it, so every reference to `i.blas` in the match becomes a reference to the
// name the recursion advances.
struct ReplaceExpr : public ir::Mutator {
    ir::Expr from, to;

    ReplaceExpr(ir::Expr f, ir::Expr t)
        : from(std::move(f)), to(std::move(t)) {}

    using ir::Mutator::mutate;

    ir::Expr mutate(const ir::Expr &expr) override {
        if (expr.defined() && ir::equals(expr, from)) {
            return to;
        }
        return ir::Mutator::mutate(expr);
    }
};

struct WrapMatchInRecLoop : public ir::Mutator {
    std::vector<ir::TypedVar> trees;
    size_t nested = 0;
    size_t depth = 0;

    WrapMatchInRecLoop(std::vector<ir::TypedVar> trees)
        : trees(std::move(trees)) {}

    ir::Stmt visit(const ir::Match *node) override {
        // Descend first, because a query over a set held in an element's field
        // has a match inside a match -- and the two are different kinds of
        // nesting, told apart below.
        const bool outermost = depth == 0;
        depth++;
        ir::Stmt body = ir::Mutator::visit(node);
        depth--;

        const ir::Match *inner = body.as<ir::Match>();
        internal_assert(inner);

        if (outermost) {
            return ir::RecLoop::make(trees, std::move(body));
        }

        // A nested match on a *name* is a tree coiterated with the enclosing
        // one -- what `product` builds, where both trees advance together in a
        // single recursion over their children's Cartesian product. It belongs
        // to the recursion already wrapped around it.
        if (inner->loc.as<ir::Var>()) {
            return body;
        }

        // A nested match on an *expression* -- `i.blas` -- is a second
        // traversal, of a tree reached from an element of the first. It gets a
        // recursion of its own, which needs a name to advance, so the root is
        // bound to one and the match rewritten to use it. This is where the
        // second stack begins, and where a schedule will later decide what a
        // suspension keeps.
        const std::string name = "_subtree" + std::to_string(nested++);
        ir::Expr root = ir::Var::make(inner->loc.type(), name);
        ir::Stmt bind = ir::LetStmt::make(ir::WriteLoc(name, inner->loc.type()),
                                          inner->loc);
        ir::Stmt matched = ReplaceExpr(inner->loc, root).mutate(body);
        std::vector<ir::TypedVar> args = {{name, inner->loc.type()}};
        return ir::Sequence::make(
            {std::move(bind),
             ir::RecLoop::make(std::move(args), std::move(matched))});
    }
};

// Records the first access of the form `<anything>.<field of `elem`>` whose own
// base is `elem` -- the nested tree's root augmentation, `i.blas.AABB`.
struct FindRootVolume : public ir::Visitor {
    const std::string &set_field;
    ir::Expr found;

    FindRootVolume(const std::string &set_field) : set_field(set_field) {}

    using ir::Visitor::visit;

    void visit(const ir::Access *node) override {
        if (!found.defined()) {
            if (const ir::Var *base = node->value.as<ir::Var>();
                base != nullptr && base->name == set_field) {
                found = node;
                return;
            }
        }
        ir::Visitor::visit(node);
    }
};

// Gives a match over a tree reached from an element the function that carries
// that tree's bounds into the element's frame.
//
// The function is not asked for a second time: an element's `with extent`
// already says it. `extent = transform(render_from_instance, blas.AABB)` states
// both what the element's own extent is *and* how its tree's frame relates to
// its own, because the tree's root bound appears inside it. Abstracting that
// root bound out -- `blas.AABB` becomes the parameter -- leaves exactly the
// map. Deriving it means the extent and the map cannot disagree, which they
// could if both were written down.
struct SetNestedVolumeMaps : public ir::Mutator {
    const std::map<std::string, ir::Expr> &extents;

    SetNestedVolumeMaps(const std::map<std::string, ir::Expr> &extents)
        : extents(extents) {}

    using ir::Mutator::visit;

    ir::Stmt visit(const ir::Match *node) override {
        ir::Stmt mutated = ir::Mutator::visit(node);
        const ir::Match *match = mutated.as<ir::Match>();
        internal_assert(match);
        if (match->volume_map.defined()) {
            return mutated;
        }

        // Only a tree reached through an element has a frame of its own.
        const ir::Access *access = match->loc.as<ir::Access>();
        if (access == nullptr) {
            return mutated;
        }
        const ir::Struct_t *element = access->value.type().as<ir::Struct_t>();
        if (element == nullptr) {
            return mutated;
        }
        const auto iter = extents.find(element->name);
        if (iter == extents.cend()) {
            // No extent: nothing relates the two frames, so the tree's own
            // bounds are the only ones there are. Sound when the query does
            // not move its elements, and when it does, the missing map is
            // caught by `with extent` being missing rather than here.
            return mutated;
        }

        // Abstract the tree's root bound out of the extent *first*: it is the
        // one subterm that cannot survive being rebuilt, since a set has no
        // field to look a geometry up by -- only the annotation gave it one.
        FindRootVolume finder(access->field);
        iter->second.accept(&finder);
        if (!finder.found.defined()) {
            // An extent that does not mention the tree's root bound says
            // nothing about the tree's frame.
            return mutated;
        }

        const std::string name = "_vol";
        const ir::Type volume_type = finder.found.type();
        std::map<ir::Expr, ir::Expr, ir::ExprLessThan> abstraction;
        abstraction[finder.found] = ir::Var::make(volume_type, name);
        ir::Expr body = replace(abstraction, iter->second);

        // What is left is over the element's other fields; bind them to this
        // element.
        std::map<std::string, ir::Expr> fields;
        for (const auto &field : element->fields) {
            if (field.name == access->field) {
                continue; // abstracted away above
            }
            fields[field.name] = ir::Access::make(field.name, access->value);
        }
        body = replace(fields, body);

        ir::Expr volume_map =
            ir::Lambda::make({{name, volume_type}}, std::move(body));

        return ir::Match::make(match->loc, match->arms, std::move(volume_map));
    }
};

ir::Stmt set_nested_volume_maps(
    ir::Stmt body, const std::map<std::string, ir::Expr> &extents) {
    return SetNestedVolumeMaps(extents).mutate(std::move(body));
}

struct LowerBVH : public ir::Mutator {
    const ir::TypeMap &tree_types;
    const std::map<std::string, ir::Expr> &extents;
    ir::FuncMap new_funcs;

    LowerBVH(const ir::TypeMap &tree_types,
             const std::map<std::string, ir::Expr> &extents)
        : tree_types(tree_types), extents(extents) {}

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
        ir::Stmt body = build_traversal(expr, tree_types, extents, intervals);
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
    return build_base_scan(ir::Var::make(bvh_t, name), bvh_t);
}

// The same, rooted at an expression rather than at a name.
//
// A top-level set is a variable and its tree is reached by naming it. A set
// held in a field -- `i.blas`, an instance's own acceleration structure -- has
// no name of its own: its root is wherever the element that holds it says, so
// the traversal has to be built against that expression.
ir::Stmt build_base_scan(ir::Expr bvh_expr, const ir::BVH_t *bvh_t) {
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
            stmts.back() = ir::Scan::make(ir::Expr(), make_tuple(cs));
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

    LowerBVH converter(tree_types, program.extents);

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
