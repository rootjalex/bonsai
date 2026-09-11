#include "Opt/PullQueries.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace opt {

using namespace ir;

namespace {

// Whether moving both operands of `op` by the same motion leaves it unchanged.
// The topological relations are, under any bijection; the metrics are, by the
// contract an `untransform` for the query type takes on (GeomOp::untransform);
// the ordering predicates are not, since they ask about the world's axes.
bool commutes_with_motion(GeomOp::OpType op) {
    const std::string name = GeomOp::intrinsic_name(op);
    return !is_geometric_motion(name) && !is_ordering_predicate(name);
}

// Whether the program supplies the function LowerGeometrics will look up for
// undoing `motion` on a `query`. Absent, the query stays as written.
bool can_untransform(const FuncMap &funcs, const Type &motion,
                     const Type &query) {
    const std::string m = geometric_element_name(motion);
    const std::string q = geometric_element_name(query);
    if (m.empty() || q.empty()) {
        return false;
    }
    return funcs.contains("untransform_" + m + "_" + q);
}

// Every name a statement binds. A term that mentions one of these cannot be
// moved ahead of the statement.
struct BoundNames : public Visitor {
    std::set<std::string> names;

    using Visitor::visit;

    void visit(const LetStmt *node) override {
        names.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const Allocate *node) override {
        names.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const ForEach *node) override {
        names.insert(node->name);
        Visitor::visit(node);
    }
    void visit(const ForAll *node) override {
        names.insert(node->index);
        Visitor::visit(node);
    }
    void visit(const ParFor *node) override {
        names.insert(node->index);
        Visitor::visit(node);
    }
    void visit(const RecLoop *node) override {
        for (const auto &arg : node->args) {
            names.insert(arg.var.name);
        }
        Visitor::visit(node);
    }
    void visit(const Lambda *node) override {
        for (const auto &arg : node->args) {
            names.insert(arg.name);
        }
        Visitor::visit(node);
    }
};

// The distinct `untransform` terms in a body that mention none of `fixed`, in
// the order first seen. A term that qualifies is taken whole -- its operands
// are not searched, since they leave with it.
struct FindPullable : public Visitor {
    const std::set<std::string> &fixed;
    std::vector<Expr> found;

    FindPullable(const std::set<std::string> &fixed) : fixed(fixed) {}

    using Visitor::visit;

    void visit(const GeomOp *node) override {
        if (node->op == GeomOp::untransform) {
            const Expr term(node);
            const std::vector<TypedVar> free = gather_free_vars(term);
            const bool movable =
                std::none_of(free.begin(), free.end(), [&](const TypedVar &v) {
                    return fixed.contains(v.name);
                });
            if (movable) {
                const bool seen =
                    std::any_of(found.begin(), found.end(),
                                [&](const Expr &e) { return equals(e, term); });
                if (!seen) {
                    found.push_back(term);
                }
                return;
            }
        }
        Visitor::visit(node);
    }
};

// Replaces every expression structurally equal to a key with its value.
struct ReplaceTerms : public Mutator {
    const std::map<Expr, Expr, ExprLessThan> &terms;

    ReplaceTerms(const std::map<Expr, Expr, ExprLessThan> &terms)
        : terms(terms) {}

    using Mutator::mutate;

    Expr mutate(const Expr &expr) override {
        if (expr.defined()) {
            if (const auto it = terms.find(expr); it != terms.end()) {
                return it->second;
            }
        }
        return Mutator::mutate(expr);
    }
};

// The rewrite: rel(q, transform(m, x)) => rel(untransform(m, q), x), and its
// mirror. Run over a whole body before anything is hoisted, so that the hoist
// sees every term the rewrite made.
struct RewriteRelations : public Mutator {
    const FuncMap &funcs;

    RewriteRelations(const FuncMap &funcs) : funcs(funcs) {}

    using Mutator::visit;

    Expr visit(const GeomOp *node) override {
        Expr a = mutate(node->a);
        Expr b = mutate(node->b);

        if (commutes_with_motion(node->op)) {
            if (const GeomOp *moved = b.as<GeomOp>();
                moved != nullptr && moved->op == GeomOp::transform &&
                can_untransform(funcs, moved->a.type(), a.type())) {
                Expr pulled =
                    GeomOp::make(GeomOp::untransform, moved->a, std::move(a));
                return GeomOp::make(node->op, std::move(pulled), moved->b);
            }
            if (const GeomOp *moved = a.as<GeomOp>();
                moved != nullptr && moved->op == GeomOp::transform &&
                can_untransform(funcs, moved->a.type(), b.type())) {
                Expr pulled =
                    GeomOp::make(GeomOp::untransform, moved->a, std::move(b));
                return GeomOp::make(node->op, moved->b, std::move(pulled));
            }
        }

        if (a.same_as(node->a) && b.same_as(node->b)) {
            return node;
        }
        return GeomOp::make(node->op, std::move(a), std::move(b));
    }
};

// The hoist: each distinct pulled term bound once, where its scope begins.
struct HoistPulled : public Mutator {
    size_t counter = 0;

    using Mutator::visit;

    // The distinct pulled terms of `body` that mention none of `fixed`, each
    // given a name -- `let _pulledN = untransform(m, q)` -- with every
    // occurrence in the body replaced by it and the sort keys inside pointed
    // at it (see RepointKeys). The lets come back separately from the body, so
    // that the caller can put them where the scope it is binding for begins.
    struct Bound {
        std::vector<Stmt> lets;
        Stmt body;
    };

    Bound bind_pulled(const Stmt &body, const std::set<std::string> &fixed) {
        FindPullable find(fixed);
        body.accept(&find);
        if (find.found.empty()) {
            return Bound{{}, body};
        }

        Bound bound;
        std::map<Expr, Expr, ExprLessThan> names;
        // The query each pulled term stands in for, so that anything still
        // written over the query inside can be pointed at the pulled one.
        std::map<Expr, Expr, ExprLessThan> queries;
        for (const Expr &term : find.found) {
            const std::string name = "_pulled" + std::to_string(counter++);
            const Expr var = Var::make(term.type(), name);
            names[term] = var;
            const GeomOp *pull = term.as<GeomOp>();
            internal_assert(pull && pull->op == GeomOp::untransform) << term;
            queries[pull->b] = var;
            bound.lets.push_back(
                LetStmt::make(WriteLoc(name, term.type()), term));
        }
        bound.body = ReplaceTerms(names).mutate(body);
        bound.body = RepointKeys(queries).mutate(std::move(bound.body));
        return bound;
    }

    // Everything a statement binds or writes: what a term mentioning any of
    // it cannot be moved ahead of.
    static std::set<std::string> bound_within(const Stmt &body) {
        BoundNames bound;
        body.accept(&bound);
        std::set<std::string> fixed = std::move(bound.names);
        const std::set<std::string> written = mutated_variables(body);
        fixed.insert(written.begin(), written.end());
        return fixed;
    }

    // Hoist what the rewrite above produced out of the recursion, where that
    // is legal: a term may move ahead of the recursion when it mentions
    // nothing the recursion carries, nothing its body binds, and nothing its
    // body writes. For a tree held in an element that lands the pulled ray
    // exactly where pbrt puts it -- inside the loop over the leaf's instances,
    // once per instance, ahead of the walk of that instance's tree. The
    // outer recursion then finds the same term, mentioning the instance its
    // own body binds, and correctly leaves it where it is.
    Stmt visit(const RecLoop *node) override {
        Stmt rewritten = Mutator::visit(node);
        const RecLoop *loop = rewritten.as<RecLoop>();
        internal_assert(loop) << "Rewriting a RecLoop gave: " << rewritten;

        std::set<std::string> fixed = bound_within(loop->body);
        for (const auto &arg : loop->args) {
            fixed.insert(arg.var.name);
        }
        Bound bound = bind_pulled(loop->body, fixed);
        if (bound.lets.empty()) {
            return rewritten;
        }
        std::vector<Stmt> stmts = std::move(bound.lets);
        stmts.push_back(RecLoop::make(loop->args, std::move(bound.body)));
        return Sequence::make(std::move(stmts));
    }

    // And bind it once at the top of each arm of a match that uses it.
    //
    // A tree of mixed primitives dispatches on the primitive inside the leaf's
    // element loop, and an arm that holds no tree -- pbrt's
    // GeometricPrimitive -- has no recursion for the rule above to hoist
    // ahead of. Its tests are then each written over `untransform(p, r)`, the
    // same term as many times as the query tests the element, and what that
    // costs is not the pull -- for this arm the motion holds still, and the
    // term is `r` -- but that the tests no longer share an operand: the shape
    // test was run once for `intersects` and once for `distmin` and, written
    // over four copies of a term the backend could not prove equal, ran once
    // per copy. Bound once at the top of the arm, the term is what pbrt's
    // `Primitive::Intersect` has in each of its arms: one ray, computed on
    // entry, and every test over it. Inside the arm the match has settled
    // which variant the value is, so the term's own dispatch folds to nothing
    // for the arm that moves nothing.
    //
    // Per arm rather than once ahead of the match, on purpose: bound before
    // the dispatch, the term would be evaluated on every path, and the arm
    // that does nothing with a pulled ray would still compute one.
    //
    // Terms that mention something the arm binds or writes stay put; the
    // recursion inside the arm, if there is one, hoists what it can. The
    // arm's own bindings count as bound, since a match statement the program
    // wrote names its fields that way.
    Stmt visit(const MatchVariant *node) override {
        const Expr value = mutate(node->value);
        std::vector<MatchVariant::Arm> arms;
        arms.reserve(node->arms.size());
        bool not_changed = value.same_as(node->value);
        for (const auto &arm : node->arms) {
            std::set<std::string> fixed = bound_within(arm.body);
            fixed.insert(arm.bindings.begin(), arm.bindings.end());
            Bound bound = bind_pulled(arm.body, fixed);
            Stmt body = mutate(bound.body);
            if (!bound.lets.empty()) {
                std::vector<Stmt> stmts = std::move(bound.lets);
                stmts.push_back(std::move(body));
                body = Sequence::make(std::move(stmts));
            }
            not_changed = not_changed && body.same_as(arm.body);
            arms.push_back(
                MatchVariant::Arm{arm.variant, arm.bindings, std::move(body)});
        }
        if (not_changed) {
            return node;
        }
        return MatchVariant::make(value, std::move(arms));
    }

    // A schedule's sort keys, written over the query, follow it into the
    // frame the walk now runs in.
    //
    // `sort(Instance.blas.Interior, |i, r, axis| select((1 / r.d)[axis] < 0,
    // ..))` is pbrt's front-to-back rule, and pbrt applies it to the ray *after*
    // `ApplyInverse`: the split axis is an axis of the instance's frame, and the
    // sign of the world-space direction along it says nothing about which child
    // is nearer once the instance is rotated. Every relation in this recursion
    // has just been moved onto the pulled ray; a key left on the world ray
    // would order the walk by a ray the walk is no longer testing against --
    // the right rule applied to the wrong ray, and not visibly wrong, since an
    // argmin does not depend on the order. Only the keys are touched: what a
    // `from` recurses into is a node, not a query.
    struct RepointKeys : public Mutator {
        const std::map<Expr, Expr, ExprLessThan> &queries;

        RepointKeys(const std::map<Expr, Expr, ExprLessThan> &queries)
            : queries(queries) {}

        using Mutator::visit;

        Stmt visit(const YieldFrom *node) override {
            if (node->keys.empty()) {
                return node;
            }
            ReplaceTerms repoint(queries);
            std::vector<Expr> keys;
            keys.reserve(node->keys.size());
            bool not_changed = true;
            for (const Expr &key : node->keys) {
                Expr moved = repoint.mutate(key);
                not_changed = not_changed && moved.same_as(key);
                keys.push_back(std::move(moved));
            }
            if (not_changed) {
                return node;
            }
            return YieldFrom::make(node->value, std::move(keys));
        }
    };
};

} // namespace

FuncMap PullQueries::run(FuncMap funcs, const CompilerOptions &options) const {
    RewriteRelations rewrite(funcs);
    HoistPulled hoist;
    for (auto &[name, func] : funcs) {
        func->body = hoist.mutate(rewrite.mutate(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
