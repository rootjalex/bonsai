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

struct PullQueriesImpl : public Mutator {
    const FuncMap &funcs;
    size_t counter = 0;

    PullQueriesImpl(const FuncMap &funcs) : funcs(funcs) {}

    using Mutator::visit;

    // rel(q, transform(m, x)) => rel(untransform(m, q), x), and its mirror.
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

        std::set<std::string> fixed;
        for (const auto &arg : loop->args) {
            fixed.insert(arg.var.name);
        }
        BoundNames bound;
        loop->body.accept(&bound);
        fixed.insert(bound.names.begin(), bound.names.end());
        const std::set<std::string> written = mutated_variables(loop->body);
        fixed.insert(written.begin(), written.end());

        FindPullable find(fixed);
        loop->body.accept(&find);
        if (find.found.empty()) {
            return rewritten;
        }

        std::vector<Stmt> stmts;
        std::map<Expr, Expr, ExprLessThan> names;
        for (const Expr &term : find.found) {
            const std::string name = "_pulled" + std::to_string(counter++);
            names[term] = Var::make(term.type(), name);
            stmts.push_back(
                LetStmt::make(WriteLoc(name, term.type()), term));
        }
        Stmt body = ReplaceTerms(names).mutate(loop->body);
        stmts.push_back(RecLoop::make(loop->args, std::move(body)));
        return Sequence::make(std::move(stmts));
    }
};

} // namespace

FuncMap PullQueries::run(FuncMap funcs, const CompilerOptions &options) const {
    PullQueriesImpl pull(funcs);
    for (auto &[name, func] : funcs) {
        func->body = pull.mutate(func->body);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
