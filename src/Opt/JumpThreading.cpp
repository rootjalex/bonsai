#include "Opt/JumpThreading.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Rename.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace opt {

using namespace ir;

namespace {

// The most statements a run following a branch may hold and still be copied
// into both of its arms.
constexpr size_t kMaxDuplicated = 256;
// How deep threadings may nest, since each one copies what the ones inside
// it have already copied.
constexpr size_t kMaxDepth = 3;

//===--------------------------------------------------------------------===//
// What a condition and a run of statements look like
//===--------------------------------------------------------------------===//

// A condition pure enough to be known: arithmetic over variables and
// constants, with nothing in it that runs or reads memory.
struct IsSimple : Visitor {
    bool simple = true;

    void visit(const Call *) override { simple = false; }
    void visit(const CallStmt *) override { simple = false; }
    void visit(const Intrinsic *) override { simple = false; }
    void visit(const MatchExpr *) override { simple = false; }
    void visit(const Generator *) override { simple = false; }
    void visit(const Lambda *) override { simple = false; }
    void visit(const GeomOp *) override { simple = false; }
    void visit(const SetOp *) override { simple = false; }
    void visit(const AggOp *) override { simple = false; }
    void visit(const Deref *) override { simple = false; }
    void visit(const PtrTo *) override { simple = false; }
    void visit(const Construct *) override { simple = false; }
    void visit(const UnionOf *) override { simple = false; }
    void visit(const Unwrap *) override { simple = false; }
    void visit(const VectorReduce *) override { simple = false; }
    void visit(const VectorShuffle *) override { simple = false; }
    void visit(const Ramp *) override { simple = false; }
};

bool is_simple(const Expr &e) {
    IsSimple check;
    e.accept(&check);
    return check.simple;
}

// The names an expression reads.
std::set<std::string> reads(const Expr &e) {
    std::set<std::string> names;
    for (const TypedVar &v : gather_free_vars(e)) {
        names.insert(v.name);
    }
    return names;
}

// Does anything in `stmt` assign or bind one of `names`? A `let` counts:
// binding a name again changes what a later read of it means.
struct Writes : Visitor {
    explicit Writes(const std::set<std::string> &names) : names(names) {}
    const std::set<std::string> &names;
    bool writes = false;

    void check(const WriteLoc &loc) { writes = writes || names.count(loc.base) > 0; }

    void visit(const Store *node) override {
        check(node->loc);
        Visitor::visit(node);
    }
    void visit(const Accumulate *node) override {
        check(node->loc);
        Visitor::visit(node);
    }
    void visit(const Allocate *node) override {
        check(node->loc);
        Visitor::visit(node);
    }
    void visit(const LetStmt *node) override {
        check(node->loc);
        Visitor::visit(node);
    }
};

// Does `stmt` contain a `continue`?
bool leaves_loop(const Stmt &stmt) {
    struct Check : Visitor {
        bool found = false;
        void visit(const Continue *) override { found = true; }
    };
    Check check;
    stmt.accept(&check);
    return check.found;
}

bool writes_any(const Stmt &stmt, const std::set<std::string> &names) {
    if (!stmt.defined() || names.empty()) {
        return false;
    }
    Writes check(names);
    stmt.accept(&check);
    return check.writes;
}

// Is `cond` the condition `c`, or its negation? Nothing, when neither.
std::optional<bool> same_or_negated(const Expr &cond, const Expr &c) {
    if (equals(cond, c)) {
        return true;
    }
    if (const auto *n = cond.as<UnOp>(); n && n->op == UnOp::OpType::Not &&
                                          equals(n->a, c)) {
        return false;
    }
    if (const auto *n = c.as<UnOp>();
        n && n->op == UnOp::OpType::Not && equals(cond, n->a)) {
        return false;
    }
    return std::nullopt;
}

// Does `stmt` branch on `c` anywhere in it?
struct Tests : Visitor {
    explicit Tests(const Expr &c) : c(c) {}
    const Expr &c;
    bool found = false;

    void visit(const IfElse *node) override {
        found = found || same_or_negated(node->cond, c).has_value();
        Visitor::visit(node);
    }
    void visit(const Select *node) override {
        found = found || same_or_negated(node->cond, c).has_value();
        Visitor::visit(node);
    }
};

// The shape of a run of statements: how many there are, whether any is a
// loop or a construct the query lowering owns, and the names it binds.
struct Shape : Visitor {
    size_t statements = 0;
    bool plain = true;
    std::set<std::string> bound;

    void visit(const LetStmt *node) override {
        statements++;
        bound.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const Allocate *node) override {
        statements++;
        bound.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const Store *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const Accumulate *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const Return *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const IfElse *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const CallStmt *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const Print *node) override {
        statements++;
        Visitor::visit(node);
    }

    // A loop is named by the schedule, so there can be one of it.
    void visit(const While *) override { plain = false; }
    void visit(const DoWhile *) override { plain = false; }
    void visit(const ForAll *) override { plain = false; }
    void visit(const ForEach *) override { plain = false; }
    void visit(const ParFor *) override { plain = false; }
    void visit(const RecLoop *) override { plain = false; }
    void visit(const MultiRecurse *) override { plain = false; }
    void visit(const Match *) override { plain = false; }
    void visit(const MatchVariant *) override { plain = false; }
    void visit(const Yield *) override { plain = false; }
    void visit(const Iterate *) override { plain = false; }
    void visit(const Scan *) override { plain = false; }
    void visit(const YieldFrom *) override { plain = false; }
    void visit(const Launch *) override { plain = false; }
    void visit(const Append *) override { plain = false; }
    void visit(const Label *) override { plain = false; }
    void visit(const Free *) override { plain = false; }
    void visit(const Continue *) override { plain = false; }
};

// The names a function can assign: its mutable locals and the parameters it
// takes by reference. A condition over anything else cannot change behind
// the code's back, whatever that code calls.
std::set<std::string> mutable_names(const Function &func) {
    struct Allocations : Visitor {
        std::set<std::string> names;
        void visit(const Allocate *node) override {
            names.insert(node->loc.base);
            Visitor::visit(node);
        }
    };
    Allocations allocations;
    func.body.accept(&allocations);
    for (const Function::Argument &arg : func.args) {
        if (arg.mutating) {
            allocations.names.insert(arg.name);
        }
    }
    return allocations.names;
}

//===--------------------------------------------------------------------===//
// The threading
//===--------------------------------------------------------------------===//

class Threader : public Mutator {
  public:
    Threader(std::set<std::string> mutables, size_t &counter)
        : mutables(std::move(mutables)), counter(counter) {}

    // A branch whose condition is known folds to the arm taken; one that is
    // not is mutated with its condition known in each arm.
    Stmt visit(const IfElse *node) override {
        if (std::optional<bool> value = known(node->cond)) {
            const Stmt &arm = *value ? node->then_body : node->else_body;
            return arm.defined() ? mutate(arm) : Stmt();
        }
        Expr cond = mutate(node->cond);
        const bool knowable = is_knowable(cond, node->then_body) &&
                              is_knowable(cond, node->else_body);
        Stmt then_body = with_known(cond, true, knowable, node->then_body);
        Stmt else_body = with_known(cond, false, knowable, node->else_body);
        if (cond.same_as(node->cond) && then_body.same_as(node->then_body) &&
            else_body.same_as(node->else_body)) {
            return node;
        }
        // An arm that was one branch on a known condition folds to nothing.
        // A branch needs a then-arm, so what is left goes under the negated
        // condition, and a branch with nothing in either arm goes.
        if (!then_body.defined()) {
            if (!else_body.defined()) {
                return Stmt();
            }
            return IfElse::make(UnOp::make(UnOp::OpType::Not, std::move(cond)),
                                std::move(else_body));
        }
        return IfElse::make(std::move(cond), std::move(then_body),
                            std::move(else_body));
    }

    // A select is a function of three values, and evaluates both of its
    // arms; so a known condition picks the arm only when the other one did
    // nothing, since a draw from a sampler in the arm not taken still
    // advanced the sampler.
    Expr visit(const Select *node) override {
        if (std::optional<bool> value = known(node->cond)) {
            const Expr &taken = *value ? node->tvalue : node->fvalue;
            const Expr &dropped = *value ? node->fvalue : node->tvalue;
            if (is_simple(dropped)) {
                return mutate(taken);
            }
        }
        return Mutator::visit(node);
    }

    // A branch followed by statements that test its condition again takes
    // those statements into both of its arms, where the second test is
    // decided.
    Stmt visit(const Sequence *node) override {
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());
        bool changed = false;
        for (const Stmt &s : node->stmts) {
            Stmt m = mutate(s);
            changed = changed || !m.same_as(s);
            if (!m.defined()) {
                changed = true;
                continue;
            }
            if (const auto *inner = m.as<Sequence>()) {
                // A folded branch leaves its arm's sequence behind; opened
                // out, so that a `let` in it scopes over what follows.
                changed = true;
                stmts.insert(stmts.end(), inner->stmts.begin(),
                             inner->stmts.end());
                continue;
            }
            stmts.push_back(std::move(m));
        }

        for (size_t i = 0; i + 1 < stmts.size(); i++) {
            const auto *branch = stmts[i].as<IfElse>();
            if (branch == nullptr || !threadable(*branch, stmts, i + 1)) {
                continue;
            }
            std::vector<Stmt> rest(stmts.begin() + i + 1, stmts.end());
            const Expr cond = branch->cond;
            Stmt then_body = thread_into(branch->then_body, rest, cond, true);
            Stmt else_body = thread_into(branch->else_body, rest, cond, false);
            stmts.resize(i);
            // A copy may fold to nothing at all.
            if (then_body.defined()) {
                stmts.push_back(IfElse::make(cond, std::move(then_body),
                                             std::move(else_body)));
            } else if (else_body.defined()) {
                stmts.push_back(IfElse::make(
                    UnOp::make(UnOp::OpType::Not, cond), std::move(else_body)));
            }
            changed = true;
            break;
        }

        if (!changed) {
            return node;
        }
        if (stmts.empty()) {
            return Stmt();
        }
        if (stmts.size() == 1) {
            return stmts[0];
        }
        return Sequence::make(std::move(stmts));
    }

  private:
    const std::set<std::string> mutables;
    // Shared across a program's functions, so the names made here are unique
    // in every one of them.
    size_t &counter;
    // The conditions whose values are decided where the mutation is.
    std::vector<std::pair<Expr, bool>> facts;
    size_t depth = 0;

    std::optional<bool> known(const Expr &cond) const {
        for (const auto &[fact, value] : facts) {
            if (std::optional<bool> same = same_or_negated(cond, fact)) {
                return *same ? value : !value;
            }
        }
        return std::nullopt;
    }

    // Whether `cond` means the same thing throughout `arm`: it is pure, and
    // nothing can assign what it reads -- not in the arm, and not anywhere,
    // since a call inside the arm could otherwise do it.
    bool is_knowable(const Expr &cond, const Stmt &arm) const {
        if (!is_simple(cond)) {
            return false;
        }
        const std::set<std::string> names = reads(cond);
        for (const std::string &name : names) {
            if (mutables.count(name)) {
                return false;
            }
        }
        return !writes_any(arm, names);
    }

    Stmt with_known(const Expr &cond, bool value, bool knowable,
                    const Stmt &arm) {
        if (!arm.defined()) {
            return arm;
        }
        if (!knowable) {
            return mutate(arm);
        }
        facts.emplace_back(cond, value);
        Stmt out = mutate(arm);
        facts.pop_back();
        return out;
    }

    // Whether the statements from `from` on can go into the arms of `branch`.
    bool threadable(const IfElse &branch, const std::vector<Stmt> &stmts,
                    size_t from) const {
        if (depth >= kMaxDepth) {
            return false;
        }
        if (!is_simple(branch.cond)) {
            return false;
        }
        const std::set<std::string> names = reads(branch.cond);
        for (const std::string &name : names) {
            if (mutables.count(name)) {
                return false;
            }
        }
        if (writes_any(branch.then_body, names) ||
            writes_any(branch.else_body, names)) {
            return false;
        }
        Shape shape;
        Tests tests(branch.cond);
        for (size_t i = from; i < stmts.size(); i++) {
            stmts[i].accept(&shape);
            stmts[i].accept(&tests);
            if (writes_any(stmts[i], names)) {
                return false;
            }
        }
        return tests.found && shape.plain && shape.statements <= kMaxDuplicated;
    }

    // `arm` followed by a copy of `rest` with `cond` known to be `value`, its
    // bindings renamed so the two copies bind different names. An arm that
    // always returns never reaches `rest` and is left as it is.
    Stmt thread_into(const Stmt &arm, const std::vector<Stmt> &rest,
                     const Expr &cond, bool value) {
        if (arm.defined() && always_returns(arm)) {
            return arm;
        }
        // An arm that leaves its loop early likewise; the copy would sit
        // behind the `continue`.
        if (arm.defined() && leaves_loop(arm)) {
            return arm;
        }
        Shape shape;
        for (const Stmt &s : rest) {
            s.accept(&shape);
        }
        const std::string tag = "$t" + std::to_string(counter++);
        std::map<std::string, std::string> renames;
        for (const std::string &name : shape.bound) {
            renames[name] = name + tag;
        }
        std::vector<Stmt> copy;
        if (arm.defined()) {
            copy.push_back(arm);
        }
        for (const Stmt &s : rest) {
            copy.push_back(rename_bindings(s, renames));
        }
        Stmt body = Sequence::make(std::move(copy));

        facts.emplace_back(cond, value);
        depth++;
        Stmt out = mutate(body);
        depth--;
        facts.pop_back();
        return out;
    }
};

} // namespace

FuncMap JumpThreading::run(FuncMap funcs, const CompilerOptions &) const {
    size_t counter = 0;
    for (auto &[name, func] : funcs) {
        if (!func->body.defined()) {
            continue;
        }
        Threader threader(mutable_names(*func), counter);
        Stmt body = threader.mutate(func->body);
        if (body.defined()) {
            func->body = std::move(body);
        }
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
