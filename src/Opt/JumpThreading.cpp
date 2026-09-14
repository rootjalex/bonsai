#include "Opt/JumpThreading.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Rename.h"
#include "IR/Visitor.h"
#include "Opt/Simplify.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace opt {

using namespace ir;

namespace {

// The most statements one threading may add to a function (see the header).
// GCC's bound for the same quantity, `max-jump-thread-duplication-stmts`,
// defaults to 15.
constexpr size_t kMaxGrowth = 15;

//===--------------------------------------------------------------------===//
// What a run of statements looks like
//===--------------------------------------------------------------------===//

// Does `stmt` branch on `c`, or its negation, anywhere in it?
struct Tests : Visitor {
    explicit Tests(const Expr &c) : c(c) {}
    const Expr &c;
    bool found = false;

    void visit(const IfElse *node) override {
        found = found || truth_under(node->cond, c).has_value();
        Visitor::visit(node);
    }
    void visit(const Select *node) override {
        found = found || truth_under(node->cond, c).has_value();
        Visitor::visit(node);
    }
};

// The shape of a run of statements: how many there are, whether any is a
// loop or a construct the query lowering owns, and the names it binds.
//
// Given facts, it is the shape of the run as a copy under those facts would
// be: a branch a fact decides keeps only the arm taken and is not itself
// counted, which is what the simplifier does to the copy.
struct Shape : Visitor {
    explicit Shape(const Facts *facts = nullptr) : facts(facts) {}
    const Facts *facts;
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
        if (facts != nullptr) {
            if (std::optional<bool> value = decided(*facts, node->cond)) {
                const Stmt &arm = *value ? node->then_body : node->else_body;
                if (arm.defined()) {
                    arm.accept(this);
                }
                return;
            }
        }
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

// Whether an arm never goes on to what follows its branch: a copy of the
// run would sit behind its `return` or its `continue`.
bool never_continues(const Stmt &arm) {
    return arm.defined() && (always_returns(arm) || leaves_loop(arm));
}

void append(std::vector<Stmt> &out, Stmt stmt) {
    if (!stmt.defined()) {
        return;
    }
    if (const auto *seq = stmt.as<Sequence>()) {
        out.insert(out.end(), seq->stmts.begin(), seq->stmts.end());
        return;
    }
    out.push_back(std::move(stmt));
}

Stmt sequence(std::vector<Stmt> stmts) {
    if (stmts.empty()) {
        return Stmt();
    }
    if (stmts.size() == 1) {
        return stmts[0];
    }
    return Sequence::make(std::move(stmts));
}

//===--------------------------------------------------------------------===//
// The threading
//===--------------------------------------------------------------------===//

class Threader : public Mutator {
  public:
    Threader(const Function &func, const std::set<std::string> &effectful,
             size_t &counter)
        : knowledge{assignable_names(func), effectful, {}}, counter(counter) {}

    // A branch's condition is known in its arms. Kept here so that a branch
    // an enclosing one has already decided is not threaded: the simplifier
    // folds it for nothing.
    Stmt visit(const IfElse *node) override {
        Expr cond = mutate(node->cond);
        const bool learn = learnable(cond);
        Stmt then_body = with_fact(cond, true, learn, node->then_body);
        Stmt else_body = with_fact(cond, false, learn, node->else_body);
        if (cond.same_as(node->cond) && then_body.same_as(node->then_body) &&
            else_body.same_as(node->else_body)) {
            return node;
        }
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

    // The first branch in a sequence that the statements after it test again
    // takes those statements into its arms. Whatever came before it is
    // threaded on its own; whatever came after is now inside it.
    Stmt visit(const Sequence *node) override {
        const std::vector<Stmt> &stmts = node->stmts;
        for (size_t i = 0; i + 1 < stmts.size(); i++) {
            const auto *branch = stmts[i].as<IfElse>();
            if (branch == nullptr) {
                continue;
            }
            const std::vector<Stmt> rest(stmts.begin() + i + 1, stmts.end());
            if (!threadable(*branch, rest)) {
                continue;
            }
            std::vector<Stmt> out;
            for (size_t j = 0; j < i; j++) {
                append(out, mutate(stmts[j]));
            }
            append(out, thread(*branch, rest, knowledge.known));
            return sequence(std::move(out));
        }

        // Nothing to thread at this level; each statement on its own.
        std::vector<Stmt> out;
        bool changed = false;
        for (const Stmt &s : stmts) {
            Stmt m = mutate(s);
            changed = changed || !m.same_as(s);
            append(out, std::move(m));
        }
        if (!changed) {
            return node;
        }
        return sequence(std::move(out));
    }

  private:
    // The names the function can assign and the functions with effects,
    // fixed; and the facts decided where the mutation is, which move.
    Simplify::Knowledge knowledge;
    // Shared across a program's functions, so the names made here are unique
    // in every one of them.
    size_t &counter;

    bool learnable(const Expr &cond) const {
        if (!is_pure_value(cond)) {
            return false;
        }
        for (const TypedVar &v : gather_free_vars(cond)) {
            if (knowledge.assignable.count(v.name)) {
                return false;
            }
        }
        return true;
    }

    Stmt with_fact(const Expr &cond, bool value, bool learn, const Stmt &arm) {
        if (!arm.defined() || !learn) {
            return mutate(arm);
        }
        knowledge.known.emplace_back(cond, value);
        Stmt out = mutate(arm);
        knowledge.known.pop_back();
        return out;
    }

    // The branch `arm` is, when it is one branch alone and `rest` tests its
    // condition too: the chain a `match` lowers to, followed link by link so
    // that the run is copied once per case. Nothing otherwise.
    const IfElse *link(const Stmt &arm, const std::vector<Stmt> &rest) const {
        if (!arm.defined()) {
            return nullptr;
        }
        const IfElse *branch = arm.as<IfElse>();
        if (const auto *seq = arm.as<Sequence>();
            seq != nullptr && seq->stmts.size() == 1) {
            branch = seq->stmts[0].as<IfElse>();
        }
        if (branch == nullptr || !learnable(branch->cond)) {
            return nullptr;
        }
        Tests tests(branch->cond);
        for (const Stmt &s : rest) {
            s.accept(&tests);
        }
        return tests.found ? branch : nullptr;
    }

    // The facts at each leaf of `branch`'s chain that a copy of `rest` would
    // go into: one per arm that goes on to what follows.
    void leaves(const IfElse &branch, const std::vector<Stmt> &rest,
                const Facts &path, std::vector<Facts> &out) const {
        for (const bool value : {true, false}) {
            const Stmt &arm = value ? branch.then_body : branch.else_body;
            Facts facts = path;
            facts.emplace_back(branch.cond, value);
            if (const IfElse *inner = link(arm, rest)) {
                leaves(*inner, rest, facts, out);
            } else if (!never_continues(arm)) {
                out.push_back(std::move(facts));
            }
        }
    }

    // Whether the statements after `branch` can go into its arms: its
    // condition can be known there and is not already, they test it, they
    // hold no loop, and the copies add no more than the bound.
    bool threadable(const IfElse &branch,
                    const std::vector<Stmt> &rest) const {
        if (!learnable(branch.cond) ||
            decided(knowledge.known, branch.cond).has_value()) {
            return false;
        }
        Shape shape;
        Tests tests(branch.cond);
        for (const Stmt &s : rest) {
            s.accept(&shape);
            s.accept(&tests);
        }
        if (!tests.found || !shape.plain) {
            return false;
        }
        std::vector<Facts> paths;
        leaves(branch, rest, knowledge.known, paths);
        size_t copied = 0;
        for (const Facts &path : paths) {
            Shape survives(&path);
            for (const Stmt &s : rest) {
                s.accept(&survives);
            }
            copied += survives.statements;
        }
        return copied <= shape.statements + kMaxGrowth;
    }

    // `branch` with `rest` copied into each leaf of its chain; `path` holds
    // what is known above it.
    Stmt thread(const IfElse &branch, const std::vector<Stmt> &rest,
                const Facts &path) {
        Stmt then_body = into(branch.then_body, rest, path, branch.cond, true);
        Stmt else_body =
            into(branch.else_body, rest, path, branch.cond, false);
        if (!then_body.defined()) {
            if (!else_body.defined()) {
                return Stmt();
            }
            return IfElse::make(UnOp::make(UnOp::OpType::Not, branch.cond),
                                std::move(else_body));
        }
        return IfElse::make(branch.cond, std::move(then_body),
                            std::move(else_body));
    }

    // An arm, and after it -- unless it never gets there -- a copy of `rest`
    // with its bindings renamed, the whole simplified under what the arm
    // knows. The arm is threaded further; the copy is final.
    Stmt into(const Stmt &arm, const std::vector<Stmt> &rest, Facts path,
              const Expr &cond, bool value) {
        path.emplace_back(cond, value);
        if (const IfElse *inner = link(arm, rest)) {
            return thread(*inner, rest, path);
        }
        std::vector<Stmt> body;
        if (arm.defined()) {
            const size_t known_before = knowledge.known.size();
            knowledge.known.emplace_back(cond, value);
            append(body, mutate(arm));
            knowledge.known.resize(known_before);
        }
        if (!never_continues(arm)) {
            Shape shape;
            for (const Stmt &s : rest) {
                s.accept(&shape);
            }
            const std::string tag = "$t" + std::to_string(counter++);
            std::map<std::string, std::string> renames;
            for (const std::string &name : shape.bound) {
                renames[name] = name + tag;
            }
            for (const Stmt &s : rest) {
                append(body, rename_bindings(s, renames));
            }
        }
        Stmt out = sequence(std::move(body));
        if (!out.defined()) {
            return out;
        }
        Simplify::Knowledge at_leaf{knowledge.assignable, knowledge.effectful,
                                    std::move(path)};
        return Simplify::simplify(std::move(out), at_leaf);
    }
};

} // namespace

FuncMap JumpThreading::run(FuncMap funcs, const CompilerOptions &) const {
    // Templated functions are left alone, and left out of the analysis: a
    // call in one names an instantiation, not a function.
    FuncMap concrete;
    for (const auto &[name, func] : funcs) {
        if (func->interfaces.empty() && func->body.defined()) {
            concrete[name] = func;
        }
    }
    const std::set<std::string> effectful = find_side_effects(concrete);
    size_t counter = 0;
    for (auto &[name, func] : concrete) {
        // What a branch already decides is folded first, so that only a test
        // no fact reaches is worth a copy.
        const Simplify::Knowledge knowledge{assignable_names(*func), effectful,
                                            {}};
        func->body = Simplify::simplify(func->body, knowledge);
        Threader threader(*func, effectful, counter);
        Stmt body = threader.mutate(func->body);
        if (body.defined()) {
            func->body = std::move(body);
        }
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
