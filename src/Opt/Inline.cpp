#include "Opt/Inline.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Rename.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Lower/TopologicalOrder.h"
#include "Opt/CSE.h"
#include "Opt/Simplify.h"
#include "Utils.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace bonsai {
namespace opt {

namespace {

// The largest body, in statements, that is inlined without being asked to.
// Below this a function is a few relations' worth of code -- a slab test, a
// variant dispatch, a hit test unwrapped to a distance -- and copying it to a
// call site costs less than the call and buys what the call hid: the same
// test in two of them, which CSE can then merge. Above it the copy is the
// cost, and `[[inline]]` is how a program asks for one anyway.
constexpr size_t kMaxInlinedStatements = 40;

//===--------------------------------------------------------------------===//
// What a body looks like, for deciding whether it can be inlined.
//===--------------------------------------------------------------------===//

bool mentions_return(const ir::Stmt &stmt) {
    struct Check : ir::Visitor {
        bool found = false;
        void visit(const ir::Return *) override { found = true; }
    };
    Check check;
    stmt.accept(&check);
    return check.found;
}

// Counts the statements of a body, collects the names it binds, and notes
// what would rule it out: a loop or a construct the query lowering owns.
struct BodyShape : ir::Visitor {
    size_t statements = 0;
    bool plain = true;
    std::set<std::string> bound;

    void visit(const ir::LetStmt *node) override {
        statements++;
        bound.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const ir::Allocate *node) override {
        statements++;
        bound.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const ir::Store *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::Accumulate *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::Return *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::IfElse *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::SwitchStmt *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::CallStmt *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::Print *node) override {
        statements++;
        Visitor::visit(node);
    }

    // Loops, and everything a query lowers through. A Return inside a loop
    // would need a flag to become an assignment, and the rest are not things
    // a small relation has in it; none of them is worth the cases.
    void visit(const ir::While *) override { plain = false; }
    void visit(const ir::DoWhile *) override { plain = false; }
    void visit(const ir::ForAll *) override { plain = false; }
    void visit(const ir::ForEach *) override { plain = false; }
    void visit(const ir::ParFor *) override { plain = false; }
    void visit(const ir::RecLoop *) override { plain = false; }
    void visit(const ir::MultiRecurse *) override { plain = false; }
    void visit(const ir::Match *) override { plain = false; }
    void visit(const ir::MatchVariant *) override { plain = false; }
    void visit(const ir::Yield *) override { plain = false; }
    void visit(const ir::Iterate *) override { plain = false; }
    void visit(const ir::Scan *) override { plain = false; }
    void visit(const ir::YieldFrom *) override { plain = false; }
    void visit(const ir::Launch *) override { plain = false; }
    void visit(const ir::Append *) override { plain = false; }
    void visit(const ir::Label *) override { plain = false; }
    void visit(const ir::Free *) override { plain = false; }
    void visit(const ir::Continue *) override { plain = false; }
};

// A body as a flat list of statements, nested sequences opened out.
std::vector<ir::Stmt> flatten(const ir::Stmt &stmt) {
    std::vector<ir::Stmt> out;
    if (!stmt.defined()) {
        return out;
    }
    if (const auto *seq = stmt.as<ir::Sequence>()) {
        for (const ir::Stmt &s : seq->stmts) {
            std::vector<ir::Stmt> inner = flatten(s);
            out.insert(out.end(), inner.begin(), inner.end());
        }
        return out;
    }
    out.push_back(stmt);
    return out;
}

// Whether every `return` in a list of statements can become an assignment to
// a result variable with nothing else to mark that the function is done.
//
// A return at the end of a sequence can, plainly. A return in the middle can
// only if it sits in an arm of an `if` that always returns, and the other arm
// never does: the statements after the `if` then move into the other arm,
// where they run exactly when the function would have gone on to them. An
// arm that returns on some paths and not others would need a flag checked
// after the `if`, and that is where this stops.
bool nestable(const std::vector<ir::Stmt> &stmts) {
    for (size_t i = 0; i < stmts.size(); i++) {
        const ir::Stmt &s = stmts[i];
        if (s.as<ir::Return>()) {
            // Whatever follows is unreachable.
            return true;
        }
        if (const auto *sw = s.as<ir::SwitchStmt>()) {
            // The arms of a switch, by the same rule: an arm that mentions a
            // return must return on every path, and the statements after the
            // switch move into every arm that does not return.
            bool any_mentions = false;
            bool all_return = true;
            for (const ir::Stmt &arm : sw->arms) {
                const bool mentions = arm.defined() && mentions_return(arm);
                const bool returns = arm.defined() && ir::always_returns(arm);
                if (mentions && (!returns || !nestable(flatten(arm)))) {
                    return false;
                }
                any_mentions = any_mentions || mentions;
                all_return = all_return && returns;
            }
            if (!any_mentions) {
                continue;
            }
            if (all_return) {
                // Every arm is done; whatever follows is unreachable.
                return true;
            }
            std::vector<ir::Stmt> rest(stmts.begin() + i + 1, stmts.end());
            return nestable(rest);
        }
        const auto *ie = s.as<ir::IfElse>();
        if (ie == nullptr) {
            // Anything else may not hide a return: loops are excluded
            // upstream, and this keeps the rest honest.
            if (mentions_return(s)) {
                return false;
            }
            continue;
        }
        const bool t_mentions = mentions_return(ie->then_body);
        const bool e_mentions =
            ie->else_body.defined() && mentions_return(ie->else_body);
        if (!t_mentions && !e_mentions) {
            continue;
        }
        if (t_mentions && !nestable(flatten(ie->then_body))) {
            return false;
        }
        if (e_mentions && !nestable(flatten(ie->else_body))) {
            return false;
        }
        const bool t_returns = ir::always_returns(ie->then_body);
        const bool e_returns =
            ie->else_body.defined() && ir::always_returns(ie->else_body);
        if (t_returns && e_returns) {
            // Both arms are done; whatever follows is unreachable.
            return true;
        }
        std::vector<ir::Stmt> rest(stmts.begin() + i + 1, stmts.end());
        if (t_returns && !e_mentions) {
            return nestable(rest);
        }
        if (e_returns && !t_mentions) {
            return nestable(rest);
        }
        return false;
    }
    return true;
}

// Rewrites a nestable body so that every `return e` becomes `result = e`, by
// the rule `nestable` checks.
struct Nester {
    // Empty for a function returning nothing.
    std::string result;
    ir::Type type;

    ir::Stmt rewrite(const std::vector<ir::Stmt> &stmts) {
        std::vector<ir::Stmt> out;
        for (size_t i = 0; i < stmts.size(); i++) {
            const ir::Stmt &s = stmts[i];
            if (const auto *r = s.as<ir::Return>()) {
                if (r->value.defined()) {
                    internal_assert(!result.empty())
                        << "A value returned from a function of no value: "
                        << s;
                    out.push_back(ir::Store::make(ir::WriteLoc(result, type),
                                                  r->value));
                }
                return sequence(std::move(out));
            }
            if (const auto *sw = s.as<ir::SwitchStmt>()) {
                bool any_mentions = false;
                bool all_return = true;
                for (const ir::Stmt &arm : sw->arms) {
                    any_mentions = any_mentions ||
                                   (arm.defined() && mentions_return(arm));
                    all_return = all_return &&
                                 (arm.defined() && ir::always_returns(arm));
                }
                if (!any_mentions) {
                    out.push_back(s);
                    continue;
                }
                // The statements after the switch go into every arm that
                // falls out of it, where they run exactly when the function
                // would have gone on to them (see `nestable`); into more
                // than one arm when more than one does.
                std::vector<ir::Stmt> rest(stmts.begin() + i + 1, stmts.end());
                std::vector<ir::Stmt> arms;
                arms.reserve(sw->arms.size());
                for (const ir::Stmt &arm : sw->arms) {
                    std::vector<ir::Stmt> arm_stmts = flatten(arm);
                    if (!arm.defined() || !ir::always_returns(arm)) {
                        internal_assert(!arm.defined() || !mentions_return(arm))
                            << "An arm that returns on some paths only: " << s;
                        if (!all_return) {
                            arm_stmts.insert(arm_stmts.end(), rest.begin(),
                                             rest.end());
                        }
                    }
                    arms.push_back(rewrite(arm_stmts));
                }
                out.push_back(ir::SwitchStmt::make(sw->value, std::move(arms)));
                return sequence(std::move(out));
            }
            const auto *ie = s.as<ir::IfElse>();
            if (ie == nullptr) {
                internal_assert(!mentions_return(s))
                    << "A return where nesting cannot reach it: " << s;
                out.push_back(s);
                continue;
            }
            const bool t_mentions = mentions_return(ie->then_body);
            const bool e_mentions =
                ie->else_body.defined() && mentions_return(ie->else_body);
            if (!t_mentions && !e_mentions) {
                out.push_back(s);
                continue;
            }
            const bool t_returns = ir::always_returns(ie->then_body);
            const bool e_returns =
                ie->else_body.defined() && ir::always_returns(ie->else_body);
            std::vector<ir::Stmt> then_stmts = flatten(ie->then_body);
            std::vector<ir::Stmt> else_stmts = flatten(ie->else_body);
            std::vector<ir::Stmt> rest(stmts.begin() + i + 1, stmts.end());
            if (t_returns && e_returns) {
                // Nothing after the `if` is reachable.
            } else if (t_returns) {
                internal_assert(!e_mentions);
                else_stmts.insert(else_stmts.end(), rest.begin(), rest.end());
            } else {
                internal_assert(e_returns && !t_mentions)
                    << "An arm that returns on some paths only: " << s;
                then_stmts.insert(then_stmts.end(), rest.begin(), rest.end());
            }
            ir::Stmt then_body = rewrite(then_stmts);
            ir::Stmt else_body = rewrite(else_stmts);
            if (then_body.defined()) {
                out.push_back(ir::IfElse::make(ie->cond, std::move(then_body),
                                               std::move(else_body)));
            } else if (else_body.defined()) {
                out.push_back(ir::IfElse::make(
                    ir::UnOp::make(ir::UnOp::OpType::Not, ie->cond),
                    std::move(else_body)));
            }
            return sequence(std::move(out));
        }
        return sequence(std::move(out));
    }

    static ir::Stmt sequence(std::vector<ir::Stmt> stmts) {
        if (stmts.empty()) {
            return ir::Stmt();
        }
        return ir::Sequence::make(std::move(stmts));
    }
};

//===--------------------------------------------------------------------===//
// The inliner.
//===--------------------------------------------------------------------===//

using StatementFunctions =
    std::map<std::string, std::shared_ptr<ir::Function>>;

class Inliner : public ir::Mutator {
  public:
    Inliner(const ir::FuncMap &functions,
            const std::unordered_map<std::string, ir::Expr> &function_to_expr,
            const StatementFunctions &statement_functions, size_t &counter)
        : functions(functions), function_to_expr(function_to_expr),
          statement_functions(statement_functions), counter(counter) {}

    // How many calls were replaced by a body or an expression.
    size_t inlined = 0;

    ir::Expr visit(const ir::Call *node) override {
        // Arguments first, so that a call inside an argument is inlined --
        // and its statements hoisted -- before the call it is an argument of.
        std::vector<ir::Expr> args;
        args.reserve(node->args.size());
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        const ir::Var *v = node->func.as<ir::Var>();
        if (v == nullptr) {
            return ir::Call::make(node->func, std::move(args));
        }
        const std::string &function_name = v->name;

        if (auto it = function_to_expr.find(function_name);
            it != function_to_expr.end()) {
            // A function that is one expression: the call is the expression,
            // with the arguments put in for the parameters. Whatever the
            // expression calls in turn is the next round's.
            auto f = functions.find(function_name);
            internal_assert(f != functions.end());
            const std::vector<ir::Function::Argument> &params = f->second->args;
            internal_assert(params.size() == args.size())
                << "mismatch in function argument size: " << params.size()
                << " and call argument size: " << args.size()
                << " for function: " << function_name;
            std::map<std::string, ir::Expr> repls;
            for (size_t i = 0; i < params.size(); ++i) {
                repls[params[i].name] = args[i];
            }
            inlined++;
            return replace(repls, it->second);
        }

        if (auto it = statement_functions.find(function_name);
            it != statement_functions.end() && hoistable) {
            return inline_statements(*it->second, std::move(args));
        }
        return ir::Call::make(node->func, std::move(args));
    }

    // A call as a statement: the same, with no result to bind.
    ir::Stmt visit(const ir::CallStmt *node) override {
        std::vector<ir::Stmt> pre;
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate_hoisting(arg, pre));
        }
        const ir::Var *v = node->func.as<ir::Var>();
        if (v != nullptr) {
            if (auto it = statement_functions.find(v->name);
                it != statement_functions.end()) {
                std::vector<ir::Stmt> *saved = prelude;
                prelude = &pre;
                inline_statements(*it->second, std::move(args));
                prelude = saved;
                return Nester::sequence(std::move(pre));
            }
        }
        return with(std::move(pre),
                    ir::CallStmt::make(node->func, std::move(args)));
    }

    // The statements whose expressions a call can be hoisted out of: the
    // inlined body goes in front of the statement, and the statement reads
    // the result. Not a loop's condition or bounds, which are evaluated on
    // every trip or before the loop and would not be by anything hoisted to
    // one place; a call there stays a call.
    ir::Stmt visit(const ir::LetStmt *node) override {
        std::vector<ir::Stmt> pre;
        ir::Expr value = mutate_hoisting(node->value, pre);
        return with(std::move(pre), ir::LetStmt::make(node->loc, value));
    }
    ir::Stmt visit(const ir::Store *node) override {
        std::vector<ir::Stmt> pre;
        auto [loc, _] = mutate_writeloc(node->loc);
        ir::Expr value = mutate_hoisting(node->value, pre);
        ir::Expr mask =
            node->mask.defined() ? mutate_hoisting(node->mask, pre) : node->mask;
        return with(std::move(pre),
                    ir::Store::make(std::move(loc), value, mask));
    }
    ir::Stmt visit(const ir::Allocate *node) override {
        if (!node->value.defined()) {
            return node;
        }
        std::vector<ir::Stmt> pre;
        ir::Expr value = mutate_hoisting(node->value, pre);
        return with(std::move(pre),
                    ir::Allocate::make(node->loc, value, node->memory,
                                       node->unaliased));
    }
    ir::Stmt visit(const ir::Accumulate *node) override {
        std::vector<ir::Stmt> pre;
        auto [loc, _] = mutate_writeloc(node->loc);
        ir::Expr value = mutate_hoisting(node->value, pre);
        return with(std::move(pre),
                    ir::Accumulate::make(std::move(loc), node->op, value,
                                         node->atomic));
    }
    ir::Stmt visit(const ir::Return *node) override {
        if (!node->value.defined()) {
            return node;
        }
        std::vector<ir::Stmt> pre;
        ir::Expr value = mutate_hoisting(node->value, pre);
        return with(std::move(pre), ir::Return::make(value));
    }
    ir::Stmt visit(const ir::Print *node) override {
        std::vector<ir::Stmt> pre;
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate_hoisting(arg, pre));
        }
        return with(std::move(pre), ir::Print::make(std::move(args)));
    }
    ir::Stmt visit(const ir::IfElse *node) override {
        std::vector<ir::Stmt> pre;
        ir::Expr cond = mutate_hoisting(node->cond, pre);
        ir::Stmt then_body = mutate(node->then_body);
        ir::Stmt else_body =
            node->else_body.defined() ? mutate(node->else_body) : ir::Stmt();
        return with(std::move(pre),
                    ir::IfElse::make(cond, then_body, else_body));
    }
    ir::Stmt visit(const ir::SwitchStmt *node) override {
        std::vector<ir::Stmt> pre;
        ir::Expr value = mutate_hoisting(node->value, pre);
        std::vector<ir::Stmt> arms;
        arms.reserve(node->arms.size());
        for (const ir::Stmt &arm : node->arms) {
            arms.push_back(mutate(arm));
        }
        return with(std::move(pre),
                    ir::SwitchStmt::make(std::move(value), std::move(arms)));
    }

    // A nested sequence is opened into its parent, so that a `let` a
    // statement turned into `{inlined body; let}` still scopes over what
    // follows it.
    ir::Stmt visit(const ir::Sequence *node) override {
        std::vector<ir::Stmt> out;
        bool changed = false;
        for (const ir::Stmt &s : node->stmts) {
            ir::Stmt m = mutate(s);
            changed = changed || !m.same_as(s);
            if (const auto *inner = m.as<ir::Sequence>()) {
                changed = true;
                out.insert(out.end(), inner->stmts.begin(),
                           inner->stmts.end());
            } else if (m.defined()) {
                out.push_back(m);
            }
        }
        if (!changed) {
            return node;
        }
        return Nester::sequence(std::move(out));
    }

    // The arms of a select and the right side of a short-circuit are
    // evaluated on a condition, and a body hoisted in front of the statement
    // would be evaluated on none: a call there stays a call.
    ir::Expr visit(const ir::Select *node) override {
        ir::Expr cond = mutate(node->cond);
        const bool saved = hoistable;
        hoistable = false;
        ir::Expr tvalue = mutate(node->tvalue);
        ir::Expr fvalue = mutate(node->fvalue);
        hoistable = saved;
        return ir::Select::make(cond, tvalue, fvalue);
    }
    ir::Expr visit(const ir::BinOp *node) override {
        if (node->op != ir::BinOp::LAnd && node->op != ir::BinOp::LOr) {
            return Mutator::visit(node);
        }
        ir::Expr a = mutate(node->a);
        const bool saved = hoistable;
        hoistable = false;
        ir::Expr b = mutate(node->b);
        hoistable = saved;
        return ir::BinOp::make(node->op, a, b);
    }

  private:
    const ir::FuncMap &functions;
    const std::unordered_map<std::string, ir::Expr> &function_to_expr;
    const StatementFunctions &statement_functions;
    // Shared across the functions of a program and across rounds, so that
    // the names made here are unique in every one of them.
    size_t &counter;

    // Where the statements of an inlined body go, while an expression that
    // may hold a call is being mutated; null anywhere else, which is what
    // makes a call in a loop's condition stay a call.
    std::vector<ir::Stmt> *prelude = nullptr;
    bool hoistable = false;
    size_t depth = 0;

    ir::Expr mutate_hoisting(const ir::Expr &expr, std::vector<ir::Stmt> &pre) {
        std::vector<ir::Stmt> *saved_prelude = prelude;
        const bool saved_hoistable = hoistable;
        prelude = &pre;
        hoistable = true;
        ir::Expr result = mutate(expr);
        prelude = saved_prelude;
        hoistable = saved_hoistable;
        return result;
    }

    static ir::Stmt with(std::vector<ir::Stmt> pre, ir::Stmt stmt) {
        if (pre.empty()) {
            return stmt;
        }
        pre.push_back(std::move(stmt));
        return ir::Sequence::make(std::move(pre));
    }

    // Puts a copy of `f`'s body into the prelude, its arguments bound to
    // `args` and its returns turned into assignments to a fresh result
    // variable, and returns that variable -- or nothing, for a function of
    // no value.
    ir::Expr inline_statements(const ir::Function &f,
                               std::vector<ir::Expr> args) {
        internal_assert(prelude != nullptr);
        internal_assert(f.args.size() == args.size())
            << "mismatch in function argument size: " << f.args.size()
            << " and call argument size: " << args.size()
            << " for function: " << f.name;
        inlined++;
        const std::string tag = "$i" + std::to_string(counter++);

        BodyShape shape;
        f.body.accept(&shape);
        std::map<std::string, std::string> renames;
        for (const ir::Function::Argument &arg : f.args) {
            renames[arg.name] = arg.name + tag;
        }
        for (const std::string &name : shape.bound) {
            renames[name] = name + tag;
        }

        // An argument that is a variable or a constant stands in for the
        // parameter directly; anything else is bound once, by name, so that
        // it is evaluated once however often the body reads it.
        std::vector<ir::Stmt> pre;
        std::map<std::string, ir::Expr> substitutions;
        for (size_t i = 0; i < args.size(); i++) {
            const std::string &param = renames.at(f.args[i].name);
            if (args[i].as<ir::Var>() || is_const(args[i])) {
                substitutions[param] = args[i];
                continue;
            }
            pre.push_back(ir::LetStmt::make(
                ir::WriteLoc(param, f.args[i].type), std::move(args[i])));
        }
        const bool has_value = !f.ret_type.is<ir::Void_t>();
        std::string result;
        if (has_value) {
            result = f.name + "$r" + tag.substr(2);
            pre.push_back(ir::Allocate::make(ir::WriteLoc(result, f.ret_type),
                                             ir::Allocate::Memory::Stack));
        }

        // The variables the body binds, and its arguments, renamed so that
        // the copy can sit beside anything. Whatever the copy calls stays a
        // call until the next round, which is what lets the merge between
        // rounds see it (see Opt/Inline.h).
        ir::Stmt body = ir::rename_bindings(f.body, renames);
        if (!substitutions.empty()) {
            body = replace(substitutions, body);
        }
        Nester nester{result, f.ret_type};
        ir::Stmt nested = nester.rewrite(flatten(body));
        if (nested.defined()) {
            pre.push_back(std::move(nested));
        }
        prelude->insert(prelude->end(), pre.begin(), pre.end());

        if (!has_value) {
            return ir::Expr();
        }
        return ir::Var::make(f.ret_type, result);
    }
};

} // namespace

ir::FuncMap Inline::run(ir::FuncMap funcs,
                        const CompilerOptions &options) const {
    // A function on a cycle of the call graph is never inlined: a copy of
    // its body holds the call that leads back to it, and the rounds below
    // would never run out of calls to copy.
    const std::set<std::string> recursive = lower::recursive_functions(funcs);

    size_t counter = 0;
    for (size_t round = 0;; round++) {
        // One level a round, callers first: a body is copied as the round
        // found it, its own calls left as calls for the next round. The
        // inlinable functions form a DAG, so the rounds end after as many
        // as it has levels.
        internal_assert(round <= funcs.size())
            << "inlining did not settle after " << round << " rounds";

        // A function that is one expression is replaced by that expression
        // at every call, whatever its size.
        std::unordered_map<std::string, ir::Expr> function_to_expr;
        // A function of a few statements with returns that nest is copied to
        // its call sites, its returns becoming assignments to a result
        // variable. Copies of the functions: the round inlines into these
        // functions too, and what it copies out of one has to be the body as
        // the round found it.
        StatementFunctions statement_functions;
        for (const auto &[name, func] : funcs) {
            if (recursive.contains(name) || func->is_kernel() ||
                func->is_vectorized() || func->is_noinline()) {
                continue;
            }
            if (!func->ret_type.is<ir::Void_t>()) {
                if (const auto *body = func->body.as<ir::Return>()) {
                    internal_assert(body->value.defined());
                    function_to_expr[name] = body->value;
                    continue;
                }
            }
            // A mutating argument is a reference to the caller's variable,
            // and binding it to a fresh name would copy it instead.
            bool mutating = false;
            for (const ir::Function::Argument &arg : func->args) {
                mutating = mutating || arg.mutating;
            }
            if (mutating) {
                continue;
            }
            BodyShape shape;
            func->body.accept(&shape);
            if (!shape.plain) {
                continue;
            }
            // `[[inline]]` asks for a copy whatever the size.
            const bool asked = func->is_inlined() || func->is_always_inlined();
            if (!asked && shape.statements > kMaxInlinedStatements) {
                continue;
            }
            if (!nestable(flatten(func->body))) {
                continue;
            }
            statement_functions[name] = std::make_shared<ir::Function>(*func);
        }

        // Whether the round copied anything, counted rather than read off
        // the bodies: the mutator rebuilds a statement it hoisted nothing
        // into, and two bodies that mean the same are not the same node.
        size_t inlined = 0;
        for (auto &[name, func] : funcs) {
            if (!func->body.defined()) {
                continue;
            }
            Inliner inliner(funcs, function_to_expr, statement_functions,
                            counter);
            func->body = inliner.mutate(func->body);
            inlined += inliner.inlined;
        }
        if (inlined == 0) {
            break;
        }

        // The merge between levels: two copies that now make the same call
        // make it once before the next round copies that call in.
        funcs = Simplify().run(std::move(funcs), options);
        funcs = CSE().run(std::move(funcs), options);
    }

    return funcs;
}

} // namespace opt
} // namespace bonsai
