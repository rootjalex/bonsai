#include "Opt/Inline.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Lower/TopologicalOrder.h"
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
// what would rule it out: a loop or a construct the query lowering owns, or a
// call to the function itself.
struct BodyShape : ir::Visitor {
    explicit BodyShape(std::string self) : self(std::move(self)) {}

    const std::string self;
    size_t statements = 0;
    bool plain = true;
    bool recursive = false;
    bool allocates = false;
    std::set<std::string> bound;

    void visit(const ir::LetStmt *node) override {
        statements++;
        bound.insert(node->loc.base);
        Visitor::visit(node);
    }
    void visit(const ir::Allocate *node) override {
        statements++;
        allocates = true;
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
    void visit(const ir::CallStmt *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::Print *node) override {
        statements++;
        Visitor::visit(node);
    }
    void visit(const ir::Call *node) override {
        if (const auto *v = node->func.as<ir::Var>(); v && v->name == self) {
            recursive = true;
        }
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

// Renames the variables a body binds, and its arguments, so that a copy of
// it can sit beside anything. Both the uses and the bindings: a let, an
// allocation, and the base of every store.
struct Renamer : ir::Mutator {
    explicit Renamer(const std::map<std::string, std::string> &renames)
        : renames(renames) {}

    const std::map<std::string, std::string> &renames;

    std::string of(const std::string &name) const {
        auto it = renames.find(name);
        return it == renames.end() ? name : it->second;
    }

    ir::Expr visit(const ir::Var *node) override {
        auto it = renames.find(node->name);
        if (it == renames.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

    std::pair<ir::WriteLoc, bool>
    mutate_writeloc(const ir::WriteLoc &loc) override {
        auto [renamed, not_changed] = Mutator::mutate_writeloc(loc);
        auto it = renames.find(renamed.base);
        if (it == renames.end()) {
            return {renamed, not_changed};
        }
        renamed.base = it->second;
        return {renamed, false};
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        auto [loc, _] = mutate_writeloc(node->loc);
        return ir::LetStmt::make(std::move(loc), mutate(node->value));
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        auto [loc, _] = mutate_writeloc(node->loc);
        if (node->value.defined()) {
            return ir::Allocate::make(std::move(loc), mutate(node->value),
                                      node->memory, node->unaliased);
        }
        return ir::Allocate::make(std::move(loc), node->memory);
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
            const StatementFunctions &statement_functions, std::string self,
            size_t &counter)
        : functions(functions), function_to_expr(function_to_expr),
          statement_functions(statement_functions), self(std::move(self)),
          counter(counter) {}

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
            // with the arguments put in for the parameters.
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
            ir::Expr body = replace(repls, it->second);
            // The expression may itself call something inlinable. Bounded,
            // since a chain of expression functions that call each other
            // round in a circle would otherwise never end.
            if (depth < kMaxDepth) {
                depth++;
                body = mutate(body);
                depth--;
            }
            return body;
        }

        if (auto it = statement_functions.find(function_name);
            it != statement_functions.end() && hoistable &&
            function_name != self && depth < kMaxDepth &&
            still_small(*it->second)) {
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
                it != statement_functions.end() && v->name != self &&
                depth < kMaxDepth && still_small(*it->second)) {
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
    static constexpr size_t kMaxDepth = 8;

    const ir::FuncMap &functions;
    const std::unordered_map<std::string, ir::Expr> &function_to_expr;
    const StatementFunctions &statement_functions;
    const std::string self;
    // Shared across the functions of a program, so that the names made here
    // are unique in every one of them.
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

    // Whether `f` is still worth copying. It was small when the candidates
    // were chosen, but its callees have been inlined into it since -- it is
    // processed before its callers -- and a body that has grown past the
    // bound is a call again, or every copy would carry the whole chain
    // beneath it and the program would grow with the depth of its calls.
    static bool still_small(const ir::Function &f) {
        if (f.is_inlined() || f.is_always_inlined()) {
            return true;
        }
        BodyShape shape(f.name);
        f.body.accept(&shape);
        return shape.statements <= kMaxInlinedStatements;
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
        const std::string tag = "$i" + std::to_string(counter++);

        BodyShape shape(f.name);
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

        Renamer renamer(renames);
        ir::Stmt body = renamer.mutate(f.body);
        if (!substitutions.empty()) {
            body = replace(substitutions, body);
        }
        // Whatever the copy calls is inlined in turn, where it can be.
        depth++;
        body = mutate(body);
        depth--;
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
    // A function that is one expression is replaced by that expression at
    // every call, whatever its size.
    std::unordered_map<std::string, ir::Expr> function_to_expr;
    // A function of a few statements with returns that nest is copied to its
    // call sites, its returns becoming assignments to a result variable.
    StatementFunctions statement_functions;
    for (const auto &[name, func] : funcs) {
        if (func->is_kernel() || func->is_vectorized() || func->is_noinline()) {
            continue;
        }
        if (!func->ret_type.is<ir::Void_t>()) {
            if (const auto *body = func->body.as<ir::Return>()) {
                internal_assert(body->value.defined());
                function_to_expr[name] = body->value;
                continue;
            }
        }
        // A mutating argument is a reference to the caller's variable, and
        // binding it to a fresh name would copy it instead.
        bool mutating = false;
        for (const ir::Function::Argument &arg : func->args) {
            mutating = mutating || arg.mutating;
        }
        if (mutating) {
            continue;
        }
        BodyShape shape(name);
        func->body.accept(&shape);
        if (!shape.plain || shape.recursive) {
            continue;
        }
        // A body with mutable locals of its own is left as a call unless the
        // program asks, or this is the second round (see Opt/Inline.h).
        // Copied, its state is a set of variables at every call site, and CSE
        // sees through none of them: two copies of the same call are two
        // computations for good. As a call it is one value, and two calls
        // with the same arguments are one -- which is the point of inlining
        // the small functions that make the call. `aabb_span` stays a call
        // through the first round for exactly this reason; `intersects` and
        // `distmin` over an AABB, which each call it, are copied, CSE leaves
        // the traversal asking the box once, and the second round copies that
        // one call in.
        // `[[inline]]` asks for a copy whatever the body looks like.
        const bool asked = func->is_inlined() || func->is_always_inlined();
        if (!asked && !with_locals && shape.allocates) {
            continue;
        }
        if (!asked && shape.statements > kMaxInlinedStatements) {
            continue;
        }
        if (!nestable(flatten(func->body))) {
            continue;
        }
        statement_functions[name] = func;
    }

    // Callees before callers, so that a body copied into a caller has
    // already had its own calls inlined. A cycle gets some order; nothing in
    // one is inlinable anyway.
    const std::vector<std::string> order =
        lower::func_topological_order(funcs);
    size_t counter = 0;
    for (const std::string &name : order) {
        auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue;
        }
        Inliner inliner(funcs, function_to_expr, statement_functions, name,
                        counter);
        it->second->body = inliner.mutate(it->second->body);
    }

    return funcs;
}

} // namespace opt
} // namespace bonsai
