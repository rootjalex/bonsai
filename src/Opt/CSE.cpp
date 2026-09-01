#include "Opt/CSE.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Opt/DCE.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace bonsai {
namespace opt {

namespace {
// Prefix for a temporary variable.
static constexpr char T_PREFIX[] = "_t";
// For unique variable renaming.
static int64_t counter = 0;

std::string fresh_name() { return T_PREFIX + std::to_string(counter++); }

// Stack of mutable variable names for a given function.
using MutableVariableStack = ir::SetStack<std::string>;

// A set of expressions using IR comparison rather than pointer comparison.
using ExprSet = std::set<ir::Expr, ir::ExprLessThan>;

// For checking whether an expression can legally be CSE'd. This is used in two
// different classes (rename analysis and LVN), so we leave it here.
struct CseLegalChecker : public ir::Visitor {
    CseLegalChecker(const std::set<std::string> &side_effect_functions,
                    const std::set<std::string> &mutable_arguments,
                    const MutableVariableStack &mutable_variables)
        : side_effect_functions(side_effect_functions),
          mutable_arguments(mutable_arguments),
          mutable_variables(mutable_variables) {}

    void visit(const ir::Var *node) override {
        // We cannot CSE with mutable variables since mutations may have
        // occurred between. In the future, we can rename mutated
        // variables to overcome this. For example,
        // a = x + 1; #1
        // x += 1;
        // b = x + 1; #2 (same expression, but x has changed value)
        is_legal &= !mutable_variables.contains(node->name) &&
                    !mutable_arguments.contains(node->name);
    }

    void visit(const ir::Call *node) override {
        // We cannot CSE with side effecting function calls. For
        // example,
        // a = print_and_return(x); #1
        // b = print_and_return(x); #2 (same, but would only print once)
        const auto *v = node->func.as<ir::Var>();
        if (v == nullptr) {
            return;
        }
        is_legal &= !side_effect_functions.contains(v->name);
    }

    bool is_legal = true;

  private:
    const std::set<std::string> &side_effect_functions;
    const std::set<std::string> &mutable_arguments;
    const MutableVariableStack &mutable_variables;
};

// Retrieves all variables that have been seen more than once.
class RenameAnalysis : public ir::Visitor {
  public:
    RenameAnalysis(const std::set<std::string> &side_effect_functions,
                   const std::set<std::string> &mutable_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_arguments(mutable_arguments) {}

    void visit(const ir::LetStmt *node) override {
        var_to_e.add_to_frame(node->loc.base, node->value);
    }

    void visit(const ir::BinOp *node) override {
        update_count(node);
        substitute(node->a).accept(this);
        substitute(node->b).accept(this);
    }

    void visit(const ir::Select *node) override {
        update_count(node);
        substitute(node->cond).accept(this);
        substitute(node->tvalue).accept(this);
        substitute(node->fvalue).accept(this);
    }

    void visit(const ir::Access *node) override {
        update_count(node);
        substitute(node->value).accept(this);
    }

    void visit(const ir::Build *node) override {
        update_count(node);
        for (const ir::Expr &v : node->values) {
            substitute(v).accept(this);
        }
    }

    void visit(const ir::Cast *node) override {
        update_count(node);
        substitute(node->value).accept(this);
    }

    void visit(const ir::Extract *node) override {
        update_count(node);
        substitute(node->vec).accept(this);
        substitute(node->idx).accept(this);
    }

    void visit(const ir::Call *node) override {
        update_count(node);
        for (const ir::Expr &arg : node->args) {
            substitute(arg).accept(this);
        }
    }

    void visit(const ir::CallStmt *node) override {
        for (const ir::Expr &arg : node->args) {
            substitute(arg).accept(this);
        }
    }

    void visit(const ir::Intrinsic *node) override {
        update_count(node);
        for (const ir::Expr &arg : node->args) {
            substitute(arg).accept(this);
        }
    }

    void visit(const ir::IfElse *node) override {
        node->cond.accept(this);
        push_frame();
        node->then_body.accept(this);
        pop_frame();

        push_frame();
        if (node->else_body.defined()) {
            node->else_body.accept(this);
        }
        pop_frame();
    }

    void visit(const ir::Store *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        substitute(node->value).accept(this);
    }

    void visit(const ir::Accumulate *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        substitute(node->value).accept(this);
    }

    void visit(const ir::ForEach *node) override {
        push_frame();
        node->iter.accept(this);
        node->body.accept(this);
        pop_frame();
    }

    void visit(const ir::ForAll *node) override {
        push_frame();
        const ir::ForAll::Slice &slice = node->slice;
        slice.begin.accept(this);
        slice.end.accept(this);
        slice.stride.accept(this);
        node->body.accept(this);
        pop_frame();
    }

    void visit(const ir::DoWhile *node) override {
        push_frame();
        node->body.accept(this);
        node->cond.accept(this);
        pop_frame();
    }

    void visit(const ir::While *node) override {
        push_frame();
        node->cond.accept(this);
        node->body.accept(this);
        pop_frame();
    }

    void visit(const ir::YieldFrom *node) override { node->value.accept(this); }

    // Yields a set of expressions that have been seen more than 1 time.
    ExprSet post_process() {
        ExprSet set;
        for (const auto &[e, count] : expression_count) {
            if (count > 1) {
                set.insert(e);
            }
        }
        return set;
    }

  private:
    void push_frame() { var_to_e.push_frame(); }
    void pop_frame() { var_to_e.pop_frame(); }

    void update_count(ir::Expr e) {
        if (e.is<ir::Var>() || is_const(e) || !is_cse_legal(e)) {
            return;
        }
        ++expression_count[e];
    }
    // The count of each expression.
    std::map<ir::Expr, int64_t, ir::ExprLessThan> expression_count;

    ir::Expr substitute(ir::Expr e) {
        const auto *v = e.as<ir::Var>();
        if (v == nullptr) {
            return e;
        }
        if (std::optional<ir::Expr> f = var_to_e.from_frames(v->name)) {
            return substitute(*f);
        }
        return e;
    }
    // variable -> expression
    ir::MapStack<std::string, ir::Expr> var_to_e;

    // A list of variable names that should stop CSE if found within an
    // expression. This includes mutable assignments, and references to
    // allocations.
    MutableVariableStack mutable_variables;
    // A list of functions that may have side effects. This is "whole program
    // analysis", so doesn't require a frame stack.
    const std::set<std::string> &side_effect_functions;
    // A list of function arguments that are mutable. This is "whole function
    // analysis", so doesn't require a frame stack.
    const std::set<std::string> &mutable_arguments;
    // Returns whether this is supported in our simplistic variant of CSE.
    bool is_cse_legal(ir::Expr e) {
        CseLegalChecker checker(side_effect_functions, mutable_arguments,
                                mutable_variables);
        e.accept(&checker);
        return checker.is_legal;
    }
};

struct IfElseHandler : public ir::Mutator {

    std::set<ir::Expr, ir::ExprLessThan> get_evaled(const ir::Expr &expr) {
        struct AbsolutelyEvaluatedGetter : ir::Visitor {
            std::set<ir::Expr, ir::ExprLessThan> expressions;

            void visit(const ir::BinOp *node) override {
                switch (node->op) {
                // Logical variables cannot safely emit temporary variables.
                // We could eventually special case for the left most operand of
                // the logical operation, which will always execute.
                case ir::BinOp::OpType::LAnd:
                case ir::BinOp::OpType::LOr:
                    // node->a's expressions are *always* evaluated,
                    expressions.insert(node->a);
                    node->a.accept(this);
                    return;
                default:
                    expressions.insert(node); // can duplicate for nested BinOps
                    expressions.insert(node->a);
                    expressions.insert(node->b);
                    node->a.accept(this);
                    node->b.accept(this);
                    return;
                }
            }

            // Adopt Halide semantics of both sides are always evaluated.
            void visit(const ir::Select *node) override {
                expressions.insert(node->cond);
                node->cond.accept(this);
                node->tvalue.accept(this);
                node->fvalue.accept(this);
            }

            void visit(const ir::Intrinsic *node) override {
                expressions.insert(node);
                for (const auto &expr : node->args) {
                    expressions.insert(expr);
                    expr.accept(this);
                }
            }
        };

        AbsolutelyEvaluatedGetter getter;
        expr.accept(&getter);
        return getter.expressions;
    }

    std::map<ir::Expr, int64_t, ir::ExprLessThan>
    get_freq(const ir::Expr &expr) {
        struct GetFrequencies : ir::Mutator {
            std::map<ir::Expr, int64_t, ir::ExprLessThan> frequencies;

            ir::Expr mutate(const ir::Expr &expr) override {
                frequencies[expr]++;
                return ir::Mutator::mutate(expr);
            }
        };

        GetFrequencies getter;
        getter.mutate(expr);
        return getter.frequencies;
    }

    bool should_rename(const ir::Expr &e) {
        return !e.is<ir::Var>() && !is_const(e);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        const ir::BinOp *bop = node->cond.as<ir::BinOp>();
        if (!bop ||
            !(bop->op == ir::BinOp::LAnd || bop->op == ir::BinOp::LOr)) {
            return ir::Mutator::visit(node);
        }

        auto exprs = get_evaled(node->cond);
        if (exprs.empty()) {
            return ir::Mutator::visit(node);
        }

        auto freqs = get_freq(node->cond);

        std::vector<ir::Expr> candidates;
        candidates.reserve(exprs.size());
        // TODO: only legal without side-effects.

        for (const auto &c : exprs) {
            if (should_rename(c) && freqs[c] > 1) {
                candidates.push_back(c);
            }
        }

        if (candidates.empty()) {
            return ir::Mutator::visit(node);
        }

        // Sort candidates by size (smaller first) to avoid overlapping
        // replacements.
        // TODO: cache sizes, this could be expensive.
        std::sort(candidates.begin(), candidates.end(),
                  [&](const ir::Expr &a, const ir::Expr &b) {
                      return ast_size(a) < ast_size(b);
                  });

        std::map<ir::Expr, ir::Expr, ir::ExprLessThan> replacements;
        std::vector<ir::Stmt> stmts;

        for (const ir::Expr &e : candidates) {
            ir::Expr repl = replace(replacements, e);
            auto name = fresh_name();
            ir::WriteLoc location(name, repl.type());
            stmts.push_back(ir::LetStmt::make(location, std::move(repl)));
            replacements[e] = ir::Var::make(e.type(), name);
        }

        ir::Expr new_cond = replace(replacements, node->cond);

        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);

        stmts.push_back(ir::IfElse::make(std::move(new_cond), std::move(th),
                                         std::move(el)));
        return ir::Sequence::make(std::move(stmts));
    }
};

// Gives an expression its own variable name if it fits certain criteria. For
// example,
//   g(foo(i), bar(j));
//   ->
//   let _t0 = foo(i) in
//   let _t1 = bar(j) in
//   g(_t0, _t1);
struct Rename : public ir::Mutator {
    Rename(const ExprSet &to_rename) : to_rename(to_rename) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        if (node->loc.base == "tid") {
            // TODO(cgyurgyik): do *not* rename TID for now; the CUDA backend
            // assumes this name!
            return node;
        }
        if (is_simple(node->value)) {
            // Non-compounding statements can just use the O.G. variable name.
            return node;
        }
        return make(ir::LetStmt::make(node->loc, mutate(node->value)));
    }
    ir::Stmt visit(const ir::Allocate *node) override {
        return make(
            ir::Allocate::make(node->loc, mutate(node->value), node->memory));
    }
    ir::Stmt visit(const ir::Store *node) override {
        return make(ir::Store::make(node->loc, mutate(node->value)));
    }
    ir::Stmt visit(const ir::Accumulate *node) override {
        return make(ir::Accumulate::make(node->loc, node->op,
                                         mutate(node->value), node->atomic));
    }
    ir::Stmt visit(const ir::Return *node) override {
        if (!node->value.defined()) {
            return node;
        }
        return make(ir::Return::make(mutate(node->value)));
    }
    ir::Stmt visit(const ir::Print *node) override {
        std::vector<ir::Expr> values;
        for (int i = 0, e = node->args.size(); i < e; ++i) {
            values.push_back(mutate(node->args[i]));
        }
        return make(ir::Print::make(std::move(values)));
    }
    ir::Stmt visit(const ir::CallStmt *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        return make(ir::CallStmt::make(node->func, std::move(args)));
    }
    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);
        // This should be lowered after so that any expressions generated are
        // not placed in the `then` or `else` body.
        ir::Expr cond = mutate(node->cond);
        return make(
            ir::IfElse::make(std::move(cond), std::move(th), std::move(el)));
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        ir::Expr iter = mutate(node->iter);
        ir::Stmt body = mutate(node->body);
        return make(
            ir::ForEach::make(node->name, std::move(iter), std::move(body)));
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ir::Stmt body = mutate(node->body);
        // This should be lowered after so that any expressions generated are
        // not placed in the `body`.
        ir::ForAll::Slice slice = ir::ForAll::Slice{
            .begin = mutate(node->slice.begin),
            .end = mutate(node->slice.end),
            .stride = mutate(node->slice.stride),
        };
        return make(
            ir::ForAll::make(node->index, std::move(slice), std::move(body)));
    }

    ir::Stmt visit(const ir::DoWhile *node) override {
        ir::Stmt body = mutate(node->body);
        ir::Expr cond = mutate(node->cond);
        return make(ir::DoWhile::make(std::move(body), std::move(cond)));
    }

    ir::Stmt visit(const ir::While *node) override {
        ir::Expr cond = mutate(node->cond);
        ir::Stmt body = mutate(node->body);
        return make(ir::While::make(std::move(cond), std::move(body)));
    }

    ir::Stmt visit(const ir::YieldFrom *node) override {
        return make(ir::YieldFrom::make(mutate(node->value)));
    }

    ir::Expr visit(const ir::BinOp *node) override {
        switch (node->op) {
        // Logical variables cannot safely emit temporary variables.
        // We could eventually special case for the left most operand of the
        // logical operation, which will always execute.
        case ir::BinOp::OpType::LAnd:
        case ir::BinOp::OpType::LOr:
            return node;
        default:
            const bool rename = should_rename(node);
            ir::Expr a = mutate(node->a);
            ir::Expr b = mutate(node->b);
            internal_assert(a.defined() && b.defined());
            ir::Expr op = ir::BinOp::make(node->op, std::move(a), std::move(b));
            if (!rename) {
                return op;
            }
            ir::WriteLoc location(fresh_name(), node->type);
            stmts.push_back(ir::LetStmt::make(location, std::move(op)));
            return ir::Var::make(node->type, location.base);
        }
    }

    ir::Expr visit(const ir::Intrinsic *node) override {
        const bool rename = should_rename(node);
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        ir::Expr intrinsic = ir::Intrinsic::make(node->op, std::move(args));
        if (!rename) {
            return intrinsic;
        }
        ir::WriteLoc location(fresh_name(), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(intrinsic)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Access *node) override {
        const bool rename = should_rename(node);
        ir::Expr value = mutate(node->value);
        ir::Expr access = ir::Access::make(node->field, std::move(value));
        if (!rename) {
            return access;
        }
        ir::WriteLoc location(fresh_name(), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(access)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Build *node) override {
        const bool rename = should_rename(node);
        std::vector<ir::Expr> values;
        for (const ir::Expr &value : node->values) {
            values.push_back(mutate(value));
        }
        ir::Expr build = ir::Build::make(node->type, std::move(values));
        if (!rename) {
            return build;
        }
        ir::WriteLoc location(fresh_name(), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(build)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Cast *node) override {
        const bool rename = should_rename(node);
        ir::Expr value = mutate(node->value);
        ir::Expr cast =
            ir::Cast::make(node->type, std::move(value), node->mode);
        if (!rename) {
            return cast;
        }
        ir::WriteLoc location(fresh_name(), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(cast)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Extract *node) override {
        const bool rename = should_rename(node);
        ir::Expr vec = mutate(node->vec);
        ir::Expr idx = mutate(node->idx);
        ir::Expr extract = ir::Extract::make(std::move(vec), std::move(idx));
        if (!rename) {
            return extract;
        }
        ir::WriteLoc location(fresh_name(), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(extract)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Call *node) override {
        const bool rename = should_rename(node);
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        ir::Expr call = ir::Call::make(node->func, std::move(args));
        if (!rename) {
            return call;
        }
        ir::WriteLoc loc(fresh_name(), node->type);
        stmts.push_back(ir::LetStmt::make(loc, std::move(call)));
        return ir::Var::make(node->type, loc.base);
    }

    // Do not CSE out of a lambda
    ir::Expr visit(const ir::Lambda *node) override { return node; }

  private:
    bool is_trivial(const ir::Expr &e) {
        return e.is<ir::Var>() || is_const(e);
    }
    // Returns whether this expression is comprised of just variables and
    // constants, and thus does not need to be renamed.
    bool is_simple(const ir::Expr &e) {
        if (is_trivial(e)) {
            return true;
        }
        if (const auto *unop = e.as<ir::UnOp>()) {
            return is_trivial(unop->a);
        }
        if (const auto *cast = e.as<ir::Cast>()) {
            return is_trivial(cast->value);
        }
        if (const auto *intrinsic = e.as<ir::Intrinsic>()) {
            return std::all_of(
                intrinsic->args.begin(), intrinsic->args.end(),
                [&](const ir::Expr &v) { return is_trivial(v); });
        }
        if (const auto *binop = e.as<ir::BinOp>()) {
            return is_trivial(binop->a) && is_trivial(binop->b);
        }
        if (const auto *access = e.as<ir::Access>()) {
            return is_trivial(access->value);
        }
        if (const auto *extract = e.as<ir::Extract>()) {
            return is_trivial(extract->vec) && is_trivial(extract->idx);
        }
        if (const auto *call = e.as<ir::Call>()) {
            return std::all_of(
                call->args.begin(), call->args.end(),
                [&](const ir::Expr &v) { return is_trivial(v); });
        }
        if (const auto *build = e.as<ir::Build>()) {
            return std::all_of(
                build->values.begin(), build->values.end(),
                [&](const ir::Expr &v) { return is_trivial(v); });
        }
        return false;
    }
    // Whether we should give this sub-expression its own variable.
    bool should_rename(const ir::Expr &e) {
        return !e.is<ir::Var>() && to_rename.contains(e) && !is_const(e);
    }
    // A set of expressions that should be renamed in this pass.
    const ExprSet &to_rename;
    // A list of intermediate statements generated for subexpressions.
    std::vector<ir::Stmt> stmts;

    // Pushes this `statement` onto the list of generated statements and returns
    // a sequence.
    ir::Stmt make(ir::Stmt statement) {
        stmts.push_back(std::move(statement));
        ir::Stmt sequence = ir::Sequence::make(std::move(stmts));
        stmts.clear();
        return sequence;
    }
};

// A variant of local value numbering [1], where we explore scopes rather than
// basic blocks. Expressions are fully substituted before given their value
// number.
//
//  1: https://en.wikipedia.org/wiki/Value_numbering
class LVN : public ir::Mutator {
  public:
    LVN(const std::set<std::string> &side_effect_functions,
        const std::set<std::string> &mutable_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_arguments(mutable_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = mutate(node->value);
        const std::optional<int64_t> vn = get_value_number(value);
        if (!vn.has_value()) {
            return node; // We cannot legally CSE this.
        }
        if (std::optional<std::string> name = vn_to_var.from_frames(*vn)) {
            ir::Expr v = ir::Var::make(value.type(), *name);
            var_to_e.add_to_frame(node->loc.base, v);
            return ir::LetStmt::make(node->loc, std::move(v));
        }
        var_to_e.add_to_frame(node->loc.base, std::move(value));
        vn_to_var.add_to_frame(*vn, node->loc.base);
        return node;
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return ir::Store::make(node->loc, mutate(node->value));
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return ir::Allocate::make(node->loc, mutate(node->value), node->memory);
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return ir::Accumulate::make(node->loc, node->op, mutate(node->value),
                                    node->atomic);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        push_frame();
        ir::Expr cond = mutate(node->cond);
        ir::Stmt th = mutate(node->then_body);
        pop_frame();
        push_frame();
        ir::Stmt el = mutate(node->else_body);
        pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        push_frame();
        ir::Expr iter = mutate(node->iter);
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::ForEach::make(node->name, std::move(iter), std::move(body));
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ir::ForAll::Slice slice = ir::ForAll::Slice{
            .begin = mutate(node->slice.begin),
            .end = mutate(node->slice.end),
            .stride = mutate(node->slice.stride),
        };
        push_frame();
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::ForAll::make(node->index, std::move(slice), std::move(body));
    }

    ir::Stmt visit(const ir::DoWhile *node) override {
        push_frame();
        ir::Stmt body = mutate(node->body);
        ir::Expr cond = mutate(node->cond);
        pop_frame();
        return ir::DoWhile::make(std::move(body), std::move(cond));
    }

    ir::Stmt visit(const ir::While *node) override {
        push_frame();
        ir::Expr cond = mutate(node->cond);
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::While::make(std::move(cond), std::move(body));
    }

    ir::Stmt visit(const ir::YieldFrom *node) override {
        return ir::YieldFrom::make(mutate(node->value));
    }

    ir::Expr visit(const ir::Call *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(cse(arg));
        }
        return ir::Call::make(node->func, std::move(args));
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(cse(arg));
        }
        return ir::CallStmt::make(node->func, std::move(args));
    }

    ir::Expr visit(const ir::BinOp *node) override {
        return ir::BinOp::make(node->op, cse(node->a), cse(node->b));
    }

    ir::Expr visit(const ir::Access *node) override {
        return ir::Access::make(node->field, cse(node->value));
    }

    ir::Expr visit(const ir::Build *node) override {
        std::vector<ir::Expr> values;
        for (const ir::Expr &value : node->values) {
            values.push_back(cse(value));
        }
        return ir::Build::make(node->type, std::move(values));
    }

    ir::Expr visit(const ir::Extract *node) override {
        return ir::Extract::make(cse(node->vec), cse(node->idx));
    }

    ir::Expr visit(const ir::Cast *node) override {
        return ir::Cast::make(node->type, cse(node->value), node->mode);
    }

    // TODO(cgyurgyik): Add LVN for bodies of lambda expressions.
    ir::Expr visit(const ir::Lambda *node) override { return node; }

  private:
    // A list of functions that may have side effects. This is "whole program
    // analysis", so doesn't require a frame stack.
    const std::set<std::string> &side_effect_functions;
    // A list of function arguments that are mutable. This is "whole function
    // analysis", so doesn't require a frame stack.
    const std::set<std::string> &mutable_arguments;
    // A list of variable names that should stop CSE if found within an
    // expression. This includes mutable assignments, and references to
    // allocations.
    MutableVariableStack mutable_variables;

    // The "local value number" for expressions in this function. We just use
    // the same number in conjunction with a stack to ensure the values remain
    // truly local to their scope.
    int64_t value_number = 0;
    // expression -> value number
    ir::MapStack<ir::Expr, int64_t, ir::ExprLessThan> e_to_vn;
    // variable -> expression
    ir::MapStack<std::string, ir::Expr> var_to_e;
    // value number -> variable (for subsequent replacement)
    ir::MapStack<int64_t, std::string> vn_to_var;

    void push_frame() {
        mutable_variables.push_frame();
        e_to_vn.push_frame();
        var_to_e.push_frame();
        vn_to_var.push_frame();
    }
    void pop_frame() {
        mutable_variables.pop_frame();
        e_to_vn.pop_frame();
        var_to_e.pop_frame();
        vn_to_var.pop_frame();
    }

    // Returns the lcoal value number for expression `e` if this is legal to
    // CSE, and {} otherwise.
    std::optional<int64_t> get_value_number(ir::Expr e) {
        if (!is_cse_legal(e)) {
            return {};
        }
        e = substitute(e);
        if (std::optional<int64_t> vn = e_to_vn.from_frames(e)) {
            return *vn;
        }
        const int64_t vn = value_number++;
        e_to_vn.add_to_frame(e, vn);
        return vn;
    }

    // Finds a common subexpression replacement for `e`, or returns the original
    // expression otherwise.
    ir::Expr cse(ir::Expr e) {
        std::optional<int64_t> vn = e_to_vn.from_frames(substitute(e));
        if (!vn.has_value()) {
            return e;
        }
        std::optional<std::string> name = vn_to_var.from_frames(*vn);
        if (!name.has_value()) {
            return e;
        }
        return ir::Var::make(e.type(), std::move(*name));
    }

    // For simplicity, we replace intermediate variables with their value
    // expression. There might be a better way to do this, but this is how I've
    // implemented LVN for SSA, where use-def chains are immediately available.
    // To illustrate this with an example, `a` and `c` would not have the same
    // value number unless substitution occurred:
    //  a = x + 42   #1
    //  b = 42       #2
    //  c = x + b    #3  <-- should be #2
    ir::Expr substitute(ir::Expr e) {
        if (const auto *b = e.as<ir::Build>()) {
            std::vector<ir::Expr> values;
            for (const ir::Expr &v : b->values) {
                values.push_back(substitute(v));
            }
            return ir::Build::make(b->type, std::move(values));
        }
        if (const auto *c = e.as<ir::Call>()) {
            std::vector<ir::Expr> args;
            for (const ir::Expr &a : c->args) {
                args.push_back(substitute(a));
            }
            return ir::Call::make(c->func, std::move(args));
        }
        if (const auto *o = e.as<ir::BinOp>()) {
            return ir::BinOp::make(o->op, substitute(o->a), substitute(o->b));
        }
        if (const auto *op = e.as<ir::Intrinsic>()) {
            std::vector<ir::Expr> args = op->args;
            for (int i = 0, e = args.size(); i < e; ++i) {
                args[i] = substitute(args[i]);
            }
            return ir::Intrinsic::make(op->op, std::move(args));
        }
        if (const auto *op = e.as<ir::Access>()) {
            return ir::Access::make(op->field, substitute(op->value));
        }
        if (const auto *op = e.as<ir::Cast>()) {
            return ir::Cast::make(op->type, substitute(op->value), op->mode);
        }
        if (const auto *op = e.as<ir::Extract>()) {
            return ir::Extract::make(substitute(op->vec), substitute(op->idx));
        }
        if (const auto *v = e.as<ir::Var>()) {
            if (std::optional<ir::Expr> f = var_to_e.from_frames(v->name)) {
                return substitute(*f);
            }
        }
        return e;
    }

    // Returns whether this is supported in our simplistic variant of CSE.
    bool is_cse_legal(ir::Expr e) {
        CseLegalChecker checker(side_effect_functions, mutable_arguments,
                                mutable_variables);
        e.accept(&checker);
        return checker.is_legal;
    }
};

// Implements copy propagation, e.g.,
//  let _t0 = a + b in
//  let _t1 = a + b in
//  in f(_t0, _t1)
//  ->
//  let _t0 = a + b in
//  let _t1 = _t0 in
//  in f(_t0, _t0)
class CopyPropagation : public ir::Mutator {
  public:
    CopyPropagation(const std::set<std::string> &mutable_arguments)
        : mutable_arguments(mutable_arguments) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
        ir::Expr value = mutate(node->value);
        internal_assert(value.defined());
        const auto *variable = value.as<ir::Var>();
        ir::WriteLoc lhs = node->loc;
        if (variable == nullptr) {
            // This is an expression, do nothing.
            return ir::LetStmt::make(std::move(lhs), std::move(value));
        }
        internal_assert(!lhs_to_rhs.contains(lhs.base));
        lhs_to_rhs.add_to_frame(lhs.base, variable->name);
        return ir::LetStmt::make(std::move(lhs), std::move(value));
    }

    ir::Expr visit(const ir::Var *node) override {
        if (mutable_arguments.contains(node->name)) {
            return node;
        }
        std::optional<std::string> name = lhs_to_rhs.from_frames(node->name);
        if (!name.has_value()) {
            return node;
        }
        return ir::Var::make(node->type, *name);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = mutate(node->cond);
        push_frame();
        ir::Stmt th = mutate(node->then_body);
        pop_frame();
        push_frame();
        ir::Stmt el = mutate(node->else_body);
        pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        push_frame();
        ir::Expr iter = mutate(node->iter);
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::ForEach::make(node->name, std::move(iter), std::move(body));
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ir::ForAll::Slice slice = ir::ForAll::Slice{
            .begin = mutate(node->slice.begin),
            .end = mutate(node->slice.end),
            .stride = mutate(node->slice.stride),
        };
        push_frame();
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::ForAll::make(node->index, std::move(slice), std::move(body));
    }
    ir::Stmt visit(const ir::DoWhile *node) override {
        push_frame();
        ir::Stmt body = mutate(node->body);
        ir::Expr cond = mutate(node->cond);
        pop_frame();
        return ir::DoWhile::make(std::move(body), std::move(cond));
    }

    ir::Stmt visit(const ir::While *node) override {
        push_frame();
        ir::Expr cond = mutate(node->cond);
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::While::make(std::move(cond), std::move(body));
    }

    // Cannot propagate copies through mutable variables.
    ir::Stmt visit(const ir::Store *node) override { return node; }
    ir::Stmt visit(const ir::Allocate *node) override { return node; }
    ir::Stmt visit(const ir::Accumulate *node) override { return node; }

  private:
    void push_frame() { lhs_to_rhs.push_frame(); }
    void pop_frame() { lhs_to_rhs.pop_frame(); }

    // A list of mutable arguments for this function.
    const std::set<std::string> &mutable_arguments;

    // A mapping from the lhs to rhs assignment of variable names, e.g.,
    //   x: i32 = y; // {x, y}
    //   z: i32 = x; // {z, x}
    ir::MapStack<std::string, std::string> lhs_to_rhs;
};

// Substitute single-use temporary variables for their values, e.g.,
// let _t0 = x.y in
// use(_t0)
// ->
// use(x.y)
ir::Stmt substitute_temporaries(ir::Stmt body) {
    // Count single uses of (compiler-generated) temporary variables created
    // during the CSE process. This does not require a frame stack because
    // every temporary variable is given a unique name.
    struct CountSingleUses : public ir::Visitor {
        void visit(const ir::Var *node) override { count(node->name); }

        std::unordered_set<std::string> post_process() {
            std::unordered_set<std::string> set;
            for (const auto &[name, count] : variable_to_count) {
                if (count == 1) {
                    set.insert(name);
                }
            }
            return set;
        }

      private:
        void count(std::string name) {
            if (!name.starts_with(T_PREFIX)) {
                return;
            }
            ++variable_to_count[name];
        }
        std::unordered_map<std::string, int64_t> variable_to_count;
    };
    CountSingleUses count;
    body.accept(&count);
    std::unordered_set<std::string> single_use_variables = count.post_process();

    // Substitute single uses of temporary values with its original value.
    struct Substitute : public ir::Mutator {
        Substitute(std::unordered_set<std::string> single_use_variables)
            : single_use_variables(std::move(single_use_variables)) {}

        ir::Stmt visit(const ir::LetStmt *node) override {
            const std::string &base = node->loc.base;
            if (!single_use_variables.contains(base)) {
                return ir::Mutator::visit(node);
            }
            variable_to_expr[base] = mutate(node->value);
            return ir::Mutator::visit(node);
        }
        ir::Expr visit(const ir::Var *node) override {
            auto it = variable_to_expr.find(node->name);
            if (it == variable_to_expr.end()) {
                return node;
            }
            return it->second;
        }
        std::unordered_set<std::string> single_use_variables;
        std::unordered_map<std::string, ir::Expr> variable_to_expr;
    };
    return Substitute{single_use_variables}.mutate(std::move(body));
}

} // namespace

ir::FuncMap CSE::run(ir::FuncMap funcs, const CompilerOptions &options) const {
    std::set<std::string> side_effect_functions = find_side_effects(funcs);
    for (auto &[name, func] : funcs) {
        // TODO(cgyurgyik): there are some compiler breakages here; need to
        // investigate. Not a high priority since we're not focused on tree
        // building and I don't think CSE will help much here.
        if (name.starts_with("rec_build") || name.starts_with("build") ||
            name.starts_with("rec_count") || name.starts_with("count")) {
            continue;
        }

        // Handle IfElse statements
        // func->body = IfElseHandler().mutate(func->body);

        const std::set<std::string> &mutable_arguments = func->mutable_args();
        RenameAnalysis analysis(side_effect_functions, mutable_arguments);
        // Find expressions that have been seen > 1  times.
        func->body.accept(&analysis);
        ExprSet to_rename = analysis.post_process();

        // Give each of these expressions its own name.
        Rename rename(to_rename);
        func->body = rename.mutate(func->body);

        // Perform local value numbering, as well as the common subexpression
        // elimination.
        LVN lvn(side_effect_functions, mutable_arguments);
        func->body = lvn.mutate(func->body);

        // Propagate copies.
        CopyPropagation cp(mutable_arguments);
        func->body = cp.mutate(func->body);

        // Temporaries that appear once should be rewritten to their respective
        // value. We perform dead code elimination first to get rid of unused
        // references to a temporary variable.
        func->body = opt::dce(std::move(func->body), mutable_arguments,
                              side_effect_functions);
        func->body = substitute_temporaries(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
