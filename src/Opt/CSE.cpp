#include "Opt/CSE.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {
// Stack of mutable variable names for a given function.
using MutableVariableStack = ir::SetStack<std::string>;
using ExprSet = std::set<ir::Expr, ir::ExprLessThan>;

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
        // variables to overcome this. For example, a = x + 1; #1 x +=
        // 1; b = x + 1; #2 (same expression, but x has changed value)
        is_legal &= !mutable_variables.contains(node->name) &&
                    !mutable_arguments.contains(node->name);
    }

    void visit(const ir::Call *node) override {
        // We cannot CSE with side effecting function calls. For
        // example, a = print_and_return(x); #1 b = print_and_return(x);
        // #2 (same, but would only print once)
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
// TODO(cgyurgyik): This should be stack-based. This requires being able to
// mutate values in these stack data structures, which isn't possible yet.
//
// TODO(cgyurgyik): Still not working for variable substitution:
// a = x + y + 2; <--
// b = x + y;
// c = b + 2;     <--
class RenameAnalysis : public ir::Visitor {
  public:
    RenameAnalysis(const std::set<std::string> &side_effect_functions,
                   const std::set<std::string> &mutable_arguments)
        : side_effect_functions(side_effect_functions),
          mutable_arguments(mutable_arguments) {}

    void visit(const ir::LetStmt *node) override {
        if (node->value.is<ir::Access>()) {
            return;
        }
        var_to_e.add_to_frame(node->loc.base, node->value);
    }

    void visit(const ir::BinOp *node) override {
        update_count(node);
        ir::Expr a = substitute(node->a);
        a.accept(this);
        ir::Expr b = substitute(node->b);
        b.accept(this);
    }

    void visit(const ir::Access *node) override {
        update_count(node);
        ir::Expr value = substitute(node->value);
        value.accept(this);
    }

    void visit(const ir::Build *node) override {
        update_count(node);
        for (const ir::Expr &v : node->values) {
            ir::Expr value = substitute(v);
            value.accept(this);
        }
    }

    void visit(const ir::Cast *node) override {
        update_count(node);
        ir::Expr value = substitute(node->value);
        value.accept(this);
    }

    void visit(const ir::Extract *node) override {
        update_count(node);
        ir::Expr vec = substitute(node->vec);
        ir::Expr idx = substitute(node->idx);
        vec.accept(this);
        idx.accept(this);
    }

    void visit(const ir::Call *node) override {
        update_count(node);
        for (const ir::Expr &arg : node->args) {
            arg.accept(this);
        }
    }

    void visit(const ir::CallStmt *node) override {
        for (const ir::Expr &arg : node->args) {
            arg.accept(this);
        }
    }

    void visit(const ir::Intrinsic *node) override {
        update_count(node);
        for (const ir::Expr &arg : node->args) {
            arg.accept(this);
        }
    }

    void visit(const ir::IfElse *node) override {
        node->cond.accept(this);
        var_to_e.new_frame();
        node->then_body.accept(this);
        var_to_e.pop_frame();

        var_to_e.new_frame();
        if (node->else_body.defined()) {
            node->else_body.accept(this);
        }
        var_to_e.pop_frame();
    }

    void visit(const ir::Assign *node) override {
        if (node->mutating && !mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
    }

    void visit(const ir::Store *node) override {
        if (!mutable_variables.contains(node->name)) {
            mutable_variables.add_to_frame(node->name);
        }
    }

    void visit(const ir::Accumulate *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
    }

    void visit(const ir::ForEach *node) override {
        var_to_e.new_frame();
        node->iter.accept(this);
        node->body.accept(this);
        var_to_e.pop_frame();
    }

    void visit(const ir::ForAll *node) override {
        var_to_e.new_frame();
        const ir::ForAll::Slice &slice = node->slice;
        slice.begin.accept(this);
        slice.end.accept(this);
        slice.stride.accept(this);
        if (node->header.defined()) {
            node->header.accept(this);
        }
        node->body.accept(this);
        var_to_e.pop_frame();
    }

    void visit(const ir::DoWhile *node) override {
        var_to_e.new_frame();
        node->body.accept(this);
        node->cond.accept(this);
        var_to_e.pop_frame();
    }

    ExprSet post_process() {
        ExprSet es;
        for (const auto &[e, count] : expression_count) {
            if (count > 1) {
                es.insert(e);
            }
        }
        return es;
    }

  private:
    void update_count(ir::Expr e) {
        if (!is_cse_legal(e)) {
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
    // A list of variable names that should stop CSE if found within an
    // expression. This includes mutable assignments, and references to
    // allocations.
    // Returns whether this is supported in our simplistic variant of CSE.
    bool is_cse_legal(ir::Expr e) {
        CseLegalChecker checker(side_effect_functions, mutable_arguments,
                                mutable_variables);
        e.accept(&checker);
        return checker.is_legal;
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
        return make(ir::LetStmt::make(node->loc, mutate(node->value)));
    }
    ir::Stmt visit(const ir::Assign *node) override {
        return make(
            ir::Assign::make(node->loc, mutate(node->value), node->mutating));
    }
    ir::Stmt visit(const ir::Accumulate *node) override {
        return make(
            ir::Accumulate::make(node->loc, node->op, mutate(node->value)));
    }
    ir::Stmt visit(const ir::Return *node) override {
        if (!node->value.defined()) {
            return node;
        }
        return make(ir::Return::make(mutate(node->value)));
    }
    ir::Stmt visit(const ir::Print *node) override {
        ir::Expr value = mutate(node->value);
        return make(ir::Print::make(std::move(value)));
    }
    ir::Stmt visit(const ir::Store *node) override {
        ir::Expr value = mutate(node->value);
        ir::Expr index = mutate(node->index);
        return make(
            ir::Store::make(node->name, std::move(index), std::move(value)));
    }
    ir::Stmt visit(const ir::CallStmt *node) override {
        std::vector<ir::Expr> args;
        for (const ir::Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        return make(ir::CallStmt::make(node->func, std::move(args)));
    }

    ir::Expr visit(const ir::BinOp *node) override {
        bool rename = should_rename(node);
        ir::Expr a = mutate(node->a), b;
        switch (node->op) {
        // Logical variables cannot safely emit temporary variables.
        case ir::BinOp::OpType::LAnd:
        case ir::BinOp::OpType::LOr:
            if (const auto *inner = node->a.as<ir::BinOp>();
                inner && ((inner->op == ir::BinOp::OpType::LAnd) ||
                          (inner->op == ir::BinOp::OpType::LOr))) {
                return node;
            }
            b = node->b;
            break;
        default:
            b = mutate(node->b);
            break;
        }
        internal_assert(a.defined() && b.defined());
        // Check before nested renames occur.
        ir::Expr op = ir::BinOp::make(node->op, std::move(a), std::move(b));
        if (!rename) {
            return op;
        }
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(op)));
        return ir::Var::make(node->type, location.base);
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
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
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
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
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
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
        stmts.push_back(ir::LetStmt::make(location, std::move(build)));
        return ir::Var::make(node->type, location.base);
    }

    ir::Expr visit(const ir::Cast *node) override {
        const bool rename = should_rename(node);
        ir::Expr value = mutate(node->value);
        ir::Expr cast = ir::Cast::make(node->type, std::move(value));
        if (!rename) {
            return cast;
        }
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
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
        ir::WriteLoc location("_t" + std::to_string(counter++), node->type);
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
        ir::WriteLoc loc("_t" + std::to_string(counter++), node->type);
        stmts.push_back(ir::LetStmt::make(loc, std::move(call)));
        return ir::Var::make(node->type, loc.base);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Stmt th = mutate(node->then_body);
        ir::Stmt el = mutate(node->else_body);
        ir::Expr cond = mutate(node->cond);
        return make(
            ir::IfElse::make(std::move(cond), std::move(th), std::move(el)));
    }

    // Skip the body of a lambda expression.
    ir::Expr visit(const ir::Lambda *node) override { return node; }

    ir::Stmt visit(const ir::ForEach *node) override {
        ir::Expr iter = mutate(node->iter);
        ir::Stmt body = mutate(node->body);
        return make(
            ir::ForEach::make(node->name, std::move(iter), std::move(body)));
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ir::Stmt header = node->header;
        if (header.defined()) {
            header = mutate(header);
        }
        ir::Stmt body = mutate(node->body);
        ir::ForAll::Slice slice = ir::ForAll::Slice{
            .begin = mutate(node->slice.begin),
            .end = mutate(node->slice.end),
            .stride = mutate(node->slice.stride),
        };
        return make(ir::ForAll::make(node->index, std::move(header),
                                     std::move(slice), std::move(body)));
    }

    ir::Stmt visit(const ir::DoWhile *node) override {
        ir::Stmt body = mutate(node->body);
        ir::Expr cond = mutate(node->cond);
        return make(ir::DoWhile::make(std::move(body), std::move(cond)));
    }

  private:
    // Whether we should give this sub-expression its own variable.
    // TODO(cgyurgyik): What else?
    bool should_rename(const ir::Expr &e) {
        return !e.is<ir::Var>() && !is_const(e) && to_rename.contains(e);
    }
    const ExprSet &to_rename;
    // A list of intermediate statements generated for subexpressions.
    std::vector<ir::Stmt> stmts;
    // For unique variable renaming.
    int64_t counter = 0;

    // Pushes this `statement` onto the list of generated statements and returns
    // a sequence.
    ir::Stmt make(ir::Stmt statement) {
        stmts.push_back(std::move(statement));
        ir::Stmt sequence = ir::Sequence::make(std::move(stmts));
        stmts.clear();
        return sequence;
    }
};

class CseImpl : public ir::Mutator {
  public:
    CseImpl(const std::set<std::string> &side_effect_functions,
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

    ir::Stmt visit(const ir::Assign *node) override {
        if (node->mutating && !mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return node;
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!mutable_variables.contains(node->name)) {
            mutable_variables.add_to_frame(node->name);
        }
        return node;
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        if (!mutable_variables.contains(node->loc.base)) {
            mutable_variables.add_to_frame(node->loc.base);
        }
        return node;
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ir::Expr cond = mutate(node->cond);
        new_frame();
        ir::Stmt th = mutate(node->then_body);
        pop_frame();
        new_frame();
        ir::Stmt el = mutate(node->else_body);
        pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        new_frame();
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
        ir::Stmt header = node->header;
        new_frame();
        if (header.defined()) {
            header = mutate(header);
        }
        ir::Stmt body = mutate(node->body);
        pop_frame();
        return ir::ForAll::make(node->index, std::move(header),
                                std::move(slice), std::move(body));
    }
    ir::Stmt visit(const ir::DoWhile *node) override {
        new_frame();
        ir::Stmt body = mutate(node->body);
        ir::Expr cond = mutate(node->cond);
        pop_frame();
        return ir::DoWhile::make(std::move(body), std::move(cond));
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
        return ir::Cast::make(node->type, cse(node->value));
    }

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
    int64_t local_value_number = 0;
    // expression -> value number
    ir::MapStack<ir::Expr, int64_t, ir::ExprLessThan> e_to_vn;
    // variable -> expression
    ir::MapStack<std::string, ir::Expr> var_to_e;
    // value number -> variable (for subsequent replacement)
    ir::MapStack<int64_t, std::string> vn_to_var;

    void new_frame() {
        mutable_variables.new_frame();
        e_to_vn.new_frame();
        var_to_e.new_frame();
        vn_to_var.new_frame();
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
        const int64_t vn = local_value_number++;
        e_to_vn.add_to_frame(e, vn);
        return vn;
    }

    // Finds a common subexpression replacement for `e`, or returns the original
    // expression otherwise.
    ir::Expr cse(ir::Expr e) {
        ir::Expr cse_e = substitute(e);
        std::optional<int64_t> vn = e_to_vn.from_frames(cse_e);
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
    ir::Expr substitute(ir::Expr e) {
        if (const auto *b = e.as<ir::Build>()) {
            std::vector<ir::Expr> values;
            for (const ir::Expr &v : b->values) {
                values.push_back(substitute(v));
            }
            return ir::Build::make(b->type, std::move(values));
        } else if (const auto *c = e.as<ir::Call>()) {
            std::vector<ir::Expr> args;
            for (const ir::Expr &a : c->args) {
                args.push_back(substitute(a));
            }
            return ir::Call::make(c->func, std::move(args));
        } else if (const auto *o = e.as<ir::BinOp>()) {
            return ir::BinOp::make(o->op, substitute(o->a), substitute(o->b));
        } else if (const auto *op = e.as<ir::Intrinsic>()) {
            std::vector<ir::Expr> args = op->args;
            for (int i = 0, e = args.size(); i < e; ++i) {
                args[i] = substitute(args[i]);
            }
            return ir::Intrinsic::make(op->op, std::move(args));
        } else if (const auto *op = e.as<ir::Access>()) {
            return ir::Access::make(op->field, substitute(op->value));
        } else if (const auto *op = e.as<ir::Cast>()) {
            return ir::Cast::make(op->type, substitute(op->value));
        } else if (const auto *op = e.as<ir::Extract>()) {
            return ir::Extract::make(substitute(op->vec), substitute(op->idx));
        } else if (const auto *v = e.as<ir::Var>()) {
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
        lhs_to_rhs.new_frame();
        ir::Stmt th = mutate(node->then_body);
        lhs_to_rhs.pop_frame();
        lhs_to_rhs.new_frame();
        ir::Stmt el = mutate(node->else_body);
        lhs_to_rhs.pop_frame();
        return ir::IfElse::make(std::move(cond), std::move(th), std::move(el));
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        lhs_to_rhs.new_frame();
        ir::Expr iter = mutate(node->iter);
        ir::Stmt body = mutate(node->body);
        lhs_to_rhs.pop_frame();
        return ir::ForEach::make(node->name, std::move(iter), std::move(body));
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ir::ForAll::Slice slice = ir::ForAll::Slice{
            .begin = mutate(node->slice.begin),
            .end = mutate(node->slice.end),
            .stride = mutate(node->slice.stride),
        };
        ir::Stmt header = node->header;
        lhs_to_rhs.new_frame();
        if (header.defined()) {
            header = mutate(header);
        }
        ir::Stmt body = mutate(node->body);
        lhs_to_rhs.pop_frame();
        return ir::ForAll::make(node->index, std::move(header),
                                std::move(slice), std::move(body));
    }
    ir::Stmt visit(const ir::DoWhile *node) override {
        lhs_to_rhs.new_frame();
        ir::Stmt body = mutate(node->body);
        ir::Expr cond = mutate(node->cond);
        lhs_to_rhs.pop_frame();
        return ir::DoWhile::make(std::move(body), std::move(cond));
    }

    // Cannot propagate copies through mutable variables.
    ir::Stmt visit(const ir::Assign *node) override { return node; }
    ir::Stmt visit(const ir::Accumulate *node) override { return node; }
    // Don't propagate through lambda bodies.
    ir::Expr visit(const ir::Lambda *node) override { return node; }

  private:
    const std::set<std::string> &mutable_arguments;
    // A mapping from the lhs to rhs assignment of variable names, e.g.,
    //   x: i32 = y; // {x, y}
    //   z: i32 = x; // {z, x}
    ir::MapStack<std::string, std::string> lhs_to_rhs;
};

} // namespace

ir::FuncMap CSE::run(ir::FuncMap funcs) const {
    std::set<std::string> side_effect_functions = find_side_effects(funcs);
    for (auto &[name, func] : funcs) {
        std::set<std::string> mutable_arguments = get_mutable_arguments(*func);
        RenameAnalysis rename_analysis(side_effect_functions,
                                       mutable_arguments);
        func->body.accept(&rename_analysis);
        ExprSet to_rename = rename_analysis.post_process();
        Rename rename(to_rename);
        func->body = rename.mutate(std::move(func->body));

        CseImpl cse(side_effect_functions, mutable_arguments);
        func->body = cse.mutate(std::move(func->body));

        CopyPropagation cp(mutable_arguments);
        func->body = cp.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
