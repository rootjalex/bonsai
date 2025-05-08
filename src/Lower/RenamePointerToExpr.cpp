#include "Lower/RenamePointerToExpr.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {
// Prefix for a temporary variable, `lv` for l-value.
static constexpr char PREFIX[] = "_lv";

// A set of expressions using IR comparison rather than pointer comparison.
using ExprSet = std::set<ir::Expr, ir::ExprLessThan>;

// Retrieves all variables that take the address of an r-value.
ExprSet retrieve_variables(const ir::Stmt &body) {
    struct RenameAnalysis : public ir::Visitor {
        void visit(const ir::PtrTo *node) {
            if (!node->expr.is<ir::Var, ir::Access>()) {
                set.insert(node->expr);
            }
        }

        ExprSet set;
    };
    RenameAnalysis analysis;
    body.accept(&analysis);
    return analysis.set;
}

// Gives an expression its own variable name if it is found in `to_rename`.
// For example,
//   to_rename = { Sphere(j) };
//   f(&Sphere(j));
//   ->
//   let _lv0 = Sphere(j) in
//   f(&_lv0);
struct Rename : public ir::Mutator {
    Rename(const ExprSet &to_rename) : to_rename(to_rename) {}

    ir::Stmt visit(const ir::LetStmt *node) override {
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
            ir::WriteLoc location(PREFIX + std::to_string(counter++),
                                  node->type);
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
        ir::WriteLoc location(PREFIX + std::to_string(counter++), node->type);
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
        ir::WriteLoc location(PREFIX + std::to_string(counter++), node->type);
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
        ir::WriteLoc location(PREFIX + std::to_string(counter++), node->type);
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
        ir::WriteLoc location(PREFIX + std::to_string(counter++), node->type);
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
        ir::WriteLoc location(PREFIX + std::to_string(counter++), node->type);
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
        ir::WriteLoc loc(PREFIX + std::to_string(counter++), node->type);
        stmts.push_back(ir::LetStmt::make(loc, std::move(call)));
        return ir::Var::make(node->type, loc.base);
    }

  private:
    // Whether we should give this sub-expression its own variable.
    bool should_rename(const ir::Expr &e) { return to_rename.contains(e); }

    // A set of expressions that should be renamed in this pass.
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

} // namespace

ir::FuncMap RenamePointerToExpr::run(ir::FuncMap functions,
                                     const CompilerOptions &options) const {
    for (auto &[_, f] : functions) {
        ExprSet s = retrieve_variables(f->body);
        f->body = Rename(s).mutate(std::move(f->body));
    }
    return functions;
}

} // namespace lower
} // namespace bonsai
