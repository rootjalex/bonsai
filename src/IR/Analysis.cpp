#include "IR/Analysis.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include <set>

namespace bonsai {
namespace ir {

namespace {

struct GatherFreeVars : public Visitor {
    // return in seen-order
    std::vector<std::pair<std::string, Type>> free_vars;
    // no duplicates
    std::set<std::string> seen_vars;

    void visit(const Var *node) override {
        if (seen_vars.count(node->name) == 0) {
            free_vars.emplace_back(node->name, node->type);
            seen_vars.insert(node->name);
        }
    }

    void visit(const Store *node) override {
        if (seen_vars.count(node->name) == 0) {
            Type ptr_t = Ptr_t::make(node->value.type());
            free_vars.emplace_back(node->name, std::move(ptr_t));
            seen_vars.insert(node->name);
        }
        // TODO: consider implications on recursive definition.
        Visitor::visit(node);
    }

    void visit(const LetStmt *node) override {
        // TODO: fix this!! use SSA.
        seen_vars.insert(node->loc.base);
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<Expr>(value)) {
                std::get<Expr>(value).accept(this);
            }
        }
        node->value.accept(this);
        // node->body.accept(this);
        // seen_vars.erase(node->name);
    }

    void visit(const Assign *node) override {
        seen_vars.insert(node->loc.base);
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<Expr>(value)) {
                std::get<Expr>(value).accept(this);
            }
        }
        node->value.accept(this);
    }

    void visit(const Accumulate *node) override {
        seen_vars.insert(node->loc.base);
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<Expr>(value)) {
                std::get<Expr>(value).accept(this);
            }
        }
        node->value.accept(this);
    }
};

struct AlwaysReturns : public Visitor {
    bool returns = false;
    void visit(const Return *) override { returns = true; }

    void visit(const Store *) override { returns = false; }

    void visit(const LetStmt *node) override {
        // TODO: fix this!!
        returns = false;
        // node->body.accept(this);
    }

    void visit(const IfElse *node) override {
        if (node->else_body.defined()) {
            node->then_body.accept(this);
            if (returns) {
                node->else_body.accept(this);
            }
        } else {
            returns = false;
        }
    }

    void visit(const Sequence *node) override {
        for (size_t i = 0; i < node->stmts.size() - 1; i++) {
            const Stmt &stmt = node->stmts[i];
            stmt.accept(this);
            internal_assert(!returns)
                << "Sequence always returns in the middle of computation: "
                << node;
        }
        node->stmts.back().accept(this);
    }

    void visit(const Assign *node) override {
        // TODO: fix this!!
        returns = false;
        // node->body.accept(this);
    }

    void visit(const Accumulate *node) override {
        // TODO: fix this!!
        returns = false;
        // node->body.accept(this);
    }
};

struct ReturnType : public Visitor {
    Type type;

    void visit(const Return *node) override { type = node->value.type(); }

    void visit(const Store *) override { type = Type(); }

    void visit(const LetStmt *node) override {
        // TODO: fix this!! bring back SSA
        // node->body.accept(this);
    }

    void visit(const Assign *node) override {
        // TODO: fix this!! bring back SSA
        // node->body.accept(this);
    }

    void visit(const Accumulate *node) override {
        // TODO: fix this!! bring back SSA
        // node->body.accept(this);
    }

    void visit(const IfElse *node) override {
        node->then_body.accept(this);
        Type then_type = std::move(type);
        type = Type();
        if (node->else_body.defined()) {
            node->else_body.accept(this);
        }
        if (then_type.defined() && type.defined()) {
            internal_assert(equals(then_type, type))
                << "IfElse returns two separate types:" << then_type << " vs. "
                << type << " in\n"
                << node;
            // TODO: structural equality.
        } else if (then_type.defined()) {
            type = then_type;
        }
    }

    void visit(const Sequence *node) override {
        Type prev_type;
        for (size_t i = 0; i < node->stmts.size(); i++) {
            const Stmt &stmt = node->stmts[i];
            type = Type();
            stmt.accept(this);
            if (type.defined()) {
                internal_assert(!prev_type.defined() || equals(type, prev_type))
                    << "Sequence returns two separate types:\n"
                    << prev_type << " vs. " << type << " in\n"
                    << node;
                prev_type = type;
            }
        }
        type = prev_type; // any possible return type.
    }
};

struct GatherStructTypes : public Visitor {
    // return in seen-order
    std::vector<const Struct_t *> struct_types;
    // no duplicates
    std::set<std::string> seen_structs;

    void visit(const Struct_t *node) override {
        if (seen_structs.count(node->name) == 0) {
            struct_types.push_back(node);
            seen_structs.insert(node->name);
        }
    }

    // TODO: override all Expr/Stmt ops that might interact
    // with a Struct_t and make sure the Visitor recurses

    void visit(const Build *node) override {
        node->type.accept(this);
        Visitor::visit(node);
    }

    void visit(const Access *node) override {
        node->value.type().accept(this);
        Visitor::visit(node);
    }
};

} // namespace

std::vector<std::pair<std::string, Type>> gather_free_vars(const Expr &expr) {
    GatherFreeVars gather;
    expr.accept(&gather);
    return std::move(gather.free_vars);
}

std::vector<std::pair<std::string, Type>> gather_free_vars(const Stmt &stmt) {
    GatherFreeVars gather;
    stmt.accept(&gather);
    return std::move(gather.free_vars);
}

bool always_returns(const Stmt &stmt) {
    AlwaysReturns check;
    stmt.accept(&check);
    return check.returns;
}

Type get_return_type(const Stmt &stmt) {
    internal_assert(always_returns(stmt))
        << "get_return_type does not work for stmt that does not always "
           "return:\n"
        << stmt;
    ReturnType getter;
    stmt.accept(&getter);
    // Cannot assert that getter.type is defined, type inference might not have
    // run yet.
    return getter.type;
}

std::vector<const Struct_t *> gather_struct_types(const Program &program) {
    GatherStructTypes gather;

    // TODO: can externs contain optionals?

    for (const auto &[_, f] : program.funcs) {
        for (const auto &arg : f->args) {
            arg.type.accept(&gather);
            if (arg.default_value.defined()) {
                arg.default_value.accept(&gather); // is this necessary?
            }
        }
        f->ret_type.accept(&gather);
        f->body.accept(&gather);
    }

    for (const auto &[_, t] : program.types) {
        t.accept(&gather);
    }

    if (program.main_body.defined()) {
        // TODO: main should *always* be defined!
        program.main_body.accept(&gather);
    }

    return std::move(gather.struct_types);
}

// TODO: merge with is_const ?
bool is_constant_expr(const Expr &expr) {
    // TODO: constant fold first?
    if (expr.is<IntImm>() || expr.is<UIntImm>() || expr.is<FloatImm>() ||
        expr.is<BoolImm>()) {
        return true;
    } else if (expr.is<Broadcast>()) {
        return is_constant_expr(expr.as<Broadcast>()->value);
    } else if (expr.is<VectorReduce>()) {
        return is_constant_expr(expr.as<VectorReduce>()->value);
        // } else if (expr.is<VectorShuffle>()) {
        //     return is_constant_expr(expr.as<VectorShuffle>()->value) &&
        //     std::all_of(expr.as<VectorShuffle>()->idxs.cbegin(),
        //     expr.as<VectorShuffle>()->idxs.cend(), [](const auto &e) { return
        //     is_constant_expr(e); });
    } else if (expr.is<Ramp>()) {
        return is_constant_expr(expr.as<Ramp>()->base) &&
               is_constant_expr(expr.as<Ramp>()->stride);
    } else if (expr.is<Build>()) {
        for (const auto &value : expr.as<Build>()->values) {
            if (!is_constant_expr(value)) {
                return false;
            }
        }
        return true;
    } else if (expr.is<Access>()) {
        // TODO: this is unnecessarily restrictive for now,
        // should only be checking if the accessed field is a constant.
        return is_constant_expr(expr.as<Access>()->value);
    } else {
        internal_error << "is_constant_expr() called on: " << expr;
        return false;
        // TODO: Intrinsic, Lambda, GeomOp, SetOp, Call (constant folding)
    }
}

} // namespace ir
} // namespace bonsai
