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
    std::vector<const Var *> free_vars;
    // no duplicates
    std::set<std::string> seen_vars;

    void visit(const Var *node) override {
        // Function calls are not free vars.
        if (seen_vars.count(node->name) == 0 && !node->type.is_func()) {
            free_vars.push_back(node);
            seen_vars.insert(node->name);
        }
    }

    void visit(const LetStmt *node) override {
        seen_vars.insert(node->loc.base);
        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<Expr>(value)) {
                std::get<Expr>(value).accept(this);
            }
        }
        node->value.accept(this);
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

    void visit(const ir::Lambda *node) override {
        for (const auto &arg : node->args) {
            internal_assert(!seen_vars.contains(arg.name));
            seen_vars.insert(arg.name);
        }
        node->value.accept(this);
        for (const auto &arg : node->args) {
            seen_vars.erase(arg.name);
        }
    }

    RESTRICT_VISITOR(Store);

    void visit(const ir::ForAll *node) override {
        node->slice.begin.accept(this);
        node->slice.end.accept(this);
        node->slice.stride.accept(this);

        // Now insert iteration var.
        internal_assert(!seen_vars.contains(node->index));
        seen_vars.insert(node->index);

        if (node->header.defined()) {
            node->header.accept(this);
        }
        node->body.accept(this);

        // Erase iteration var.
        seen_vars.erase(node->index);
    }

    void visit(const ir::ForEach *node) override {
        node->iter.accept(this);

        // Now insert iteration var.
        internal_assert(!seen_vars.contains(node->name));
        seen_vars.insert(node->name);

        node->body.accept(this);

        // Erase iteration var.
        seen_vars.erase(node->name);
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

    RESTRICT_VISITOR(DoWhile);

    void visit(const Sequence *node) override {
        for (size_t i = 0; i < node->stmts.size() - 1; i++) {
            const Stmt &stmt = node->stmts[i];
            stmt.accept(this);
            internal_assert(!returns)
                << "Sequence always returns in the middle of computation: "
                << Stmt(node);
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

    RESTRICT_VISITOR(Match);
    RESTRICT_VISITOR(Yield);
    RESTRICT_VISITOR(Scan);
    RESTRICT_VISITOR(YieldFrom);
};

struct ReturnType : public Visitor {
    Type type;

    void visit(const Return *node) override {
        ir::Expr value = node->value;
        type = value.defined() ? value.type() : ir::Void_t::make();
    }

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

    RESTRICT_VISITOR(Match);
    RESTRICT_VISITOR(Yield);
    RESTRICT_VISITOR(Scan);
    RESTRICT_VISITOR(YieldFrom);
    RESTRICT_VISITOR(DoWhile);

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

struct FindReads : Visitor {
    // TODO: enable early outs once `found` is set.
    const std::set<std::string> &vars;
    bool found = false;

    FindReads(const std::set<std::string> &vars) : vars(vars) {}

    void visit(const Var *node) override {
        if (vars.contains(node->name)) {
            found = true;
        }
    }
};

} // namespace

std::vector<const ir::Var *> gather_free_vars(const Expr &expr) {
    GatherFreeVars gather;
    expr.accept(&gather);
    return std::move(gather.free_vars);
}

// std::vector<const ir::Var *> gather_free_vars(const Stmt &stmt) {
//     GatherFreeVars gather;
//     stmt.accept(&gather);
//     return std::move(gather.free_vars);
// }

std::vector<const ir::Var *> gather_free_vars(const Function &func) {
    GatherFreeVars gather;
    for (const auto &arg : func.args) {
        gather.seen_vars.insert(arg.name);
    }
    func.body.accept(&gather);
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

    return std::move(gather.struct_types);
}

// TODO: merge with is_const ?
bool is_constant_expr(const Expr &expr) {
    // TODO: constant fold first?
    if (expr.is<IntImm, UIntImm, FloatImm, BoolImm, Infinity>()) {
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
        // TODO: Intrinsic, Lambda, GeomOp, SetOp, Call (constant folding)
    }
}

bool contains_generics(const Type &type, const TypeMap &types) {
    struct ContainsGenerics : public Visitor {
        ContainsGenerics(const TypeMap &types) : types(types) {}

        void visit(const Generic_t *node) override {
            if (types.contains(node->name)) {
                seen_types.insert(node->name);
            }
        }

        const TypeMap &types;
        std::set<std::string> seen_types;
    };

    ContainsGenerics checker(types);
    type.accept(&checker);
    return !checker.seen_types.empty();
}

std::set<std::string> mutated_variables(Stmt stmt) {
    struct Gather : Visitor {
        std::set<std::string> mutated;

        void visit(const Store *node) override { mutated.insert(node->name); }

        void visit(const Assign *node) override {
            if (node->mutating) {
                mutated.insert(node->loc.base);
            }
        }
    };

    Gather g;
    stmt.accept(&g);
    return std::move(g.mutated);
}

bool reads(Expr expr, const std::set<std::string> &vars) {
    FindReads f(vars);
    expr.accept(&f);
    return f.found;
}

bool reads(Stmt stmt, const std::set<std::string> &vars) {
    FindReads f(vars);
    stmt.accept(&f);
    return f.found;
}

std::set<std::string> assigned_variables(Stmt stmt) {
    struct Gather : Visitor {
        std::set<std::string> mutated;

        void visit(const Store *node) override { mutated.insert(node->name); }

        void visit(const Assign *node) override {
            mutated.insert(node->loc.base);
        }

        void visit(const LetStmt *node) override {
            mutated.insert(node->loc.base);
        }
    };

    Gather g;
    stmt.accept(&g);
    return std::move(g.mutated);
}

} // namespace ir
} // namespace bonsai
