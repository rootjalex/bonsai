#include "IR/Analysis.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "Lower/TopologicalOrder.h"

#include <set>

namespace bonsai {
namespace ir {

namespace {

struct GatherFreeVars : public Visitor {
    // return in seen-order
    std::vector<TypedVar> free_vars;
    // no duplicates
    std::set<std::string> seen_vars;

    void visit(const Var *node) override {
        // Function calls are not free vars.
        if (seen_vars.count(node->name) == 0 && !node->type.is_func()) {
            // Visit sizes, might be a free var
            if (node->type.is<Array_t>()) {
                node->type.as<Array_t>()->size.accept(this);
            }
            free_vars.push_back({node->name, node->type});
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

    void visit(const Allocate *node) override {
        seen_vars.insert(node->loc.base);
        internal_assert(node->loc.accesses.empty());
        if (node->value.defined()) {
            node->value.accept(this);
        }
    }

    void visit(const Store *node) override {
        if (!seen_vars.contains(node->loc.base)) {
            free_vars.push_back({node->loc.base, node->loc.base_type});
            seen_vars.insert(node->loc.base);
        }
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

    void visit(const ir::ForAll *node) override {
        node->slice.begin.accept(this);
        node->slice.end.accept(this);
        node->slice.stride.accept(this);

        // Now insert iteration var.
        internal_assert(!seen_vars.contains(node->index));
        seen_vars.insert(node->index);

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

    void visit(const ir::RecLoop *node) override {
        for (const auto &arg : node->args) {
            internal_assert(!seen_vars.contains(arg.name));
            seen_vars.insert(arg.name);
        }
        node->body.accept(this);
        for (const auto &arg : node->args) {
            seen_vars.erase(arg.name);
        }
    }
};

struct AlwaysReturns : public Visitor {
    bool returns = false;
    void visit(const Return *) override { returns = true; }

    void visit(const IfElse *node) override {
        if (node->else_body.defined()) {
            node->then_body.accept(this);
            if (returns) {
                returns = false;
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
                << Stmt(node);
        }
        node->stmts.back().accept(this);
    }

    void visit(const CallStmt *node) override { returns = false; }
    void visit(const LetStmt *node) override { returns = false; }
    void visit(const Allocate *node) override { returns = false; }
    void visit(const Store *node) override { returns = false; }
    void visit(const Accumulate *node) override { returns = false; }
    void visit(const Print *node) override { returns = false; }

    // Assume these don't always-return, otherwise
    // it should simplify.
    void visit(const ForAll *node) override { returns = false; }
    void visit(const DoWhile *node) override { returns = false; }

    RESTRICT_VISITOR(RecLoop);
    RESTRICT_VISITOR(ForEach);
    RESTRICT_VISITOR(Match);
    RESTRICT_VISITOR(Yield);
    RESTRICT_VISITOR(Scan);
    RESTRICT_VISITOR(YieldFrom);
    RESTRICT_VISITOR(Continue);
    RESTRICT_VISITOR(Launch);
};

struct ReturnType : public Visitor {
    Type type;

    void visit(const Return *node) override {
        ir::Expr value = node->value;
        type = value.defined() ? value.type() : ir::Void_t::make();
    }

    void visit(const LetStmt *node) override {}
    void visit(const Allocate *node) override {}
    void visit(const Store *node) override {}
    void visit(const Accumulate *node) override {}

    RESTRICT_VISITOR(Match);
    RESTRICT_VISITOR(Yield);
    RESTRICT_VISITOR(Scan);
    RESTRICT_VISITOR(YieldFrom);
    RESTRICT_VISITOR(DoWhile);
    RESTRICT_VISITOR(Launch);

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

struct HasSideEffects : ir::Visitor {
    bool found = false;
    const std::set<std::string> &function_has_side_effects;

    HasSideEffects(const std::set<std::string> &side_effects_functions)
        : function_has_side_effects(side_effects_functions) {}

    void visit(const ir::Print *node) override {
        if (found) {
            return;
        }
        found = true;
    }

    // TODO: use Halide's pure implementation
    void visit(const ir::Intrinsic *node) override {
        if (found) {
            return;
        }
        found = node->op == ir::Intrinsic::rand;
    }

    void visit(const ir::Call *node) override {
        if (found) {
            return;
        }
        const auto *var = node->func.as<ir::Var>();
        internal_assert(var) << node;
        if (var->type.is<ir::Function_t>() &&
            function_has_side_effects.contains(var->name)) {
            found = true;
        }
    }

    void visit(const ir::CallStmt *node) override {
        if (found) {
            return;
        }
        const auto *var = node->func.as<ir::Var>();
        internal_assert(var) << node;
        if (var->type.is<ir::Function_t>() &&
            function_has_side_effects.contains(var->name)) {
            found = true;
        }
    }
};

} // namespace

std::vector<TypedVar> gather_free_vars(const Expr &expr) {
    GatherFreeVars gather;
    expr.accept(&gather);
    return std::move(gather.free_vars);
}

std::vector<TypedVar> gather_free_vars(const Stmt &stmt) {
    GatherFreeVars gather;
    stmt.accept(&gather);
    return std::move(gather.free_vars);
}

std::vector<TypedVar> gather_free_vars(const Function &func) {
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

bool can_be_empty(const Expr &expr) {
    internal_assert(expr.type().is<Set_t>());
    if (const SetOp *op = expr.as<SetOp>()) {
        if (op->op == SetOp::filter) {
            return true;
        } else if (op->op == SetOp::map) {
            return can_be_empty(op->b);
        }
    }
    return false;
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

        void visit(const Store *node) override {
            mutated.insert(node->loc.base);
        }

        void visit(const Accumulate *node) override {
            mutated.insert(node->loc.base);
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

bool has_side_effects(const ir::Expr &expr,
                      const std::set<std::string> &side_effect_functions) {
    HasSideEffects checker(side_effect_functions);
    expr.accept(&checker);
    return checker.found;
}

std::set<std::string> find_side_effects(const ir::FuncMap &functions) {
    const std::vector<std::string> topo_order =
        lower::func_topological_order(functions, /*undef_calls=*/false);
    std::set<std::string> side_effects;
    HasSideEffects checker(side_effects);
    for (const std::string &f : topo_order) {
        internal_assert(!checker.function_has_side_effects.contains(f))
            << "Found cycle involving: " << f;
        checker.found = false;
        const auto func = functions.at(f);
        // Conservatively say that funcs with mutable arguments have side
        // effects.
        if (std::any_of(func->args.cbegin(), func->args.cend(),
                        [](const auto &arg) { return arg.mutating; })) {
            side_effects.insert(f);
            continue;
        }
        // Otherwise search for side effecting statements.
        func->body.accept(&checker);
        if (checker.found) {
            side_effects.insert(f);
        }
        checker.found = false;
    }
    return side_effects;
}

} // namespace ir
} // namespace bonsai
