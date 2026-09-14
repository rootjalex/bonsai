#pragma once

#include "Expr.h"
#include "Program.h"
#include "Stmt.h"
#include "Type.h"
#include "Visitor.h"

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {

std::vector<TypedVar> gather_free_vars(const Expr &expr);
std::vector<TypedVar> gather_free_vars(const Stmt &stmt);
// std::vector<const Var *> gather_free_vars(const Stmt &stmt);
std::vector<TypedVar> gather_free_vars(const Function &func);
std::vector<TypedVar> gather_write_vars(const WriteLoc &loc);

bool always_returns(const Stmt &stmt);
Type get_return_type(const Stmt &stmt);

// Whether `expr` is a value of its free variables alone: arithmetic over
// variables, constants and reads of storage, with nothing in it that runs a
// function, draws from a generator or goes through a pointer. Two evaluations
// of such an expression agree unless something assigns a variable it reads.
bool is_pure_value(const Expr &expr);

// The names a function can assign: its mutable locals and the parameters it
// takes by reference. A pure value over any other name means the same thing
// wherever the function reads it, whatever the function calls in between.
std::set<std::string> assignable_names(const Function &func);

// The value `cond` takes when `fact` holds: true when it is `fact`, false
// when one is the other negated, and nothing when neither is so.
std::optional<bool> truth_under(const Expr &cond, const Expr &fact);

// Conditions known to hold (true) or not to (false) at some point of a
// program, most recently learned last.
using Facts = std::vector<std::pair<Expr, bool>>;

// The value `cond` takes under `facts`, if one of them decides it.
std::optional<bool> decided(const Facts &facts, const Expr &cond);

std::vector<const Struct_t *> gather_struct_types(const Program &program);

bool is_constant_expr(const Expr &expr);

bool can_be_empty(const Expr &expr);

bool contains_generics(const Type &type, const TypeMap &types);

template <typename IRNode>
bool contains(const Expr &expr) {
    static_assert(std::is_base_of<BaseExprNode, IRNode>::value,
                  "IRNode must be a subclass of BaseExprNode");
    struct Checker : public Visitor {
        bool found = false;

        void visit(const IRNode *node) override { found = true; }
    };
    Checker checker;
    expr.accept(&checker);
    return checker.found;
}

template <typename IRNode>
bool contains(const Type &type) {
    static_assert(std::is_base_of<BaseTypeNode, IRNode>::value,
                  "IRNode must be a subclass of BaseTypeNode");
    struct Checker : public Visitor {
        bool found = false;

        void visit(const IRNode *node) override { found = true; }
    };
    Checker checker;
    type.accept(&checker);
    return checker.found;
}

template <typename IRNode>
bool contains(const Stmt &stmt) {
    static_assert(std::is_base_of<BaseStmtNode, IRNode>::value ||
                      std::is_base_of<BaseExprNode, IRNode>::value,
                  "IRNode must be a subclass of BaseStmtNode or BaseExprNode");
    struct Checker : public Visitor {
        bool found = false;

        void visit(const IRNode *node) override { found = true; }
    };
    Checker checker;
    stmt.accept(&checker);
    return checker.found;
}

template <typename IRNode>
uint64_t count(const Stmt &stmt) {
    static_assert(std::is_base_of<BaseStmtNode, IRNode>::value ||
                      std::is_base_of<BaseExprNode, IRNode>::value,
                  "IRNode must be a subclass of BaseStmtNode or BaseExprNode");
    struct Counter : public Visitor {
        uint64_t c = 0;

        void visit(const IRNode *node) override {
            c++;
            Visitor::visit(node);
        }
    };
    Counter counter;
    stmt.accept(&counter);
    return counter.c;
}

template <typename IRNode>
uint64_t count(const Expr &expr) {
    static_assert(std::is_base_of<BaseStmtNode, IRNode>::value ||
                      std::is_base_of<BaseExprNode, IRNode>::value,
                  "IRNode must be a subclass of BaseStmtNode or BaseExprNode");
    struct Counter : public Visitor {
        uint64_t c = 0;

        void visit(const IRNode *node) override {
            c++;
            Visitor::visit(node);
        }
    };
    Counter counter;
    expr.accept(&counter);
    return counter.c;
}

std::set<std::string> mutated_variables(Stmt stmt);

// The names of the functions `stmt` calls, directly.
//
// Only the ones named by a Var, which is every call to a function the program
// declares. A call through anything else names nothing to return.
std::set<std::string> called_functions(Stmt stmt);

bool reads(Expr expr, const std::set<std::string> &vars);
bool reads(Stmt stmt, const std::set<std::string> &vars);

// Returns the set of function names that have side effects.
std::set<std::string> find_side_effects(const ir::FuncMap &functions);

// Returns whether `expr` has side effects.
bool has_side_effects(const ir::Expr &expr,
                      const std::set<std::string> &side_effect_functions);

uint64_t ast_size(const Expr &expr);

} // namespace ir
} // namespace bonsai
