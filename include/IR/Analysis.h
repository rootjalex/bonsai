#pragma once

#include "Expr.h"
#include "Program.h"
#include "Stmt.h"
#include "Type.h"
#include "Visitor.h"

#include <set>
#include <string>

namespace bonsai {
namespace ir {

std::vector<TypedVar> gather_free_vars(const Expr &expr);
// std::vector<const Var *> gather_free_vars(const Stmt &stmt);
std::vector<TypedVar> gather_free_vars(const Function &func);

bool always_returns(const Stmt &stmt);
Type get_return_type(const Stmt &stmt);

std::vector<const Struct_t *> gather_struct_types(const Program &program);

bool is_constant_expr(const Expr &expr);

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

std::set<std::string> mutated_variables(Stmt stmt);

bool reads(Expr expr, const std::set<std::string> &vars);
bool reads(Stmt stmt, const std::set<std::string> &vars);

std::set<std::string> assigned_variables(Stmt stmt);

} // namespace ir
} // namespace bonsai
