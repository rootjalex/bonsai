#pragma once

#include "Expr.h"
#include "Program.h"
#include "Stmt.h"
#include "Type.h"

namespace bonsai {
namespace ir {

std::vector<std::pair<std::string, Type>> gather_free_vars(const Expr &expr);
std::vector<std::pair<std::string, Type>> gather_free_vars(const Stmt &stmt);

bool always_returns(const Stmt &stmt);
Type get_return_type(const Stmt &stmt);

std::vector<const Struct_t *> gather_struct_types(const Program &program);

bool is_constant_expr(const Expr &expr);

bool contains_generics(const Type &type, const TypeMap &types);

} // namespace ir
} // namespace bonsai
