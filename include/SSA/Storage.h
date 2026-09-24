#pragma once

#include "IR/Expr.h"
#include "IR/Type.h"
#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// Storage for a value of `type`, appended to `block`: what a `mut` local is
// in this form -- an Alloca whose value is the pointer to it, or the array
// itself for an array (see the Allocate visitor in SSA/Convert.cpp). Named
// `name` when one is given, which has to be a name the function does not
// use, so that the generated code says what the storage is. An allocation
// has no operands: a run-time size is a value named in its type (`f32[n]`,
// made with as_expr below), which the code generators and the passes that
// move storage (SSA/HoistAllocations.h) resolve by that name.
std::shared_ptr<Value> make_alloca(Function &func,
                                   const std::shared_ptr<Block> &block,
                                   const Type &type,
                                   const std::string &name = "");

// An SSA value as an expression a type may carry: the size of an entry array
// is one, and it is a run-time value named after the SSA value, or the
// constant itself.
Expr as_expr(const std::shared_ptr<Value> &v);

// A value's name may be in a *type*: an array whose length is not a constant
// is `f32[n]` for the value `n`, and the code generator finds `n` by that
// name when it allocates or indexes the array. A pass that renames or moves a
// value has to rename it inside the types too, or the types keep naming a
// value that no longer exists, or is not in scope where the type is read.
// This rewrites the sizes of the arrays in `type` by `renames`; the base
// mutator leaves them alone.
Type rename_in_type(const Type &type, const std::map<std::string, Expr> &renames);

// The same, in every type `func` holds: the types of its instructions and of
// what they ask the size of, its blocks' parameters, and the copies of a
// parameter that its values carry.
void rename_in_types(Function &func, const std::map<std::string, Expr> &renames);

// The names of the values a type carries: what has to be in scope wherever
// storage of that type is made.
std::vector<std::string> names_in_type(const Type &type);

} // namespace ssa
} // namespace ir
} // namespace bonsai
