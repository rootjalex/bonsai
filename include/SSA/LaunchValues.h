#pragma once

#include "IR/Expr.h"
#include "SSA/SSA.h"

#include <memory>
#include <optional>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// A value of a bound loop's body as the launch knows it.
//
// A kernel is the body of a loop bound to the GPU, and some of what the
// launch has to know before the kernel starts is a value of that body: the
// extent of a thread loop inside a block loop, for the block size; the size
// of an allocation in the block loop's body, for the block's shared memory
// (CodeGen_PTX::allocate_shared). Such a value may be a constant, a capture
// -- an argument of the body's head block, which the launch hands the
// kernel -- or an instruction the body computes from those: a trip count's
// subtraction, the cast of a queue's capacity to the loop's index type. The
// host cannot run the body, but it can compute again whatever is a pure
// function of the captures, and this is that function, as an expression
// over the captures' names (the body head's argument names, which the
// kernel binds its parameters to and the launch maps to its own operands)
// and constants. Briggs, Cooper and Torczon's rematerialization (PLDI 1992)
// once more, with the launch for the register file: a value computed again
// where it is wanted rather than carried there.
//
// None when the value is no such function -- it reads memory, calls a
// function, is the loop's own index, or is merged or carried inside the
// body -- with `why` saying so for the error the caller raises.
std::optional<Expr> launch_value(const Function &func, const Block &body_head,
                                 const std::string &block,
                                 const std::shared_ptr<Value> &v, std::string *why);

// The same for an expression a body's allocation is sized by (SSA/Storage.h,
// as_expr): a constant, or a variable naming a value of the body, resolved
// as `block` refers to it.
std::optional<Expr> launch_value(const Function &func, const Block &body_head,
                                 const std::string &block, const Expr &e,
                                 std::string *why);

} // namespace ssa
} // namespace ir
} // namespace bonsai
