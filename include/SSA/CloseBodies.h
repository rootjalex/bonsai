#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

// Makes every parfor body's arguments its captures: each value the body uses
// that is defined outside it -- an instruction of the blocks before the loop,
// a parameter of the function -- becomes an argument of the body's first
// block, supplied by the loop's edge into it, under the name it already has.
//
// The SSA form does not need this to mean what it means: a block may use any
// value that dominates it, and inside one function the code generator finds
// the value by name wherever it was defined. What needs it is a body that
// becomes a function of its own -- the kernel a `bind(i, GPUThread)` or
// `bind(i, CPUThread)` makes of it -- which is handed its captures and nothing
// else (see CodeGen_LLVM::launch_captures). The builder threads every value
// through block arguments as it goes, so a freshly converted body is closed;
// the rewrites after it need not keep it so, and do not: promoting a `mut`
// local to a value (SSA/PromoteAllocas.h) replaces the body's load of it with
// the value that reaches the loop, and collapse() builds a block whose jump
// into the body names the header's values. So the bodies are closed once,
// after every rewrite has run and before anything is generated from them.
//
// Names are what a threaded value keeps (Block::get_value), and what the code
// generator binds a kernel's captures under, so the body's uses need no
// rewriting: they name what the new argument is called. Returns how many
// values were threaded in.
size_t close_parfor_bodies(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
