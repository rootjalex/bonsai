#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Replaces a parallel ForAll loop with a packetized variant that follows
// all control flow.
// Inserts new functions into the func map (packetized versions).
ir::Stmt packetize_forall(const std::string &loop_idx, ir::Stmt body,
                          ir::FuncMap &funcs, ir::TypeMap &types);
} // namespace opt
} // namespace bonsai
