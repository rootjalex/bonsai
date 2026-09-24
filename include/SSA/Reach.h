#pragma once

#include "SSA/SSA.h"

#include <memory>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// `v` as `block` may refer to it. A block refers only to its own
// instructions and arguments in this form; a value defined elsewhere is
// threaded in as an argument along every path from its definition
// (Block::get_value), and this is where that is asked for. What every
// rewrite that builds new blocks uses to hand them the values they read
// (SSA/Defer.cpp, SSA/ReorderLoops.cpp).
std::shared_ptr<Value> reach(const std::shared_ptr<Block> &block,
                             const std::shared_ptr<Value> &v);

std::vector<std::shared_ptr<Value>>
reach_all(const std::shared_ptr<Block> &block,
          const std::vector<std::shared_ptr<Value>> &vs);

} // namespace ssa
} // namespace ir
} // namespace bonsai
