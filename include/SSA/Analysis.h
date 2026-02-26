#pragma once

#include "SSA/SSA.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// block -> [arg0.is_mutable, arg1.is_mutable, ...]
using ArgMutabilityMap = std::map<std::string, std::vector<bool>>;

ArgMutabilityMap get_mutability_map(const ssa::Function &func);

using BlockMap = std::map<std::string, std::shared_ptr<Block>>;

BlockMap make_block_map(const std::shared_ptr<Function> &func);
BlockMap make_block_map(const Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
