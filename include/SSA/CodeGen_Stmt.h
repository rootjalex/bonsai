#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

std::shared_ptr<ir::Function> codegen_stmt(const ssa::Function &func,
                                           const TypeMap &func_type_map);

} // namespace ssa
} // namespace ir
} // namespace bonsai
