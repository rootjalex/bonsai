#pragma once

#include "IR/Stmt.h"

namespace bonsai {
namespace opt {

ir::Stmt dce(const ir::Stmt &stmt);

} // namespace opt
} // namespace bonsai
