#include "IR/TypeEnforcement.h"

#include <stdexcept>

namespace bonsai {
namespace ir {

namespace {
static bool TYPE_ENFORCEMENT = true;
}

void global_disable_type_enforcement() { TYPE_ENFORCEMENT = false; }

void global_enable_type_enforcement() { TYPE_ENFORCEMENT = true; }

bool type_enforcement_enabled() { return TYPE_ENFORCEMENT; }

} // namespace ir
} // namespace bonsai
