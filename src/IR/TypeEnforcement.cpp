#include "IR/TypeEnforcement.h"

#include <stdexcept>

namespace bonsai {
namespace ir {

namespace {
static bool TYPE_ENFORCMENT = true;
}

void global_disable_type_enforcement() { TYPE_ENFORCMENT = false; }

void global_enable_type_enforcement() { TYPE_ENFORCMENT = true; }

bool type_enforcement_enabled() { return TYPE_ENFORCMENT; }

} // namespace ir
} // namespace bonsai
