#pragma once

namespace bonsai {
namespace ir {

void global_disable_type_enforcement();
void global_enable_type_enforcement();
bool type_enforcement_enabled();

} // namespace ir
} // namespace bonsai
