#pragma once

#include "Layout.h"
#include "Type.h"

namespace bonsai {
namespace ir {

// internal_asserts or internal_errors if `layout` does not implement `type`
// assumes `type` is a BVH_t
void validate_layout(const Layout &layout, const Type &bvh_t);

} // namespace ir
} // namespace bonsai
