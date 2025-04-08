#pragma once

#include <map>
#include <string>

#include "Layout.h"
#include "ScheduleRewrite.h"
#include "Type.h"

namespace bonsai {
namespace ir {

struct Schedule {
    // BVH specifications
    TypeMap tree_types;
    // Tree layouts
    LayoutMap tree_layouts;
    // In-order scheduling rewrites to be applied.
    std::vector<schedule::Rewrite> rewrites;
};

} // namespace ir
} // namespace bonsai
