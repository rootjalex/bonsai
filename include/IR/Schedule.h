#pragma once

#include <map>
#include <string>

#include "Type.h"

namespace bonsai {
namespace ir {

struct Schedule {
    // TODO: should this be part of the schedule?
    TypeMap tree_types;
    // TODO: support tree layouts
    // TODO: support function scheduling
};

} // namespace ir
} // namespace bonsai
