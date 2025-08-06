#pragma once

#include "Layout.h"
#include "Type.h"

namespace bonsai {
namespace ir {

using Path = std::map<std::string, Type>;

[[maybe_unused]] std::ostream &operator<<(std::ostream &os,
                                          const std::vector<Path> &paths);

// internal_asserts or internal_errors if `layout` does not implement `type`
// assumes `type` is a BVH_t
// Returns a mapping from node name to the concrete types of it's fields.
std::map<std::string, Path> validate_layout(const Layout &layout);

} // namespace ir
} // namespace bonsai
