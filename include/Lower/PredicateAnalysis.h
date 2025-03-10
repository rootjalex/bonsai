#pragma once

#include "IR/Expr.h"

namespace bonsai {
namespace lower {

struct Interval {
    ir::Expr min, max;
};

using VolumeMap = std::map<std::string, ir::Expr>;

Interval predicate_analysis(const ir::Expr &expr, const VolumeMap &bounds);

} // namespace lower
} // namespace bonsai
