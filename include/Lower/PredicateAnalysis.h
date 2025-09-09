#pragma once

#include "IR/Equality.h"
#include "IR/Expr.h"

namespace bonsai {
namespace lower {

struct Interval {
    ir::Expr min, max;

    bool is_single_point(const ir::Expr &a) const;
    bool is_single_point() const;
    bool has_upper_bound() const;
    bool has_lower_bound() const;
    bool is_bounded() const;
    void include(const ir::Expr &e);

    static Interval single_point(ir::Expr a) { return Interval{a, a}; }
};

using VolumeMap = std::map<std::string, ir::Expr>;
using IntervalMap = std::map<ir::Expr, Interval, ir::ExprLessThan>;

Interval predicate_analysis(const ir::Expr &expr, const VolumeMap &bounds,
                            const IntervalMap &intervals = {});

} // namespace lower
} // namespace bonsai
