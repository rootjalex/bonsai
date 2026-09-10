#pragma once

#include "IR/Equality.h"
#include "IR/Expr.h"

#include <map>
#include <string>

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

// What varies over the subtree a predicate is being bounded on, and by what.
//
// A name that is absent does not vary: the expression it names has one value
// over the whole subtree and is its own bound. A name that is present varies,
// and is bounded by the volume it maps to -- or by nothing, when that volume
// is undefined, in which case anything built from it is unbounded too.
//
// `exprs` says the same about an expression taken whole, which a promise can
// bound where no rule about its parts could. `with extent` on an element says
// where the element *places* what is reached through it: at a node of the tree
// of instances it is `transform(i.render_from_instance, tri)` that lies within
// the node's box, while `tri` on its own is in the instance's frame and the box
// says nothing about it. The analysis looks an operand up here first.
//
// `fixed` names the parameters of a query that do not vary over this subtree
// *because the traversal is at one element of theirs* -- inside an instance's
// tree, the instance -- and says which element. They are absent from `names`,
// since they do not vary; but a bound that mentions them has to be a bound the
// traversal can evaluate, and the traversal knows the element by its own name
// (`_iter0`), not by the lambda's (`i`). The analysis substitutes as it goes.
struct VolumeMap {
    std::map<std::string, ir::Expr> names;
    std::map<ir::Expr, ir::Expr, ir::ExprLessThan> exprs;
    std::map<std::string, ir::Expr> fixed;

    ir::Expr &operator[](const std::string &name) { return names[name]; }
    auto find(const std::string &name) const { return names.find(name); }
    auto cend() const { return names.cend(); }
};

using IntervalMap = std::map<ir::Expr, Interval, ir::ExprLessThan>;

Interval predicate_analysis(const ir::Expr &expr, const VolumeMap &bounds,
                            const IntervalMap &intervals = {});

} // namespace lower
} // namespace bonsai
