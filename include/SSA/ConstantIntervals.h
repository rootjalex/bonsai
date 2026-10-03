#pragma once

#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// A constant interval: what a scalar value is known to lie between, as two
// optional bounds. Halide's ConstantInterval (src/ConstantInterval.h, what
// its constant_integer_bounds answers with), extended to floats. A bound is
// a double: an integer bound is held exactly up to 2^53 in magnitude and
// dropped past it, a float bound may be an infinity, and an integer
// interval is kept within its type's range (an operation that may overflow
// gives the whole range, as Halide's cast_to does). A NaN is outside the
// lattice: the bounds say where a value is when it is a number, and a
// client that may meet one reasons about that itself -- the sort key's
// lanes that are read are the hits', whose box test is false for a NaN.
struct ConstantInterval {
    double min = 0, max = 0;
    bool min_defined = false, max_defined = false;

    ConstantInterval() = default;
    ConstantInterval(double min, double max)
        : min(min), max(max), min_defined(true), max_defined(true) {}

    static ConstantInterval everything() { return {}; }
    static ConstantInterval single_point(double x) { return {x, x}; }
    static ConstantInterval bounded_below(double min);
    static ConstantInterval bounded_above(double max);
    // The range of a type: an integer type's, a bool's [0, 1]; a float's is
    // everything (its infinities are values, so no finite bound holds).
    static ConstantInterval bounds_of_type(const Type &t);

    bool is_everything() const { return !min_defined && !max_defined; }
    bool is_bounded() const { return min_defined && max_defined; }
    bool is_single_point() const {
        return is_bounded() && min == max;
    }
    bool operator==(const ConstantInterval &o) const;

    // Widens this to hold `i` too: the union.
    void include(const ConstantInterval &i);
    static ConstantInterval make_union(const ConstantInterval &a,
                                       const ConstantInterval &b);
    static ConstantInterval make_intersection(const ConstantInterval &a,
                                              const ConstantInterval &b);
    // This interval as a value of type `t`: an integer bound past 2^53 is
    // dropped, an interval the type cannot hold is the type's range, and a
    // float's bounds are converted to its precision outward.
    void cast_to(const Type &t);
};

ConstantInterval operator+(const ConstantInterval &a, const ConstantInterval &b);
ConstantInterval operator-(const ConstantInterval &a, const ConstantInterval &b);
ConstantInterval operator-(const ConstantInterval &a);
ConstantInterval operator*(const ConstantInterval &a, const ConstantInterval &b);
// Quotient and remainder, by a divisor that is known to be positive or known
// to be negative; everything otherwise.
ConstantInterval operator/(const ConstantInterval &a, const ConstantInterval &b);
ConstantInterval operator%(const ConstantInterval &a, const ConstantInterval &b);
ConstantInterval min(const ConstantInterval &a, const ConstantInterval &b);
ConstantInterval max(const ConstantInterval &a, const ConstantInterval &b);
ConstantInterval abs(const ConstantInterval &a);
std::ostream &operator<<(std::ostream &os, const ConstantInterval &i);

// What every value of a program's functions is known to lie between, by one
// linear pass over each function in reverse postorder, repeated a few
// times for what one pass cannot see -- Halide's constant_integer_bounds
// (src/ConstantBounds.cpp) over this SSA form.
//
// A constant is itself; an instruction is its operation on its operands'
// intervals; a block argument is the union of what every edge hands it,
// except that an edge from inside the loop the block heads makes it
// everything (one pass cannot say what a value becomes round a loop), a
// value handed back unchanged being left out of the union since it adds
// nothing. A load reads the memory the pointer names: a local allocation's
// contents are the union of everything stored into it (one entry per
// field, an array's elements as one), everything if its address escapes
// to something unknown. Across functions, a parameter is the union of the
// arguments every call site passes (field-sensitive: the ray's `tnear` is
// one field of `r`), an exported function's being everything; a call's
// result is the union of the callee's returns, and what the callee stores
// through a pointer parameter is what the caller's memory holds after the
// call. Every summary starts as everything and narrows round by round, so
// that stopping after any round is sound; a parameter a recursion passes
// through unchanged is left out of its own union, which is what lets a
// traversal's ray keep what its caller said of it. Weak through diverging
// control flow and loops by design: this answers the questions a rewrite
// asks ("is the key non-negative", "is the dividend below twice the
// divisor"), not a bounds inference.
class ConstantIntervals {
  public:
    using FuncMap = std::map<std::string, std::shared_ptr<Function>>;

    // Over every function of `funcs`, interprocedurally.
    explicit ConstantIntervals(const FuncMap &funcs);
    // Over one function alone, its parameters everything.
    explicit ConstantIntervals(const Function &func);

    // What `v`, as `block` of `func` refers to it, lies between: a scalar's
    // or a vector's lanes' interval; for an array, the one interval all its
    // elements lie in; everything for anything else.
    ConstantInterval of(const Function &func, const Block &block,
                        const std::shared_ptr<Value> &v) const;

    // What was found about `func`, for reading: its parameters' intervals
    // (a struct's by field), every block argument's and instruction's that
    // is narrower than its type, and the contents of its local allocations.
    // Printed for every function when BONSAI_INTERVALS is set in the
    // environment, as BONSAI_RELOOPER prints the relooper's reading.
    void dump(std::ostream &os, const Function &func) const;

  private:
    struct State;
    std::map<const Function *, std::shared_ptr<State>> states;
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
