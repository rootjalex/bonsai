#include "SSA/ConstantIntervals.h"

#include "Error.h"
#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "Utils.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <cmath>
#include <limits>
#include <optional>
#include <set>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::string;
using std::vector;
using ValuePtr = shared_ptr<Value>;

// Past this an integer is not exact in a double, and a bound is dropped.
constexpr double kExactInteger = 9007199254740992.0; // 2^53
constexpr double kInf = std::numeric_limits<double>::infinity();

bool is_integer_type(const Type &t) {
    return t.is_int_or_uint() || (t.is_vector() && t.element_of().is_int_or_uint());
}

Type scalar_of(const Type &t) { return t.is_vector() ? t.element_of() : t; }

// The same value: one instruction, one argument by name, or equal constants.
bool same_value(const shared_ptr<Value> &a, const shared_ptr<Value> &b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    if (const auto *ia = std::get_if<shared_ptr<Instruction>>(&a->data)) {
        const auto *ib = std::get_if<shared_ptr<Instruction>>(&b->data);
        return ib != nullptr && ia->get() == ib->get();
    }
    if (const auto *aa = std::get_if<Argument>(&a->data)) {
        const auto *ab = std::get_if<Argument>(&b->data);
        return ab != nullptr && aa->name == ab->name;
    }
    if (const auto *ca = std::get_if<Constant>(&a->data)) {
        const auto *cb = std::get_if<Constant>(&b->data);
        return cb != nullptr && ca->data == cb->data && equals(ca->type, cb->type);
    }
    return false;
}

} // namespace

//===--------------------------------------------------------------------===//
// ConstantInterval
//===--------------------------------------------------------------------===//

ConstantInterval ConstantInterval::bounded_below(double min) {
    ConstantInterval i;
    i.min = min;
    i.min_defined = true;
    return i;
}

ConstantInterval ConstantInterval::bounded_above(double max) {
    ConstantInterval i;
    i.max = max;
    i.max_defined = true;
    return i;
}

ConstantInterval ConstantInterval::single_point(double x) {
    ConstantInterval i{x, x};
    i.number = !std::isnan(x);
    i.sign_clear = i.nonnegative = !std::signbit(x);
    return i;
}

ConstantInterval ConstantInterval::bounds_of_type(const Type &type) {
    const Type t = scalar_of(type);
    if (t.is_bool()) {
        ConstantInterval i{0, 1};
        i.number = i.nonnegative = i.sign_clear = true;
        return i;
    }
    if (t.is_uint()) {
        ConstantInterval i = bounded_below(0);
        if (t.bits() < 53) {
            i.max = std::ldexp(1.0, int(t.bits())) - 1;
            i.max_defined = true;
        }
        i.number = i.nonnegative = i.sign_clear = true;
        return i;
    }
    if (t.is_int()) {
        ConstantInterval i;
        if (t.bits() <= 53) {
            const double top = std::ldexp(1.0, int(t.bits()) - 1);
            i = {-top, top - 1};
        }
        i.number = true;
        return i;
    }
    return everything();
}

bool ConstantInterval::operator==(const ConstantInterval &o) const {
    return min_defined == o.min_defined && max_defined == o.max_defined &&
           (!min_defined || min == o.min) && (!max_defined || max == o.max) &&
           number == o.number && nonnegative == o.nonnegative &&
           sign_clear == o.sign_clear;
}

void ConstantInterval::include(const ConstantInterval &i) {
    if (max_defined && i.max_defined) {
        max = std::max(max, i.max);
    } else {
        max_defined = false;
    }
    if (min_defined && i.min_defined) {
        min = std::min(min, i.min);
    } else {
        min_defined = false;
    }
    number = number && i.number;
    nonnegative = nonnegative && i.nonnegative;
    sign_clear = sign_clear && i.sign_clear;
}

ConstantInterval ConstantInterval::make_union(const ConstantInterval &a,
                                              const ConstantInterval &b) {
    ConstantInterval r = a;
    r.include(b);
    return r;
}

ConstantInterval ConstantInterval::make_intersection(const ConstantInterval &a,
                                                     const ConstantInterval &b) {
    ConstantInterval r;
    r.min_defined = a.min_defined || b.min_defined;
    r.max_defined = a.max_defined || b.max_defined;
    r.min = a.min_defined && b.min_defined ? std::max(a.min, b.min)
            : a.min_defined                ? a.min
                                           : b.min;
    r.max = a.max_defined && b.max_defined ? std::min(a.max, b.max)
            : a.max_defined                ? a.max
                                           : b.max;
    r.number = a.number || b.number;
    r.nonnegative = a.nonnegative || b.nonnegative;
    r.sign_clear = a.sign_clear || b.sign_clear;
    r.settle_flags();
    if (r.is_bounded() && r.min > r.max) {
        // Two intervals that do not meet: the value is not anywhere, which
        // this lattice has no word for, so it is everywhere.
        return everything();
    }
    return r;
}

void ConstantInterval::cast_to(const Type &type) {
    const Type t = scalar_of(type);
    if (t.is_int_or_uint() || t.is_bool()) {
        if (min_defined && (std::isnan(min) || std::fabs(min) > kExactInteger)) {
            min_defined = false;
        }
        if (max_defined && (std::isnan(max) || std::fabs(max) > kExactInteger)) {
            max_defined = false;
        }
        const ConstantInterval range = bounds_of_type(t);
        const bool within =
            (!range.min_defined || (min_defined && min >= range.min)) &&
            (!range.max_defined || (max_defined && max <= range.max));
        if (!within) {
            *this = range;
        }
        // An integer is a number, and its sign is clear where it is not
        // negative.
        number = true;
        nonnegative = sign_clear = min_defined && min >= 0;
        return;
    }
    if (t.is_float() && t.bits() <= 32) {
        // Outward to the type's precision: the bound as a float that is no
        // nearer than the double was.
        if (min_defined && !std::isinf(min)) {
            float f = float(min);
            if (double(f) > min) {
                f = std::nextafterf(f, -kInf);
            }
            min = f;
        }
        if (max_defined && !std::isinf(max)) {
            float f = float(max);
            if (double(f) < max) {
                f = std::nextafterf(f, kInf);
            }
            max = f;
        }
    }
    if (min_defined && std::isnan(min)) {
        min_defined = false;
    }
    if (max_defined && std::isnan(max)) {
        max_defined = false;
    }
}

namespace {

// A bound that may be missing, as a signed limit: an undefined lower bound
// is -inf and an undefined upper bound +inf, remembered as undefined so
// that a result it reaches is undefined too.
struct Limit {
    double value;
    bool defined;
};

Limit lower(const ConstantInterval &i) {
    return i.min_defined ? Limit{i.min, true} : Limit{-kInf, false};
}
Limit upper(const ConstantInterval &i) {
    return i.max_defined ? Limit{i.max, true} : Limit{kInf, false};
}

// The interval whose bounds are the least and the greatest of `corners`,
// each corner a value with whether it is a defined bound; a NaN among them
// is a product that is not a number (zero times an infinity), a case the
// lattice leaves out.
ConstantInterval from_corners(const vector<Limit> &corners) {
    ConstantInterval r;
    bool first = true;
    Limit lo{0, true}, hi{0, true};
    for (const Limit &c : corners) {
        if (std::isnan(c.value)) {
            // A corner that is not a number -- zero times an infinity -- is
            // a case the lattice leaves out, as it leaves out every NaN
            // (see ConstantInterval): the bounds say where the value is
            // when it is a number, and this corner is none. A product whose
            // every corner is one is never a number, and nothing is said.
            continue;
        }
        if (first || c.value < lo.value || (c.value == lo.value && !c.defined)) {
            lo = c;
        }
        if (first || c.value > hi.value || (c.value == hi.value && !c.defined)) {
            hi = c;
        }
        first = false;
    }
    if (first) {
        return ConstantInterval::everything();
    }
    if (lo.defined) {
        r.min = lo.value;
        r.min_defined = true;
    }
    if (hi.defined) {
        r.max = hi.value;
        r.max_defined = true;
    }
    return r;
}

Limit times(const Limit &a, const Limit &b) {
    // An undefined limit stands for a value of unknown size, so a defined
    // zero times it is a defined zero -- `[0, ..) * [0, ..)` is bounded
    // below by zero, whatever the sizes -- where zero times a defined
    // infinity is a NaN, a corner from_corners leaves out, so that `[0,
    // inf] * [0, inf]` is `[0, inf]` where it is a number at all. (An
    // undefined limit of a float may be an infinity, and zero times that is
    // not a number either: outside the lattice, as every NaN is.)
    if ((a.value == 0 && a.defined && !b.defined) ||
        (b.value == 0 && b.defined && !a.defined)) {
        return {0, true};
    }
    return {a.value * b.value, a.defined && b.defined};
}

Limit over(const Limit &a, const Limit &b) {
    if (!b.defined) {
        // Divided by something of unknown size (but of the sign the caller
        // checked): toward zero, and no nearer.
        return {0, false};
    }
    return {a.value / b.value, a.defined};
}

} // namespace

// Whether `i` may hold `v`, an infinity or zero: nothing says it does not.
bool may_hold(const ConstantInterval &i, double v) {
    return (!i.min_defined || i.min <= v) && (!i.max_defined || i.max >= v);
}

ConstantInterval operator+(const ConstantInterval &a, const ConstantInterval &b) {
    ConstantInterval r;
    if (a.min_defined && b.min_defined) {
        r.min = a.min + b.min;
        r.min_defined = !std::isnan(r.min);
    }
    if (a.max_defined && b.max_defined) {
        r.max = a.max + b.max;
        r.max_defined = !std::isnan(r.max);
    }
    // A sum of numbers is a number unless it may be an infinity less
    // itself; of two non-negative values, non-negative where a number.
    r.number = a.number && b.number &&
               !(may_hold(a, kInf) && may_hold(b, -kInf)) &&
               !(may_hold(a, -kInf) && may_hold(b, kInf));
    r.nonnegative = a.nonnegative && b.nonnegative;
    r.settle_flags();
    return r;
}

ConstantInterval operator-(const ConstantInterval &a) {
    ConstantInterval r;
    r.min = -a.max;
    r.min_defined = a.max_defined;
    r.max = -a.min;
    r.max_defined = a.min_defined;
    r.number = a.number;
    return r;
}

ConstantInterval operator-(const ConstantInterval &a, const ConstantInterval &b) {
    return a + (-b);
}

ConstantInterval operator*(const ConstantInterval &a, const ConstantInterval &b) {
    ConstantInterval r;
    if (!(a.is_everything() && b.is_everything())) {
        r = from_corners({times(lower(a), lower(b)), times(lower(a), upper(b)),
                          times(upper(a), lower(b)), times(upper(a), upper(b))});
    }
    // A product of numbers is a number unless it may be zero times an
    // infinity; where it is a number its sign is the signs' exclusive or,
    // so two factors with their signs clear make one.
    const bool infinite_a = may_hold(a, kInf) || may_hold(a, -kInf);
    const bool infinite_b = may_hold(b, kInf) || may_hold(b, -kInf);
    r.number = a.number && b.number && !(may_hold(a, 0) && infinite_b) &&
               !(infinite_a && may_hold(b, 0));
    r.nonnegative = a.nonnegative && b.nonnegative;
    r.settle_flags();
    return r;
}

ConstantInterval operator/(const ConstantInterval &a, const ConstantInterval &b) {
    ConstantInterval r;
    if (b.min_defined && b.min > 0) {
        r = from_corners({over(lower(a), lower(b)), over(lower(a), upper(b)),
                          over(upper(a), lower(b)), over(upper(a), upper(b))});
    } else if (b.max_defined && b.max < 0) {
        r = -(a / -b);
    }
    // A quotient of numbers is a number unless it may be zero over zero or
    // an infinity over an infinity; its sign is the signs' exclusive or.
    const bool infinite_a = may_hold(a, kInf) || may_hold(a, -kInf);
    const bool infinite_b = may_hold(b, kInf) || may_hold(b, -kInf);
    r.number = a.number && b.number && !(may_hold(a, 0) && may_hold(b, 0)) &&
               !(infinite_a && infinite_b);
    r.nonnegative = a.nonnegative && b.nonnegative;
    r.settle_flags();
    return r;
}

ConstantInterval operator%(const ConstantInterval &a, const ConstantInterval &b) {
    // The remainder's sign is the dividend's, and its magnitude is below the
    // divisor's and no more than the dividend's.
    ConstantInterval d = b;
    if (b.max_defined && b.max < 0) {
        d = -b;
    } else if (!(b.min_defined && b.min > 0)) {
        return ConstantInterval::everything();
    }
    ConstantInterval magnitude = ConstantInterval::bounded_below(0);
    if (d.max_defined) {
        magnitude.max = d.max - 1;
        magnitude.max_defined = true;
    }
    if (a.min_defined && a.min >= 0) {
        if (a.max_defined) {
            magnitude.max = magnitude.max_defined ? std::min(magnitude.max, a.max)
                                                  : a.max;
            magnitude.max_defined = true;
        }
        return magnitude;
    }
    if (a.max_defined && a.max <= 0) {
        if (a.min_defined) {
            magnitude.max = magnitude.max_defined ? std::min(magnitude.max, -a.min)
                                                  : -a.min;
            magnitude.max_defined = true;
        }
        return -magnitude;
    }
    return ConstantInterval::make_union(magnitude, -magnitude);
}

ConstantInterval min(const ConstantInterval &a, const ConstantInterval &b) {
    ConstantInterval r;
    r.max_defined = a.max_defined || b.max_defined;
    r.min_defined = a.min_defined && b.min_defined;
    if (a.max_defined && b.max_defined) {
        r.max = std::min(a.max, b.max);
    } else if (a.max_defined) {
        r.max = a.max;
    } else if (b.max_defined) {
        r.max = b.max;
    }
    if (r.min_defined) {
        r.min = std::min(a.min, b.min);
    }
    // One of the two, bit for bit (`b < a ? b : a`): a number if both
    // are, its sign clear if both are.
    r.number = a.number && b.number;
    r.nonnegative = a.nonnegative && b.nonnegative;
    r.sign_clear = a.sign_clear && b.sign_clear;
    return r;
}

ConstantInterval max(const ConstantInterval &a, const ConstantInterval &b) {
    ConstantInterval r;
    r.min_defined = a.min_defined || b.min_defined;
    r.max_defined = a.max_defined && b.max_defined;
    if (a.min_defined && b.min_defined) {
        r.min = std::max(a.min, b.min);
    } else if (a.min_defined) {
        r.min = a.min;
    } else if (b.min_defined) {
        r.min = b.min;
    }
    if (r.max_defined) {
        r.max = std::max(a.max, b.max);
    }
    r.number = a.number && b.number;
    r.nonnegative = a.nonnegative && b.nonnegative;
    r.sign_clear = a.sign_clear && b.sign_clear;
    return r;
}

ConstantInterval abs(const ConstantInterval &a) {
    ConstantInterval r;
    if (a.is_bounded()) {
        r.max = std::max(-a.min, a.max);
        r.max_defined = true;
    }
    r.min_defined = true;
    if (a.min_defined && a.min > 0) {
        r.min = a.min;
    } else if (a.max_defined && a.max < 0) {
        r.min = -a.max;
    } else {
        r.min = 0;
    }
    // The sign bit is cleared whatever the value, a NaN's included.
    r.number = a.number;
    r.nonnegative = r.sign_clear = true;
    return r;
}

// `[min, max]`, then ` n` where the value is a number, ` +` where its sign
// is clear in every value, and ` (+)` where it is non-negative only as a
// number.
std::ostream &operator<<(std::ostream &os, const ConstantInterval &i) {
    os << "[";
    if (i.min_defined) {
        os << i.min;
    } else {
        os << "-inf";
    }
    os << ", ";
    if (i.max_defined) {
        os << i.max;
    } else {
        os << "inf";
    }
    os << "]";
    if (i.number) {
        os << " n";
    }
    if (i.sign_clear) {
        os << " +";
    } else if (i.nonnegative) {
        os << " (+)";
    }
    return os;
}

//===--------------------------------------------------------------------===//
// The analysis
//===--------------------------------------------------------------------===//

namespace {

// What is known of one value: an interval for a scalar or a vector (its
// lanes'), one of these per field for a struct. `bottom` is no information
// at all -- the state of a union nothing has been added to -- which is not
// everything: everything is a fact (the value is somewhere in the type's
// range), bottom is the absence of one, and the union of bottom and `x` is
// `x`.
struct Abstract {
    ConstantInterval interval;
    vector<Abstract> fields;
    bool bottom = false;

    static Abstract top(const Type &t) {
        Abstract a;
        if (const Struct_t *s = t.as<Struct_t>()) {
            for (const auto &f : s->fields) {
                a.fields.push_back(top(f.type));
            }
        } else {
            a.interval = ConstantInterval::bounds_of_type(t);
        }
        return a;
    }
    static Abstract none() {
        Abstract a;
        a.bottom = true;
        return a;
    }
    // Every scalar zero, down through the fields.
    static Abstract zero(const Type &t) {
        Abstract a;
        if (const Struct_t *s = t.as<Struct_t>()) {
            for (const auto &f : s->fields) {
                a.fields.push_back(zero(f.type));
            }
        } else {
            a.interval = ConstantInterval::single_point(0);
            a.interval.cast_to(t);
        }
        return a;
    }
    static Abstract point(const Type &t, double x) {
        Abstract a;
        a.interval = ConstantInterval::single_point(x);
        a.interval.cast_to(t);
        return a;
    }
    static Abstract of_interval(ConstantInterval i) {
        Abstract a;
        a.interval = std::move(i);
        return a;
    }

    void include(const Abstract &o) {
        if (o.bottom) {
            return;
        }
        if (bottom) {
            *this = o;
            return;
        }
        if (fields.size() != o.fields.size()) {
            // Shapes that disagree: nothing to say of any of it.
            fields.clear();
            interval = ConstantInterval::everything();
            return;
        }
        interval.include(o.interval);
        for (size_t k = 0; k < fields.size(); k++) {
            fields[k].include(o.fields[k]);
        }
    }

    // Narrows this to what it and `o` both allow: what a value's own
    // computation says of it, and what a condition on the way adds.
    void intersect(const Abstract &o) {
        if (bottom || o.bottom || fields.size() != o.fields.size()) {
            return;
        }
        interval = ConstantInterval::make_intersection(interval, o.interval);
        for (size_t k = 0; k < fields.size(); k++) {
            fields[k].intersect(o.fields[k]);
        }
    }

    bool operator==(const Abstract &o) const {
        return bottom == o.bottom && interval == o.interval && fields == o.fields;
    }

    // A struct's field, or this for anything else (an array's elements are
    // one Abstract, and an index step stays where it is).
    Abstract &at(const vector<int> &path, const Type &type) {
        Abstract *a = this;
        Type t = type;
        for (int step : path) {
            if (step < 0) {
                if (t.is_reference() || t.is<Ptr_t>()) {
                    t = t.element_of();
                }
                continue;
            }
            const Struct_t *s = t.as<Struct_t>();
            if (s == nullptr || size_t(step) >= s->fields.size()) {
                return *a;
            }
            if (a->fields.size() != s->fields.size()) {
                *a = a->bottom ? Abstract::none() : top(t);
                if (a->bottom) {
                    a->bottom = false;
                    a->fields.assign(s->fields.size(), Abstract::none());
                }
            }
            a = &a->fields[size_t(step)];
            t = s->fields[size_t(step)].type;
        }
        return *a;
    }
    const Abstract &at(const vector<int> &path, const Type &type) const {
        return const_cast<Abstract *>(this)->at(path, type);
    }
};

// The name a value is known by throughout a function: an instruction's, or
// a block argument's. A value carried from the block that computes it into
// the blocks below is an argument of each under the instruction's own name
// (the name is what says two references are one definition, see
// SSA/Rewrite.cpp), so a fact learned of it in one guise is found in the
// other by the name. A constant has none.
string name_of(const ValuePtr &v) {
    if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data)) {
        return (*in)->name;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return a->name;
    }
    return "";
}

// What a condition says of the values it compares, where it is known to
// hold: in the block a branch on it leads to, and in the arm of a select it
// chooses (see State::learn), by the value's name. What LLVM's
// LazyValueInfo reads off a branch's condition for the block below it, and
// what Halide's simplifier learns of a condition's operands inside `a && b`
// (Simplify_And.cpp, learn_true), over this lattice.
struct Facts {
    std::map<string, ConstantInterval> values;

    bool empty() const { return values.empty(); }

    const ConstantInterval *about(const ValuePtr &v) const {
        const string name = name_of(v);
        if (name.empty()) {
            return nullptr;
        }
        const auto it = values.find(name);
        return it == values.end() ? nullptr : &it->second;
    }

    // Narrows the fact about a value to what `i` says too.
    void add(const ValuePtr &v, const ConstantInterval &i) {
        const string name = name_of(v);
        if (name.empty()) {
            return;
        }
        auto [it, fresh] = values.emplace(name, i);
        if (!fresh) {
            it->second = ConstantInterval::make_intersection(it->second, i);
        }
    }

    // What holds on every one of several ways in: the facts both have, each
    // as wide as either says.
    void meet(const Facts &o) {
        for (auto it = values.begin(); it != values.end();) {
            const auto found = o.values.find(it->first);
            if (found == o.values.end()) {
                it = values.erase(it);
            } else {
                it->second = ConstantInterval::make_union(it->second, found->second);
                ++it;
            }
        }
    }
};

// What a function says to its callers: what its parameters have been
// handed, what each pointer or array parameter pointed at when it was
// entered, what it returns, and what it stores through each pointer or
// array parameter. Everything to begin with; narrowed round by round.
struct Summary {
    vector<Abstract> params;
    // Per parameter, the contents of what a pointer or array parameter
    // names as the callers hand it over: the union over the call sites of
    // what the caller's memory holds there. The argmin's running best,
    // read by the traversal through the pointer its caller passed, is what
    // this is for: the caller stored infinity, and every store since is
    // the traversal's own.
    vector<Abstract> pointees;
    Abstract ret;
    vector<Abstract> stores; // per parameter; bottom for one it writes nothing through
    bool operator==(const Summary &o) const {
        return params == o.params && pointees == o.pointees && ret == o.ret &&
               stores == o.stores;
    }
};

using Summaries = std::map<string, Summary>;

// Where a pointer or an array handle points: a local allocation, a
// parameter, or nowhere this analysis follows.
struct Root {
    const Instruction *local = nullptr; // an Alloca or Alloc
    string param;                       // a pointer or array parameter
    vector<int> path;                   // field indices, -1 for an element
    bool known() const { return local != nullptr || !param.empty(); }
};

// The pointee type of what an allocation or a parameter hands out.
Type held_type_of(const Type &t) {
    if (t.is<Ptr_t>() || t.is_reference()) {
        return t.element_of();
    }
    return t;
}

} // namespace

struct ConstantIntervals::State {
    const Function *func = nullptr;
    // What each instruction was read as, kept under the instruction itself:
    // a rewrite that asks after the analysis (ConstantIntervals::of) frees
    // instructions and makes new ones, and a new one may take a freed one's
    // address, so an entry answers for a pointer only while the instruction
    // it was made for is alive -- the weak reference says, and a pointer
    // reused is a value not yet read.
    struct Known {
        std::weak_ptr<const Instruction> of;
        Abstract value;
    };
    std::unordered_map<const Instruction *, Known> instrs;
    const Abstract *recorded(const shared_ptr<Instruction> &i) const {
        const auto it = instrs.find(i.get());
        if (it == instrs.end()) {
            return nullptr;
        }
        const auto alive = it->second.of.lock();
        return alive != nullptr && alive.get() == i.get() ? &it->second.value
                                                           : nullptr;
    }
    void record(const shared_ptr<Instruction> &i, Abstract a) {
        instrs[i.get()] = Known{i, std::move(a)};
    }
    std::map<std::pair<string, string>, Abstract> args;
    // The contents of each local allocation, by the instruction that made it
    // (an array's elements as one), and of what this function stores through
    // each pointer or array parameter, by the parameter's name.
    std::unordered_map<const Instruction *, Abstract> memory;
    std::map<string, Abstract> param_memory;
    // Allocations and parameters whose address went somewhere this analysis
    // does not follow: their contents are everything.
    std::set<const Instruction *> escaped;
    std::set<string> escaped_params;
    Abstract ret = Abstract::none();
    std::unique_ptr<Definitions> defs;
    std::unique_ptr<Cfg> cfg;
    std::unique_ptr<DomTree> dom;
    std::map<string, size_t> rpo_index;
    // The function's local allocations by name, for an argument that
    // carries one by its name (see root_of).
    std::map<string, const Instruction *> allocas_by_name;

    // The block that declares `name` as an argument: `block` itself, or
    // the nearest block dominating it that does -- a gang's blocks name the
    // values of the blocks above them without taking them as arguments
    // (SSA/Vectorize.cpp), and so does a loop's body the loop's entry's --
    // or the entry for a parameter; null where none does.
    const Block *declaring(const string &block, const string &name) const {
        if (!cfg) {
            return nullptr;
        }
        for (BlockId b = cfg->find(block); b != NO_BLOCK;) {
            const shared_ptr<Block> &held = cfg->block(b);
            for (const Argument &a : held->args) {
                if (a.name == name) {
                    return held.get();
                }
            }
            if (!dom || b == dom->root || b >= dom->idom.size()) {
                break;
            }
            b = dom->idom[b];
        }
        return nullptr;
    }

    // The memory in force for this pass: the previous pass's contents, read
    // by loads, while `memory` collects this pass's stores.
    const std::unordered_map<const Instruction *, Abstract> *seen_memory = nullptr;
    const std::set<const Instruction *> *seen_escaped = nullptr;
    const Summaries *summaries = nullptr;
    // What each pointer or array parameter pointed at on entry, as the
    // callers' summaries say (Summary::pointees), by parameter index; and
    // the previous pass's stores through the parameters, which a load
    // through one reads along with it.
    vector<Abstract> pointees;
    const std::map<string, Abstract> *seen_param_memory = nullptr;
    const std::set<string> *seen_escaped_params = nullptr;
    // What the conditions on the way to each block say of the values read
    // in it (see learn), by block name, for this pass.
    std::map<string, Facts> facts;
    // How deep a select's arms are being read under their condition, so
    // that a select inside an arm inside an arm does not compound.
    int refining = 0;

    //--- values -----------------------------------------------------------

    // What `v` is known to lie between as `block` reads it: what its
    // computation says, narrowed by what a condition on the way to the
    // block says of it.
    Abstract value(const string &block, const ValuePtr &v) const {
        if (v == nullptr) {
            return Abstract::top(Type());
        }
        const Facts *known = nullptr;
        if (const auto f = facts.find(block); f != facts.end() && !f->second.empty()) {
            known = &f->second;
        }
        Abstract r = std::visit(
            overloads{
                [&](const Constant &c) { return constant(c); },
                [&](const shared_ptr<Instruction> &i) {
                    const Abstract *k = recorded(i);
                    return k != nullptr ? *k : Abstract::top(i->type);
                },
                [&](const Argument &a) {
                    auto it = args.find({block, a.name});
                    if (it == args.end()) {
                        // Named here without being an argument here: the
                        // block that declares it is above (see declaring).
                        if (const Block *decl = declaring(block, a.name)) {
                            it = args.find({decl->name, a.name});
                        }
                    }
                    return it != args.end() ? it->second : Abstract::top(a.type);
                },
            },
            v->data);
        if (known != nullptr) {
            if (const ConstantInterval *fact = known->about(v)) {
                r.intersect(Abstract::of_interval(*fact));
            }
        }
        return r;
    }

    //--- facts from conditions --------------------------------------------

    // A bool known in every lane: a constant, or a broadcast of one.
    static std::optional<bool> uniform_bool(const ValuePtr &v) {
        if (const auto *c = std::get_if<Constant>(&v->data)) {
            if (const auto *b = std::get_if<bool>(&c->data)) {
                return *b;
            }
            return std::nullopt;
        }
        if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data);
            in != nullptr && (*in)->op == Instruction::Op::Bc &&
            !(*in)->operands.empty()) {
            return uniform_bool((*in)->operands[0]);
        }
        return std::nullopt;
    }

    // What `cond` being `truth` says of the values it is made of, into
    // `into`: both sides of an `&&` that holds, both of an `||` that does
    // not, the operand of a `!` the other way, a `select(c, true, false)`
    // as `c` and `select(c, false, true)` as `!c` (the linearizer's
    // negation), a broadcast as what it broadcasts; and at the bottom a
    // comparison, which bounds each side by the other's interval -- `x < y`
    // puts `x` at most `y`'s greatest and `y` at least `x`'s least, one
    // closer over integers -- or an equality, which gives each side the
    // other's interval. Nothing of a comparison that fails to hold between
    // floats says where they are (either may be a NaN), so a false `<` is
    // read as a true `>=` and no more. Bounded in depth.
    void learn(const string &block, const ValuePtr &cond, bool truth,
               Facts &into, int depth = 0) const {
        const auto *held = std::get_if<shared_ptr<Instruction>>(&cond->data);
        if (held == nullptr || depth > 8) {
            return;
        }
        const Instruction &in = **held;
        switch (in.op) {
        case Instruction::Op::LAnd:
            if (truth) {
                for (const ValuePtr &o : in.operands) {
                    learn(block, o, true, into, depth + 1);
                }
            }
            return;
        case Instruction::Op::LOr:
            if (!truth) {
                for (const ValuePtr &o : in.operands) {
                    learn(block, o, false, into, depth + 1);
                }
            }
            return;
        case Instruction::Op::Not:
            if (in.operands.size() == 1) {
                learn(block, in.operands[0], !truth, into, depth + 1);
            }
            return;
        case Instruction::Op::Bc:
            if (!in.operands.empty()) {
                learn(block, in.operands[0], truth, into, depth + 1);
            }
            return;
        case Instruction::Op::Select: {
            if (in.operands.size() != 3) {
                return;
            }
            const std::optional<bool> t = uniform_bool(in.operands[1]);
            const std::optional<bool> f = uniform_bool(in.operands[2]);
            if (t.has_value() && f.has_value() && *t != *f) {
                learn(block, in.operands[0], *t ? truth : !truth, into, depth + 1);
            }
            return;
        }
        case Instruction::Op::Lt:
        case Instruction::Op::Leq:
        case Instruction::Op::Eq:
        case Instruction::Op::Ne: {
            if (in.operands.size() != 2) {
                return;
            }
            const Type t = scalar_of(in.operands[0]->get_type());
            if (!(t.is_int_or_uint() || t.is_float())) {
                return;
            }
            const bool integer = t.is_int_or_uint();
            const ValuePtr &x = in.operands[0];
            const ValuePtr &y = in.operands[1];
            const ConstantInterval xi = value(block, x).interval;
            const ConstantInterval yi = value(block, y).interval;
            // A comparison that holds -- `<`, `<=`, `==` -- was between
            // numbers: a NaN passes none. A failed `!=` is a held `==`; a
            // failed `<` is a held `>=`, between numbers too. (A failed `==`
            // or a held `!=` says nothing: a NaN fails every equality.)
            const bool between_numbers =
                in.op == Instruction::Op::Lt || in.op == Instruction::Op::Leq ||
                (in.op == Instruction::Op::Eq) == truth;
            if (between_numbers) {
                ConstantInterval a_number;
                a_number.number = true;
                into.add(x, a_number);
                into.add(y, a_number);
                // And so were the values each is computed from, through the
                // operations a NaN comes out of.
                numbers_beneath(x, into, 0);
                numbers_beneath(y, into, 0);
            }
            // The least value above a bound, or the greatest below one, in
            // the compared type: one over integers, the next float over
            // 32-bit floats, nothing otherwise.
            const auto above = [&](double v) {
                if (integer) {
                    return v + 1;
                }
                if (t.bits() == 32) {
                    float f = float(v);
                    return double(f > v ? f : std::nextafterf(f, kInf));
                }
                return v;
            };
            const auto below = [&](double v) {
                if (integer) {
                    return v - 1;
                }
                if (t.bits() == 32) {
                    float f = float(v);
                    return double(f < v ? f : std::nextafterf(f, -kInf));
                }
                return v;
            };
            // `lo < hi` or `lo <= hi`, strict or not, as the comparison and
            // its truth make it. A number above a non-negative one, or at
            // least a positive one, is positive: its sign is clear.
            const auto ordered = [&](const ValuePtr &lo, const ConstantInterval &loi,
                                     const ValuePtr &hi, const ConstantInterval &hii,
                                     bool strict) {
                if (hii.max_defined) {
                    into.add(lo, ConstantInterval::bounded_above(
                                     strict ? below(hii.max) : hii.max));
                }
                if (loi.min_defined) {
                    ConstantInterval at_least = ConstantInterval::bounded_below(
                        strict ? above(loi.min) : loi.min);
                    at_least.number = true;
                    at_least.nonnegative = strict ? loi.min >= 0 : loi.min > 0;
                    at_least.settle_flags();
                    into.add(hi, at_least);
                }
            };
            if (in.op == Instruction::Op::Lt) {
                if (truth) {
                    ordered(x, xi, y, yi, true);
                } else {
                    ordered(y, yi, x, xi, false);
                }
            } else if (in.op == Instruction::Op::Leq) {
                if (truth) {
                    ordered(x, xi, y, yi, false);
                } else {
                    ordered(y, yi, x, xi, true);
                }
            } else if ((in.op == Instruction::Op::Eq) == truth) {
                into.add(x, yi);
                into.add(y, xi);
            }
            return;
        }
        default:
            return;
        }
    }

    // A value known to be a number was computed from numbers, through the
    // operations a NaN would have come out of: a sum, difference, product or
    // quotient of a NaN is a NaN, as are its negation, absolute value,
    // reciprocal, square root and broadcast. Not through min, max or
    // select, which can leave a NaN behind, nor through a reinterpretation.
    // Bounded in depth.
    void numbers_beneath(const ValuePtr &v, Facts &into, int depth) const {
        const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
        if (held == nullptr || depth > 4) {
            return;
        }
        const Instruction &in = **held;
        bool carries = false;
        switch (in.op) {
        case Instruction::Op::Add:
        case Instruction::Op::Sub:
        case Instruction::Op::Mul:
        case Instruction::Op::Div:
        case Instruction::Op::Abs:
        case Instruction::Op::Bc:
            carries = true;
            break;
        case Instruction::Op::Intrinsic:
            switch (in.intrinsic) {
            case ir::Intrinsic::abs:
            case ir::Intrinsic::rcp:
            case ir::Intrinsic::rcp_estimate:
            case ir::Intrinsic::sqrt:
            case ir::Intrinsic::sqr:
            case ir::Intrinsic::fma:
                carries = true;
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
        if (!carries) {
            return;
        }
        ConstantInterval a_number;
        a_number.number = true;
        for (const ValuePtr &o : in.operands) {
            if (scalar_of(o->get_type()).is_float()) {
                into.add(o, a_number);
                numbers_beneath(o, into, depth + 1);
            }
        }
    }

    // Whether lane `i` of `in` depends on lane `i` of its vector operands
    // and on its scalar operands alone, so that a fact about one lane of a
    // condition carries through it to that lane of the result: what reading
    // a select's arm under the select's mask needs. A reduction, a shuffle,
    // a lane read at an index, a compress or a change of lane count moves
    // lanes and is not.
    static bool lanewise(const Instruction &in) {
        switch (in.op) {
        case Instruction::Op::Abs:
        case Instruction::Op::Add:
        case Instruction::Op::Bc:
        case Instruction::Op::BwAnd:
        case Instruction::Op::BwOr:
        case Instruction::Op::Div:
        case Instruction::Op::Eq:
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr:
        case Instruction::Op::Leq:
        case Instruction::Op::LoadField:
        case Instruction::Op::Lt:
        case Instruction::Op::MakeStruct:
        case Instruction::Op::Max:
        case Instruction::Op::Min:
        case Instruction::Op::Mod:
        case Instruction::Op::Mul:
        case Instruction::Op::Ne:
        case Instruction::Op::Not:
        case Instruction::Op::Select:
        case Instruction::Op::Shl:
        case Instruction::Op::Shr:
        case Instruction::Op::Sub:
        case Instruction::Op::Xor:
            return true;
        case Instruction::Op::Cast:
        case Instruction::Op::Reinterpret: {
            if (in.operands.size() != 1) {
                return false;
            }
            const Type &from = in.operands[0]->get_type();
            if (in.type.is_vector() && from.is_vector()) {
                return in.type.lanes() == from.lanes();
            }
            return !in.type.is_vector() && !from.is_vector();
        }
        case Instruction::Op::Intrinsic:
            switch (in.intrinsic) {
            case ir::Intrinsic::permute:
            case ir::Intrinsic::compress:
                return false;
            default:
                return true;
            }
        default:
            return false;
        }
    }

    // Whether `in` is a value computed from its operands alone -- no memory
    // read, nothing kept on the instruction (a reduction's kind, a shuffle's
    // order, an intrinsic's name are read by eval as they are) -- so that it
    // can be read again under facts about what feeds it.
    static bool recomputable(const Instruction &in) {
        switch (in.op) {
        case Instruction::Op::Load:
        case Instruction::Op::Alloca:
        case Instruction::Op::Alloc:
        case Instruction::Op::GEP:
        case Instruction::Op::FieldPtr:
        case Instruction::Op::AddressOf:
        case Instruction::Op::Set:
            return false;
        default:
            return !in.name.empty();
        }
    }

    using Lookup = std::function<Abstract(const ValuePtr &)>;

    // What `v` lies between with `known` taken as true of the values it
    // names: the value's own reading narrowed by the fact where it has one,
    // and the values computed from such a value read again from what feeds
    // them -- through pure value computations alone, to a budget, and
    // lanewise ones where the facts are a mask's -- and never wider than
    // the plain reading. Halide's simplifier reads an expression under what
    // it has learned the same way.
    Abstract under(const string &block, const ValuePtr &v, const Facts &known,
                   bool vector_facts, size_t &budget,
                   std::map<const Instruction *, Abstract> &memo) {
        Abstract plain = value(block, v);
        if (const ConstantInterval *fact = known.about(v)) {
            plain.intersect(Abstract::of_interval(*fact));
            return plain;
        }
        const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
        if (held == nullptr) {
            return plain;
        }
        const Instruction &in = **held;
        if (const auto it = memo.find(&in); it != memo.end()) {
            return it->second;
        }
        if (budget == 0 || !recomputable(in) || (vector_facts && !lanewise(in))) {
            return plain;
        }
        budget--;
        const Lookup lookup = [&](const ValuePtr &o) {
            return under(block, o, known, vector_facts, budget, memo);
        };
        Abstract r = eval(block, in, &lookup);
        r.intersect(plain);
        memo[&in] = r;
        return r;
    }

    // `arm` of a select read with the select's condition taken as `truth`
    // -- `select(lo < t, t * r, inf)`: inside the first arm `t` is at least
    // `lo`, so the arm is at least `lo * r` -- or read plainly where the
    // condition says nothing, or where arms are already being read two
    // deep.
    Abstract refined(const string &block, const ValuePtr &arm, const ValuePtr &cond,
                     bool truth, const Lookup *lookup) {
        const Abstract plain = lookup != nullptr ? (*lookup)(arm) : value(block, arm);
        if (refining >= 2) {
            return plain;
        }
        Facts known;
        learn(block, cond, truth, known);
        if (known.empty()) {
            return plain;
        }
        refining++;
        size_t budget = 64;
        std::map<const Instruction *, Abstract> memo;
        Abstract r = under(block, arm, known, cond->get_type().is_vector(), budget, memo);
        refining--;
        r.intersect(plain);
        return r;
    }

    // The facts in force in `block`: what every way in agrees on. A
    // predecessor that branches on a bool and reaches the block by one of
    // its two edges alone adds that edge's reading of the condition to its
    // own facts; any other edge hands its facts on as they are; a block a
    // back edge reaches has none (one pass cannot say what holds round a
    // loop), nor has the entry.
    void settle_facts(const shared_ptr<Block> &block, BlockId b) {
        Facts result;
        bool first = true;
        for (const BlockId p : cfg->preds[b]) {
            const shared_ptr<Block> pred = cfg->block(p);
            const auto order = rpo_index.find(pred->name);
            if (order == rpo_index.end()) {
                continue; // never runs
            }
            if (order->second >= rpo_index.at(block->name)) {
                facts.erase(block->name);
                return; // a back edge
            }
            Facts along;
            if (const auto f = facts.find(pred->name); f != facts.end()) {
                along = f->second;
            }
            if (const auto *d = std::get_if<Terminator::Dispatch>(&pred->terminator.data);
                d != nullptr && d->cond != nullptr && d->targets.size() == 2 &&
                scalar_of(d->cond->get_type()).is_bool() &&
                !d->cond->get_type().is_vector()) {
                const bool to_false = d->targets[0].name == block->name;
                const bool to_true = d->targets[1].name == block->name;
                if (to_false != to_true) {
                    learn(pred->name, d->cond, to_true, along);
                }
            }
            if (first) {
                result = std::move(along);
                first = false;
            } else {
                result.meet(along);
            }
        }
        if (result.empty()) {
            facts.erase(block->name);
        } else {
            facts[block->name] = std::move(result);
        }
    }

    static Abstract constant(const Constant &c) {
        return std::visit(
            overloads{
                [&](bool b) { return Abstract::point(c.type, b ? 1.0 : 0.0); },
                [&](int64_t i) { return Abstract::point(c.type, double(i)); },
                [&](uint64_t u) { return Abstract::point(c.type, double(u)); },
                [&](double d) { return Abstract::point(c.type, d); },
                [&](const string &) { return Abstract::top(c.type); },
                [&](const Undefined &) { return Abstract::top(c.type); },
            },
            c.data);
    }

    //--- memory -----------------------------------------------------------

    // What `v`, a pointer or array handle as `block` refers to it, points at.
    Root root_of(const string &block, const ValuePtr &v, int depth = 0) {
        Root none;
        if (v == nullptr || depth > 32) {
            return none;
        }
        if (const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data)) {
            const Instruction &in = **held;
            switch (in.op) {
            case Instruction::Op::Alloca:
            case Instruction::Op::Alloc: {
                Root r;
                r.local = &in;
                return r;
            }
            case Instruction::Op::GEP: {
                if (in.operands.empty()) {
                    return none;
                }
                Root r = root_of(block, in.operands[0], depth + 1);
                if (r.known()) {
                    r.path.push_back(-1);
                }
                return r;
            }
            case Instruction::Op::FieldPtr: {
                if (in.operands.size() != 2) {
                    return none;
                }
                const auto *c = std::get_if<Constant>(&in.operands[1]->data);
                const auto *k = c ? std::get_if<uint64_t>(&c->data) : nullptr;
                if (k == nullptr) {
                    return none;
                }
                Root r = root_of(block, in.operands[0], depth + 1);
                if (r.known()) {
                    r.path.push_back(int(*k));
                }
                return r;
            }
            case Instruction::Op::Set:
                return in.operands.empty() ? none
                                           : root_of(block, in.operands[0], depth + 1);
            default:
                return none;
            }
        }
        if (const auto *named = std::get_if<Argument>(&v->data)) {
            // A name is one value throughout a function (a block carries a
            // value into the blocks below as an argument under the name it
            // was defined by, SSA/Rewrite.cpp; a merge of different values
            // has a name of its own), so an argument named as a parameter
            // is the parameter, and one named as a local allocation is the
            // allocation -- however many blocks it reached its reader
            // through, which following each edge would read as a merge of
            // as many different values.
            for (const Argument &param : func->blocks.front()->args) {
                if (param.name == named->name) {
                    Root r;
                    r.param = named->name;
                    return r;
                }
            }
            if (const auto local = allocas_by_name.find(named->name);
                local != allocas_by_name.end()) {
                Root r;
                r.local = local->second;
                return r;
            }
            const Definition d = defs->of(block, v);
            if (d.value == nullptr) {
                return none;
            }
            if (std::holds_alternative<shared_ptr<Instruction>>(d.value->data)) {
                return root_of(d.block, d.value, depth + 1);
            }
            if (std::holds_alternative<Argument>(d.value->data)) {
                // Named here without being an argument here: the block
                // that declares it is above (see declaring), and the name
                // is resolved from there.
                if (const Block *decl = declaring(d.block, named->name);
                    decl != nullptr && decl->name != d.block) {
                    return root_of(decl->name, v, depth + 1);
                }
            }
        }
        return none;
    }

    Type root_type(const Root &r) const {
        if (r.local != nullptr) {
            return held_type_of(r.local->type);
        }
        for (const Argument &a : func->blocks.front()->args) {
            if (a.name == r.param) {
                return held_type_of(a.type);
            }
        }
        return Type();
    }

    // What the memory `v` points at holds, as the previous pass saw it: a
    // local allocation's contents, or, through a pointer or array
    // parameter, what the callers handed over (Summary::pointees) together
    // with what this function has stored through it since. Everything in
    // the first pass, before the stores are known, and for anything whose
    // address went somewhere this does not follow.
    Abstract loaded(const string &block, const ValuePtr &v, const Type &as) {
        const Root r = root_of(block, v);
        if (seen_memory == nullptr) {
            return Abstract::top(as);
        }
        const Type rt = root_type(r);
        if (r.local != nullptr) {
            if (seen_escaped->count(r.local)) {
                return Abstract::top(as);
            }
            const auto it = seen_memory->find(r.local);
            if (it == seen_memory->end()) {
                return Abstract::top(as);
            }
            const Abstract &a = it->second.at(r.path, rt);
            return a.bottom ? Abstract::top(as) : a;
        }
        if (!r.param.empty()) {
            if (seen_escaped_params == nullptr || seen_escaped_params->count(r.param)) {
                return Abstract::top(as);
            }
            Abstract contents = Abstract::none();
            const vector<Argument> &params = func->blocks.front()->args;
            for (size_t k = 0; k < params.size() && k < pointees.size(); k++) {
                if (params[k].name == r.param) {
                    contents.include(pointees[k]);
                }
            }
            if (seen_param_memory != nullptr) {
                if (const auto it = seen_param_memory->find(r.param);
                    it != seen_param_memory->end()) {
                    contents.include(it->second);
                }
            }
            if (contents.bottom) {
                return Abstract::top(as);
            }
            const Abstract &a = contents.at(r.path, rt);
            return a.bottom ? Abstract::top(as) : a;
        }
        return Abstract::top(as);
    }

    void stored(const string &block, const ValuePtr &where, const Abstract &what) {
        const Root r = root_of(block, where);
        if (r.local != nullptr) {
            local_contents(r.local).at(r.path, root_type(r)).include(what);
        } else if (!r.param.empty()) {
            param_contents(r.param).at(r.path, root_type(r)).include(what);
        }
    }

    // The contents of an allocation or a parameter's pointee, nothing until
    // the first store (not everything: that is what a load of it reads
    // when no store was seen, and it must not swallow the first store).
    Abstract &local_contents(const Instruction *root) {
        auto it = memory.find(root);
        if (it == memory.end()) {
            it = memory.emplace(root, Abstract::none()).first;
        }
        return it->second;
    }
    Abstract &param_contents(const string &param) {
        auto it = param_memory.find(param);
        if (it == param_memory.end()) {
            it = param_memory.emplace(param, Abstract::none()).first;
        }
        return it->second;
    }

    void escaped_through(const string &block, const ValuePtr &v) {
        const Root r = root_of(block, v);
        if (r.local != nullptr) {
            escaped.insert(r.local);
        } else if (!r.param.empty()) {
            escaped_params.insert(r.param);
        }
    }

    //--- instructions -----------------------------------------------------

    // What `in` computes from what its operands lie between: as `block`
    // reads them, or as `lookup` says of them when one is given (a select's
    // arm read under the select's condition, see under).
    Abstract eval(const string &block, const Instruction &in,
                  const Lookup *lookup = nullptr) {
        const Type &type = in.type;
        const auto A = [&](size_t k) {
            if (k >= in.operands.size()) {
                return Abstract::top(Type());
            }
            return lookup != nullptr ? (*lookup)(in.operands[k])
                                     : value(block, in.operands[k]);
        };
        const auto I = [&](size_t k) { return A(k).interval; };
        const auto fit = [&](ConstantInterval i) {
            i.cast_to(type);
            return Abstract::of_interval(i);
        };
        const auto top = [&] { return Abstract::top(type); };
        const uint32_t lanes = type.is_vector() ? type.lanes() : 1;
        switch (in.op) {
        case Instruction::Op::Add:
            return fit(I(0) + I(1));
        case Instruction::Op::Sub:
            return fit(I(0) - I(1));
        case Instruction::Op::Mul:
            return fit(I(0) * I(1));
        case Instruction::Op::Div: {
            ConstantInterval q = I(0) / I(1);
            if (is_integer_type(type)) {
                // Toward zero, which never moves a bound outward.
                if (q.min_defined) {
                    q.min = std::trunc(q.min);
                }
                if (q.max_defined) {
                    q.max = std::trunc(q.max);
                }
            }
            return fit(q);
        }
        case Instruction::Op::Mod:
            return is_integer_type(type) ? fit(I(0) % I(1)) : top();
        case Instruction::Op::Min:
            return fit(min(I(0), I(1)));
        case Instruction::Op::Max:
            return fit(max(I(0), I(1)));
        case Instruction::Op::Abs:
            return fit(abs(I(0)));
        case Instruction::Op::Select: {
            if (in.operands.size() != 3) {
                return top();
            }
            // Each arm read with the condition as it is there: true in the
            // first, false in the second (see refined).
            Abstract r = refined(block, in.operands[1], in.operands[0], true, lookup);
            r.include(refined(block, in.operands[2], in.operands[0], false, lookup));
            // `select(x < c, x, x - c)` is x brought below c once: at most
            // c - 1 where x was below c already, and x's bound less c where
            // it was not -- the shape a remainder by a constant becomes
            // (SSA/InvariantDivision.cpp), and what the next remainder in a
            // chain of them is asked about.
            const auto *cond =
                std::get_if<shared_ptr<Instruction>>(&in.operands[0]->data);
            const auto *sub =
                std::get_if<shared_ptr<Instruction>>(&in.operands[2]->data);
            if (cond != nullptr && sub != nullptr &&
                (*cond)->op == Instruction::Op::Lt &&
                (*sub)->op == Instruction::Op::Sub &&
                (*cond)->operands.size() == 2 && (*sub)->operands.size() == 2 &&
                same_value((*cond)->operands[0], in.operands[1]) &&
                same_value((*sub)->operands[0], in.operands[1]) &&
                same_value((*cond)->operands[1], (*sub)->operands[1])) {
                const ConstantInterval c = value(block, (*cond)->operands[1]).interval;
                const ConstantInterval x = I(1);
                if (c.is_single_point() && c.min > 0 && x.max_defined &&
                    r.fields.empty()) {
                    const double below = std::min(c.min - 1, x.max);
                    const double above = x.max - c.min;
                    r.interval.max = std::max(below, above);
                    r.interval.max_defined = true;
                    if (x.min_defined && x.min >= 0) {
                        r.interval.min = std::max(r.interval.min_defined ? r.interval.min : 0.0, 0.0);
                        r.interval.min_defined = true;
                    }
                }
            }
            return r;
        }
        case Instruction::Op::Cast: {
            if (in.operands.empty()) {
                return top();
            }
            const Type from = in.operands[0]->get_type();
            ConstantInterval i = I(0);
            if (from.is_bool() || scalar_of(from).is_bool()) {
                i = ConstantInterval::make_intersection(i, {0, 1});
            }
            if (is_integer_type(type) && !is_integer_type(from)) {
                // A float to an integer is truncated toward zero.
                if (i.min_defined) {
                    i.min = std::trunc(i.min);
                }
                if (i.max_defined) {
                    i.max = std::trunc(i.max);
                }
            } else if (!is_integer_type(type) && is_integer_type(from)) {
                // An integer made a float is a number, non-negative where
                // the integer was.
                i.number = true;
                i.nonnegative = i.sign_clear = i.min_defined && i.min >= 0;
            }
            return fit(i);
        }
        case Instruction::Op::Reinterpret: {
            if (in.operands.size() != 1) {
                return top();
            }
            const Type from = in.operands[0]->get_type();
            if (equals(from, type)) {
                return A(0);
            }
            // A float's bits as a 32-bit integer order as the float does
            // where the float is not negative, and the other way round:
            // [0, inf] as floats is [0, 0x7f800000] as bits, the bits of
            // +inf the greatest. A NaN is outside this lattice as it is
            // outside every float interval (see ConstantInterval), so its
            // bits, above inf's, are not what these bounds speak of -- the
            // reading the sort's flip rule takes of a key that cannot be
            // negative, and the one Embree's slab test on bits (`maxi`,
            // `mini`, `asInt(tNear) <= asInt(tFar)`) rests on, which is
            // what this is for: the bits of `max(.., asInt(tnear))` with
            // `tnear` in [0, inf] are non-negative, and read back as a float
            // the distance is in [0, inf] again, so the key made of it
            // needs no flip.
            const Type f = scalar_of(from), t = scalar_of(type);
            if (!f.is_scalar() || !t.is_scalar()) {
                return top(); // an array viewed as another, say
            }
            const ConstantInterval i = I(0);
            constexpr double kInfBits = 0x7f800000;
            if (f.is_float() && f.bits() == 32 && is_integer_type(t) &&
                t.bits() == 32) {
                // A non-negative number's bits are between its bounds' bits,
                // +inf's the greatest; a value whose sign is clear but may
                // be a NaN has bits up to 0x7fffffff; anything else -- a
                // negative zero's bits are the integer's least -- is
                // everything.
                if (!i.sign_clear) {
                    return top();
                }
                float lo = i.min_defined ? float(std::max(i.min, 0.0)) : 0.0f;
                if (double(lo) > i.min) {
                    lo = std::nextafterf(lo, 0.0f);
                }
                double hi_bits = 0x7fffffff;
                if (i.number) {
                    hi_bits = kInfBits;
                    if (i.max_defined && std::isfinite(i.max)) {
                        float hi = float(i.max);
                        if (double(hi) < i.max) {
                            hi = std::nextafterf(hi, kInf);
                        }
                        hi_bits = std::isfinite(hi)
                                      ? double(std::bit_cast<uint32_t>(hi))
                                      : kInfBits;
                    }
                }
                return fit(ConstantInterval(double(std::bit_cast<uint32_t>(lo)),
                                            hi_bits));
            }
            if (is_integer_type(f) && f.bits() == 32 && t.is_float() &&
                t.bits() == 32) {
                // Bits with the sign clear, [0, 0x7fffffff], read as floats
                // from +0 up: those past inf's are NaNs, outside the
                // lattice. An unsigned value that may reach the sign bit
                // (a word whose sign a xor flips, `asInt(x) ^ asInt(sgn)`)
                // may be any float at all.
                constexpr double kSignBit = 0x80000000;
                if (!(i.min_defined && i.min >= 0 && i.min <= kInfBits) ||
                    !(i.max_defined && i.max < kSignBit)) {
                    return top();
                }
                const float lo = std::bit_cast<float>(uint32_t(i.min));
                const double hi = i.max <= kInfBits
                                      ? double(std::bit_cast<float>(uint32_t(i.max)))
                                      : kInf;
                // Bits with the sign clear read as a float with its sign
                // clear in every value, a number where none lie past
                // infinity's.
                ConstantInterval r(double(lo), hi);
                r.sign_clear = r.nonnegative = true;
                r.number = i.max <= kInfBits;
                return fit(r);
            }
            return top();
        }
        case Instruction::Op::Bc:
            return fit(I(0));
        case Instruction::Op::Ramp: {
            // base + stride * k for k in [0, lanes).
            const ConstantInterval steps(0, double(lanes > 0 ? lanes - 1 : 0));
            return fit(I(0) + I(1) * steps);
        }
        case Instruction::Op::ExtractIdx: {
            if (in.operands.empty()) {
                return top();
            }
            const Type &container = in.operands[0]->get_type();
            if (container.is_vector()) {
                return fit(I(0));
            }
            if (container.is_reference()) {
                Abstract a = loaded(block, in.operands[0], type);
                return a;
            }
            return top();
        }
        case Instruction::Op::Reduce: {
            const ConstantInterval i = I(0);
            const uint32_t n = in.operands.empty() ? 1
                               : in.operands[0]->get_type().is_vector()
                                   ? in.operands[0]->get_type().lanes()
                                   : 1;
            switch (in.reduce) {
            case ir::VectorReduce::Min:
            case ir::VectorReduce::Max:
            case ir::VectorReduce::And:
            case ir::VectorReduce::Or:
                return fit(i);
            case ir::VectorReduce::Add:
                return fit(i * ConstantInterval::single_point(double(n)));
            case ir::VectorReduce::Idxmin:
            case ir::VectorReduce::Idxmax:
                return fit({0, double(n > 0 ? n - 1 : 0)});
            default:
                return top();
            }
        }
        case Instruction::Op::Shuffle: {
            Abstract r = Abstract::none();
            for (size_t k = 0; k < in.operands.size(); k++) {
                r.include(Abstract::of_interval(I(k)));
            }
            return r.bottom ? top() : fit(r.interval);
        }
        case Instruction::Op::LoadField: {
            if (in.operands.size() != 2) {
                return top();
            }
            const auto *c = std::get_if<Constant>(&in.operands[1]->data);
            const auto *k = c ? std::get_if<uint64_t>(&c->data) : nullptr;
            const Abstract s = A(0);
            if (k == nullptr || *k >= s.fields.size() || s.fields[*k].bottom) {
                return top();
            }
            return s.fields[*k];
        }
        case Instruction::Op::MakeStruct: {
            // Built with no operands at all -- an option's empty variant,
            // `make_struct<_option>()` -- a struct is all zeros (the
            // backends' null value); built with some, the fields given are
            // those and the rest are the struct's declared defaults, which
            // are not followed here.
            if (in.operands.empty()) {
                return Abstract::zero(type);
            }
            Abstract r = Abstract::top(type);
            if (in.operands.size() > r.fields.size()) {
                return r;
            }
            for (size_t k = 0; k < in.operands.size(); k++) {
                r.fields[k] = A(k);
                if (r.fields[k].bottom) {
                    r.fields[k] = Abstract::top(in.operands[k]->get_type());
                }
            }
            return r;
        }
        case Instruction::Op::Load:
            return in.operands.empty() ? top() : loaded(block, in.operands[0], type);
        case Instruction::Op::Set:
            return A(0);
        case Instruction::Op::Popcount:
            return fit({0, double(in.operands.empty() ? 0
                                  : in.operands[0]->get_type().is_vector()
                                      ? in.operands[0]->get_type().lanes()
                                      : 1)});
        case Instruction::Op::Any:
        case Instruction::Op::Vote:
        case Instruction::Op::Lt:
        case Instruction::Op::Leq:
        case Instruction::Op::Eq:
        case Instruction::Op::Ne:
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr:
            return fit({0, 1});
        case Instruction::Op::Not:
            return scalar_of(type).is_bool() ? fit({0, 1}) : top();
        case Instruction::Op::BwAnd: {
            // Below either operand that is non-negative, whatever the other
            // holds -- the non-negative one's sign bit is clear, so the
            // result's is, and no bit is set the operand has clear.
            ConstantInterval r = ConstantInterval::bounds_of_type(type);
            bool any = false;
            for (size_t k = 0; k < 2 && k < in.operands.size(); k++) {
                const ConstantInterval i = I(k);
                if (i.min_defined && i.min >= 0) {
                    any = true;
                    r.min = 0;
                    r.min_defined = true;
                    if (i.max_defined) {
                        r.max = r.max_defined ? std::min(r.max, i.max) : i.max;
                        r.max_defined = true;
                    }
                }
            }
            return any ? fit(r) : top();
        }
        case Instruction::Op::BwOr:
        case Instruction::Op::Xor: {
            // Of two non-negative values, below the power of two above the
            // larger: no bit is set that neither has.
            const ConstantInterval a = I(0), b = I(1);
            if (a.min_defined && a.min >= 0 && b.min_defined && b.min >= 0 &&
                a.max_defined && b.max_defined) {
                const double m = std::max(a.max, b.max);
                const double top_bit = std::ldexp(1.0, int(std::ceil(std::log2(m + 1))));
                return fit({0, top_bit - 1});
            }
            if (a.min_defined && a.min >= 0 && b.min_defined && b.min >= 0) {
                return fit(ConstantInterval::bounded_below(0));
            }
            return top();
        }
        case Instruction::Op::Shl: {
            const ConstantInterval a = I(0), k = I(1);
            if (a.min_defined && a.min >= 0 && k.is_bounded() && k.min >= 0 &&
                k.max < 64) {
                ConstantInterval r = ConstantInterval::bounded_below(
                    a.min * std::ldexp(1.0, int(k.min)));
                if (a.max_defined) {
                    r.max = a.max * std::ldexp(1.0, int(k.max));
                    r.max_defined = true;
                }
                return fit(r);
            }
            return top();
        }
        case Instruction::Op::Shr: {
            // A shift right is a floor division by a power of two, which
            // is monotone in the value for either sign.
            const ConstantInterval a = I(0), k = I(1);
            if (k.is_bounded() && k.min >= 0 && k.max < 64) {
                ConstantInterval r;
                if (a.min_defined) {
                    r.min = std::floor(a.min / std::ldexp(1.0, int(a.min < 0 ? k.min : k.max)));
                    r.min_defined = true;
                }
                if (a.max_defined) {
                    r.max = std::floor(a.max / std::ldexp(1.0, int(a.max < 0 ? k.max : k.min)));
                    r.max_defined = true;
                }
                return fit(r);
            }
            return top();
        }
        case Instruction::Op::Inf:
            return Abstract::of_interval(ConstantInterval::single_point(kInf));
        case Instruction::Op::Eps: {
            const Type t = scalar_of(type);
            const double e = t.is_float() ? (t.bits() == 64 ? std::numeric_limits<double>::epsilon()
                                             : t.bits() == 32 ? double(std::numeric_limits<float>::epsilon())
                                                              : std::ldexp(1.0, -10))
                                          : 1.0;
            return Abstract::of_interval(ConstantInterval::single_point(e));
        }
        case Instruction::Op::Intrinsic:
            return intrinsic(block, in, lookup);
        default:
            return top();
        }
    }

    Abstract intrinsic(const string &block, const Instruction &in,
                       const Lookup *lookup = nullptr) {
        const Type &type = in.type;
        const auto I = [&](size_t k) {
            if (k >= in.operands.size()) {
                return ConstantInterval::everything();
            }
            return lookup != nullptr ? (*lookup)(in.operands[k]).interval
                                     : value(block, in.operands[k]).interval;
        };
        const auto fit = [&](ConstantInterval i) {
            i.cast_to(type);
            return Abstract::of_interval(i);
        };
        const auto top = [&] { return Abstract::top(type); };
        switch (in.intrinsic) {
        case ir::Intrinsic::abs:
            return fit(abs(I(0)));
        case ir::Intrinsic::min:
            return fit(min(I(0), I(1)));
        case ir::Intrinsic::max:
            return fit(max(I(0), I(1)));
        case ir::Intrinsic::fma:
            return fit(I(0) * I(1) + I(2));
        case ir::Intrinsic::sqrt: {
            // Of a negative there is no number (a NaN), so the bounds are
            // those of the non-negative part.
            const ConstantInterval a = I(0);
            ConstantInterval r = ConstantInterval::bounded_below(
                a.min_defined && a.min > 0 ? std::sqrt(a.min) : 0.0);
            if (a.max_defined) {
                r.max = std::sqrt(std::max(a.max, 0.0));
                r.max_defined = true;
            }
            // A number where the operand is one and not negative (the root
            // of a negative zero is a negative zero); the sign the operand's.
            r.number = a.number && a.min_defined && a.min >= 0;
            r.nonnegative = a.nonnegative;
            r.settle_flags();
            return fit(r);
        }
        case ir::Intrinsic::sqr:
            return fit(I(0) * I(0));
        case ir::Intrinsic::rcp:
        case ir::Intrinsic::rcp_estimate: {
            // The reciprocal of an interval on one side of zero lies between
            // the reciprocals of its ends, the other way round, and goes to
            // infinity at zero (rcp(+0) is +inf, rcp(-0) is -inf: a lower
            // bound of zero is read as +0, the lattice's reading of a
            // non-negative float, see ConstantInterval); an interval that
            // crosses zero gives everything. The instruction is an estimate
            // and a Newton step (CodeGen_X86), within a few units in the last
            // place of the quotient, so each end is moved outward by more
            // than that before it is used; the bare estimate is within a
            // relative 1.5 * 2^-12 of it (`rcpps`, the coarser of the two
            // instructions), and is moved outward by more than that.
            const ConstantInterval a = I(0);
            const double tolerance =
                in.intrinsic == ir::Intrinsic::rcp ? 0x1p-20 : 0x1p-10;
            const auto outward = [tolerance](double v, bool down) {
                const double slack = std::abs(v) * tolerance;
                return down ? v - slack : v + slack;
            };
            // A number where the operand is one (the reciprocal of zero is
            // an infinity, of an infinity zero), with the operand's sign.
            ConstantInterval r;
            if (a.min_defined && a.min >= 0) {
                r = ConstantInterval::bounded_below(
                    a.max_defined && a.max > 0 && std::isfinite(a.max)
                        ? outward(1.0 / a.max, true)
                        : 0.0);
                if (a.min > 0) {
                    r.max = outward(1.0 / a.min, false);
                    r.max_defined = true;
                }
            } else if (a.max_defined && a.max <= 0) {
                r = ConstantInterval::bounded_above(
                    a.min_defined && a.min < 0 && std::isfinite(a.min)
                        ? outward(1.0 / a.min, false)
                        : 0.0);
                if (a.max < 0) {
                    r.min = outward(1.0 / a.max, true);
                    r.min_defined = true;
                }
            }
            r.number = a.number;
            r.nonnegative = a.nonnegative;
            r.settle_flags();
            return fit(r);
        }
        case ir::Intrinsic::exp: {
            // Positive where it is a number, which it is where the operand
            // is (an infinity gives zero or an infinity).
            ConstantInterval r = ConstantInterval::bounded_below(0);
            r.number = I(0).number;
            r.nonnegative = true;
            r.settle_flags();
            return fit(r);
        }
        case ir::Intrinsic::cos:
        case ir::Intrinsic::sin: {
            // A number where the operand is a finite one.
            const ConstantInterval a = I(0);
            ConstantInterval r{-1, 1};
            r.number = a.number && a.is_bounded() && std::isfinite(a.min) &&
                       std::isfinite(a.max);
            return fit(r);
        }
        case ir::Intrinsic::clz:
        case ir::Intrinsic::ctz:
            return fit({0, double(scalar_of(type).bits())});
        case ir::Intrinsic::permute:
            return fit(I(0));
        case ir::Intrinsic::compress:
            // With a fill, the packed lanes and the fill's; without one the
            // lanes past the packed ones hold anything.
            return in.operands.size() == 3
                       ? fit(ConstantInterval::make_union(I(0), I(2)))
                       : top();
        default:
            return top();
        }
    }

    //--- a pass over the function -------------------------------------------

    // The arguments a jump from `pred` hands `block`, as what each parameter
    // of `block` is handed: the callee's return first for a call's
    // continuation that keeps it, and the loop index first for a parfor's
    // body.
    void arrive(const shared_ptr<Block> &pred, const Terminator::Jump &jump,
                const shared_ptr<Block> &block, bool retreating,
                vector<Abstract> &into, vector<bool> &everything_in) {
        size_t offset = 0;
        const auto *call = std::get_if<Terminator::Call>(&pred->terminator.data);
        const auto *multi = std::get_if<Terminator::MultiCall>(&pred->terminator.data);
        const auto *par = std::get_if<Terminator::ParFor>(&pred->terminator.data);
        if (call != nullptr && &jump == &call->cont && !call->drop) {
            offset = 1;
            if (!block->args.empty()) {
                into[0].include(returned(call->call.name, block->args[0].type));
            }
        } else if (multi != nullptr && &jump == &multi->cont && !multi->drop) {
            offset = 1;
            if (!block->args.empty()) {
                into[0].include(returned(multi->call.name, block->args[0].type));
            }
        } else if (par != nullptr && &jump == &par->body) {
            offset = 1;
            if (!block->args.empty()) {
                const ConstantInterval start = value(pred->name, par->start).interval;
                const ConstantInterval end = value(pred->name, par->end).interval;
                ConstantInterval index;
                if (start.min_defined) {
                    index.min = start.min;
                    index.min_defined = true;
                }
                if (end.max_defined) {
                    index.max = end.max - 1;
                    index.max_defined = true;
                }
                index.cast_to(block->args[0].type);
                into[0].include(Abstract::of_interval(index));
            }
        }
        for (size_t i = offset; i < block->args.size(); i++) {
            const size_t j = i - offset;
            if (j >= jump.args.size()) {
                everything_in[i] = true;
                continue;
            }
            const ValuePtr &v = jump.args[j];
            if (retreating) {
                // Round a loop: a value handed back as itself adds nothing;
                // anything else is more than one pass can follow.
                const auto *a = std::get_if<Argument>(&v->data);
                if (a != nullptr && a->name == block->args[i].name) {
                    continue;
                }
                everything_in[i] = true;
                continue;
            }
            into[i].include(value(pred->name, v));
        }
    }

    Abstract returned(const string &callee, const Type &as) const {
        if (summaries != nullptr) {
            const auto it = summaries->find(callee);
            if (it != summaries->end() && !it->second.ret.bottom) {
                return it->second.ret;
            }
        }
        return Abstract::top(as);
    }

    // A call of `callee` with `args`, made from `block`: what it stores
    // through the pointers it is handed lands in the caller's memory, and a
    // pointer handed to a callee this analysis has no summary of has gone
    // somewhere unknown.
    void called(const string &block, const string &callee,
                const vector<ValuePtr> &args) {
        const Summary *s = nullptr;
        if (summaries != nullptr) {
            const auto it = summaries->find(callee);
            if (it != summaries->end()) {
                s = &it->second;
            }
        }
        for (size_t k = 0; k < args.size(); k++) {
            const Type &t = args[k]->get_type();
            if (!(t.is<Ptr_t>() || t.is_reference())) {
                continue;
            }
            if (s == nullptr || k >= s->stores.size()) {
                escaped_through(block, args[k]);
                continue;
            }
            // This function handing its own parameter to itself: what the
            // callee stores through it is what this function stores through
            // it, collected here already. Taking the summary's word for it
            // too would hand each round the round before's, which began at
            // everything, and nothing would ever narrow.
            if (callee == func->blocks.front()->name) {
                const vector<Argument> &params = func->blocks.front()->args;
                if (const auto *a = std::get_if<Argument>(&args[k]->data);
                    a != nullptr && k < params.size() && a->name == params[k].name) {
                    continue;
                }
            }
            if (!s->stores[k].bottom) {
                const Root r = root_of(block, args[k]);
                if (r.local != nullptr) {
                    // The callee's stores are the pointee's whole shape, laid
                    // over what the pointer names in the local.
                    local_contents(r.local).at(r.path, root_type(r)).include(s->stores[k]);
                } else if (!r.param.empty()) {
                    param_contents(r.param).at(r.path, root_type(r)).include(s->stores[k]);
                }
            }
        }
    }

    // One pass: every block in reverse postorder, its arguments from its
    // predecessors and its instructions in order; the stores it meets go to
    // `memory` for the next pass, and `ret` collects the returns.
    void pass(const vector<Abstract> &params) {
        instrs.clear();
        args.clear();
        facts.clear();
        memory.clear();
        param_memory.clear();
        escaped.clear();
        escaped_params.clear();
        ret = Abstract::none();
        const shared_ptr<Block> &entry = func->blocks.front();
        for (size_t i = 0; i < entry->args.size(); i++) {
            args[{entry->name, entry->args[i].name}] =
                i < params.size() && !params[i].bottom
                    ? params[i]
                    : Abstract::top(entry->args[i].type);
        }
        for (const BlockId b : cfg->rpo) {
            const shared_ptr<Block> block = cfg->block(b);
            if (block != entry) {
                vector<Abstract> into(block->args.size(), Abstract::none());
                vector<bool> everything_in(block->args.size(), false);
                for (const BlockId p : cfg->preds[b]) {
                    const shared_ptr<Block> pred = cfg->block(p);
                    // A predecessor the entry cannot reach never runs, and
                    // hands this block nothing.
                    const auto order = rpo_index.find(pred->name);
                    if (order == rpo_index.end()) {
                        continue;
                    }
                    const bool retreating = order->second >= rpo_index.at(block->name);
                    for (const Terminator::Jump *jump : jumps_of(*pred)) {
                        if (jump->name == block->name) {
                            arrive(pred, *jump, block, retreating, into, everything_in);
                        }
                    }
                }
                for (size_t i = 0; i < block->args.size(); i++) {
                    args[{block->name, block->args[i].name}] =
                        everything_in[i] || into[i].bottom
                            ? Abstract::top(block->args[i].type)
                            : into[i];
                }
                settle_facts(block, b);
            }
            for (const shared_ptr<Instruction> &in : block->instrs) {
                switch (in->op) {
                case Instruction::Op::Store:
                    // Where it stores and what; a vectorized store's third
                    // operand is its mask, which changes which lanes are
                    // written and not what may be.
                    if (in->operands.size() >= 2) {
                        stored(block->name, in->operands[0],
                               value(block->name, in->operands[1]));
                        // A pointer put into memory has gone somewhere this
                        // does not follow.
                        const Type &t = in->operands[1]->get_type();
                        if (t.is<Ptr_t>() || t.is_reference()) {
                            escaped_through(block->name, in->operands[1]);
                        }
                    }
                    break;
                case Instruction::Op::AccArgmin:
                case Instruction::Op::AccArgmax:
                case Instruction::Op::AccMin:
                case Instruction::Op::AccMax:
                    // The least or the greatest of what was there and what
                    // arrives is one of the two, so what the memory holds
                    // after it is within the union of both: a store, for
                    // these bounds. The argmin's running best is kept this
                    // way, and begins at the infinity its caller stored.
                    if (in->operands.size() >= 2) {
                        stored(block->name, in->operands[0],
                               value(block->name, in->operands[1]));
                        const Type &t = in->operands[1]->get_type();
                        if (t.is<Ptr_t>() || t.is_reference()) {
                            escaped_through(block->name, in->operands[1]);
                        }
                        break;
                    }
                    [[fallthrough]];
                case Instruction::Op::AtomicAdd:
                case Instruction::Op::AccAdd:
                case Instruction::Op::AccMul:
                case Instruction::Op::AccSub:
                case Instruction::Op::Append:
                case Instruction::Op::Push:
                case Instruction::Op::AddressOf:
                case Instruction::Op::Reinterpret:
                case Instruction::Op::Cast:
                    for (const ValuePtr &v : in->operands) {
                        const Type &t = v->get_type();
                        if (t.is<Ptr_t>() || t.is_reference()) {
                            escaped_through(block->name, v);
                        }
                    }
                    break;
                case Instruction::Op::Intrinsic:
                    if (in->intrinsic != ir::Intrinsic::prefetch) {
                        for (const ValuePtr &v : in->operands) {
                            const Type &t = v->get_type();
                            if (t.is<Ptr_t>() || t.is_reference()) {
                                escaped_through(block->name, v);
                            }
                        }
                    }
                    break;
                default:
                    break;
                }
                if (!in->name.empty()) {
                    record(in, eval(block->name, *in));
                }
            }
            std::visit(
                overloads{
                    [&](const Terminator::Return &r) {
                        if (r.value != nullptr) {
                            ret.include(value(block->name, r.value));
                        }
                    },
                    [&](const Terminator::Call &c) {
                        called(block->name, c.call.name, c.call.args);
                    },
                    [&](const Terminator::MultiCall &c) {
                        for (size_t k = 0; k < c.varying.size(); k++) {
                            called(block->name, c.call.name, c.call_args(k));
                        }
                    },
                    [&](const auto &) {},
                },
                block->terminator.data);
        }
    }

    // The function analysed under `params`: passes until the memory the
    // loads read is the memory the stores made, each sound on its own since
    // a load first reads everything. A value that reaches a load through a
    // chain of stores -- the leaf's hit into one option, that into the
    // other, the distance read from it into the argmin's best behind the
    // pointer, then the best read at the node -- needs a pass per link,
    // so the cap is eight; a function with less to carry stops sooner.
    void analyze(const vector<Abstract> &params, const vector<Abstract> &entry_pointees,
                 const Summaries *all) {
        summaries = all;
        pointees = entry_pointees;
        if (!cfg) {
            cfg = std::make_unique<Cfg>(*func);
            dom = std::make_unique<DomTree>(compute_dominator_tree(*cfg));
            for (const shared_ptr<Block> &block : func->blocks) {
                for (const shared_ptr<Instruction> &in : block->instrs) {
                    if ((in->op == Instruction::Op::Alloca ||
                         in->op == Instruction::Op::Alloc) &&
                        !in->name.empty()) {
                        allocas_by_name[in->name] = in.get();
                    }
                }
            }
            for (size_t i = 0; i < cfg->rpo.size(); i++) {
                rpo_index[cfg->name(cfg->rpo[i])] = i;
            }
            defs = std::make_unique<Definitions>(*func, /*lenient=*/true);
        }
        std::unordered_map<const Instruction *, Abstract> last_memory;
        std::set<const Instruction *> last_escaped;
        std::map<string, Abstract> last_param_memory;
        std::set<string> last_escaped_params;
        for (int round = 0; round < 8; round++) {
            seen_memory = round == 0 ? nullptr : &last_memory;
            seen_escaped = round == 0 ? nullptr : &last_escaped;
            seen_param_memory = round == 0 ? nullptr : &last_param_memory;
            seen_escaped_params = round == 0 ? nullptr : &last_escaped_params;
            pass(params);
            // Everything an escaped allocation holds is unknown.
            for (const Instruction *e : escaped) {
                memory.erase(e);
            }
            const bool same = round > 0 && memory == last_memory &&
                              escaped == last_escaped &&
                              param_memory == last_param_memory &&
                              escaped_params == last_escaped_params;
            last_memory = memory;
            last_escaped = escaped;
            last_param_memory = param_memory;
            last_escaped_params = escaped_params;
            if (same) {
                break;
            }
        }
        // Keep the last pass's memory as what loads were read against, for
        // queries after the fact.
        kept_memory = last_memory;
        kept_escaped = last_escaped;
        kept_param_memory = last_param_memory;
        kept_escaped_params = last_escaped_params;
        seen_memory = &kept_memory;
        seen_escaped = &kept_escaped;
        seen_param_memory = &kept_param_memory;
        seen_escaped_params = &kept_escaped_params;
    }
    std::unordered_map<const Instruction *, Abstract> kept_memory;
    std::set<const Instruction *> kept_escaped;
    std::map<string, Abstract> kept_param_memory;
    std::set<string> kept_escaped_params;

    // What the memory a pointer or array argument `v` names holds as this
    // function hands it to a callee, from `block`: a local's contents, or
    // what this function's own parameter pointed at together with what it
    // stored through it. Everything where the address went somewhere
    // unknown or nothing was ever stored; nothing (bottom) for a parameter
    // of this function handed on as itself, since what it points at is the
    // callee's own question.
    Abstract handed(const string &block, const ValuePtr &v, const string &callee) {
        const Type held = held_type_of(v->get_type());
        const Root r = root_of(block, v);
        if (r.local != nullptr) {
            if (kept_escaped.count(r.local)) {
                return Abstract::top(held);
            }
            const auto m = kept_memory.find(r.local);
            if (m == kept_memory.end()) {
                return Abstract::top(held);
            }
            const Abstract &a = m->second.at(r.path, root_type(r));
            return a.bottom ? Abstract::top(held) : a;
        }
        if (!r.param.empty()) {
            if (callee == func->blocks.front()->name && r.path.empty()) {
                if (const auto *a = std::get_if<Argument>(&v->data);
                    a != nullptr && a->name == r.param) {
                    return Abstract::none();
                }
            }
            if (kept_escaped_params.count(r.param)) {
                return Abstract::top(held);
            }
            Abstract contents = Abstract::none();
            const vector<Argument> &params = func->blocks.front()->args;
            for (size_t k = 0; k < params.size() && k < pointees.size(); k++) {
                if (params[k].name == r.param) {
                    contents.include(pointees[k]);
                }
            }
            if (const auto m = kept_param_memory.find(r.param);
                m != kept_param_memory.end()) {
                contents.include(m->second);
            }
            if (contents.bottom) {
                return Abstract::top(held);
            }
            const Abstract &a = contents.at(r.path, root_type(r));
            return a.bottom ? Abstract::top(held) : a;
        }
        return Abstract::top(held);
    }

    // This function's summary for its callers, from the last pass.
    Summary summary(const Summaries &previous) const {
        Summary s;
        const shared_ptr<Block> &entry = func->blocks.front();
        s.params.assign(entry->args.size(), Abstract::none());
        s.pointees.assign(entry->args.size(), Abstract::none());
        s.ret = ret;
        s.stores.assign(entry->args.size(), Abstract::none());
        for (size_t k = 0; k < entry->args.size(); k++) {
            const Argument &a = entry->args[k];
            if (escaped_params.count(a.name)) {
                s.stores[k] = Abstract::top(held_type_of(a.type));
                continue;
            }
            const auto it = param_memory.find(a.name);
            if (it != param_memory.end()) {
                s.stores[k] = it->second;
            }
        }
        (void)previous;
        return s;
    }
};

namespace {

bool is_exported(const Function &f) {
    return std::find(f.attributes.begin(), f.attributes.end(),
                     ir::Function::Attribute::exported) != f.attributes.end();
}

Summary everything_of(const Function &f) {
    Summary s;
    const shared_ptr<Block> &entry = f.blocks.front();
    for (const Argument &a : entry->args) {
        s.params.push_back(Abstract::top(a.type));
        s.pointees.push_back(Abstract::top(held_type_of(a.type)));
        s.stores.push_back(Abstract::top(held_type_of(a.type)));
    }
    s.ret = Abstract::top(f.ret_type);
    return s;
}

} // namespace

ConstantIntervals::ConstantIntervals(const FuncMap &funcs) {
    Summaries summaries;
    for (const auto &[name, f] : funcs) {
        if (f == nullptr || f->blocks.empty()) {
            continue;
        }
        summaries[name] = everything_of(*f);
        auto state = std::make_shared<State>();
        state->func = f.get();
        states[f.get()] = state;
    }
    // The functions that run: the exported ones and what they call. A
    // function nothing reaches -- a traversal the lowering extracted and
    // then inlined into every caller, kept in the map -- still calls what
    // the live ones call, with parameters nobody narrowed, and saying its
    // calls happen would widen the callee's parameters to everything.
    std::set<string> live;
    {
        vector<string> work;
        for (const auto &[name, f] : funcs) {
            if (f != nullptr && !f->blocks.empty() && is_exported(*f)) {
                work.push_back(name);
            }
        }
        while (!work.empty()) {
            const string name = work.back();
            work.pop_back();
            if (!live.insert(name).second) {
                continue;
            }
            for (const shared_ptr<Block> &block : funcs.at(name)->blocks) {
                if (const auto *callee = block->terminator.callee();
                    callee != nullptr && funcs.count(callee->name)) {
                    work.push_back(callee->name);
                }
            }
        }
    }
    // Round by round: every function under the summaries so far, then the
    // summaries from what the functions did. Each round is sound on its
    // own, and the parameters, returns and stores only narrow; a few rounds
    // carry a fact down a chain of calls as deep as a traversal's.
    for (int round = 0; round < 6; round++) {
        for (const auto &[name, f] : funcs) {
            if (f == nullptr || f->blocks.empty()) {
                continue;
            }
            states.at(f.get())->analyze(summaries.at(name).params,
                                        summaries.at(name).pointees, &summaries);
        }
        Summaries next;
        for (const auto &[name, f] : funcs) {
            if (f == nullptr || f->blocks.empty()) {
                continue;
            }
            next[name] = states.at(f.get())->summary(summaries);
        }
        // The parameters: what every call site hands them, and for a pointer
        // or array parameter what the memory it names holds there. A call a
        // function makes of itself with its own parameter adds nothing to
        // that parameter and is left out, which is what lets a traversal's
        // ray keep what its caller said of it -- and nothing to what the
        // parameter points at, the function's own stores through it being
        // read along with the pointee at every load (see loaded).
        for (const auto &[name, f] : funcs) {
            if (f == nullptr || f->blocks.empty() || !live.count(name)) {
                continue;
            }
            State &state = *states.at(f.get());
            for (const shared_ptr<Block> &block : f->blocks) {
                const auto feed = [&](const string &callee,
                                      const vector<ValuePtr> &args) {
                    const auto it = next.find(callee);
                    if (it == next.end()) {
                        return;
                    }
                    Summary &s = it->second;
                    const Function &g = *funcs.at(callee);
                    const vector<Argument> &params = g.blocks.front()->args;
                    for (size_t k = 0; k < params.size() && k < args.size(); k++) {
                        if (callee == name) {
                            if (const auto *a = std::get_if<Argument>(&args[k]->data);
                                a != nullptr && a->name == params[k].name) {
                                continue;
                            }
                        }
                        s.params[k].include(state.value(block->name, args[k]));
                        const Type &t = args[k]->get_type();
                        if (t.is<Ptr_t>() || t.is_reference()) {
                            s.pointees[k].include(state.handed(block->name, args[k], callee));
                        }
                    }
                };
                std::visit(
                    overloads{
                        [&](const Terminator::Call &c) { feed(c.call.name, c.call.args); },
                        [&](const Terminator::MultiCall &c) {
                            for (size_t k = 0; k < c.varying.size(); k++) {
                                feed(c.call.name, c.call_args(k));
                            }
                        },
                        [&](const auto &) {},
                    },
                    block->terminator.data);
            }
        }
        for (const auto &[name, f] : funcs) {
            if (f == nullptr || f->blocks.empty()) {
                continue;
            }
            Summary &s = next.at(name);
            const Summary top = everything_of(*f);
            for (size_t k = 0; k < s.params.size(); k++) {
                if (is_exported(*f) || s.params[k].bottom) {
                    s.params[k] = top.params[k];
                }
                if (is_exported(*f) || s.pointees[k].bottom) {
                    s.pointees[k] = top.pointees[k];
                }
            }
            if (s.ret.bottom) {
                s.ret = top.ret;
            }
        }
        const bool settled = next == summaries;
        summaries = std::move(next);
        if (settled) {
            break;
        }
    }
}

ConstantIntervals::ConstantIntervals(const Function &func) {
    auto state = std::make_shared<State>();
    state->func = &func;
    states[&func] = state;
    vector<Abstract> params, pointees;
    for (const Argument &a : func.blocks.front()->args) {
        params.push_back(Abstract::top(a.type));
        pointees.push_back(Abstract::top(held_type_of(a.type)));
    }
    state->analyze(params, pointees, nullptr);
}

namespace {

void dump_abstract(std::ostream &os, const Abstract &a, const Type &t) {
    if (a.bottom) {
        os << "nothing";
        return;
    }
    if (!a.fields.empty()) {
        const Struct_t *s = t.as<Struct_t>();
        os << "{";
        for (size_t k = 0; k < a.fields.size(); k++) {
            if (k > 0) {
                os << ", ";
            }
            if (s != nullptr && k < s->fields.size()) {
                os << s->fields[k].name << ": ";
                dump_abstract(os, a.fields[k], s->fields[k].type);
            } else {
                dump_abstract(os, a.fields[k], Type());
            }
        }
        os << "}";
        return;
    }
    os << a.interval;
}

bool narrower_than_type(const Abstract &a, const Type &t) {
    if (a.bottom) {
        return false;
    }
    if (!a.fields.empty()) {
        const Struct_t *s = t.as<Struct_t>();
        for (size_t k = 0; k < a.fields.size(); k++) {
            if (narrower_than_type(a.fields[k],
                                   s != nullptr && k < s->fields.size()
                                       ? s->fields[k].type
                                       : Type())) {
                return true;
            }
        }
        return false;
    }
    return !(a.interval == ConstantInterval::bounds_of_type(t));
}

} // namespace

void ConstantIntervals::dump(std::ostream &os, const Function &func) const {
    const auto it = states.find(&func);
    if (it == states.end() || func.blocks.empty()) {
        return;
    }
    const State &state = *it->second;
    os << "intervals of " << func.blocks.front()->name << ":\n";
    for (const shared_ptr<Block> &block : func.blocks) {
        for (const Argument &a : block->args) {
            const auto found = state.args.find({block->name, a.name});
            if (found == state.args.end() || !narrower_than_type(found->second, a.type)) {
                continue;
            }
            os << "  " << block->name << "(" << a.name << ") : ";
            dump_abstract(os, found->second, a.type);
            os << "\n";
        }
        for (const shared_ptr<Instruction> &in : block->instrs) {
            const Abstract *found = state.recorded(in);
            if (found == nullptr || !narrower_than_type(*found, in->type)) {
                continue;
            }
            os << "  " << in->name << " : ";
            dump_abstract(os, *found, in->type);
            os << "\n";
        }
    }
    for (const auto &[alloc, contents] : state.kept_memory) {
        os << "  *" << alloc->name << " : ";
        dump_abstract(os, contents, held_type_of(alloc->type));
        os << "\n";
    }
    for (const Instruction *e : state.kept_escaped) {
        os << "  *" << e->name << " escapes\n";
    }
    for (const auto &[param, contents] : state.param_memory) {
        os << "  *" << param << " (stored through) : ";
        dump_abstract(os, contents, Type());
        os << "\n";
    }
    const vector<Argument> &params = func.blocks.front()->args;
    for (size_t k = 0; k < params.size() && k < state.pointees.size(); k++) {
        const Type held = held_type_of(params[k].type);
        if (!(params[k].type.is<Ptr_t>() || params[k].type.is_reference()) ||
            !narrower_than_type(state.pointees[k], held)) {
            continue;
        }
        os << "  *" << params[k].name << " (on entry) : ";
        dump_abstract(os, state.pointees[k], held);
        os << "\n";
    }
}

ConstantInterval ConstantIntervals::of(const Function &func, const Block &block,
                                       const std::shared_ptr<Value> &v) const {
    const auto it = states.find(&func);
    if (it == states.end() || v == nullptr) {
        return ConstantInterval::everything();
    }
    State &state = *it->second;
    const Type &t = v->get_type();
    if (t.is_reference()) {
        return state.loaded(block.name, v, t.element_of()).interval;
    }
    // An instruction made after the analysis -- by the rewrite asking --
    // is read off its operands here, once, as the pass would have read it,
    // its operands first where they are new as well.
    const std::function<void(const ValuePtr &, int)> prepare =
        [&](const ValuePtr &u, int depth) {
            const auto *in = std::get_if<shared_ptr<Instruction>>(&u->data);
            if (in == nullptr || (*in)->name.empty() || depth > 16 ||
                state.recorded(*in) != nullptr) {
                return;
            }
            for (const ValuePtr &o : (*in)->operands) {
                prepare(o, depth + 1);
            }
            state.record(*in, state.eval(block.name, **in));
        };
    prepare(v, 0);
    const Abstract a = state.value(block.name, v);
    return a.bottom || !a.fields.empty() ? ConstantInterval::everything()
                                         : a.interval;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
