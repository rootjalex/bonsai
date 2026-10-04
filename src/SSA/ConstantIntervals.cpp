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
#include <iostream>
#include <cmath>
#include <limits>
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

ConstantInterval ConstantInterval::bounds_of_type(const Type &type) {
    const Type t = scalar_of(type);
    if (t.is_bool()) {
        return {0, 1};
    }
    if (t.is_uint()) {
        ConstantInterval i = bounded_below(0);
        if (t.bits() < 53) {
            i.max = std::ldexp(1.0, int(t.bits())) - 1;
            i.max_defined = true;
        }
        return i;
    }
    if (t.is_int()) {
        if (t.bits() <= 53) {
            const double top = std::ldexp(1.0, int(t.bits()) - 1);
            return {-top, top - 1};
        }
        return everything();
    }
    return everything();
}

bool ConstantInterval::operator==(const ConstantInterval &o) const {
    return min_defined == o.min_defined && max_defined == o.max_defined &&
           (!min_defined || min == o.min) && (!max_defined || max == o.max);
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
// is a product no interval holds (zero times an infinity), and gives
// everything.
ConstantInterval from_corners(const vector<Limit> &corners) {
    ConstantInterval r;
    bool first = true;
    Limit lo{0, true}, hi{0, true};
    for (const Limit &c : corners) {
        if (std::isnan(c.value)) {
            return ConstantInterval::everything();
        }
        if (first || c.value < lo.value || (c.value == lo.value && !c.defined)) {
            lo = c;
        }
        if (first || c.value > hi.value || (c.value == hi.value && !c.defined)) {
            hi = c;
        }
        first = false;
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
    // An undefined limit stands for a finite value of unknown size, so zero
    // times it is zero, where zero times a defined infinity is a NaN that
    // from_corners reads as everything.
    if ((a.value == 0 && !b.defined) || (b.value == 0 && !a.defined)) {
        return {0, a.defined && b.defined};
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
    return r;
}

ConstantInterval operator-(const ConstantInterval &a) {
    ConstantInterval r;
    r.min = -a.max;
    r.min_defined = a.max_defined;
    r.max = -a.min;
    r.max_defined = a.min_defined;
    return r;
}

ConstantInterval operator-(const ConstantInterval &a, const ConstantInterval &b) {
    return a + (-b);
}

ConstantInterval operator*(const ConstantInterval &a, const ConstantInterval &b) {
    if (a.is_everything() && b.is_everything()) {
        return ConstantInterval::everything();
    }
    return from_corners({times(lower(a), lower(b)), times(lower(a), upper(b)),
                         times(upper(a), lower(b)), times(upper(a), upper(b))});
}

ConstantInterval operator/(const ConstantInterval &a, const ConstantInterval &b) {
    if (b.min_defined && b.min > 0) {
        return from_corners({over(lower(a), lower(b)), over(lower(a), upper(b)),
                             over(upper(a), lower(b)), over(upper(a), upper(b))});
    }
    if (b.max_defined && b.max < 0) {
        return -(a / -b);
    }
    return ConstantInterval::everything();
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
    return r;
}

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
    return os << "]";
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

// What a function says to its callers: what its parameters have been
// handed, what it returns, and what it stores through each pointer or
// array parameter. Everything to begin with; narrowed round by round.
struct Summary {
    vector<Abstract> params;
    Abstract ret;
    vector<Abstract> stores; // per parameter; bottom for one it writes nothing through
    bool operator==(const Summary &o) const {
        return params == o.params && ret == o.ret && stores == o.stores;
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
    std::unordered_map<const Instruction *, Abstract> instrs;
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
    std::map<string, size_t> rpo_index;

    // The memory in force for this pass: the previous pass's contents, read
    // by loads, while `memory` collects this pass's stores.
    const std::unordered_map<const Instruction *, Abstract> *seen_memory = nullptr;
    const std::set<const Instruction *> *seen_escaped = nullptr;
    const Summaries *summaries = nullptr;

    //--- values -----------------------------------------------------------

    Abstract value(const string &block, const ValuePtr &v) const {
        if (v == nullptr) {
            return Abstract::top(Type());
        }
        return std::visit(
            overloads{
                [&](const Constant &c) { return constant(c); },
                [&](const shared_ptr<Instruction> &i) {
                    const auto it = instrs.find(i.get());
                    return it != instrs.end() ? it->second : Abstract::top(i->type);
                },
                [&](const Argument &a) {
                    const auto it = args.find({block, a.name});
                    return it != args.end() ? it->second : Abstract::top(a.type);
                },
            },
            v->data);
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
        if (std::holds_alternative<Argument>(v->data)) {
            const Definition d = defs->of(block, v);
            if (d.value == nullptr) {
                return none;
            }
            if (std::holds_alternative<shared_ptr<Instruction>>(d.value->data)) {
                return root_of(d.block, d.value, depth + 1);
            }
            if (const auto *a = std::get_if<Argument>(&d.value->data);
                a != nullptr && d.block == func->blocks.front()->name) {
                Root r;
                r.param = a->name;
                return r;
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

    // What the memory `v` points at holds, as the previous pass saw it.
    Abstract loaded(const string &block, const ValuePtr &v, const Type &as) {
        const Root r = root_of(block, v);
        if (r.local == nullptr || seen_memory == nullptr ||
            seen_escaped->count(r.local)) {
            return Abstract::top(as);
        }
        const auto it = seen_memory->find(r.local);
        if (it == seen_memory->end()) {
            return Abstract::top(as);
        }
        const Type rt = root_type(r);
        const Abstract &a = it->second.at(r.path, rt);
        if (a.bottom) {
            return Abstract::top(as);
        }
        return a;
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

    Abstract eval(const string &block, const Instruction &in) {
        const Type &type = in.type;
        const auto A = [&](size_t k) {
            return k < in.operands.size() ? value(block, in.operands[k])
                                          : Abstract::top(Type());
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
            Abstract r = A(1);
            r.include(A(2));
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
                if (!(i.min_defined && i.min >= 0)) {
                    return top();
                }
                float lo = float(i.min);
                if (double(lo) > i.min) {
                    lo = std::nextafterf(lo, 0.0f);
                }
                double hi_bits = kInfBits;
                if (i.max_defined && std::isfinite(i.max)) {
                    float hi = float(i.max);
                    if (double(hi) < i.max) {
                        hi = std::nextafterf(hi, kInf);
                    }
                    hi_bits = std::isfinite(hi) ? double(std::bit_cast<uint32_t>(hi))
                                                : kInfBits;
                }
                return fit(ConstantInterval(double(std::bit_cast<uint32_t>(lo)),
                                            hi_bits));
            }
            if (is_integer_type(f) && f.bits() == 32 && t.is_float() &&
                t.bits() == 32) {
                if (!(i.min_defined && i.min >= 0 && i.min <= kInfBits)) {
                    return top();
                }
                const float lo = std::bit_cast<float>(uint32_t(i.min));
                const double hi = i.max_defined && i.max <= kInfBits
                                      ? double(std::bit_cast<float>(uint32_t(i.max)))
                                      : kInf;
                return fit(ConstantInterval(double(lo), hi));
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
            return intrinsic(block, in);
        default:
            return top();
        }
    }

    Abstract intrinsic(const string &block, const Instruction &in) {
        const Type &type = in.type;
        const auto I = [&](size_t k) {
            return k < in.operands.size() ? value(block, in.operands[k]).interval
                                          : ConstantInterval::everything();
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
            return fit(r);
        }
        case ir::Intrinsic::sqr:
            return fit(I(0) * I(0));
        case ir::Intrinsic::exp:
            return fit(ConstantInterval::bounded_below(0));
        case ir::Intrinsic::cos:
        case ir::Intrinsic::sin:
            return fit({-1, 1});
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
            }
            for (const shared_ptr<Instruction> &in : block->instrs) {
                switch (in->op) {
                case Instruction::Op::Store:
                    if (in->operands.size() == 2) {
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
                case Instruction::Op::AtomicAdd:
                case Instruction::Op::AccAdd:
                case Instruction::Op::AccMul:
                case Instruction::Op::AccSub:
                case Instruction::Op::AccArgmin:
                case Instruction::Op::AccArgmax:
                case Instruction::Op::AccMin:
                case Instruction::Op::AccMax:
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
                    instrs[in.get()] = eval(block->name, *in);
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
    // loads read is the memory the stores made, three at most, each sound
    // on its own since a load first reads everything.
    void analyze(const vector<Abstract> &params, const Summaries *all) {
        summaries = all;
        if (!cfg) {
            cfg = std::make_unique<Cfg>(*func);
            for (size_t i = 0; i < cfg->rpo.size(); i++) {
                rpo_index[cfg->name(cfg->rpo[i])] = i;
            }
            defs = std::make_unique<Definitions>(*func, /*lenient=*/true);
        }
        std::unordered_map<const Instruction *, Abstract> last_memory;
        std::set<const Instruction *> last_escaped;
        for (int round = 0; round < 3; round++) {
            seen_memory = round == 0 ? nullptr : &last_memory;
            seen_escaped = round == 0 ? nullptr : &last_escaped;
            pass(params);
            // Everything an escaped allocation holds is unknown.
            for (const Instruction *e : escaped) {
                memory.erase(e);
            }
            const bool same = round > 0 && memory == last_memory && escaped == last_escaped;
            last_memory = memory;
            last_escaped = escaped;
            if (same) {
                break;
            }
        }
        seen_memory = &last_memory;
        seen_escaped = &last_escaped;
        // Keep the last pass's memory as what loads were read against, for
        // queries after the fact.
        kept_memory = last_memory;
        kept_escaped = last_escaped;
        seen_memory = &kept_memory;
        seen_escaped = &kept_escaped;
    }
    std::unordered_map<const Instruction *, Abstract> kept_memory;
    std::set<const Instruction *> kept_escaped;

    // This function's summary for its callers, from the last pass.
    Summary summary(const Summaries &previous) const {
        Summary s;
        const shared_ptr<Block> &entry = func->blocks.front();
        s.params.assign(entry->args.size(), Abstract::none());
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
            states.at(f.get())->analyze(summaries.at(name).params, &summaries);
        }
        Summaries next;
        for (const auto &[name, f] : funcs) {
            if (f == nullptr || f->blocks.empty()) {
                continue;
            }
            next[name] = states.at(f.get())->summary(summaries);
        }
        // The parameters: what every call site hands them. A call a function
        // makes of itself with its own parameter adds nothing to that
        // parameter and is left out, which is what lets a traversal's ray
        // keep what its caller said of it.
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
    vector<Abstract> params;
    for (const Argument &a : func.blocks.front()->args) {
        params.push_back(Abstract::top(a.type));
    }
    state->analyze(params, nullptr);
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
            const auto found = state.instrs.find(in.get());
            if (found == state.instrs.end() || !narrower_than_type(found->second, in->type)) {
                continue;
            }
            os << "  " << in->name << " : ";
            dump_abstract(os, found->second, in->type);
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
    // is read off its operands here, once, as the pass would have read it.
    if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data);
        in != nullptr && !state.instrs.count(in->get()) && !(*in)->name.empty()) {
        state.instrs[in->get()] = state.eval(block.name, **in);
    }
    const Abstract a = state.value(block.name, v);
    return a.bottom || !a.fields.empty() ? ConstantInterval::everything()
                                         : a.interval;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
