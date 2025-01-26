#include "IR/Equality.h"

namespace bonsai {
namespace ir {

namespace {

enum class Cmp { Less, Equals, Greater };

template <typename T>
Cmp compare_primitives(const T &t0, const T &t1) {
    if (t0 == t1) {
        return Cmp::Equals;
    } else if (t0 < t1) {
        return Cmp::Less;
    } else {
        return Cmp::Greater;
    }
}

Cmp compare_types(const Type &t0, const Type &t1) {
    if (!t0.defined()) {
        if (!t1.defined()) {
            return Cmp::Equals;
        } else {
            return Cmp::Less;
        }
    } else if (!t1.defined()) {
        return Cmp::Greater;
    }

    // Must both be defined.

    if (t0.node_type() < t1.node_type()) {
        return Cmp::Less;
    } else if (t0.node_type() > t1.node_type()) {
        return Cmp::Greater;
    }

    // Must both be the same node type.
    switch (t0.node_type()) {
    case IRTypeEnum::Int_t: {
        return compare_primitives(t0.as<Int_t>()->bits, t1.as<Int_t>()->bits);
    }
    case IRTypeEnum::UInt_t: {
        return compare_primitives(t0.as<UInt_t>()->bits, t1.as<UInt_t>()->bits);
    }
    case IRTypeEnum::Float_t: {
        return compare_primitives(t0.as<Float_t>()->bits,
                                  t1.as<Float_t>()->bits);
    }
    case IRTypeEnum::Bool_t: {
        return Cmp::Equals;
    }
    case IRTypeEnum::Ptr_t: {
        return compare_types(t0.as<Ptr_t>()->etype, t1.as<Ptr_t>()->etype);
    }
    case IRTypeEnum::Vector_t: {
        const Vector_t *v0 = t0.as<Vector_t>();
        const Vector_t *v1 = t1.as<Vector_t>();
        if (v0->lanes == v1->lanes) {
            return compare_types(v0->etype, v1->etype);
        } else {
            return compare_primitives(v0->lanes, v1->lanes);
        }
    }
    case IRTypeEnum::Struct_t: {
        const Struct_t *s0 = t0.as<Struct_t>();
        const Struct_t *s1 = t1.as<Struct_t>();
        if (s0->name != s1->name) {
            return compare_primitives(s0->name, s1->name);
        }

        // TODO: can name ever match without fields matching?
        if (s0->fields.size() != s1->fields.size()) {
            return compare_primitives(s0->fields.size(), s1->fields.size());
        }
        const size_t n = s0->fields.size();
        for (size_t i = 0; i < n; i++) {
            if (s0->fields[i].first != s1->fields[i].first) {
                return compare_primitives(s0->fields[i].first,
                                          s1->fields[i].first);
            }
            auto rec =
                compare_types(s0->fields[i].second, s1->fields[i].second);
            if (rec != Cmp::Equals) {
                return rec;
            }
        }
        return Cmp::Equals;
    }
    case IRTypeEnum::Tuple_t: {
        const Tuple_t *tt0 = t0.as<Tuple_t>();
        const Tuple_t *tt1 = t1.as<Tuple_t>();
        if (tt0->etypes.size() != tt1->etypes.size()) {
            return compare_primitives(tt0->etypes.size(), tt1->etypes.size());
        }
        const size_t n = tt0->etypes.size();
        for (size_t i = 0; i < n; i++) {
            auto rec = compare_types(tt0->etypes[i], tt1->etypes[i]);
            if (rec != Cmp::Equals) {
                return rec;
            }
        }
        return Cmp::Equals;
    }
    case IRTypeEnum::Option_t: {
        return compare_types(t0.as<Option_t>()->etype,
                             t1.as<Option_t>()->etype);
    }
    case IRTypeEnum::Set_t: {
        return compare_types(t0.as<Set_t>()->etype, t1.as<Set_t>()->etype);
    }
    case IRTypeEnum::Function_t: {
        const Function_t *f0 = t0.as<Function_t>();
        const Function_t *f1 = t1.as<Function_t>();
        if (f0->arg_types.size() != f1->arg_types.size()) {
            return compare_primitives(f0->arg_types.size(),
                                      f1->arg_types.size());
        }
        const size_t n = f0->arg_types.size();
        for (size_t i = 0; i < n; i++) {
            auto rec = compare_types(f0->arg_types[i], f1->arg_types[i]);
            if (rec != Cmp::Equals) {
                return rec;
            }
        }
        return compare_types(f0->ret_type, f1->ret_type);
    }
    }
}

} // namespace

bool equals(const Type &t0, const Type &t1) {
    return compare_types(t0, t1) == Cmp::Equals;
}

bool TypeLessThan::operator()(const Type &t0, const Type &t1) const {
    return compare_types(t0, t1) == Cmp::Less;
}

} // namespace ir
} // namespace bonsai
