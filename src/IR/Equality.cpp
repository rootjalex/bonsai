#include "IR/Equality.h"

#include "IR/Printer.h"

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

template <typename T>
std::optional<Cmp> compare_node_types(const T &a, const T &b) {
    if (!a.defined()) {
        if (!b.defined()) {
            return Cmp::Equals;
        } else {
            return Cmp::Less;
        }
    } else if (!b.defined()) {
        return Cmp::Greater;
    }

    // Must both be defined.

    if (a.node_type() < b.node_type()) {
        return Cmp::Less;
    } else if (a.node_type() > b.node_type()) {
        return Cmp::Greater;
    }
    return {};
}

Cmp compare_interfaces(const Interface &i0, const Interface &i1) {
    if (std::optional<Cmp> nodes_cmp = compare_node_types(i0, i1)) {
        return *nodes_cmp;
    }

    // Must both be the same node type.
    switch (i0.node_type()) {
    case IRInterfaceEnum::IEmpty:
        return Cmp::Equals;
    case IRInterfaceEnum::IFloat:
        return Cmp::Equals;
    case IRInterfaceEnum::IVector: {
        const IVector *v0 = i0.as<IVector>();
        const IVector *v1 = i1.as<IVector>();
        return compare_interfaces(v0->etype, v1->etype);
    }
    }
}

Cmp compare_types(const Type &t0, const Type &t1) {
    if (std::optional<Cmp> nodes_cmp = compare_node_types(t0, t1)) {
        return *nodes_cmp;
    }

    // Must both be the same node type.
    switch (t0.node_type()) {
    case IRTypeEnum::Void_t: {
        return Cmp::Equals;
    }
    case IRTypeEnum::Int_t: {
        return compare_primitives(t0.as<Int_t>()->bits, t1.as<Int_t>()->bits);
    }
    case IRTypeEnum::UInt_t: {
        return compare_primitives(t0.as<UInt_t>()->bits, t1.as<UInt_t>()->bits);
    }
    case IRTypeEnum::Float_t: {
        const auto *f0 = t0.as<Float_t>(), *f1 = t1.as<Float_t>();
        if (f0->exponent == f1->exponent) {
            return compare_primitives(f0->mantissa, f1->mantissa);
        }
        return compare_primitives(f0->exponent, f1->exponent);
    }
    case IRTypeEnum::Bool_t: {
        return Cmp::Equals;
    }
    case IRTypeEnum::Ptr_t: {
        return compare_types(t0.as<Ptr_t>()->etype, t1.as<Ptr_t>()->etype);
    }
    case IRTypeEnum::Ref_t: {
        return compare_primitives(t0.as<Ref_t>()->name, t1.as<Ref_t>()->name);
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
            if (const Cmp rec =
                    compare_types(s0->fields[i].second, s1->fields[i].second);
                rec != Cmp::Equals) {
                return rec;
            }
        }

        // TODO: compare defaults?
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
            if (const Cmp rec = compare_types(tt0->etypes[i], tt1->etypes[i]);
                rec != Cmp::Equals) {
                return rec;
            }
        }
        return Cmp::Equals;
    }
    case IRTypeEnum::Array_t: {
        // TODO: check size equality?
        return compare_types(t0.as<Array_t>()->etype, t1.as<Array_t>()->etype);
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
            if (const Cmp rec =
                    compare_types(f0->arg_types[i], f1->arg_types[i]);
                rec != Cmp::Equals) {
                return rec;
            }
        }
        return compare_types(f0->ret_type, f1->ret_type);
    }
    case IRTypeEnum::Generic_t: {
        const Generic_t *g0 = t0.as<Generic_t>();
        const Generic_t *g1 = t1.as<Generic_t>();
        if (g0->name != g1->name) {
            return compare_primitives(g0->name, g1->name);
        }
        // TODO: can we ever have two generics of the same name with different
        // interfaces??
        return compare_interfaces(g0->interface, g1->interface);
    }
    case IRTypeEnum::BVH_t: {
        const BVH_t *b0 = t0.as<BVH_t>();
        const BVH_t *b1 = t1.as<BVH_t>();
        if (b0->name != b1->name) {
            return compare_primitives(b0->name, b1->name);
        }

        static const auto compare_volumes =
            [](const std::optional<BVH_t::Volume> &vol0,
               const std::optional<BVH_t::Volume> &vol1) {
                if (const Cmp valid =
                        compare_primitives(vol0.has_value(), vol1.has_value());
                    valid != Cmp::Equals) {
                    return valid;
                }

                if (vol0.has_value()) {
                    internal_assert(vol1.has_value());
                    const auto &v0 = *vol0;
                    const auto &v1 = *vol1;

                    if (const Cmp vtype =
                            compare_types(v0.struct_type, v1.struct_type);
                        vtype != Cmp::Equals) {
                        return vtype;
                    }
                    if (v0.initializers.size() != v1.initializers.size()) {
                        return compare_primitives(v0.initializers.size(),
                                                  v1.initializers.size());
                    }
                    const size_t isize = v0.initializers.size();
                    for (size_t i = 0; i < isize; i++) {
                        const auto &param0 = v0.initializers[i];
                        const auto &param1 = v1.initializers[i];
                        if (const Cmp rec = compare_primitives(param0, param1);
                            rec != Cmp::Equals) {
                            return rec;
                        }
                    }
                }
                return Cmp::Equals;
            };

        // Compare node types.
        if (b0->nodes.size() != b1->nodes.size()) {
            return compare_primitives(b0->nodes.size(), b1->nodes.size());
        }

        const size_t n = b0->nodes.size();
        for (size_t i = 0; i < n; i++) {
            const auto &node0 = b0->nodes[i];
            const auto &node1 = b1->nodes[i];
            if (const Cmp rec =
                    compare_types(node0.struct_type, node1.struct_type);
                rec != Cmp::Equals) {
                return rec;
            }

            if (const Cmp volumes = compare_volumes(node0.volume, node1.volume);
                volumes != Cmp::Equals) {
                return volumes;
            }
        }
        // return compare_volumes(b0->volume, b1->volume);
        return Cmp::Equals;
    }
    }
}

Cmp compare_writelocs(const WriteLoc &w0, const WriteLoc &w1) {
    if (const Cmp types = compare_types(w0.base_type, w1.base_type);
        types != Cmp::Equals) {
        return types;
    }
    if (const Cmp base_types = compare_types(w0.type, w1.type);
        base_types != Cmp::Equals) {
        return base_types;
    }
    if (const Cmp names = compare_primitives(w0.base, w1.base);
        names != Cmp::Equals) {
        return names;
    }
    if (const Cmp accs =
            compare_primitives(w0.accesses.size(), w1.accesses.size());
        accs != Cmp::Equals) {
        return accs;
    }
    const size_t n = w0.accesses.size();
    if (n == 0) {
        return Cmp::Equals;
    }
    // Compare accesses.
    for (size_t i = 0; i < n; i++) {
        const std::string *s0 = std::get_if<std::string>(&w0.accesses[i]);
        const std::string *s1 = std::get_if<std::string>(&w1.accesses[i]);
        if (s0 && s1) {
            if (const Cmp fields = compare_primitives(*s0, *s1);
                fields != Cmp::Equals) {
                return fields;
            }
        } else if (s0) {
            return Cmp::Less;
        } else if (s1) {
            return Cmp::Greater;
        } else {
            // TODO(ajr): need Expr equality to compare indexes.
            internal_error
                << "TODO: implement Expr equality for WriteLoc::accesses " << w0
                << " versus " << w1;
        }
    }

    // Same base, same types, same number of accesses.
    return Cmp::Equals;
}

} // namespace

bool equals(const Type &t0, const Type &t1) {
    return compare_types(t0, t1) == Cmp::Equals;
}

bool TypeLessThan::operator()(const Type &t0, const Type &t1) const {
    return compare_types(t0, t1) == Cmp::Less;
}

bool WriteLocLessThan::operator()(const WriteLoc &w0,
                                  const WriteLoc &w1) const {
    const auto res = compare_writelocs(w0, w1);
    return res == Cmp::Less;
}

} // namespace ir
} // namespace bonsai
