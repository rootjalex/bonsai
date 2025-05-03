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

template <typename T, typename F>
Cmp compare_lists(const std::vector<T> &l0, const std::vector<T> &l1,
                  const F &f) {
    if (l0.size() != l1.size()) {
        return compare_primitives(l0.size(), l1.size());
    }
    const size_t n = l0.size();
    for (size_t i = 0; i < n; i++) {
        if (const Cmp value = f(l0[i], l1[i]); value != Cmp::Equals) {
            return value;
        }
    }
    return Cmp::Equals;
}

template <typename T, typename F>
Cmp compare_maps(const std::map<std::string, T> &m0,
                 const std::map<std::string, T> &m1, const F &f) {
    if (m0.size() != m1.size()) {
        return compare_primitives(m0.size(), m1.size());
    }

    auto it0 = m0.begin();
    auto it1 = m1.begin();

    while (it0 != m0.end() && it1 != m1.end()) {
        if (const Cmp name = compare_primitives(it0->first, it1->first);
            name != Cmp::Equals) {
            return name;
        }

        if (const Cmp value = f(it0->second, it1->second);
            value != Cmp::Equals) {
            return value;
        }

        ++it0;
        ++it1;
    }

    return Cmp::Equals;
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

Cmp compare_exprs(const Expr &e0, const Expr &e1);
Cmp compare_types(const Type &t0, const Type &t1);

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
    case IRTypeEnum::Index_t: {
        return Cmp::Equals;
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
            if (s0->fields[i].name != s1->fields[i].name) {
                return compare_primitives(s0->fields[i].name,
                                          s1->fields[i].name);
            }
            if (const Cmp rec =
                    compare_types(s0->fields[i].type, s1->fields[i].type);
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
        return compare_lists(tt0->etypes, tt1->etypes, compare_types);
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

        auto compare_arg_sigs = [](const Function_t::ArgSig &a,
                                   const Function_t::ArgSig &b) {
            if ((int)a.is_mutable < (int)b.is_mutable) {
                return Cmp::Less;
            } else if ((int)a.is_mutable > (int)b.is_mutable) {
                return Cmp::Greater;
            } else {
                return compare_types(a.type, b.type);
            }
        };

        if (const Cmp arg_types =
                compare_lists(f0->arg_types, f1->arg_types, compare_arg_sigs);
            arg_types != Cmp::Equals) {
            return arg_types;
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

                    if (const Cmp inits =
                            compare_lists(v0.initializers, v1.initializers,
                                          compare_primitives<std::string>);
                        inits != Cmp::Equals) {
                        return inits;
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

Cmp compare_exprs(const Expr &e0, const Expr &e1) {
    if (std::optional<Cmp> nodes_cmp = compare_node_types(e0, e1)) {
        return *nodes_cmp;
    }

    internal_assert(e0.type().defined() && e1.type().defined());
    if (const Cmp types = compare_types(e0.type(), e1.type());
        types != Cmp::Equals) {
        return types;
    }

    // Must both be the same node type.
    switch (e0.node_type()) {
    case IRExprEnum::IntImm: {
        return compare_primitives(e0.as<IntImm>()->value,
                                  e1.as<IntImm>()->value);
    }
    case IRExprEnum::UIntImm: {
        return compare_primitives(e0.as<UIntImm>()->value,
                                  e1.as<UIntImm>()->value);
    }
    case IRExprEnum::IdxImm: {
        return compare_primitives(e0.as<IdxImm>()->value,
                                  e1.as<IdxImm>()->value);
    }
    case IRExprEnum::FloatImm: {
        return compare_primitives(e0.as<FloatImm>()->value,
                                  e1.as<FloatImm>()->value);
    }
    case IRExprEnum::BoolImm: {
        return compare_primitives(e0.as<BoolImm>()->value,
                                  e1.as<BoolImm>()->value);
    }
    case IRExprEnum::VecImm: {
        return compare_lists(e0.as<VecImm>()->values, e1.as<VecImm>()->values,
                             compare_exprs);
    }
    case IRExprEnum::Var: {
        return compare_primitives(e0.as<Var>()->name, e1.as<Var>()->name);
    }
    case IRExprEnum::Infinity: {
        // Equal because types are equal.
        return Cmp::Equals;
    }
    case IRExprEnum::BinOp: {
        const BinOp *b0 = e0.as<BinOp>();
        const BinOp *b1 = e1.as<BinOp>();
        if (const Cmp op = compare_primitives(b0->op, b1->op);
            op != Cmp::Equals) {
            return op;
        }
        if (const Cmp a = compare_exprs(b0->a, b1->a); a != Cmp::Equals) {
            return a;
        }
        return compare_exprs(b0->b, b1->b);
    }
    case IRExprEnum::UnOp: {
        const UnOp *u0 = e0.as<UnOp>();
        const UnOp *u1 = e1.as<UnOp>();
        if (const Cmp op = compare_primitives(u0->op, u1->op);
            op != Cmp::Equals) {
            return op;
        }
        return compare_exprs(u0->a, u1->a);
    }
    case IRExprEnum::Select: {
        const Select *s0 = e0.as<Select>();
        const Select *s1 = e1.as<Select>();
        if (const Cmp op = compare_exprs(s0->cond, s1->cond);
            op != Cmp::Equals) {
            return op;
        }
        if (const Cmp a = compare_exprs(s0->tvalue, s1->tvalue);
            a != Cmp::Equals) {
            return a;
        }
        return compare_exprs(s0->fvalue, s1->fvalue);
    }
    case IRExprEnum::Cast: {
        return compare_exprs(e0.as<Cast>()->value, e1.as<Cast>()->value);
    }
    case IRExprEnum::Broadcast: {
        return compare_exprs(e0.as<Broadcast>()->value,
                             e1.as<Broadcast>()->value);
    }
    case IRExprEnum::VectorReduce: {
        const VectorReduce *v0 = e0.as<VectorReduce>();
        const VectorReduce *v1 = e1.as<VectorReduce>();
        if (const Cmp op = compare_primitives(v0->op, v1->op);
            op != Cmp::Equals) {
            return op;
        }
        return compare_exprs(v0->value, v1->value);
    }
    case IRExprEnum::VectorShuffle: {
        const VectorShuffle *v0 = e0.as<VectorShuffle>();
        const VectorShuffle *v1 = e1.as<VectorShuffle>();
        if (const Cmp idxs = compare_lists(v0->idxs, v1->idxs, compare_exprs);
            idxs != Cmp::Equals) {
            return idxs;
        }
        return compare_exprs(v0->value, v1->value);
    }
    case IRExprEnum::Ramp: {
        const Ramp *v0 = e0.as<Ramp>();
        const Ramp *v1 = e1.as<Ramp>();
        if (const Cmp lanes = compare_primitives(v0->lanes, v1->lanes);
            lanes != Cmp::Equals) {
            return lanes;
        }
        if (const Cmp base = compare_exprs(v0->base, v1->base);
            base != Cmp::Equals) {
            return base;
        }
        return compare_exprs(v0->stride, v1->stride);
    }
    case IRExprEnum::Extract: {
        const Extract *v0 = e0.as<Extract>();
        const Extract *v1 = e1.as<Extract>();
        if (const Cmp base = compare_exprs(v0->vec, v1->vec);
            base != Cmp::Equals) {
            return base;
        }
        return compare_exprs(v0->idx, v1->idx);
    }
    case IRExprEnum::Build: {
        const Build *v0 = e0.as<Build>();
        const Build *v1 = e1.as<Build>();
        return compare_lists(v0->values, v1->values, compare_exprs);
    }
    case IRExprEnum::Access: {
        const Access *v0 = e0.as<Access>();
        const Access *v1 = e1.as<Access>();
        if (const Cmp field = compare_primitives(v0->field, v1->field);
            field != Cmp::Equals) {
            return field;
        }
        return compare_exprs(v0->value, v1->value);
    }
    case IRExprEnum::Unwrap: {
        const Unwrap *v0 = e0.as<Unwrap>();
        const Unwrap *v1 = e1.as<Unwrap>();
        if (const Cmp index = compare_primitives(v0->index, v1->index);
            index != Cmp::Equals) {
            return index;
        }
        return compare_exprs(v0->value, v1->value);
    }
    case IRExprEnum::Intrinsic: {
        const Intrinsic *v0 = e0.as<Intrinsic>();
        const Intrinsic *v1 = e1.as<Intrinsic>();
        if (const Cmp op = compare_primitives(v0->op, v1->op);
            op != Cmp::Equals) {
            return op;
        }
        return compare_lists(v0->args, v1->args, compare_exprs);
    }
    case IRExprEnum::Generator: {
        const Generator *v0 = e0.as<Generator>();
        const Generator *v1 = e1.as<Generator>();
        if (const Cmp op = compare_primitives(v0->op, v1->op);
            op != Cmp::Equals) {
            return op;
        }
        return compare_lists(v0->args, v1->args, compare_exprs);
    }
    case IRExprEnum::Lambda: {
        const Lambda *v0 = e0.as<Lambda>();
        const Lambda *v1 = e1.as<Lambda>();
        if (const Cmp value = compare_exprs(v0->value, v1->value);
            value != Cmp::Equals) {
            return value;
        }
        return compare_lists(v0->args, v1->args,
                             [](const auto &a0, const auto &a1) {
                                 return compare_types(a0.type, a1.type);
                             });
    }
    case IRExprEnum::GeomOp: {
        const GeomOp *v0 = e0.as<GeomOp>();
        const GeomOp *v1 = e1.as<GeomOp>();
        if (const Cmp op = compare_primitives(v0->op, v1->op);
            op != Cmp::Equals) {
            return op;
        }
        if (const Cmp a = compare_exprs(v0->a, v1->a); a != Cmp::Equals) {
            return a;
        }
        return compare_exprs(v0->b, v1->b);
    }
    case IRExprEnum::SetOp: {
        const SetOp *v0 = e0.as<SetOp>();
        const SetOp *v1 = e1.as<SetOp>();
        if (const Cmp op = compare_primitives(v0->op, v1->op);
            op != Cmp::Equals) {
            return op;
        }
        if (const Cmp a = compare_exprs(v0->a, v1->a); a != Cmp::Equals) {
            return a;
        }
        return compare_exprs(v0->b, v1->b);
    }
    case IRExprEnum::Call: {
        const Call *v0 = e0.as<Call>();
        const Call *v1 = e1.as<Call>();
        if (const Cmp func = compare_exprs(v0->func, v1->func);
            func != Cmp::Equals) {
            return func;
        }
        return compare_lists(v0->args, v1->args, compare_exprs);
    }
    case IRExprEnum::Instantiate: {
        const Instantiate *v0 = e0.as<Instantiate>();
        const Instantiate *v1 = e1.as<Instantiate>();
        if (const Cmp expr = compare_exprs(v0->expr, v1->expr);
            expr != Cmp::Equals) {
            return expr;
        }
        return compare_maps(v0->types, v1->types, compare_types);
    }
    case IRExprEnum::PtrTo: {
        const PtrTo *v0 = e0.as<PtrTo>();
        const PtrTo *v1 = e1.as<PtrTo>();
        return compare_exprs(v0->expr, v1->expr);
    }
    case IRExprEnum::Deref: {
        const Deref *v0 = e0.as<Deref>();
        const Deref *v1 = e1.as<Deref>();
        return compare_exprs(v0->expr, v1->expr);
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

bool equals(const Expr &e0, const Expr &e1) {
    return compare_exprs(e0, e1) == Cmp::Equals;
}

bool ExprLessThan::operator()(const Expr &e0, const Expr &e1) const {
    return compare_exprs(e0, e1) == Cmp::Less;
}

bool WriteLocLessThan::operator()(const WriteLoc &w0,
                                  const WriteLoc &w1) const {
    const auto res = compare_writelocs(w0, w1);
    return res == Cmp::Less;
}

} // namespace ir
} // namespace bonsai
