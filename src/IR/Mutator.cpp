#include "IR/Mutator.h"

#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

namespace bonsai {
namespace ir {

namespace {

template <typename T>
std::pair<std::vector<T>, bool> visit_list(Mutator *v,
                                           const std::vector<T> &l) {
    bool not_changed = true;
    const size_t n = l.size();
    std::vector<T> new_l(n);
    for (size_t i = 0; i < n; i++) {
        new_l[i] = v->mutate(l[i]);
        not_changed = not_changed && new_l[i].same_as(l[i]);
    }
    return {std::move(new_l), not_changed};
}

std::pair<WriteLoc, bool> mutate_writeloc(Mutator *v, const WriteLoc &loc) {
    WriteLoc new_loc(loc.base, loc.base_type);
    bool not_changed = true;
    for (const auto &value : loc.accesses) {
        if (std::holds_alternative<Expr>(value)) {
            Expr new_value = v->mutate(std::get<Expr>(value));
            not_changed =
                not_changed && new_value.same_as(std::get<Expr>(value));
            new_loc.add_index_access(std::move(new_value));
        } else {
            new_loc.add_struct_access(std::get<std::string>(value));
        }
    }
    return {std::move(new_loc), not_changed};
}
} // namespace

Type Mutator::mutate(const Type &type) {
    return type.defined() ? type.get()->mutate_type(this) : Type();
}

Interface Mutator::mutate(const Interface &interface) {
    return interface.defined() ? interface.get()->mutate_interface(this)
                               : Interface();
}

Expr Mutator::mutate(const Expr &expr) {
    return expr.defined() ? expr.get()->mutate_expr(this) : Expr();
}

Stmt Mutator::mutate(const Stmt &stmt) {
    return stmt.defined() ? stmt.get()->mutate_stmt(this) : Stmt();
}

Type Mutator::visit(const Void_t *node) { return node; }

Type Mutator::visit(const Int_t *node) { return node; }

Type Mutator::visit(const UInt_t *node) { return node; }

Type Mutator::visit(const Float_t *node) { return node; }

Type Mutator::visit(const Bool_t *node) { return node; }

Type Mutator::visit(const Ptr_t *node) {
    Type etype = mutate(node->etype);
    if (etype.same_as(node->etype)) {
        return node;
    }
    return Ptr_t::make(std::move(etype));
}

Type Mutator::visit(const Vector_t *node) {
    Type etype = mutate(node->etype);
    if (etype.same_as(node->etype)) {
        return node;
    }
    return Vector_t::make(std::move(etype), node->lanes);
}

Type Mutator::visit(const Struct_t *node) {
    Struct_t::Map fields = node->fields; // copy
    bool changed = false;
    // TODO: lift into helper func?
    for (auto &[key, value] : fields) {
        Type t = mutate(value);
        if (!t.same_as(value)) {
            changed = true;
            value = t; // edits mapped value
        }
    }

    // TODO: should we be recursing into defaults?
    auto defaults = node->defaults; // copy
    for (auto &[key, value] : defaults) {
        Expr e = mutate(value);
        if (!e.same_as(value)) {
            changed = true;
            value = e; // edits mapped value
        }
    }

    if (!changed) {
        return node;
    }
    return Struct_t::make(node->name, std::move(fields), std::move(defaults));
}

Type Mutator::visit(const Tuple_t *node) {
    auto [etypes, not_changed] = visit_list(this, node->etypes);
    if (not_changed) {
        return node;
    }
    return Tuple_t::make(std::move(etypes));
}

Type Mutator::visit(const Option_t *node) {
    Type etype = mutate(node->etype);
    if (etype.same_as(node->etype)) {
        return node;
    }
    return Option_t::make(std::move(etype));
}

Type Mutator::visit(const Set_t *node) {
    Type etype = mutate(node->etype);
    if (etype.same_as(node->etype)) {
        return node;
    }
    return Set_t::make(std::move(etype));
}

Type Mutator::visit(const Function_t *node) {
    Type ret_type = mutate(node->ret_type);
    auto [arg_types, not_changed] = visit_list(this, node->arg_types);
    if (ret_type.same_as(node->ret_type) && not_changed) {
        return node;
    }
    return Function_t::make(std::move(ret_type), std::move(arg_types));
}

Type Mutator::visit(const Generic_t *node) {
    Interface interface = mutate(node->interface);
    if (interface.same_as(node->interface)) {
        return node;
    }
    return Generic_t::make(node->name, std::move(interface));
}

Type Mutator::visit(const BVH_t *node) {
    // Keep track of whether any component of the type has changed or not.
    ir::Type primitive = mutate(node->primitive);
    bool not_changed = primitive.same_as(node->primitive);

    const auto visit_param = [&](const BVH_t::Param &param) {
        Type type = mutate(param.type);
        if (type.same_as(param.type)) {
            return param;
        }
        not_changed = false;
        return BVH_t::Param{param.name, std::move(type)};
    };
    const auto visit_volume = [&](const std::optional<BVH_t::Volume> &volume)
        -> std::optional<BVH_t::Volume> {
        if (!volume.has_value())
            return volume;
        Type type = mutate(volume->struct_type);
        if (type.same_as(volume->struct_type)) {
            return volume;
        }
        not_changed = false;
        return BVH_t::Volume{std::move(type), volume->initializers};
    };
    const auto visit_node = [&](const BVH_t::Node &node) {
        bool cache_changed = not_changed;
        not_changed = true;
        const size_t nparams = node.params.size();
        std::vector<BVH_t::Param> params(nparams);

        for (size_t i = 0; i < nparams; i++) {
            params[i] = visit_param(node.params[i]);
        }
        std::optional<BVH_t::Volume> volume = visit_volume(node.volume);

        if (not_changed) {
            not_changed = cache_changed;
            return node;
        }
        return BVH_t::Node{node.name, std::move(params), std::move(volume)};
    };

    // const size_t nparams = node->params.size();
    // std::vector<BVH_t::Param> params(nparams);

    // for (size_t i = 0; i < nparams; i++) {
    //     params[i] = visit_param(node->params[i]);
    // }

    const size_t nnodes = node->nodes.size();
    std::vector<BVH_t::Node> nodes(nnodes);

    for (size_t i = 0; i < nnodes; i++) {
        nodes[i] = visit_node(node->nodes[i]);
    }

    // std::optional<BVH_t::Volume> volume = visit_volume(node->volume);

    if (not_changed) {
        return node;
        // } else if (volume.has_value()) {
        //     return BVH_t::make(node->name, std::move(params),
        //     std::move(nodes),
        //                        *volume);
    } else {
        // internal_assert(params.empty());
        return BVH_t::make(std::move(primitive), node->name, std::move(nodes));
    }
}

Interface Mutator::visit(const IEmpty *node) { return node; }

Interface Mutator::visit(const IFloat *node) { return node; }

Interface Mutator::visit(const IVector *node) {
    Interface etype = mutate(node->etype);
    if (etype.same_as(node->etype)) {
        return node;
    }
    return IVector::make(std::move(etype));
}

Expr Mutator::visit(const IntImm *node) { return node; }

Expr Mutator::visit(const UIntImm *node) { return node; }

Expr Mutator::visit(const FloatImm *node) { return node; }

Expr Mutator::visit(const BoolImm *node) { return node; }

Expr Mutator::visit(const Infinity *node) { return node; }

Expr Mutator::visit(const Var *node) { return node; }

Expr Mutator::visit(const BinOp *node) {
    Expr a = mutate(node->a);
    Expr b = mutate(node->b);
    if (a.same_as(node->a) && b.same_as(node->b)) {
        return node;
    }
    return BinOp::make(node->op, std::move(a), std::move(b));
}

Expr Mutator::visit(const UnOp *node) {
    Expr a = mutate(node->a);
    if (a.same_as(node->a)) {
        return node;
    }
    return UnOp::make(node->op, std::move(a));
}

Expr Mutator::visit(const Select *node) {
    Expr cond = mutate(node->cond);
    Expr tvalue = mutate(node->tvalue);
    Expr fvalue = mutate(node->fvalue);
    if (cond.same_as(node->cond) && tvalue.same_as(node->tvalue) &&
        fvalue.same_as(node->fvalue)) {
        return node;
    }
    return Select::make(std::move(cond), std::move(tvalue), std::move(fvalue));
}

Expr Mutator::visit(const Cast *node) {
    // TODO: should we mutate node->type here?
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Cast::make(node->type, std::move(value));
}

Expr Mutator::visit(const Broadcast *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Broadcast::make(node->lanes, std::move(value));
}

Expr Mutator::visit(const VectorReduce *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return VectorReduce::make(node->op, std::move(value));
}

Expr Mutator::visit(const VectorShuffle *node) {
    Expr value = mutate(node->value);
    auto [idxs, not_changed] = visit_list(this, node->idxs);
    if (value.same_as(node->value) && not_changed) {
        return node;
    }
    return VectorShuffle::make(std::move(value), std::move(idxs));
}

Expr Mutator::visit(const Ramp *node) {
    Expr base = mutate(node->base);
    Expr stride = mutate(node->stride);
    if (base.same_as(node->base) && stride.same_as(node->stride)) {
        return node;
    }
    return Ramp::make(std::move(base), std::move(stride), node->lanes);
}

Expr Mutator::visit(const Extract *node) {
    Expr vec = mutate(node->vec);
    Expr idx = mutate(node->idx);
    if (vec.same_as(node->vec) && idx.same_as(node->idx)) {
        return node;
    }
    return Extract::make(std::move(vec), std::move(idx));
}

Expr Mutator::visit(const Build *node) {
    auto [values, not_changed] = visit_list(this, node->values);
    if (not_changed) {
        return node;
    }
    // TODO: can mutation ever change underlying type?
    // If so, need to explicitly handle imo.
    return Build::make(node->type, std::move(values));
}

Expr Mutator::visit(const Access *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Access::make(node->field, std::move(value));
}

Expr Mutator::visit(const Intrinsic *node) {
    auto [args, not_changed] = visit_list(this, node->args);
    if (not_changed) {
        return node;
    }
    return Intrinsic::make(node->op, std::move(args));
}

Expr Mutator::visit(const Lambda *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Lambda::make(node->args, std::move(value));
}

Expr Mutator::visit(const GeomOp *node) {
    Expr a = mutate(node->a);
    Expr b = mutate(node->b);
    if (a.same_as(node->a) && b.same_as(node->b)) {
        return node;
    }
    return GeomOp::make(node->op, std::move(a), std::move(b));
}

Expr Mutator::visit(const SetOp *node) {
    Expr a = mutate(node->a);
    Expr b = mutate(node->b);
    if (a.same_as(node->a) && b.same_as(node->b)) {
        return node;
    }
    return SetOp::make(node->op, std::move(a), std::move(b));
}

Expr Mutator::visit(const Call *node) {
    Expr func = mutate(node->func);
    auto [args, not_changed] = visit_list(this, node->args);
    if (func.same_as(node->func) && not_changed) {
        return node;
    }
    return Call::make(std::move(func), std::move(args));
}

Expr Mutator::visit(const Instantiate *node) {
    Expr expr = mutate(node->expr);
    // TODO: should we visit the type params? I think no,
    // we don't recurse into an Expr's type currently.
    // auto [types, not_changed] = visit_map(this, node->types);
    if (expr.same_as(node->expr)) {
        return node;
    }
    return Instantiate::make(std::move(expr), node->types);
}

Stmt Mutator::visit(const Print *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Print::make(std::move(value));
}

Stmt Mutator::visit(const Return *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Return::make(std::move(value));
}

Stmt Mutator::visit(const Store *node) {
    Expr index = mutate(node->index);
    Expr value = mutate(node->value);
    if (index.same_as(node->index) && value.same_as(node->value)) {
        return node;
    }
    return Store::make(node->name, std::move(index), std::move(value));
}

Stmt Mutator::visit(const LetStmt *node) {
    auto [loc, not_changed] = mutate_writeloc(this, node->loc);
    Expr value = mutate(node->value);
    if (not_changed && value.same_as(node->value)) {
        return node;
    }
    return LetStmt::make(std::move(loc), std::move(value));
}

Stmt Mutator::visit(const IfElse *node) {
    Expr cond = mutate(node->cond);
    Stmt then_body = mutate(node->then_body);
    Stmt else_body = mutate(node->else_body);
    if (cond.same_as(node->cond) && then_body.same_as(node->then_body) &&
        else_body.same_as(node->else_body)) {
        return node;
    }
    return IfElse::make(std::move(cond), std::move(then_body),
                        std::move(else_body));
}

Stmt Mutator::visit(const Sequence *node) {
    auto [stmts, not_changed] = visit_list(this, node->stmts);
    if (not_changed) {
        return node;
    }
    return Sequence::make(std::move(stmts));
}

Stmt Mutator::visit(const Assign *node) {
    auto [loc, not_changed] = mutate_writeloc(this, node->loc);
    Expr value = mutate(node->value);
    if (not_changed && value.same_as(node->value)) {
        return node;
    }
    return Assign::make(std::move(loc), std::move(value), node->mutating);
}

Stmt Mutator::visit(const Accumulate *node) {
    auto [loc, not_changed] = mutate_writeloc(this, node->loc);
    Expr value = mutate(node->value);
    if (not_changed && value.same_as(node->value)) {
        return node;
    }
    return Accumulate::make(std::move(loc), node->op, std::move(value));
}

Stmt Mutator::visit(const Match *node) {
    ir::Expr loc = mutate(node->loc);
    bool not_changed = loc.same_as(node->loc);
    const size_t n = node->arms.size();
    Match::Arms new_arms(n);
    for (size_t i = 0; i < n; i++) {
        ir::Stmt stmt = mutate(node->arms[i].second);
        not_changed = not_changed && stmt.same_as(node->arms[i].second);
        new_arms[i] = {node->arms[i].first, std::move(stmt)};
    }

    if (not_changed) {
        return node;
    } else {
        return Match::make(std::move(loc), std::move(new_arms));
    }
}

Stmt Mutator::visit(const Yield *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Yield::make(std::move(value));
}

Stmt Mutator::visit(const Scan *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return Scan::make(std::move(value));
}

Stmt Mutator::visit(const YieldFrom *node) {
    Expr value = mutate(node->value);
    if (value.same_as(node->value)) {
        return node;
    }
    return YieldFrom::make(std::move(value));
}

} // namespace ir
} // namespace bonsai
