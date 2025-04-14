#include "Lower/Canonicalize.h"

#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

static const ir::Type u32 = ir::UInt_t::make(32);

// Rewrite a vector of equal immediates to a broadcast.
struct RewriteVectorImmediates : public ir::Mutator {
    ir::Expr visit(const ir::VecImm *node) override {
        ir::Type type = node->type;
        internal_assert(type.lanes() > 0) << type;

        const std::vector<ir::Expr> &values = node->values;
        if (const ir::Expr &v0 = values.front(); std::all_of(
                values.begin() + 1, values.end(), [&](const ir::Expr &vn) {
                    return get_constant_value(v0) == get_constant_value(vn);
                })) {
            return ir::Broadcast::make(type.lanes(), v0);
        }
        return ir::Mutator::visit(node);
    }
};

struct RewriteVectorFields : public ir::Mutator {
    ir::Expr visit(const ir::Access *node) override {
        ir::Expr value = mutate(node->value);
        if (value.type().is_vector()) {
            const uint32_t lane = vector_field_lane(node->field);
            ir::Expr idx = make_const(u32, lane);
            return ir::Extract::make(std::move(value), std::move(idx));
        } else if (value.same_as(node->value)) {
            return node;
        } else {
            return ir::Access::make(node->field, std::move(value));
        }
    }

    std::pair<ir::WriteLoc, bool> canonicalize_loc(const ir::WriteLoc &loc) {
        ir::WriteLoc new_loc(loc.base, loc.base_type);
        bool changed = false;
        for (const auto &value : loc.accesses) {
            if (std::holds_alternative<std::string>(value)) {
                if (new_loc.type.is_vector()) {
                    const std::string field = std::get<std::string>(value);
                    const uint32_t lane = vector_field_lane(field);
                    ir::Expr idx = make_const(u32, lane);
                    internal_assert(idx.defined());
                    new_loc.add_index_access(idx);
                    changed = true;
                } else {
                    new_loc.add_struct_access(std::get<std::string>(value));
                }
            } else {
                ir::Expr idx = mutate(std::get<ir::Expr>(value));
                if (!idx.same_as(std::get<ir::Expr>(value))) {
                    changed = true;
                }
                new_loc.add_index_access(idx);
            }
        }
        return {new_loc, changed};
    }

    ir::Stmt visit(const ir::Assign *node) override {
        auto [loc, changed] = canonicalize_loc(node->loc);
        ir::Expr value = mutate(node->value);
        if (!changed && value.same_as(node->value)) {
            return node;
        } else {
            return ir::Assign::make(std::move(loc), std::move(value),
                                    node->mutating);
        }
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        auto [loc, changed] = canonicalize_loc(node->loc);
        ir::Expr value = mutate(node->value);
        if (!changed && value.same_as(node->value)) {
            return node;
        } else {
            return ir::Accumulate::make(std::move(loc), node->op,
                                        std::move(value));
        }
    }

    ir::Stmt visit(const ir::Match *node) override {
        internal_error << "TODO: implement RewriteVectorFields for Match";
        // auto [loc, changed] = canonicalize_loc(node->loc);
        // // TODO: mutate match arms?
        // if (!changed) {
        //     return node;
        // } else {
        //     return ir::Match::make(std::move(loc), node->arms);
        // }
    }
};

ir::Stmt canonicalize(ir::Stmt stmt) {
    stmt = RewriteVectorFields().mutate(std::move(stmt));
    stmt = RewriteVectorImmediates().mutate(std::move(stmt));
    // TODO: more canonicalizations.
    return stmt;
}

} // namespace

ir::FuncMap Canonicalize::run(ir::FuncMap funcs) const {
    ir::FuncMap new_funcs;

    for (const auto &[name, func] : funcs) {
        ir::Stmt body = canonicalize(func->body);
        new_funcs[name] = std::make_shared<ir::Function>(
            name, func->args, func->ret_type, body, func->interfaces);
    }

    return new_funcs;
}

} // namespace lower
} // namespace bonsai
