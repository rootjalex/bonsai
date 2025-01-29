#include "Lower/Canonicalize.h"

#include "IR/Mutator.h"

#include "Lower/Generics.h"
#include "Lower/Options.h"

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

namespace {

static const ir::Type u32 = ir::UInt_t::make(32);

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
};

ir::Stmt canonicalize(ir::Stmt stmt) {
    stmt = RewriteVectorFields().mutate(stmt);
    // TODO: more canonicalizations.
    return stmt;
}

} // namespace

ir::Program canonicalize(const ir::Program &program) {
    ir::Program new_program;
    new_program.externs = program.externs;
    new_program.types = program.types;

    for (const auto &[f, func] : program.funcs) {
        ir::Stmt body = canonicalize(func->body);
        new_program.funcs[f] = std::make_shared<ir::Function>(
            func->name, func->args, func->ret_type, body, func->interfaces);
    }

    new_program.main_body = canonicalize(program.main_body);

    new_program = lower_option(new_program);

    new_program = lower_generics(new_program);
    // TODO: more canonicalizations
    return new_program;
}

} // namespace lower
} // namespace bonsai
