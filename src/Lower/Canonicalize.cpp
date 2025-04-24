#include "Lower/Canonicalize.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>

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

ir::BinOp::OpType acc_to_bin(const ir::Accumulate::OpType op) {
    switch (op) {
    case ir::Accumulate::OpType::Add:
        return ir::BinOp::OpType::Add;
    case ir::Accumulate::OpType::Mul:
        return ir::BinOp::OpType::Mul;
    case ir::Accumulate::OpType::Sub:
        return ir::BinOp::OpType::Sub;
    case ir::Accumulate::OpType::Argmin:
    case ir::Accumulate::OpType::Argmax:
        internal_error << "[unimplemented] mapping from Accumulate::OpType to "
                          "respective BinOp::OpType: "
                       << op;
    }
}

struct RenameVariable : public ir::Mutator {
    RenameVariable(const std::set<std::string> &mutable_function_arguments)
        : mutable_function_arguments(mutable_function_arguments) {}

    const std::set<std::string> &mutable_function_arguments;
    // Tracks the old variable name to the new name.
    std::unordered_map<std::string, std::string> old_to_new_name;

    std::pair<std::string, bool> rename(std::string name) {
        auto it = old_to_new_name.find(name);
        if (should_rename) {
            std::string new_name = "_" + std::to_string(counter++) + name;
            old_to_new_name[name] = new_name;
            return {new_name, true};
        }
        if (it != old_to_new_name.end()) {
            return {it->second, true};
        }
        return {name, false};
    }

    ir::Expr visit(const ir::Var *node) override {
        auto it = old_to_new_name.find(node->name);
        if (it == old_to_new_name.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

    // x: mut i32 = 0;
    // x += 1;
    // use(x)
    // ->
    // x: mut i32 = 0;
    // _0x: mut i32 = x + 1;
    // use(_0x);
    ir::Stmt visit(const ir::Accumulate *node) override {
        ir::WriteLoc location = node->loc;
        if (!location.base_type.is_scalar()) {
            // This is a struct member update, don't rename it.
            return ir::Mutator::visit(node);
        }
        if (mutable_function_arguments.contains(location.base)) {
            // This is a mutable function argument, don't rename it.
            return ir::Mutator::visit(node);
        }
        // Save the previous name (if it exists).
        auto it = old_to_new_name.find(location.base);
        std::optional<std::string> old_name;
        if (it != old_to_new_name.end()) {
            old_name = it->second;
        }
        // Visit the value before updating the mapping.
        ir::Expr value = mutate(node->value);
        // (Potentially) rename the current assignment's name.
        auto [new_name, updated] = rename(location.base);
        if (!updated) {
            return ir::Mutator::visit(node);
        };
        std::string name = old_name.has_value() ? *old_name : location.base;
        ir::Expr lhs = ir::Var::make(location.type, std::move(name));
        return ir::Assign::make(/*loc=*/ir::WriteLoc(new_name, location.type),
                                /*value=*/
                                ir::BinOp::make(acc_to_bin(node->op),
                                                std::move(lhs),
                                                std::move(value)),
                                /*mutating=*/false);
    }

    // x: mut i32 = 0;
    // x := 1 + y;
    // use(x)
    // ->
    // x: mut i32 = 0;
    // _0x: mut i32 = 1 + y;
    // use(_0x);
    ir::Stmt visit(const ir::Assign *node) override {
        ir::WriteLoc location = node->loc;
        if (!node->mutating) {
            // This is the first occurrence, don't rename it.
            return ir::Mutator::visit(node);
        }
        if (!location.base_type.is_scalar()) {
            // This is a struct member update, don't rename it.
            return ir::Mutator::visit(node);
        }
        if (mutable_function_arguments.contains(location.base)) {
            // This is a mutable function argument, don't rename it.
            return ir::Mutator::visit(node);
        }
        // Visit the value before updating the mapping.
        ir::Expr value = mutate(node->value);
        auto [new_name, updated] = rename(location.base);
        // (Potentially) rename the current assignment's name.
        if (!updated) {
            return ir::Mutator::visit(node);
        }
        return ir::Assign::make(ir::WriteLoc(new_name, location.type),
                                std::move(value), /*mutating=*/false);
    }

    ir::Stmt visit(const ir::IfElse *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

    ir::Stmt visit(const ir::DoWhile *node) override {
        ScopedValue<bool> guard(should_rename, false);
        return ir::Mutator::visit(node);
    }

  private:
    // Whether the variable should be given a fresh name.
    bool should_rename = true;
    // For unique variable renaming.
    int64_t counter = 0;
};

} // namespace

ir::FuncMap Canonicalize::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        func->body = canonicalize(std::move(func->body));

        std::set<std::string> args = get_mutable_arguments(*func);
        RenameVariable lower(args);
        func->body = lower.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
