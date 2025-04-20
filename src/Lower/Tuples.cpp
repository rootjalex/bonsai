#include "Lower/Tuples.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

static size_t counter = 0;

std::string unique_struct_name() {
    return "tuple." + std::to_string(counter++);
}

struct TuplesToStructs : public ir::Mutator {
    std::map<ir::Type, ir::Type, ir::TypeLessThan> rewrite_map;
    ir::TypeMap new_structs;

    ir::Type build_struct(const ir::Tuple_t *node) {
        const size_t n = node->etypes.size();
        ir::Struct_t::Map fields(n);
        for (size_t i = 0; i < n; i++) {
            fields[i].first = "?field" + std::to_string(i);
            fields[i].second =
                mutate(node->etypes[i]); // in the case of nested tuples.
        }
        std::string name = unique_struct_name();
        ir::Type new_struct =
            ir::Struct_t::make(std::move(name), std::move(fields));
        rewrite_map[node] = new_struct;
        return new_struct;
    }

    ir::Type visit(const ir::Tuple_t *node) override {
        ir::Type type = ir::Type(node);
        const auto &iter = rewrite_map.find(type);
        if (iter != rewrite_map.cend()) {
            return iter->second;
        }
        return build_struct(node);
    }

    ir::Expr visit(const ir::Extract *node) override {
        const ir::Tuple_t *as_tuple = node->vec.type().as<ir::Tuple_t>();
        if (as_tuple == nullptr) {
            return ir::Mutator::visit(node);
        }
        const auto constant_idx = get_constant_value(node->idx);
        internal_assert(constant_idx.has_value())
            << "Cannot lower Extract of tuple with non-constant index: "
            << ir::Expr(node);
        uint64_t idx = *constant_idx;
        ir::Type struct_t = visit(as_tuple);
        const ir::Struct_t *as_struct = struct_t.as<ir::Struct_t>();
        internal_assert(as_struct);
        internal_assert(idx < as_struct->fields.size())
            << "Extract on tuple is out of bounds: " << ir::Expr(node)
            << " idx: " << idx << " >= " << as_struct->fields.size();
        // Possibly unnecessary safety check.
        internal_assert(ir::equals(as_struct->fields[idx].second, node->type));
        std::string field = as_struct->fields[idx].first;
        ir::Expr inner = mutate(node->vec);
        return ir::Access::make(std::move(field), std::move(inner));
    }

    ir::Expr visit(const ir::Var *node) override {
        const ir::Tuple_t *as_tuple = node->type.as<ir::Tuple_t>();
        if (as_tuple == nullptr) {
            return ir::Mutator::visit(node);
        }
        ir::Type struct_t = visit(as_tuple);
        return ir::Var::make(std::move(struct_t), node->name);
    }

    // Similar to mutate_writeloc in Mutator.cpp, but also mutates type.
    std::pair<ir::WriteLoc, bool>
    mutate_writeloc(const ir::WriteLoc &loc) override {
        ir::Type base_type = mutate(loc.base_type);
        bool not_changed = base_type.same_as(loc.base_type);
        ir::WriteLoc new_loc(loc.base, std::move(base_type));

        for (const auto &value : loc.accesses) {
            if (const ir::Expr *expr = std::get_if<ir::Expr>(&value)) {
                ir::Expr new_value = mutate(*expr);
                not_changed = not_changed && new_value.same_as(*expr);
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        return {std::move(new_loc), not_changed};
    }

    ir::Expr visit(const ir::Build *node) override {
        const ir::Tuple_t *as_tuple = node->type.as<ir::Tuple_t>();
        if (as_tuple == nullptr) {
            return ir::Mutator::visit(node);
        }
        ir::Type struct_t = visit(as_tuple);

        const size_t n = node->values.size();
        std::vector<ir::Expr> values(n);

        for (size_t i = 0; i < n; i++) {
            values[i] = mutate(node->values[i]);
        }

        return ir::Build::make(std::move(struct_t), std::move(values));
    }
};

} // namespace

ir::Program LowerTuples::run(ir::Program program) const {
    TuplesToStructs converter;

    // Externs should not contain tuples.
    for (const auto &[name, type] : program.externs) {
        internal_assert(!contains<ir::Tuple_t>(type));
    }

    // Types must be rewritten.
    for (auto &[name, type] : program.types) {
        type = converter.mutate(type);
    }

    // Arg types and function bodies must be mutated.
    for (auto &[name, f] : program.funcs) {
        for (auto &arg : f->args) {
            arg.type = converter.mutate(arg.type);
        }
        f->body = converter.mutate(f->body);
    }

    // And new structs must be added to program types.
    for (auto &[name, type] : converter.new_structs) {
        const auto [_, inserted] =
            program.types.try_emplace(name, std::move(type));
        internal_assert(inserted)
            << "Struct: " << name << " already exists in the program";
    }

    return program;
}

} // namespace lower
} // namespace bonsai
