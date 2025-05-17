#include "Lower/DynamicArrays.h"

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
std::string unique_dynamic_array_name() {
    static size_t counter = 0;
    return "__dyn_array" + std::to_string(counter++);
}

struct DynamicArraysToStructs : public ir::Mutator {
    std::map<ir::Type, ir::Type, ir::TypeLessThan> rewrite_map;
    ir::TypeMap new_structs;

    ir::Type build_struct(const ir::DynArray_t *node) {
        ir::Struct_t::Map fields;
        ir::Struct_t::DefMap defaults;
        // The pointer (which also defines the current capacity).
        fields.push_back(ir::TypedVar(
            "buffer", ir::Array_t::make(node->etype, node->capacity)));
        // Size
        ir::Type type = node->capacity.type();
        std::string name = unique_dynamic_array_name();
        fields.push_back(ir::TypedVar("size", type));
        fields.push_back(ir::TypedVar("capacity", node->capacity.type()));
        defaults["size"] = make_zero(type);

        ir::Type new_struct =
            ir::Struct_t::make(name, std::move(fields), std::move(defaults));
        rewrite_map[node] = new_struct;
        new_structs[name] = new_struct;
        return new_struct;
    }

    ir::Type visit(const ir::DynArray_t *node) override {
        ir::Type type = ir::Type(node);
        const auto &iter = rewrite_map.find(type);
        if (iter != rewrite_map.cend()) {
            return iter->second;
        }
        return build_struct(node);
    }

    ir::Expr visit(const ir::Var *node) override {
        ir::Type new_type = mutate(node->type);
        if (new_type.same_as(node->type)) {
            return node;
        }
        return ir::Var::make(std::move(new_type), node->name);
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
        const ir::DynArray_t *dynamic_array_t = node->type.as<ir::DynArray_t>();
        if (dynamic_array_t == nullptr) {
            return ir::Mutator::visit(node);
        }
        internal_error << "unimplemented: " << ir::Expr(node);
    }
};

} // namespace

ir::Program LowerDynamicArrays::run(ir::Program program,
                                    const CompilerOptions &options) const {
    DynamicArraysToStructs converter;

    // Externs should not contain dynamic arrays.
    for (const auto &[name, type] : program.externs) {
        internal_assert(!contains<ir::DynArray_t>(type)) << type;
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
        f->ret_type = converter.mutate(f->ret_type);
    }

    // And new structs must be added to program types.
    for (auto &[name, type] : converter.new_structs) {
        const auto [_, inserted] =
            program.types.try_emplace(name, std::move(type));
        internal_assert(inserted)
            << "struct: " << name << " already exists in the program";
    }

    return program;
}

} // namespace lower
} // namespace bonsai
