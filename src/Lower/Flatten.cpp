#include "Lower/Flatten.h"

#include "Lower/TopologicalOrder.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

// Retrieves the array dimension sizes from this type.
std::vector<ir::Expr> array_dimension_sizes(ir::Type type) {
    internal_assert(type.is<ir::Array_t>()) << type;
    std::vector<ir::Expr> sizes;
    while (const auto *atype = type.as<ir::Array_t>()) {
        sizes.push_back(ir::Cast::make(ir::Index_t::make(), atype->size));
        type = atype->etype;
    }
    return sizes;
}

// Returns whether this is an n-dimensional array type, where n > 1.
bool is_nd_array_type(ir::Type type) {
    const auto *atype = type.as<ir::Array_t>();
    if (atype == nullptr) {
        return false;
    }
    return atype->etype.is<ir::Array_t>();
}

// Flattens the array type, e.g.,
//   i32[M][N] -> i32[M * N]
//   i32 -> i32
ir::Type flatten_array_type(ir::Type type) {
    if (!type.is<ir::Array_t>()) {
        // Avoid array-ifying a type that is not an array.
        return type;
    }
    std::vector<ir::Expr> sizes;
    while (const auto *atype = type.as<ir::Array_t>()) {
        sizes.push_back(ir::Cast::make(ir::Index_t::make(), atype->size));
        type = atype->etype;
    }
    ir::Expr one = make_one(ir::Index_t::make());
    ir::Expr size = std::accumulate(
        sizes.begin(), sizes.end(), one,
        [](const auto &a, const auto &b) { return ir::BinOp::mul(a, b); });
    return ir::Array_t::make(type, size);
}

// Flattens an array of static arrays, where each inner array is of equal size,
// i.e., not jagged. If these conditions do not hold, then flattening fails and
// `ok` is set to false.
void flatten_static_array(ir::Expr e, std::vector<ir::Expr> &values,
                          ir::Type &etype, bool &ok) {
    if (!ok) {
        return;
    }
    if (!e.is<ir::Build>()) {
        ok = false;
        return;
    }
    const auto *build = e.as<ir::Build>();
    ir::Expr size;
    for (const auto &value : build->values) {
        const auto *type = value.type().as<ir::Array_t>();
        if (type == nullptr) {
            ok = false;
            return; // Not an array of arrays.
        }
        if (type->etype.is<ir::Array_t>()) {
            flatten_static_array(value, values, etype, ok);
            return;
        }
        etype = type->etype;
        if (!size.defined()) {
            size = type->size;
        }
        if (!(is_const(size) && is_const(type->size))) {
            ok = false;
            return; // Cannot be statically inferred.
        }
        uint64_t n = get_constant_value(type->size);
        if (get_constant_value(size) != n) {
            ok = false;
            return; // Jagged array.
        }

        for (int i = 0; i < n; ++i) {
            ir::Expr idx = ir::IdxImm::make(i);
            values.push_back(ir::Extract::make(value, std::move(idx)));
        }
    }
}

// Flattens (indices,size) to a 1-dimensional index value. Reference:
// https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general
ir::Expr flatten_index(std::vector<ir::Expr> indices,
                       std::vector<ir::Expr> sizes) {
    ir::Expr index = make_zero(ir::Index_t::make());
    internal_assert(sizes.size() == indices.size());
    for (int i = 0; i < sizes.size(); ++i) {
        ir::Expr dsize = sizes[i];
        ir::Expr dindex = indices[i];
        index = ir::BinOp::add(ir::BinOp::mul(index, dsize), dindex);
    }
    return index;
}

class FlattenStructure : public ir::Mutator {
  public:
    FlattenStructure(ir::FrameStack<ir::Type> &frames) : frames(frames) {}
    using ir::Mutator::visit;
    // Stores a mapping from name to the respective allocation's dimensions
    // (pre-flattening).
    ir::Stmt visit(const ir::Allocate *node) override {
        frames.add_to_frame(node->name, node->type);
        return ir::Allocate::make(node->name, flatten_array_type(node->type));
    }

    ir::Expr visit(const ir::Lambda *node) override {
        std::vector<ir::Lambda::Argument> args;

        frames.new_frame();
        for (const auto &arg : node->args) {
            if (!arg.type.is<ir::Array_t>()) {
                args.push_back(arg);
                continue;
            }
            frames.add_to_frame(arg.name, arg.type);
            args.push_back(ir::Lambda::Argument{
                .name = arg.name,
                .type = flatten_array_type(arg.type),
            });
        }
        ir::Expr value = mutate(node->value);
        frames.pop_frame();
        return ir::Lambda::make(std::move(args), std::move(value));
    };

    ir::Expr visit(const ir::Var *node) override {
        if (!frames.name_in_scope(node->name)) {
            if (is_nd_array_type(node->type)) {
                frames.add_to_frame(node->name, node->type);
                ir::Type type = flatten_array_type(node->type);
                return ir::Var::make(std::move(type), node->name);
            }
            return ir::Mutator::visit(node);
        }
        ir::Type type = frames.from_frames(node->name);
        return ir::Var::make(flatten_array_type(std::move(type)), node->name);
    }

    // We want arguments of a call node to be visited.
    ir::Expr visit(const ir::Call *node) override { return node; }

    ir::Expr visit(const ir::Extract *node) override {
        auto [_, inserted] = visited.insert(node);
        if (!inserted) {
            return node;
        }
        auto [indices, array] = get_metadata(node);
        const ir::Var *v = array.as<ir::Var>();
        if (!frames.name_in_scope(v->name)) {
            return ir::Mutator::visit(node);
        }

        ir::Type original_type = frames.from_frames(v->name);
        std::vector<ir::Expr> sizes = array_dimension_sizes(original_type);

        std::reverse(indices.begin(), indices.end());
        ir::Expr index = flatten_index(std::move(indices), std::move(sizes));
        return ir::Extract::make(mutate(std::move(array)), std::move(index));
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!node->index.type().is<ir::Vector_t>()) {
            return node; // This is a scalar; do nothing.
        }
        const int32_t lanes = node->index.type().as<ir::Vector_t>()->lanes;
        std::vector<ir::Expr> indices;
        indices.reserve(lanes);
        for (int32_t i = 0; i < lanes; ++i) {
            indices.push_back(ir::Extract::make(node->index, i));
        }

        internal_assert(frames.name_in_scope(node->name)) << node->name;
        ir::Type original_type = frames.from_frames(node->name);
        std::vector<ir::Expr> sizes = array_dimension_sizes(original_type);
        ir::Expr index = flatten_index(indices, sizes);
        return ir::Store::make(node->name, std::move(index), node->value);
    }

    ir::Expr visit(const ir::Build *node) override {
        if (node->values.size() <= 1) {
            return Mutator::visit(node);
        }

        std::vector<ir::Expr> flattened_values;
        bool ok = true;
        ir::Type etype;
        flatten_static_array(node, flattened_values, etype, ok);
        if (!ok) {
            return node;
        }
        internal_assert(etype.defined()) << node;
        ir::Expr size = ir::IdxImm::make(flattened_values.size());
        ir::Type type = ir::Array_t::make(std::move(etype), std::move(size));
        return ir::Build::make(std::move(type), std::move(flattened_values));
    }

  private:
    ir::FrameStack<ir::Type> &frames;
    // Used for building 1-dimensional Extract from a multi-dimensional one.
    struct ExtractMetadata {
        // The index variable(s).
        std::vector<ir::Expr> indices;
        // The original array being extracted from.
        ir::Expr array;
    };

    // Returns the metadata for this extract node.
    ExtractMetadata get_metadata(const ir::Extract *node) {
        std::vector<ir::Expr> indices;
        ir::Expr vec;
        do {
            indices.push_back(node->idx);
            vec = node->vec;
        } while ((node = node->vec.as<ir::Extract>()));

        return ExtractMetadata{.indices = indices, .array = vec};
    }

    // Tracks visited ir::Extract nodes.
    std::set<const ir::Extract *> visited;
};

std::shared_ptr<ir::Function> mutate_function(const ir::Function &f) {
    std::vector<ir::Function::Argument> args;
    args.reserve(f.args.size());

    ir::FrameStack<ir::Type> frames;
    frames.new_frame();
    for (const auto &arg : f.args) {
        if (!arg.type.is<ir::Array_t>()) {
            args.push_back(arg);
            continue;
        }
        frames.add_to_frame(arg.name, arg.type);
        args.push_back(ir::Function::Argument(
            arg.name, flatten_array_type(arg.type), arg.default_value));
    }
    ir::Type ret_type = flatten_array_type(f.ret_type);

    FlattenStructure lower(frames);
    ir::Stmt body = lower.mutate(f.body);
    frames.pop_frame();

    return std::make_shared<ir::Function>(f.name, std::move(args),
                                          std::move(ret_type), std::move(body),
                                          f.interfaces);
}

} // namespace

ir::FuncMap Flatten::run(ir::FuncMap functions) const {

    std::vector<std::string> topological_order =
        func_topological_order(functions, /*undef_calls=*/false);
    for (const std::string &name : topological_order) {
        auto it = functions.find(name);
        it->second = mutate_function(*it->second);
    }
    return functions;
}

} // namespace lower
} // namespace bonsai
