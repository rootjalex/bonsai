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
        sizes.push_back(atype->size);
        type = atype->etype;
    }
    return sizes;
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
        sizes.push_back(atype->size);
        type = atype->etype;
    }

    // Uses the bit width of the largest type for the constant one.
    auto it = std::max_element(sizes.begin(), sizes.end(),
                               [&](const auto &a, const auto &b) {
                                   return a.type().bits() < b.type().bits();
                               });
    ir::Expr one = make_one(it->type());

    ir::Expr size = std::accumulate(
        sizes.begin(), sizes.end(), one,
        [](const auto &a, const auto &b) { return ir::BinOp::mul(a, b); });
    return ir::Array_t::make(type, size);
}

// Flattens (indices,size) to a 1-dimensional index value.
// Reference:
// https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general
ir::Expr flatten_index(std::vector<ir::Expr> indices,
                       std::vector<ir::Expr> sizes) {
    // Uses the bit width of the largest type for the constant one.
    auto it = std::max_element(sizes.begin(), sizes.end(),
                               [&](const auto &a, const auto &b) {
                                   return a.type().bits() < b.type().bits();
                               });
    ir::Expr index = make_zero(it->type());
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
            return ir::Mutator::visit(node);
        }
        ir::Type type = frames.from_frames(node->name);
        return ir::Var::make(flatten_array_type(std::move(type)), node->name);
    }

    ir::Expr visit(const ir::Extract *node) override {
        auto [_, inserted] = visited.insert(node);
        if (!inserted) {
            return node;
        }
        auto [indices, array] = get_metadata(node);
        const ir::Var *v = array.as<ir::Var>();
        internal_assert(v && frames.name_in_scope(v->name)) << array;

        ir::Type original_type = frames.from_frames(v->name);
        std::vector<ir::Expr> sizes = array_dimension_sizes(original_type);

        return ir::Extract::make(/*vec=*/std::move(array),
                                 /*idx=*/flatten_index(indices, sizes));
    }

    ir::Stmt visit(const ir::Store *node) override {
        if (!node->index.type().is<ir::Vector_t>()) {
            // Scalar vector types are already flat!
            return node;
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
