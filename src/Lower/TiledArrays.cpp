// A tiled array (ir::TiledArray) is an array of elements whose storage is an
// array of tiles, each tile holding `width` elements with every field of the
// element as a vector over the tile's lanes -- Embree's Triangle4, a leaf's
// four triangles as `v0.x[4], v0.y[4], v0.z[4], e1.x[4], ...` (176 bytes),
// and the array-of-structs-of-arrays of a group nested in a group in Scion
// (Gyurgyik, Root and Kjolstad, "Decoupling Data Layouts from Bounding Volume
// Hierarchies", PLDI 2026, Fig. 7). The layout lowering names such storage
// as the array of elements it logically is (Lower/Layouts.cpp,
// field_in_layout), so that a leaf's `data = range(prims, a, n)` and a
// reference to an element (an index; Lower/ElementReferences.cpp) read it as
// they read any array: `prims[k]`.
//
// This pass makes those reads real. Element `k` is lane `k % width` of tile
// `k / width`: the tile is one read of the tile array, and the element is one
// lane of each of the tile's field vectors -- a vector field stored packed is
// converted to the kind computed with on the way, as any stored vector is --
// put together into the element. Everything after this sees ordinary arrays,
// structs and vectors. A tiled array used any other way than by indexing has
// no meaning (the elements have no representation but the tiles), and is an
// error here.
#include "Lower/TiledArrays.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

namespace {

struct ExpandTiledReads : public ir::Mutator {
    using ir::Mutator::visit;

    // The loops this read sits in, by index: what each ranges over, for
    // reading a tile and a lane off an index spelled as one.
    std::map<std::string, std::pair<ir::Expr, ir::Expr>> loops;

    ir::Stmt visit(const ir::ParFor *node) override {
        loops[node->index] = {node->slice.begin, node->slice.end};
        ir::Stmt out = ir::Mutator::visit(node);
        loops.erase(node->index);
        return out;
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        loops[node->index] = {node->slice.begin, node->slice.end};
        ir::Stmt out = ir::Mutator::visit(node);
        loops.erase(node->index);
        return out;
    }

    // Whether `e` is the index of a loop over exactly [0, width).
    bool is_lane_of(const ir::Expr &e, uint32_t width) const {
        const ir::Var *var = e.as<ir::Var>();
        if (var == nullptr) {
            return false;
        }
        const auto loop = loops.find(var->name);
        return loop != loops.end() &&
               get_constant_value<uint64_t>(loop->second.first) == 0 &&
               get_constant_value<uint64_t>(loop->second.second) == width;
    }

    ir::Expr visit(const ir::Extract *node) override {
        ir::Expr vec = mutate(node->vec);
        ir::Expr idx = mutate(node->idx);
        const ir::TiledArray *tiled = vec.as<ir::TiledArray>();
        if (tiled == nullptr) {
            if (vec.same_as(node->vec) && idx.same_as(node->idx)) {
                return node;
            }
            return ir::Extract::make(std::move(vec), std::move(idx));
        }
        internal_assert(idx.type().is_int_or_uint())
            << "An element of a tiled array is picked by an integer index, "
               "not "
            << idx;
        const ir::Expr width = make_const(idx.type(), tiled->width);

        // The tile and the lane: read off the index where it is spelled as
        // a tile's times the width plus the index of a loop over the lanes
        // -- what a loop over a tiled range becomes (Lower/ForEachs.cpp),
        // and exact, the lane being below the width -- and otherwise
        // divided out. Reading them off is what keeps a vectorized loop
        // over the lanes at one tile: the tile's index is then the same for
        // every lane, and the lane is the lane.
        ir::Expr tile_index, lane;
        if (const ir::BinOp *sum = idx.as<ir::BinOp>();
            sum != nullptr && sum->op == ir::BinOp::Add) {
            for (const auto &[product, candidate] :
                 {std::pair{sum->a, sum->b}, std::pair{sum->b, sum->a}}) {
                const ir::BinOp *scaled = product.as<ir::BinOp>();
                if (scaled != nullptr && scaled->op == ir::BinOp::Mul &&
                    get_constant_value<uint64_t>(scaled->b) == tiled->width &&
                    is_lane_of(candidate, tiled->width)) {
                    tile_index = scaled->a;
                    lane = candidate;
                }
            }
        }
        if (!tile_index.defined()) {
            tile_index = idx / width;
            lane = idx % width;
        }
        // One read of the tile; the lane picked out of each field vector.
        ir::Expr tile = ir::Extract::make(tiled->tiles, tile_index);

        const ir::Type element_t = tiled->type.element_of();
        const auto *element = element_t.as<ir::Struct_t>();
        internal_assert(element != nullptr) << element_t;
        const auto *tile_t = tile.type().as<ir::Struct_t>();
        internal_assert(tile_t != nullptr) << tile.type();

        std::vector<ir::Expr> values;
        values.reserve(element->fields.size());
        for (const auto &field : element->fields) {
            const bool held = std::any_of(
                tile_t->fields.begin(), tile_t->fields.end(),
                [&](const auto &f) { return f.name == field.name; });
            internal_assert(held)
                << "The tile " << tile.type() << " holds no field "
                << field.name << " of the element " << element_t;
            ir::Expr column = ir::Access::make(field.name, tile);
            // Stored packed, computed with unpacked (see Vector_t::packed).
            if (const auto *vector = column.type().as<ir::Vector_t>();
                vector != nullptr && vector->packed) {
                column = ir::Cast::make(
                    ir::Vector_t::make(vector->etype, vector->lanes),
                    std::move(column), ir::Cast::Mode::Convert);
            }
            ir::Expr value = ir::Extract::make(std::move(column), lane);
            internal_assert(ir::equals(value.type(), field.type))
                << "Lane " << lane << " of the tile's " << field.name << " is a "
                << value.type() << ", where the element's is a " << field.type;
            values.push_back(std::move(value));
        }
        return ir::Build::make(element_t, std::move(values));
    }
};

// What is left of a tiled array once its reads are expanded is a use of the
// whole, which has no meaning.
struct RefuseWholeTiledArrays : public ir::Visitor {
    using ir::Visitor::visit;
    void visit(const ir::TiledArray *node) override {
        internal_error << "A tiled array is used other than by reading one "
                       << "element of it: " << ir::Expr(node)
                       << ". Its elements have no representation but their "
                       << "tiles, so the array as a whole cannot be passed, "
                       << "copied or iterated as elements.";
    }
};

} // namespace

ir::FuncMap LowerTiledArrays::run(ir::FuncMap funcs,
                                  const CompilerOptions &options) const {
    ExpandTiledReads expand;
    RefuseWholeTiledArrays refuse;
    for (auto &[name, func] : funcs) {
        func->body = expand.mutate(func->body);
        func->body.accept(&refuse);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
