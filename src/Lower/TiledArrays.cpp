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
        // One read of the tile; the lane picked out of each field vector.
        ir::Expr tile = ir::Extract::make(tiled->tiles, idx / width);
        const ir::Expr lane = idx % width;

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
