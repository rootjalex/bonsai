#include "Lower/ForEachs.h"

#include "IR/Analysis.h"
#include "Lower/TiledArrays.h"
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

bool is_range_call(const ir::Expr &expr) {
    if (!expr.type().is<ir::Array_t>()) {
        return false;
    }
    const ir::Generator *call = expr.as<ir::Generator>();
    if (call == nullptr) {
        return false;
    }
    return call->op == ir::Generator::range;
}

ir::Expr get_range_offset(const ir::Expr &expr) {
    internal_assert(expr.type().is<ir::Array_t>());
    const ir::Generator *call = expr.as<ir::Generator>();
    internal_assert(call);
    internal_assert(call->op == ir::Generator::range);
    internal_assert(call->args.size() == 3);
    return call->args[1];
}

ir::Expr get_range_iterable(const ir::Expr &expr) {
    internal_assert(expr.type().is<ir::Array_t>());
    const ir::Generator *call = expr.as<ir::Generator>();
    internal_assert(call);
    internal_assert(call->op == ir::Generator::range);
    internal_assert(call->args.size() == 3);
    return call->args[0];
}

// Lowers for-each loops to for-all loops.
struct LowerToForAll : public ir::Mutator {
    uint64_t lcounter = 0; // unique identifier for iterator variables.
    // The labelled loop indices given out in the function being lowered.
    std::set<std::string> labelled;

    std::map<std::string, ir::Expr> repls;

    std::string unique_idx_name() {
        return "_idx" + std::to_string(lcounter++);
    }

    ir::Expr visit(const ir::Var *node) override {
        if (auto iter = repls.find(node->name); iter != repls.end()) {
            return iter->second;
        }
        return node;
    }

    ir::Expr get_n(const ir::Type &type) {
        if (const auto *array_t = type.as<ir::Array_t>()) {
            return array_t->size;
        } else if (const auto *vector_t = type.as<ir::Vector_t>()) {
            return vector_t->lanes;
        }
        internal_error << "Unknown iterable type: " << type;
    }

    // `e` as a multiple of `width`, when it is one by its form -- `x * w`,
    // `x << log2 w`, a constant multiple, a sum of two, or a cast of one --
    // in which case the quotient; nothing otherwise. What says a range over
    // a tiled array begins and ends on a tile: a layout that slices
    // `prims[b * 4 : b * 4 + k * 4]` has said so in those terms.
    static std::optional<ir::Expr> quotient_by(const ir::Expr &e,
                                               uint32_t width) {
        if (const auto c = get_constant_value<uint64_t>(e);
            c.has_value() && *c % width == 0) {
            return make_const(e.type(), *c / width);
        }
        if (const ir::BinOp *op = e.as<ir::BinOp>()) {
            if (op->op == ir::BinOp::Mul) {
                if (get_constant_value<uint64_t>(op->b) == width) {
                    return op->a;
                }
                if (get_constant_value<uint64_t>(op->a) == width) {
                    return op->b;
                }
            }
            if (op->op == ir::BinOp::Shl && (width & (width - 1)) == 0) {
                uint32_t log2 = 0;
                while ((1u << log2) < width) {
                    log2++;
                }
                if (get_constant_value<uint64_t>(op->b) == log2) {
                    return op->a;
                }
            }
            if (op->op == ir::BinOp::Add) {
                const std::optional<ir::Expr> a = quotient_by(op->a, width);
                const std::optional<ir::Expr> b = quotient_by(op->b, width);
                if (a.has_value() && b.has_value()) {
                    return *a + *b;
                }
            }
        }
        if (const ir::Cast *cast = e.as<ir::Cast>();
            cast != nullptr && cast->mode == ir::Cast::Mode::Convert &&
            cast->type.is_int_or_uint() && cast->value.type().is_int_or_uint()) {
            // A multiple stays one through a conversion between integers:
            // the low bits are what they were.
            if (const std::optional<ir::Expr> inner =
                    quotient_by(cast->value, width)) {
                return ir::Cast::make(cast->type, *inner, cast->mode);
            }
        }
        return std::nullopt;
    }

    // foreach x in <iterable> { body }
    //   => for idx in [0 : n) { body[x := iterable[idx]] }
    //
    // and over the elements of a tiled array (ir::TiledArray: elements kept
    // in tiles of `width`, each field of a tile one vector over its
    // elements), where the range begins and ends on a tile,
    //
    //   => for tile in [0 : n / width) {
    //        parfor <label> in [0 : width) {
    //          body[x := iterable[(first + tile) * width + <label>]]
    //        }
    //      }
    //
    // a loop over the tiles with a loop over the lanes of each inside it: the
    // lanes' loop a parfor, since the body of a loop over a set's elements
    // only accumulates into the query's result and the elements are
    // independent, so that a schedule can vectorize it into one test of the
    // tile -- Embree's test of a Triangle4 -- and named for the arm's field
    // (ForEach::label) so that the schedule can say which:
    // `trace.vectorize(tris.Leaf.data)`. The element's index is spelled as
    // the tile's times the width plus the lane, which is what the tiled
    // array's lowering reads a tile and a lane off (Lower/TiledArrays.cpp).
    //
    // A range that begins on a tile but whose count is not a multiple of the
    // width -- FCPW's leaf, `child[3]` triangles in packets of sixteen, the
    // last packet part full -- runs over as many tiles as hold the count,
    // with the lanes past the count left out,
    //
    //   => for tile in [0 : (n + width - 1) / width) {
    //        parfor <label> in [0 : width) {
    //          if (tile * width + <label> < n) { body[...] }
    //        }
    //      }
    //
    // which a vectorized lanes' loop runs as one test of the tile under a
    // lane mask -- FCPW's `W = min(WIDTH, nReferences - startReference)`.
    // A range not known to begin on a tile iterates its elements one by one,
    // as any other array.
    ir::Stmt visit(const ir::ForEach *node) override {
        ir::Expr iterable = node->iter;

        ir::Expr end = get_n(iterable.type());
        internal_assert(end.defined()) << ir::Stmt(node);
        ir::Expr begin = make_zero(end.type());
        ir::Expr stride = make_one(end.type());

        ir::Expr offset;
        if (is_range_call(iterable)) {
            offset = get_range_offset(iterable);
            iterable = get_range_iterable(iterable);
        }

        // Named for the arm's field where the loop has one, and otherwise
        // invented. A function's loops are told apart by their index, so a
        // second loop over a field of the same name -- the leaves of two
        // trees walked in one function -- is numbered after the first.
        std::string idx_name = unique_idx_name();
        if (!node->label.empty()) {
            idx_name = node->label;
            for (size_t k = 1; !labelled.insert(idx_name).second; k++) {
                idx_name = node->label + "_" + std::to_string(k);
            }
        }

        if (const ir::TiledArray *tiled = iterable.as<ir::TiledArray>()) {
            const std::optional<ir::Expr> first =
                offset.defined() ? quotient_by(offset, tiled->width)
                                 : std::optional<ir::Expr>(begin);
            const std::optional<ir::Expr> whole = quotient_by(end, tiled->width);
            if (first.has_value()) {
                const ir::Type index_t = end.type();
                const std::string tile_name = unique_idx_name();
                ir::Expr tile = ir::Var::make(index_t, tile_name);
                ir::Expr lane = ir::Var::make(index_t, idx_name);
                ir::Expr width = make_const(index_t, tiled->width);
                ir::Expr idx = (*first + tile) * width + lane;
                auto [_, inserted] =
                    repls.try_emplace(node->name, ir::Extract::make(iterable, idx));
                internal_assert(inserted)
                    << "Lowering ForEach encountered duplicate variable: "
                    << node->name;
                ir::Stmt body = mutate(node->body);
                repls.erase(node->name);
                // The lanes the tile's `where` leaves out are not elements
                // (ir::TiledArray::valid): the body runs under the predicate
                // at this lane -- a lane mask once the loop is vectorized.
                if (tiled->valid.defined()) {
                    ir::Expr tile_read =
                        ir::Extract::make(tiled->tiles, *first + tile);
                    body = ir::IfElse::make(
                        tiled_lane_value(*tiled, tile_read, lane, tiled->valid),
                        std::move(body));
                }
                // As many tiles as hold the count; the lanes past it, in the
                // last, left out -- unless the count is whole tiles, when
                // every lane is an element.
                ir::Expr tiles;
                if (whole.has_value()) {
                    tiles = *whole;
                } else {
                    tiles = (end + make_const(index_t, tiled->width - 1)) / width;
                    body = ir::IfElse::make(tile * width + lane < end,
                                            std::move(body));
                }
                ir::Stmt lanes = ir::ParFor::make(
                    idx_name,
                    ir::ParFor::Slice{make_zero(index_t), width,
                                      make_one(index_t)},
                    std::move(body));
                return ir::ForAll::make(
                    tile_name,
                    ir::ForAll::Slice{make_zero(index_t), std::move(tiles),
                                      make_one(index_t)},
                    std::move(lanes));
            }
        }

        ir::Expr idx = ir::Var::make(end.type(), idx_name);
        if (offset.defined()) {
            idx = offset + idx;
        }

        ir::Expr load = ir::Extract::make(iterable, idx);
        // `var = iterable[idx]`
        auto [_, inserted] = repls.try_emplace(node->name, load);
        internal_assert(inserted)
            << "Lowering ForEach encountered duplicate variable: "
            << node->name;

        ir::Stmt body = mutate(node->body);

        repls.erase(node->name);

        // An element-by-element loop over a tiled array whose tiles have a
        // `where`: the lane's predicate, with the tile and the lane divided
        // out of the index.
        if (const ir::TiledArray *tiled = iterable.as<ir::TiledArray>();
            tiled != nullptr && tiled->valid.defined()) {
            const ir::Expr width = make_const(idx.type(), tiled->width);
            ir::Expr tile_read = ir::Extract::make(tiled->tiles, idx / width);
            body = ir::IfElse::make(
                tiled_lane_value(*tiled, tile_read, idx % width, tiled->valid),
                std::move(body));
        }

        ir::ForAll::Slice slice{std::move(begin), std::move(end),
                                std::move(stride)};

        return ir::ForAll::make(idx_name, std::move(slice), std::move(body));
    }
};

} // namespace

ir::FuncMap LowerForEachs::run(ir::FuncMap funcs,
                               const CompilerOptions &options) const {
    LowerToForAll convert_fa;
    for (auto &[_, f] : funcs) {
        convert_fa.labelled.clear();
        f->body = convert_fa.mutate(f->body);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
