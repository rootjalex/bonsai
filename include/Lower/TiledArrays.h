#pragma once

#include "CompilerOptions.h"
#include "IR/Expr.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

// The value of `expr` -- a derived field's, or the `where` predicate's, over
// the tile's members as ir::TiledArray::placeholder names them -- at lane
// `lane` of `tile`, one read of the tile's struct: every stored member's
// lane substituted in (packed storage converted on the way), and every
// derived member's value before it, so that a derived field may read another.
ir::Expr tiled_lane_value(const ir::TiledArray &tiled, const ir::Expr &tile,
                          const ir::Expr &lane, const ir::Expr &expr);

// Spells every read of an element of a tiled array (ir::TiledArray) as the
// read of its tile it is: lane `k % width` of tile `k / width`, one lane of
// each of the tile's field vectors, assembled into the element. After
// LowerForEachs, which makes a leaf's elements indexed reads of the array its
// `data` ranges over, and after LowerElementReferences, which keeps the index
// as the reference and reads through it with the same indexed read.
class LowerTiledArrays : public Pass {
  public:
    const std::string name() const override { return "lower-tiled-arrays"; }
    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
