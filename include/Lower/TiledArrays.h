#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <string>

namespace bonsai {
namespace lower {

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
