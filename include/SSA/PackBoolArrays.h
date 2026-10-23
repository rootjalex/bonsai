#pragma once

#include "SSA/SSA.h"

#include <cstddef>

namespace bonsai {
namespace ir {
namespace ssa {

// A small local array of booleans -- `mut bool[N]`, N at most 64 -- is held
// as the bits of one integer word rather than as N bytes of the stack: an
// element written is a bit set or cleared, `w = (w & ~(1 << i)) | (b << i)`;
// an element read is a bit tested, `(w >> i) & 1`; and the array read whole,
// as the vector of N booleans, is the word's low N bits reinterpreted --
// which is the mask such a read is wanted as (SSA/MaskLanes.h: the lowest
// lane on is the word's trailing zeros counted, Embree's movemask and bsf).
//
// Why. The tree lowering collects a node's children's tests into such an
// array, one element per child in the children's loop, and the sort or the
// loopified descent reads the array whole as the mask of the children to
// visit (Lower/Trees.cpp, `_mask`). On a scalar schedule the loop stores N
// one-byte elements and the whole read loads them as one word -- a load the
// store buffer cannot forward from several smaller stores, so it waits for
// the stores to retire, ten cycles and more, at every interior node. LLVM's
// own promotion of the array to registers (SROA) refuses it: the elements are
// stored as `i1` and the whole is loaded as bytes. As a word the mask is
// assembled in a register by shifts and ors, which a children's loop the
// backend unrolls folds into `b0 | (b1 << 1) | ...`; its trailing zeros are
// a `tzcnt` of a general-purpose register, and no byte of it is in memory.
//
// Which arrays. A stack allocation of `bool[N]` whose every use is a write
// to one element or to the whole (a Store through a GEP at an index, scalar
// or the ramp over all N), a read of one element (a Load through such a GEP,
// or an ExtractIdx at a scalar index), or a read of the whole (an ExtractIdx
// at the ramp), the allocation threaded between blocks by name as the
// builder threads it. An array whose address goes anywhere else -- into a
// call, into a terminator's own reads -- is left alone; so is one written
// inside a loop a bind() put on parallel hardware while it is allocated
// outside that loop, where the elements' stores are independent and the
// word's would race; and so is one whose elements are a gang's vectors, the
// packet traversal's per-lane tests (SSA/Vectorize.cpp, widen), which is a
// different shape.
//
// Where. After the directives -- a vectorized children's loop has made its
// array the lanes of a vector by then -- and before the allocations are
// promoted (SSA/PromoteAllocas.h), so that the word becomes a value carried
// in a register where the promotion reaches, and a plain word in memory,
// which the backend's own promotion finishes, where it does not (a word
// written in a parfor's body and read after it). Returns how many arrays
// were packed.
size_t pack_bool_arrays(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
