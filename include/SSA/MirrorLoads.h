#pragma once

#include "SSA/SSA.h"

#include <cstddef>

namespace bonsai {
namespace ir {
namespace ssa {

// A pure value computed inside a loop from a cell in memory -- a `mut` local
// or a pointer parameter -- whose other inputs do not change, is computed
// where the cell is written instead, into a cell of its own that mirrors
// it, and the loop reads the mirror. The cell changes at its writes alone,
// so the value changes there alone; computed at every iteration it was
// computed as often as the loop runs, where it need only be computed as
// often as the cell is written. Promotion then makes the mirror a value
// carried round the loop (SSA/PromoteAllocas.h), a register.
//
// The nearest-hit traversal's node test compares the children's entry
// distances against `min(bits(tfar), bits(best) - 1)`, a function of the
// argmin's running best; the best changes at an accepted hit, once per
// leaf that improves on it, and the node test runs at every node. Embree
// keeps the same quantity, `tray.tfar`, broadcast once where a hit lands.
// Without this the limit was a move between register files, a decrement, a
// compare, a conditional move and a broadcast at every node.
//
// WHAT IS MIRRORED
//
// A cell is a local allocation or a pointer parameter whose address is used
// only as the pointer of loads, stores and accumulates in this function
// (threading between blocks by name aside): nothing else can write it, so
// its writes are all in view. From a load of such a cell inside a loop, the
// chain of instructions each the only reader of the one before -- the
// field read, the reinterpretation, the arithmetic, the broadcast -- is
// followed while each is a pure computation whose other operands are
// available everywhere: constants, the function's parameters, and pure
// computations of those (which are cloned where they are needed). The
// chain's last value is mirrored: a cell of its type is allocated at the
// entry, assigned the chain's value recomputed from a fresh load of the
// cell at the entry and right after each write of the cell, and the value
// in the loop becomes a load of the mirror. The chain, read by no one now,
// goes.
//
// Returns how many values were mirrored.
size_t mirror_derived_loads(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
