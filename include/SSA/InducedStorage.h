#pragma once

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// The storage a scheduling transform induces -- an array the program did not
// declare, made to carry a value from one place of the generated program to
// another -- is a struct of arrays, by the rule a queue's entry is stored by
// (docs/schedule-semantics.md, 4.5): one array per leaf of the stored type, a
// short vector one array where no gang touches it (stored_vector_type), an
// ADT's padding nowhere. pbrt's wavefront keeps every per-item state that way
// (`SOA<PixelSampleState>`, an array per member indexed by the pixel), and
// the reason is the machine's: a warp's store of one field into an array of
// 96-byte structs touches 32 half-filled sectors where its store into an
// array of that field touches four.
//
// A transform makes its storage as the one array of the stored type its
// rewrite is simplest with -- a slot of `VisibleSurface` per pixel sample --
// and registers it under a handle named for the stored thing
// (Function::InducedStorage: `rays.visible`). Once every directive has run,
// split_induced_storage() replaces each such array by its leaf arrays and
// rewrites every use the program has of its elements' addresses: the
// owner's stores and loads, the slot address a drain rebuilds from the
// index, the pointer parameters of every chain function it is handed to
// (each becomes a parameter per leaf, at every call), a field's address and
// the loads and stores through it, a reducer's accumulates. A leaf nothing
// reads is not stored and has no array -- the member of pbrt's pixel state
// that an RGB film never looks at. A use the rewrite cannot follow (an
// address compared, returned, or stored as a value) is an error that names
// it, not a silent array of structs.
//
// The handle is the one place a later directive would find the storage to
// lay it out otherwise; nothing here is the layout language's promise.

// The split, over every function's registered storage.
void split_induced_storage(FuncMap &funcs);

// Registers `storage` with its owner.
void register_induced(Function &func, Function::InducedStorage storage);

// Prints `func`'s registered storage, a line per handle, for the SSA dumps.
void dump_induced(std::ostream &os, const Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
