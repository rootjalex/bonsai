#pragma once

#include "IR/Expr.h"
#include "IR/Type.h"

#include <cstdint>
#include <vector>

namespace bonsai {
namespace lower {

// A value of any type as the 32-bit words it occupies in memory, and back.
//
// This is how a variant's payload is stored (see Lower/ADTs.cpp): not as a
// union of the arms, which every backend then has to lay out, widen and read
// through storage, but as `[[packed]] u32xN` -- N words that any arm's fields
// are read out of at their byte offsets with a reinterpret, a shift and a mask,
// and written into the same way. Words rather than bytes because a gang reads
// one word per lane in one gather, and because every field of every arm that
// matters starts on a word boundary; a byte or halfword field is a piece of a
// word, and a 64-bit one is two. The result is ordinary arithmetic that the
// vectorizer widens and the backends lower with nothing to know about unions.
//
// Offsets follow C's rule for a struct (ir::layout_bytes): each field at the
// next offset aligned for it, the whole rounded up to its alignment.

// How many 32-bit words a value of `type` occupies.
uint64_t words_of(const ir::Type &type);

// The value of `type` read out of `words`, a `[[packed]] u32xN`, starting at
// byte `offset` of it.
ir::Expr read_words(const ir::Expr &words, uint64_t offset,
                    const ir::Type &type);

// Writes `value` of `type` into `words`, one expression per word (zero to
// start with), starting at byte `offset`. A word two fields share is the OR
// of their pieces.
void write_words(std::vector<ir::Expr> &words, uint64_t offset,
                 const ir::Expr &value, const ir::Type &type);

// The type `n` words are held as.
ir::Type words_type(uint64_t n);

} // namespace lower
} // namespace bonsai
