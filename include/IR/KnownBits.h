#pragma once

#include "Expr.h"
#include "Function.h"
#include "Program.h"

#include <cstdint>

namespace bonsai {
namespace ir {

// The bits an integer expression can have set: a mask with a 1 wherever
// some evaluation of `e` may have a 1, computed from the expression's
// arithmetic -- constants, masks, shifts by constants, ors, casts, selects
// -- and through calls to the program's functions, whose bodies are read
// as straight-line code with branches (anything else, a loop or a store
// through an index, gives every bit). A sound over-approximation: a bit
// the analysis clears is never set at run time; a bit it keeps may or may
// not be. Addition, multiplication and subtraction widen to every bit up
// to where their operands could carry; a cast from a float or an unknown
// variable is every bit of its type.
//
// Written for a sorted queue's key (Lower/Sorts.cpp): a radix sort pays by
// the bit, so the sort is told how many bits of the key are live --
// `sort_indices(keys, perm, count, bits)` -- and the device sort runs that
// many digit passes rather than the key type's eight (runtime/
// bonsai_cuda_sort.cu). A Morton code of six ten-bit coordinates lives in
// 60 bits, one of six six-bit coordinates in 36; the key's arithmetic says
// which, and the compiler reads it off rather than being told.
uint64_t possible_bits(const Expr &e, const FuncMap &funcs);

// The number of bits a key can use: one past the highest bit of
// possible_bits, at least 1.
uint32_t live_bits(const Expr &e, const FuncMap &funcs);

} // namespace ir
} // namespace bonsai
