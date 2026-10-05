#ifndef BONSAI_SSA_MASK_LANES_H
#define BONSAI_SSA_MASK_LANES_H

#include "SSA/SSA.h"

#include <map>
#include <memory>

namespace bonsai {
namespace ir {
namespace ssa {

// Reading lanes off a mask of `n` booleans, one at a time, the way Embree
// reads its movemask (bvh_traverser1.h, `bscf`): where the lane count is a
// word's, the mask is the integer whose bits are its lanes (`kmov`), the
// lowest lane on is that word's trailing zeros counted (`tzcnt`, Embree's
// bsf), the highest its leading zeros (`lzcnt`, bsr), the lowest is cleared
// by `m & (m - 1)` (`blsr`), and whether any lane is left is the word
// against zero -- the flag the clear leaves. A four-wide mask is four bits,
// an integer of a width the machine has no word for, widened to the index's
// word (the C++ backend's uint4_t, runtime/u4.h, is a byte read as its low
// four bits); Embree's BVH4 reads its four-wide mask the same way,
// `movemask` to a word. Where the lane count is not a word's, a lane is the
// least or the greatest lane index the mask keeps, by a reduction over the
// lane ramp, and clearing one is a compare against it.
//
// What every lowering that takes a node's hit children one by one reads
// them with: loopify's arms over the lanes a node hit (SSA/QueueRecursion
// .cpp, Embree's traverseAnyHit) and the sort's peel of them in key order
// (SSA/SortRecursion.cpp, traverseClosestHit). The lane count decides the
// mask's word and nothing else.
class MaskLanes {
public:
    // Lanes of `n`, read as values of `index_type`, the type a lane index
    // is (a u32).
    MaskLanes(Function &func, uint32_t n, Type index_type);

    // Whether the mask's lanes are the bits of an integer.
    bool as_bits() const { return as_bits_; }
    // The word a mask is read as, where it is one.
    const Type &wide_type() const { return wide_t; }

    // The mask as lanes are read off it: the word of its bits, or the mask
    // itself where its lanes are not bits.
    std::shared_ptr<Value> whole(const std::shared_ptr<Block> &block,
                                 const std::shared_ptr<Value> &mask);
    // The least lane `m` has on (`m` as `whole` returns it).
    std::shared_ptr<Value> lowest(const std::shared_ptr<Block> &block,
                                  const std::shared_ptr<Value> &m);
    // The greatest lane `m` has on.
    std::shared_ptr<Value> highest(const std::shared_ptr<Block> &block,
                                   const std::shared_ptr<Value> &m);
    // `m` with its lowest lane on, `lane`, off.
    std::shared_ptr<Value> without_lowest(const std::shared_ptr<Block> &block,
                                          const std::shared_ptr<Value> &m,
                                          const std::shared_ptr<Value> &lane);
    // Whether `m` has any lane on.
    std::shared_ptr<Value> any(const std::shared_ptr<Block> &block,
                               const std::shared_ptr<Value> &m);
    // The lane indices 0 .. n - 1 as a vector, made once per block.
    std::shared_ptr<Value> ramp(const std::shared_ptr<Block> &block);

private:
    std::shared_ptr<Value> index_of(uint64_t k) const;
    std::shared_ptr<Value> count_zeros(const std::shared_ptr<Block> &block,
                                       const std::shared_ptr<Value> &bits,
                                       ir::Intrinsic::OpType which);
    std::shared_ptr<Value> extreme(const std::shared_ptr<Block> &block,
                                   const std::shared_ptr<Value> &m,
                                   ir::VectorReduce::OpType which);

    Function &func;
    const uint32_t n;
    const Type index_t;
    bool as_bits_ = false;
    Type bits_t;  // the integer of n bits, where the lanes are bits
    Type wide_t;  // the word it is counted at
    Type lanes_t; // a vector of n indices
    Type mask_t;  // a vector of n booleans
    std::map<const Block *, std::shared_ptr<Value>> ramps;
};

} // namespace ssa
} // namespace ir
} // namespace bonsai

#endif // BONSAI_SSA_MASK_LANES_H
