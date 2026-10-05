#include "SSA/MaskLanes.h"

#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::shared_ptr;
using std::vector;

namespace {

shared_ptr<Value> append(Function &func, const shared_ptr<Block> &block,
                         Type type, Instruction::Op op,
                         vector<shared_ptr<Value>> operands) {
    auto instr = std::make_shared<Instruction>(
        func.get_unique_name(), std::move(type), op, std::move(operands),
        block);
    block->instrs.push_back(instr);
    return std::make_shared<Value>(std::move(instr));
}

} // namespace

MaskLanes::MaskLanes(Function &func, uint32_t n, Type index_type)
    : func(func), n(n), index_t(std::move(index_type)),
      as_bits_(n == 4 || n == 8 || n == 16 || n == 32 || n == 64),
      lanes_t(Vector_t::make(index_t, n)),
      mask_t(Vector_t::make(Bool_t::make(), n)) {
    if (as_bits_) {
        bits_t = UInt_t::make(n);
        // Counted at the index's width where the mask is narrower: tzcnt
        // and lzcnt answer a zero word with its width, which is what the
        // intrinsics mean on a 32-bit word, where an 8-bit word's zero takes
        // a fixup first (an `or 0x100`, a `shl 24`) that LLVM drops only
        // where it knows the word is not zero.
        wide_t = n < index_t.bits() ? index_t : bits_t;
    }
}

shared_ptr<Value> MaskLanes::index_of(uint64_t k) const {
    return std::make_shared<Value>(Constant{index_t, k});
}

shared_ptr<Value> MaskLanes::whole(const shared_ptr<Block> &block,
                                   const shared_ptr<Value> &mask) {
    if (!as_bits_) {
        return mask;
    }
    auto bits = append(func, block, bits_t, Instruction::Op::Reinterpret, {mask});
    if (wide_t.bits() == bits_t.bits()) {
        return bits;
    }
    return append(func, block, wide_t, Instruction::Op::Cast, {bits});
}

shared_ptr<Value> MaskLanes::count_zeros(const shared_ptr<Block> &block,
                                         const shared_ptr<Value> &bits,
                                         ir::Intrinsic::OpType which) {
    auto zeros = std::make_shared<Instruction>(
        func.get_unique_name(), wide_t, Instruction::Op::Intrinsic,
        vector<shared_ptr<Value>>{bits}, block);
    zeros->intrinsic = which;
    block->instrs.push_back(zeros);
    auto counted = std::make_shared<Value>(zeros);
    if (wide_t.bits() == index_t.bits()) {
        return counted;
    }
    return append(func, block, index_t, Instruction::Op::Cast, {counted});
}

shared_ptr<Value> MaskLanes::ramp(const shared_ptr<Block> &block) {
    shared_ptr<Value> &r = ramps[block.get()];
    if (!r) {
        r = append(func, block, lanes_t, Instruction::Op::Ramp,
                   {index_of(0), index_of(1)});
    }
    return r;
}

// The least (Min) or the greatest (Max) lane index `m` has on, the lanes it
// has off standing out of the way.
shared_ptr<Value> MaskLanes::extreme(const shared_ptr<Block> &block,
                                     const shared_ptr<Value> &m,
                                     ir::VectorReduce::OpType which) {
    auto none = append(func, block, lanes_t, Instruction::Op::Bc,
                       {index_of(which == ir::VectorReduce::Min ? n : 0),
                        index_of(n)});
    auto on = append(func, block, lanes_t, Instruction::Op::Select,
                     {m, ramp(block), none});
    auto reduced = std::make_shared<Instruction>(
        func.get_unique_name(), index_t, Instruction::Op::Reduce,
        vector<shared_ptr<Value>>{on}, block);
    reduced->reduce = which;
    block->instrs.push_back(reduced);
    return std::make_shared<Value>(reduced);
}

shared_ptr<Value> MaskLanes::lowest(const shared_ptr<Block> &block,
                                    const shared_ptr<Value> &m) {
    return as_bits_ ? count_zeros(block, m, ir::Intrinsic::ctz)
                    : extreme(block, m, ir::VectorReduce::Min);
}

shared_ptr<Value> MaskLanes::highest(const shared_ptr<Block> &block,
                                     const shared_ptr<Value> &m) {
    if (!as_bits_) {
        return extreme(block, m, ir::VectorReduce::Max);
    }
    return append(func, block, index_t, Instruction::Op::Sub,
                  {index_of(wide_t.bits() - 1),
                   count_zeros(block, m, ir::Intrinsic::clz)});
}

shared_ptr<Value> MaskLanes::without_lowest(const shared_ptr<Block> &block,
                                            const shared_ptr<Value> &m,
                                            const shared_ptr<Value> &lane) {
    if (as_bits_) {
        auto less = append(func, block, wide_t, Instruction::Op::Sub,
                           {m, std::make_shared<Value>(
                                   Constant{wide_t, uint64_t(1)})});
        return append(func, block, wide_t, Instruction::Op::BwAnd, {m, less});
    }
    auto others = append(func, block, mask_t, Instruction::Op::Ne,
                         {ramp(block), append(func, block, lanes_t,
                                              Instruction::Op::Bc,
                                              {lane, index_of(n)})});
    return append(func, block, mask_t, Instruction::Op::LAnd, {m, others});
}

shared_ptr<Value> MaskLanes::any(const shared_ptr<Block> &block,
                                 const shared_ptr<Value> &m) {
    if (as_bits_) {
        return append(func, block, Bool_t::make(), Instruction::Op::Ne,
                      {m, std::make_shared<Value>(
                              Constant{wide_t, uint64_t(0)})});
    }
    return append(func, block, Bool_t::make(), Instruction::Op::Any, {m});
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
