#include "CodeGen/ExpandVectorMulHigh.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/Transforms/Utils/Local.h>

#include <vector>

namespace bonsai {

namespace {

using namespace llvm::PatternMatch;

// The 64-bit-lane high multiply `t` stands for, if it is one: the two narrow
// operands and whether the product was signed.
struct MulHigh {
    llvm::Value *a = nullptr;
    llvm::Value *b = nullptr;
    bool is_signed = false;
};

std::optional<MulHigh> match_mul_high(llvm::TruncInst &t) {
    auto *narrow = llvm::dyn_cast<llvm::FixedVectorType>(t.getType());
    auto *wide = llvm::dyn_cast<llvm::FixedVectorType>(
        t.getOperand(0)->getType());
    if (narrow == nullptr || wide == nullptr) {
        return std::nullopt;
    }
    const unsigned bits = narrow->getScalarSizeInBits();
    if (bits < 64 || wide->getScalarSizeInBits() != 2 * bits) {
        return std::nullopt;
    }
    MulHigh m;
    const llvm::APInt shift(2 * bits, bits);
    if (match(t.getOperand(0),
              m_LShr(m_Mul(m_ZExt(m_Value(m.a)), m_ZExt(m_Value(m.b))),
                     m_SpecificInt(shift)))) {
        m.is_signed = false;
    } else if (match(t.getOperand(0),
                     m_LShr(m_Mul(m_SExt(m_Value(m.a)), m_SExt(m_Value(m.b))),
                            m_SpecificInt(shift)))) {
        m.is_signed = true;
    } else {
        return std::nullopt;
    }
    if (m.a->getType() != narrow || m.b->getType() != narrow) {
        return std::nullopt;
    }
    return m;
}

// The ladder, built before `at`: see the header.
llvm::Value *expand(llvm::TruncInst &at, const MulHigh &m) {
    llvm::IRBuilder<> builder(&at);
    llvm::Type *t = at.getType();
    const std::string name = at.getName().str();
    llvm::Constant *mask32 = llvm::ConstantInt::get(t, 0xffffffffULL);
    llvm::Constant *c32 = llvm::ConstantInt::get(t, 32);
    llvm::Value *al = builder.CreateAnd(m.a, mask32, name + "_al");
    llvm::Value *ah = builder.CreateLShr(m.a, c32, name + "_ah");
    llvm::Value *bl = builder.CreateAnd(m.b, mask32, name + "_bl");
    llvm::Value *bh = builder.CreateLShr(m.b, c32, name + "_bh");
    llvm::Value *ll = builder.CreateMul(al, bl, name + "_ll");
    llvm::Value *hl = builder.CreateMul(ah, bl, name + "_hl");
    llvm::Value *lh = builder.CreateMul(al, bh, name + "_lh");
    llvm::Value *hh = builder.CreateMul(ah, bh, name + "_hh");
    // Each partial sum fits in 64 bits: the three terms of `mid` are below
    // 2^32 each.
    llvm::Value *mid = builder.CreateAdd(
        builder.CreateAdd(builder.CreateLShr(ll, c32),
                          builder.CreateAnd(hl, mask32)),
        builder.CreateAnd(lh, mask32), name + "_mid");
    llvm::Value *high = builder.CreateAdd(
        builder.CreateAdd(hh, builder.CreateLShr(hl, c32)),
        builder.CreateAdd(builder.CreateLShr(lh, c32),
                          builder.CreateLShr(mid, c32)),
        m.is_signed ? name + "_unsigned" : name);
    if (!m.is_signed) {
        return high;
    }
    // The signed high half from the unsigned one: a negative a stands for
    // a - 2^64, whose product with b is a*b - b*2^64, so the high word loses
    // b; likewise a for a negative b.
    llvm::Constant *c63 = llvm::ConstantInt::get(t, 63);
    llvm::Value *sa = builder.CreateAShr(m.a, c63, name + "_sa");
    llvm::Value *sb = builder.CreateAShr(m.b, c63, name + "_sb");
    high = builder.CreateSub(high, builder.CreateAnd(sa, m.b));
    return builder.CreateSub(high, builder.CreateAnd(sb, m.a), name);
}

} // namespace

llvm::PreservedAnalyses
ExpandVectorMulHigh::run(llvm::Function &function,
                         llvm::FunctionAnalysisManager &) {
    std::vector<llvm::TruncInst *> found;
    for (llvm::BasicBlock &block : function) {
        for (llvm::Instruction &instr : block) {
            if (auto *t = llvm::dyn_cast<llvm::TruncInst>(&instr);
                t != nullptr && match_mul_high(*t).has_value()) {
                found.push_back(t);
            }
        }
    }
    if (found.empty()) {
        return llvm::PreservedAnalyses::all();
    }
    for (llvm::TruncInst *t : found) {
        const MulHigh m = *match_mul_high(*t);
        llvm::Value *high = expand(*t, m);
        t->replaceAllUsesWith(high);
        // The trunc and, when nothing else reads them, the shift, the wide
        // product and the extensions behind it.
        llvm::RecursivelyDeleteTriviallyDeadInstructions(t);
    }
    return llvm::PreservedAnalyses::none();
}

} // namespace bonsai
