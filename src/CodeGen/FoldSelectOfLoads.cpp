#include "CodeGen/FoldSelectOfLoads.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace bonsai {

namespace {

// The base a load reads from and its constant byte offset from it, through
// whatever geps with constant indices lie between.
struct Address {
    llvm::Value *base;
    llvm::APInt offset;
};

Address address_of(llvm::LoadInst *load, const llvm::DataLayout &dl) {
    llvm::APInt offset(dl.getIndexSizeInBits(load->getPointerAddressSpace()),
                       0);
    llvm::Value *base = load->getPointerOperand()->stripAndAccumulateConstantOffsets(
        dl, offset, /*AllowNonInbounds=*/true);
    return Address{base, offset};
}

// Whether anything after `from` and before `to`, in their one block, may
// write what a load from `read` reads. A store into stack memory of the
// function's own cannot, unless that is what is read.
bool may_write_between(llvm::Instruction *from, llvm::Instruction *to,
                       const llvm::Value *read) {
    for (llvm::Instruction *in = from->getNextNode(); in != nullptr && in != to;
         in = in->getNextNode()) {
        if (!in->mayWriteToMemory()) {
            continue;
        }
        if (const auto *store = llvm::dyn_cast<llvm::StoreInst>(in)) {
            const llvm::Value *written =
                llvm::getUnderlyingObject(store->getPointerOperand());
            if (llvm::isa<llvm::AllocaInst>(written) && written != read) {
                continue;
            }
        }
        return true;
    }
    return false;
}

// Where a select of constants under `cond` may go so that every use of
// `cond` could have used it: right after `cond`'s definition, or the entry
// for an argument. Nothing for a constant, whose select would have folded.
std::optional<llvm::BasicBlock::iterator> after_definition(llvm::Value *cond,
                                                           llvm::Function &f) {
    if (auto *in = llvm::dyn_cast<llvm::Instruction>(cond)) {
        return in->getInsertionPointAfterDef();
    }
    if (llvm::isa<llvm::Argument>(cond)) {
        return f.getEntryBlock().getFirstInsertionPt();
    }
    return std::nullopt;
}

// The one bool a condition holds in every lane, or nothing: the bool
// itself, a vector splat of one, or every lane of a shuffle taken from one
// lane of a vector -- what the broadcast of one component of a vector of
// three signs becomes once LLVM has folded the extract and the insert into
// the shuffle. The last makes the lane's extract, after the vector's
// definition, so that what is made from it can go there too.
llvm::Value *uniform_condition(llvm::Value *cond, llvm::Function &f) {
    if (!cond->getType()->isVectorTy()) {
        return cond;
    }
    if (llvm::Value *splat = llvm::getSplatValue(cond)) {
        return splat;
    }
    auto *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(cond);
    if (shuffle == nullptr) {
        return nullptr;
    }
    const llvm::ArrayRef<int> mask = shuffle->getShuffleMask();
    const int lane = mask.empty() ? -1 : mask[0];
    if (lane < 0 || !llvm::all_of(mask, [&](int m) { return m == lane; })) {
        return nullptr;
    }
    llvm::Value *source = shuffle->getOperand(0);
    auto *source_t = llvm::dyn_cast<llvm::FixedVectorType>(source->getType());
    if (source_t == nullptr || unsigned(lane) >= source_t->getNumElements()) {
        return nullptr;
    }
    std::optional<llvm::BasicBlock::iterator> where = after_definition(source, f);
    if (!where.has_value()) {
        return nullptr;
    }
    llvm::IRBuilder<> at_source(where->getNodeParent(), *where);
    return at_source.CreateExtractElement(source, uint64_t(lane), "picked_sign");
}

} // namespace

llvm::PreservedAnalyses
FoldSelectOfLoads::run(llvm::Function &function,
                       llvm::FunctionAnalysisManager &) {
    const llvm::DataLayout &dl = function.getParent()->getDataLayout();
    llvm::LLVMContext &ctx = function.getContext();
    std::vector<llvm::SelectInst *> selects;
    for (llvm::BasicBlock &bb : function) {
        for (llvm::Instruction &in : bb) {
            if (auto *select = llvm::dyn_cast<llvm::SelectInst>(&in)) {
                selects.push_back(select);
            }
        }
    }
    bool changed = false;
    for (llvm::SelectInst *select : selects) {
        auto *lt = llvm::dyn_cast<llvm::LoadInst>(select->getTrueValue());
        auto *lf = llvm::dyn_cast<llvm::LoadInst>(select->getFalseValue());
        if (lt == nullptr || lf == nullptr || lt == lf || !lt->isSimple() ||
            !lf->isSimple() || lt->getType() != lf->getType() ||
            lt->getParent() != select->getParent() ||
            lf->getParent() != select->getParent()) {
            continue;
        }
        // One choice for every lane, from one base.
        llvm::Value *cond = uniform_condition(select->getCondition(), function);
        if (cond == nullptr) {
            continue;
        }
        const Address at = address_of(lt, dl), af = address_of(lf, dl);
        if (at.base != af.base) {
            continue;
        }
        llvm::LoadInst *first = lt->comesBefore(lf) ? lt : lf;
        if (may_write_between(first, select,
                              llvm::getUnderlyingObject(at.base))) {
            continue;
        }
        std::optional<llvm::BasicBlock::iterator> where =
            after_definition(cond, function);
        if (!where.has_value()) {
            continue;
        }

        llvm::IRBuilder<> at_cond(where->getNodeParent(), *where);
        llvm::Type *index_t = llvm::IntegerType::get(ctx, at.offset.getBitWidth());
        llvm::Value *offset = at_cond.CreateSelect(
            cond, llvm::ConstantInt::get(index_t, at.offset),
            llvm::ConstantInt::get(index_t, af.offset), "picked_offset");

        llvm::IRBuilder<> here(select);
        llvm::Value *address = here.CreateGEP(llvm::Type::getInt8Ty(ctx), at.base,
                                              {offset}, "picked_address");
        llvm::LoadInst *picked = here.CreateAlignedLoad(
            lt->getType(), address, std::min(lt->getAlign(), lf->getAlign()),
            "picked");
        select->replaceAllUsesWith(picked);
        select->eraseFromParent();
        for (llvm::LoadInst *load : {lt, lf}) {
            if (load->use_empty()) {
                load->eraseFromParent();
            }
        }
        changed = true;
    }
    return changed ? llvm::PreservedAnalyses::none()
                   : llvm::PreservedAnalyses::all();
}

} // namespace bonsai
