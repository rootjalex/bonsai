#include "CodeGen/MarkDeviceMemory.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Metadata.h>

#include <vector>

namespace bonsai {

namespace {

constexpr unsigned kGlobalAddressSpace = 1;

// The kernel parameter a value is, or the parameter a pointer inside a
// by-value struct parameter was extracted from: `extractvalue %param, k`,
// or an extractvalue of an extractvalue of one, for a struct in a struct.
// Null for anything else.
const llvm::Argument *parameter_root(const llvm::Value *v,
                                     const llvm::Function &kernel) {
    while (const auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(v)) {
        v = extract->getAggregateOperand();
    }
    const auto *arg = llvm::dyn_cast<llvm::Argument>(v);
    return arg != nullptr && arg->getParent() == &kernel ? arg : nullptr;
}

// Whether `v` is a root: a pointer that is a kernel parameter or comes out
// of one by extractvalue alone.
bool is_root(const llvm::Value *v, const llvm::Function &kernel) {
    return v->getType()->isPointerTy() && parameter_root(v, kernel) != nullptr;
}

// Every root the address `ptr` may be derived from, through GEPs, casts,
// phis and selects; empty if any of what it may come from is not a root.
llvm::SmallVector<const llvm::Value *, 4>
roots_of(const llvm::Value *ptr, const llvm::Function &kernel) {
    llvm::SmallVector<const llvm::Value *, 4> objects;
    llvm::getUnderlyingObjects(ptr, objects);
    for (const llvm::Value *object : objects) {
        if (!is_root(object, kernel)) {
            return {};
        }
    }
    return objects;
}

} // namespace

llvm::PreservedAnalyses
MarkDeviceMemory::run(llvm::Function &function,
                      llvm::FunctionAnalysisManager &) {
    if (function.getCallingConv() != llvm::CallingConv::PTX_Kernel) {
        return llvm::PreservedAnalyses::all();
    }
    llvm::LLVMContext &context = function.getContext();

    // The roots, and which of them the kernel writes through -- a store, an
    // atomic, a memory intrinsic's destination, or a pointer handed to a call
    // the pass cannot see into (libdevice's functions take floats; a call
    // taking a derived pointer is treated as a write, since nothing is known
    // of it). A load through a root not in this set is a load of memory that
    // does not change for the length of the launch.
    std::vector<llvm::Value *> roots;
    for (llvm::Argument &arg : function.args()) {
        if (arg.getType()->isPointerTy()) {
            roots.push_back(&arg);
        }
    }
    llvm::SmallPtrSet<const llvm::Value *, 16> written;
    const auto note_written = [&](const llvm::Value *ptr) {
        llvm::SmallVector<const llvm::Value *, 4> objects;
        llvm::getUnderlyingObjects(ptr, objects);
        for (const llvm::Value *object : objects) {
            if (is_root(object, function)) {
                written.insert(object);
            }
        }
    };
    for (llvm::Instruction &instr : llvm::instructions(function)) {
        if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instr)) {
            if (is_root(extract, function)) {
                roots.push_back(extract);
            }
        } else if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&instr)) {
            note_written(store->getPointerOperand());
        } else if (auto *rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&instr)) {
            note_written(rmw->getPointerOperand());
        } else if (auto *cas = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instr)) {
            note_written(cas->getPointerOperand());
        } else if (auto *mem = llvm::dyn_cast<llvm::MemIntrinsic>(&instr)) {
            note_written(mem->getRawDest());
        } else if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instr)) {
            if (llvm::isa<llvm::IntrinsicInst>(call)) {
                continue;
            }
            for (llvm::Value *operand : call->args()) {
                if (operand->getType()->isPointerTy()) {
                    note_written(operand);
                }
            }
        }
    }
    if (roots.empty()) {
        return llvm::PreservedAnalyses::all();
    }

    // The loads first, while a load's address still bottoms out at the root
    // itself: after the casts below it bottoms out at the same root through
    // two addrspacecasts, which getUnderlyingObjects sees through as well,
    // but there is no reason to make it.
    llvm::MDNode *invariant = llvm::MDNode::get(context, {});
    for (llvm::Instruction &instr : llvm::instructions(function)) {
        auto *load = llvm::dyn_cast<llvm::LoadInst>(&instr);
        if (load == nullptr || load->isVolatile() || load->isAtomic()) {
            continue;
        }
        const auto objects = roots_of(load->getPointerOperand(), function);
        if (objects.empty()) {
            continue;
        }
        bool read_only = true;
        for (const llvm::Value *object : objects) {
            read_only = read_only && !written.count(object);
        }
        if (read_only) {
            load->setMetadata(llvm::LLVMContext::MD_invariant_load, invariant);
        }
    }

    // The address space: each root cast to global and back to generic, its
    // uses moved onto the generic result. What NVPTXLowerArgs does for a
    // kernel's `byval` parameters, done for every root here; the backend's
    // InferAddressSpaces then rewrites the loads, stores and atomics that
    // derive from it to the global space.
    llvm::Type *generic_t = llvm::PointerType::getUnqual(context);
    llvm::Type *global_t = llvm::PointerType::get(context, kGlobalAddressSpace);
    for (llvm::Value *root : roots) {
        llvm::IRBuilder<> builder(context);
        if (auto *instr = llvm::dyn_cast<llvm::Instruction>(root)) {
            builder.SetInsertPoint(instr->getNextNode());
        } else {
            builder.SetInsertPoint(&function.getEntryBlock(),
                                   function.getEntryBlock().getFirstInsertionPt());
        }
        llvm::Value *global =
            builder.CreateAddrSpaceCast(root, global_t, root->getName() + ".global");
        llvm::Value *generic =
            builder.CreateAddrSpaceCast(global, generic_t, root->getName() + ".generic");
        root->replaceUsesWithIf(generic, [&](llvm::Use &use) {
            return use.getUser() != global;
        });
    }
    return llvm::PreservedAnalyses::none();
}

} // namespace bonsai
