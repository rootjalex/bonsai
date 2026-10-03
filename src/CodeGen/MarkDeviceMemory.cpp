#include "CodeGen/MarkDeviceMemory.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Metadata.h>

#include <vector>

namespace bonsai {

namespace {

constexpr unsigned kGlobalAddressSpace = 1;
// The address space OptiX's launch parameters live in (`.const params`,
// CodeGen_OptiX.cpp): a raygen program reads its captures out of it.
constexpr unsigned kConstantAddressSpace = 4;

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

// Whether every object an address may be derived from -- through GEPs,
// casts, phis and selects -- is in `roots`.
bool rooted(const llvm::Value *ptr,
            const llvm::SmallPtrSetImpl<const llvm::Value *> &roots) {
    llvm::SmallVector<const llvm::Value *, 4> objects;
    llvm::getUnderlyingObjects(ptr, objects);
    for (const llvm::Value *object : objects) {
        if (!roots.count(object)) {
            return false;
        }
    }
    return !objects.empty();
}

// Whether an address bottoms out at the launch parameters: the global
// named `params` in the constant address space (CodeGen_OptiX.cpp), whose
// pointer fields the launch filled with device addresses. That global
// alone: a module's other constant-space globals -- a `print`'s format
// strings -- hold no device pointers, and a pointer into one of them cast
// to the global space would read the wrong memory.
bool in_launch_parameters(const llvm::Value *ptr) {
    llvm::SmallVector<const llvm::Value *, 4> objects;
    llvm::getUnderlyingObjects(ptr, objects);
    for (const llvm::Value *object : objects) {
        const auto *global = llvm::dyn_cast<llvm::GlobalVariable>(object);
        if (global == nullptr || global->getAddressSpace() != kConstantAddressSpace ||
            global->getName() != "params") {
            return false;
        }
    }
    return !objects.empty();
}

} // namespace

llvm::PreservedAnalyses
MarkDeviceMemory::run(llvm::Function &function,
                      llvm::FunctionAnalysisManager &) {
    if (function.getCallingConv() != llvm::CallingConv::PTX_Kernel) {
        return llvm::PreservedAnalyses::all();
    }
    llvm::LLVMContext &context = function.getContext();

    // The roots: what is known to be device memory. A kernel's pointer
    // parameters and the pointers extracted from its by-value struct
    // parameters, since the launch put them there; a pointer read out of
    // the launch parameters, for a raygen program, which has no parameters
    // of its own and takes its captures from `params`; and a pointer read
    // out of device memory -- a queue's header holds its arrays' handles,
    // a scene's layout struct the tables' -- since what device memory holds
    // is device pointers: nothing of a kernel's own stack is ever stored
    // there (a hit program's context travels in payload registers, not in
    // memory). The last kind depends on the roots before it, so to a fixed
    // point.
    std::vector<llvm::Value *> roots;
    llvm::SmallPtrSet<const llvm::Value *, 32> root_set;
    const auto add_root = [&](llvm::Value *v) {
        if (root_set.insert(v).second) {
            roots.push_back(v);
        }
    };
    // The aggregate an extractvalue chain takes apart; anything else is
    // its own base.
    const auto aggregate_base = [](const llvm::Value *v) {
        while (const auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(v)) {
            v = extract->getAggregateOperand();
        }
        return v;
    };
    // An address in device memory: derived from roots alone, or from the
    // launch parameters.
    const auto device_address = [&](const llvm::Value *ptr) {
        return rooted(ptr, root_set) || in_launch_parameters(ptr);
    };
    for (llvm::Argument &arg : function.args()) {
        if (arg.getType()->isPointerTy()) {
            add_root(&arg);
        }
    }
    for (llvm::Instruction &instr : llvm::instructions(function)) {
        if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instr)) {
            if (extract->getType()->isPointerTy() &&
                parameter_root(extract, function) != nullptr) {
                add_root(extract);
            }
        }
    }
    // Pointers read out of device memory, as far as they reach: a load of
    // a pointer from a device address, and a pointer extracted from a
    // struct loaded whole from one -- a queue's header read as one value,
    // its array handles extractvalues of that load. The backend's SROA
    // splits such a load into one pointer load per field later, which is
    // too late for this pass to see; without the extractvalues as roots,
    // every drain read its entries through generic loads while the
    // handles themselves came in as `ld.global.nc`.
    for (bool grew = true; grew;) {
        grew = false;
        for (llvm::Instruction &instr : llvm::instructions(function)) {
            if (!instr.getType()->isPointerTy() || root_set.count(&instr)) {
                continue;
            }
            const llvm::LoadInst *load = nullptr;
            if (auto *direct = llvm::dyn_cast<llvm::LoadInst>(&instr)) {
                load = direct;
            } else if (auto *extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instr)) {
                load = llvm::dyn_cast<llvm::LoadInst>(aggregate_base(extract));
            }
            if (load == nullptr || load->isVolatile() || load->isAtomic()) {
                continue;
            }
            if (device_address(load->getPointerOperand())) {
                add_root(&instr);
                grew = true;
            }
        }
    }
    if (roots.empty()) {
        return llvm::PreservedAnalyses::all();
    }

    // Which roots the kernel writes through -- a store, an atomic, a memory
    // intrinsic's destination, or a pointer handed to a call the pass cannot
    // see into (libdevice's functions take floats; a call taking a derived
    // pointer is treated as a write, since nothing is known of it). A load
    // through a root not in this set is a load of memory that does not
    // change for the length of the launch. Two roots may be one buffer when
    // both were read out of the same memory -- the two slots of a
    // double-buffered queue's header, say, one read and one pushed to -- so
    // a write through a pointer read out of some object taints every
    // pointer read out of that object; a kernel's parameters are distinct
    // buffers by construction and taint only themselves.
    llvm::SmallPtrSet<const llvm::Value *, 16> written;
    llvm::SmallPtrSet<const llvm::Value *, 16> written_sources;
    const auto source_of = [&](const llvm::Value *root) -> const llvm::Value * {
        // A pointer read out of memory, directly or as a field of a struct
        // read whole: the memory it came from. A parameter, or a pointer
        // extracted from one, came from no memory the kernel can see.
        const auto *load = llvm::dyn_cast<llvm::LoadInst>(aggregate_base(root));
        if (load == nullptr) {
            return nullptr;
        }
        return llvm::getUnderlyingObject(load->getPointerOperand());
    };
    const auto note_written = [&](const llvm::Value *ptr) {
        llvm::SmallVector<const llvm::Value *, 4> objects;
        llvm::getUnderlyingObjects(ptr, objects);
        for (const llvm::Value *object : objects) {
            if (root_set.count(object)) {
                written.insert(object);
                if (const llvm::Value *source = source_of(object)) {
                    written_sources.insert(source);
                }
            }
        }
    };
    for (llvm::Instruction &instr : llvm::instructions(function)) {
        if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&instr)) {
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
    const auto is_written = [&](const llvm::Value *root) {
        if (written.count(root)) {
            return true;
        }
        const llvm::Value *source = source_of(root);
        return source != nullptr && written_sources.count(source);
    };

    // The loads first, while a load's address still bottoms out at the root
    // itself: after the casts below it bottoms out at the same root through
    // two addrspacecasts, which getUnderlyingObjects sees through as well,
    // but there is no reason to make it.
    // A load whose value is a texture unit's handle -- the operand a
    // `llvm.nvvm.tex.*` intrinsic samples through -- stays a plain load.
    // NVPTX's ReplaceImageHandles pass looks at what defines a handle and
    // accepts a load, a parameter, a global or a copy of one; the
    // non-coherent load the tag below would make of it is another
    // instruction, and the pass asserts on any it does not know
    // (NVPTXReplaceImageHandles.cpp, replaceImageHandle). The handle is
    // one word read once per lookup, so nothing is lost. The same limit
    // means a handle chosen by a select or a phi would assert too; the
    // program reads its handles from a table, one load each.
    const auto is_texture_handle = [](const llvm::LoadInst &load) {
        for (const llvm::User *user : load.users()) {
            const auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            if (call == nullptr || call->getCalledFunction() == nullptr) {
                continue;
            }
            if (call->getCalledFunction()->getName().starts_with(
                    "llvm.nvvm.tex") &&
                call->arg_size() > 0 && call->getArgOperand(0) == &load) {
                return true;
            }
        }
        return false;
    };
    llvm::MDNode *invariant = llvm::MDNode::get(context, {});
    for (llvm::Instruction &instr : llvm::instructions(function)) {
        auto *load = llvm::dyn_cast<llvm::LoadInst>(&instr);
        if (load == nullptr || load->isVolatile() || load->isAtomic()) {
            continue;
        }
        if (is_texture_handle(*load)) {
            // Including the tag the code generator put on it for reading an
            // extern's storage (CodeGen_LLVM::mark_invariant), which knows
            // the storage but not what the loaded word is for.
            load->setMetadata(llvm::LLVMContext::MD_invariant_load, nullptr);
            continue;
        }
        llvm::SmallVector<const llvm::Value *, 4> objects;
        llvm::getUnderlyingObjects(load->getPointerOperand(), objects);
        bool read_only = !objects.empty();
        for (const llvm::Value *object : objects) {
            read_only = read_only && root_set.count(object) && !is_written(object);
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
