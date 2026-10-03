#include "CodeGen/ResolveTarget.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "Utils.h"

#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>

namespace bonsai {
namespace codegen {

ir::Target resolve_target(const std::string &triple_arg,
                          const std::string &cpu_arg) {
    ir::Target target;
    // Only when nothing was named: `--triple` or `--mcpu` means the caller
    // wants a particular machine and is probably diffing the output, and a
    // triple that is not this machine's cannot take this machine's features.
    target.follows_host = triple_arg.empty() && cpu_arg.empty();
    target.triple = triple_arg.empty() ? llvm::sys::getDefaultTargetTriple()
                                       : triple_arg;
    target.cpu = cpu_arg;
    if (target.follows_host) {
        target.cpu = llvm::sys::getHostCPUName().str();
        llvm::SubtargetFeatures features;
        for (const auto &feature : llvm::sys::getHostCPUFeatures()) {
            features.AddFeature(feature.first(), feature.second);
        }
        target.llvm_features = features.getString();
    }

    const llvm::Triple triple(target.triple);
    switch (triple.getArch()) {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        target.arch = ir::Target::Arch::X86;
        break;
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::aarch64:
    case llvm::Triple::aarch64_be:
    case llvm::Triple::aarch64_32:
        target.arch = ir::Target::Arch::ARM;
        break;
    case llvm::Triple::nvptx:
    case llvm::Triple::nvptx64:
        target.arch = ir::Target::Arch::NVPTX;
        break;
    default:
        target.arch = ir::Target::Arch::Unknown;
        break;
    }
    target.bits = triple.isArch64Bit() ? 64 : 32;

    // The features by name, the CPU's implied ones included, from the same
    // subtarget description the target machine will be built on.
    CodeGen_LLVM::init_llvm();
    std::string error;
    const llvm::Target *llvm_target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    internal_assert(llvm_target != nullptr)
        << "could not find an LLVM target for " << target.triple << ": "
        << error;
    const std::unique_ptr<llvm::MCSubtargetInfo> subtarget(
        llvm_target->createMCSubtargetInfo(triple, target.cpu,
                                           target.llvm_features));
    internal_assert(subtarget != nullptr)
        << "no subtarget description for " << target.triple << " "
        << target.cpu;
    const llvm::FeatureBitset &bits = subtarget->getFeatureBits();
    for (const llvm::SubtargetFeatureKV &feature :
         subtarget->getAllProcessorFeatures()) {
        if (bits.test(feature.Value)) {
            target.features.insert(feature.key());
        }
    }
    return target;
}

} // namespace codegen
} // namespace bonsai
