#include "CompilerOptions.h"

namespace bonsai {

std::ostream &operator<<(std::ostream &os, const CompilerOptions &opt) {
    os << "-b " << backend_to_string(opt.target) << "\n";
    if (opt.is_execute) {
        os << "-e " << backend_to_string(opt.target) << "\n";
    }
    for (const std::string &input : opt.input_files) {
        os << "-i " << input << "\n";
    }
    if (!opt.output_file.empty()) {
        os << "-o " << opt.output_file << "\n";
    }
    switch (opt.level) {
    case BackendOptimizationLevel::O0:
        os << "-O0" << "\n";
        break;
    case BackendOptimizationLevel::O3:
        os << "-O3" << "\n";
        break;
    }
    return os;
}

void verify_options(const CompilerOptions &options) {
    internal_assert(!options.input_files.empty())
        << "no input file: " << options;
    switch (BackendTarget backend = options.target; backend) {
    case BackendTarget::NONE:
    case BackendTarget::ASM:
    case BackendTarget::CPP:
    case BackendTarget::CPPX:
    case BackendTarget::CUDA:
    case BackendTarget::PTX:
        internal_assert(!options.is_execute)
            << "backend: " << backend_to_string(backend)
            << " does not support execution";
    case BackendTarget::LLVM:
        break;
    }
}

std::string backend_to_string(BackendTarget target) {
    switch (target) {
    case BackendTarget::ASM:
        return "asm";
    case BackendTarget::CPP:
        return "cpp";
    case BackendTarget::CPPX:
        return "cppx";
    case BackendTarget::CUDA:
        return "cuda";
    case BackendTarget::LLVM:
        return "llvm";
    case BackendTarget::PTX:
        return "ptx";
    case BackendTarget::NONE:
        return "none";
    }
}

BackendTarget string_to_backend(std::string_view in) {
    if (in == "asm")
        return BackendTarget::ASM;
    if (in == "cpp")
        return BackendTarget::CPP;
    if (in == "cuda")
        return BackendTarget::CUDA;
    if (in == "llvm")
        return BackendTarget::LLVM;
    if (in == "ptx")
        return BackendTarget::PTX;
    if (in == "cppx")
        return BackendTarget::CPPX;
    if (in == "none")
        return BackendTarget::NONE;

    internal_error << "unexpected backend target: " << in;
}

} // namespace bonsai
