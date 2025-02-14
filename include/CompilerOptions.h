#pragma once

#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {

// This instructs the compiler which backend to target.
enum class BackendTarget {
    NONE = 0, // No backend; this will just produce Bonsai IR.
    ASM = 1,  // Generate assembly code for the host machine.
    LLVM = 2, // Generate LLVM IR.
};

// Contains information about how the compiler should be executed.
struct CompilerOptions {
    // The targeted backend for the compiler.
    BackendTarget target;

    // Whether this code should be executed after lowering. This will return a
    // failure if the chosen backend does not support execution.
    bool is_execute = false;

    // The input filename. This cannot be empty.
    std::string input_file;

    // The output file name; if this is empty, then defaults to standard I/O.
    std::string output_file;

    // The Bonsai passes to run during lowering. This may also include pass
    // aliases, which refer to a set of passes, e.g., `core`. These are run in
    // the order they are passed on the command line.
    std::vector<std::string> passes;

    friend std::ostream &operator<<(std::ostream &, const CompilerOptions &);
};

void verify_options(const CompilerOptions &);

std::string backend_to_string(BackendTarget);

BackendTarget string_to_backend(std::string_view);

} // namespace bonsai
