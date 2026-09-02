#pragma once

#include "IR/Program.h"

namespace bonsai {

// This instructs the compiler which backend to target.
enum class BackendTarget {
    NONE = 0, // No backend; this will just produce Bonsai IR.
    ASM = 1,  // Generate assembly code for the host machine.
    LLVM = 2, // Generate LLVM IR.
    CPP = 3,  // Generate C++ header with respective .o file.
    CPPX = 4, // Generate C++ header and source file.
    CUDA = 5, // Generate CUDA code.
};

enum class BackendOptimizationLevel {
    O0 = 0,
    O3 = 1,
};

// Contains information about how the compiler should be executed.
struct CompilerOptions {
    // The targeted backend for the compiler.
    BackendTarget target;

    // Whether this code should be executed after lowering. This will return a
    // failure if the chosen backend does not support execution.
    bool is_execute = false;

    // Whether this should verbosely print Bonsai IR.
    bool is_verbose = false;

    // The optimization level for the backend target.
    BackendOptimizationLevel level = BackendOptimizationLevel::O3;

    // The input filename. This cannot be empty.
    std::string input_file;

    // The output file name; if this is empty, then defaults to standard I/O.
    std::string output_file;

    // The Bonsai passes to run during lowering. This may also include pass
    // aliases, which refer to a set of passes, e.g., `core`. These are run in
    // the order they are passed on the command line.
    std::vector<std::string> passes;

    // The LLVM target to generate code for. Empty means the host, which is
    // what a normal compile wants; a test that diffs generated code sets both
    // so that its output does not depend on the machine it runs on. The CPU
    // matters as much as the triple: with an empty CPU string LLVM tunes for
    // a generic processor, but with "native" it makes different vectorization
    // choices per host.
    std::string target_triple;
    std::string target_cpu;
    // Reject a program that would allocate on the heap.
    //
    // Nothing frees a heap allocation -- see the "support deallocation" TODO
    // in CodeGen_LLVM.h -- so one made per iteration of a loop is an unbounded
    // leak, and one made per camera ray is four gigabytes. A renderer's inner
    // loop should not be allocating at all, and this is how that stops being
    // something to hope for: the property is checked, and a program that
    // breaks it fails to compile rather than running slowly and growing.
    //
    // What it costs is the things that genuinely need a heap: a `dyn_array`,
    // and a `map` or a reduction whose result is returned by value rather than
    // written into storage the caller owns.
    bool no_heap = false;

    friend std::ostream &operator<<(std::ostream &, const CompilerOptions &);
};

void verify_options(const CompilerOptions &);

std::string backend_to_string(BackendTarget);

BackendTarget string_to_backend(std::string_view);

} // namespace bonsai
