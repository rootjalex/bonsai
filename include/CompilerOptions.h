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
    // The device side of a program whose loops a schedule bound to the GPU:
    // the LLVM IR of the kernels and the functions they reach, then the PTX
    // LLVM makes of it. The host side of the same program is what `-b llvm`
    // prints, with the PTX embedded and a launch where each bound loop was.
    PTX = 6,
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

    // The input files, in the order they were given; there has to be one.
    // They are parsed in that order into one program, as if each after the
    // first were imported at the end of the one before, so a schedule can be
    // kept in a file of its own and compiled alongside the program it
    // schedules: `-i render.bonsai -i schedules/packet.bonsai`.
    std::vector<std::string> input_files;

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

    // The GPU the device code is generated for, as NVPTX names it: `sm_120`
    // for Blackwell, `sm_90` for Hopper. Empty means the GPU in this
    // machine, asked of the driver (see runtime/bonsai_cuda.h), which is
    // what a normal compile wants; a test that diffs generated PTX names
    // one so that its output does not depend on the machine it runs on.
    // Only read when some loop is bound to the GPU.
    std::string gpu_arch;

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

    // Fuse a float multiply into the add that consumes it: `a * b + c` rounds
    // once rather than twice. gcc's `-ffp-contract=fast`, which is gcc's
    // default and so what pbrt is built with.
    //
    // Off by default because it changes the arithmetic of every program that
    // asks for it, and a golden that was blessed under two roundings will not
    // match under one. It is not a fast-maths switch: nothing is reassociated,
    // no sign is dropped, and NaN behaviour is untouched. Removing a rounding
    // makes the answer *more* accurate, which is why pbrt's answers are the
    // fused ones and why matching them means doing the same.
    //
    // Applied as an SSA rewrite; see include/SSA/Contract.h for why there.
    bool ffp_contract = false;

    // Print the SSA form, which is otherwise not observable.
    //
    // Every scheduling transform is a rewrite of the block graph, and what
    // `-p ssa` prints is the *relooper's* reading of the graph the rewrite
    // left behind. Those are not the same artefact: the relooper reconstructs
    // structured control flow, so a golden of its output pins what a schedule
    // did only as far as the reconstruction happens to preserve it, and two
    // different graphs can reloop to the same statements. A test of a rewrite
    // wants the graph.
    //
    // `preschedule` is the form as built, ahead of every rewrite;
    // `postschedule` is what the schedule left, immediately before the
    // relooper runs. Asking for both is what makes a transform's golden say
    // what it changed rather than only what it ended at. Both go to stdout,
    // ahead of whatever else is printed.
    bool dump_ssa_preschedule = false;
    bool dump_ssa_postschedule = false;

    friend std::ostream &operator<<(std::ostream &, const CompilerOptions &);
};

void verify_options(const CompilerOptions &);

std::string backend_to_string(BackendTarget);

BackendTarget string_to_backend(std::string_view);

} // namespace bonsai
