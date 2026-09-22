#include "CLI/CLI.h"

#include "Bonsai.h"
#include "IR/Printer.h"

#include <fstream>
#include <iostream>

namespace {

using namespace bonsai;

// Returns a helpful message to outline the command line arguments for the
// Bonsai compiler.
std::string command_help() {
    std::stringstream s;
    s << "Bonsai Command Line:\n"
      << "-b   | --backend <backend>         | e.g., `-b llvm`\n"
      << "-p   | --pass <pass>               | e.g., `-p dce`\n"
      << "-e   | --execute,                  | e.g., `-e`\n"
      << "-i   | --input <input file name>   | e.g., `-i in.bonsai`; may be "
         "repeated, the files making one program in order\n"
      << "-o   | --output <output file name> | e.g., `-o out.bonsai`\n"
      << "-v   | --verbose                   | e.g., `-v`\n"
      << "-O<n>| n/a                         | e.g., `-O3`\n"
      << "     | --triple <target triple>    | e.g., "
         "`--triple x86_64-unknown-linux-gnu`\n"
      << "     | --mcpu <target cpu>         | e.g., `--mcpu generic`\n"
      << "     | --gpu-arch <sm>             | the GPU device code is for, "
         "e.g. `--gpu-arch sm_120`; the machine's own if not given\n"
      << "     | --gpu-max-registers <n>     | the most registers a kernel's "
         "thread may use (PTX .maxnreg; pbrt's GPU build uses 128); ptxas's "
         "own choice if not given\n"
      << "     | --fast-math                 | the platform's fast arithmetic: "
         "clang's -ffast-math on the CPU; nvcc's --use_fast_math on the GPU "
         "(flushed denormals, approximate division, square root and "
         "transcendentals), as pbrt --gpu is built. Exact if not given\n"
      << "     | --no-heap                   | reject heap allocation\n"
      << "     | --ffp-contract              | fuse `a * b + c` into one fma\n"
      << "     | --dump-ssa-preschedule      | print the SSA a schedule acts "
         "on\n"
      << "     | --dump-ssa-postschedule     | print the SSA a schedule left\n"
      << "-h   | --help";
    return s.str();
}

// Executes the Bonsai `program` with the provide compiler `options`. Upon
// success, returns zero.
int execute(const ir::Program &program, const CompilerOptions &options) {
    switch (options.target) {
    case BackendTarget::NONE: {
        if (options.output_file.empty()) {

            bonsai::ir::Printer printer(std::cout,
                                        /*verbose=*/options.is_verbose);
            printer.print(program);
            return EXIT_SUCCESS;
        }
        std::ofstream os(options.output_file);
        internal_assert(os.is_open())
            << "failed to open: " << options.output_file;
        bonsai::ir::Printer printer(os,
                                    /*verbose=*/options.is_verbose);
        printer.print(program);
        return EXIT_SUCCESS;
    }
    case BackendTarget::ASM: {
        codegen::to_asm(program, options);
        return EXIT_SUCCESS;
    }
    case BackendTarget::LLVM: {
        if (options.is_execute) {
            codegen::jit(program, options);
            return EXIT_SUCCESS;
        }
        codegen::to_llvm(program, options);
        return EXIT_SUCCESS;
    }
    case BackendTarget::CPP: {
        codegen::to_cpp(program, options);
        return EXIT_SUCCESS;
    }
    case BackendTarget::CPPX: {
        codegen::to_cppx(program, options);
        return EXIT_SUCCESS;
    }
    case BackendTarget::PTX: {
        codegen::to_ptx(program, options);
        return EXIT_SUCCESS;
    }
    }
}

} // namespace

namespace bonsai::cli {

Flags parse(int argc, char *argv[]) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }
    return parse(args);
}

Flags parse(const std::vector<std::string> &args) {
    CompilerOptions options;

    std::optional<BackendTarget> target;
    for (int i = 0; i < args.size(); ++i) {
        const std::string &arg = args[i];
        if (arg == "-h" || arg == "--help") {
            return Flags{{}, true};
        }
        if (arg == "-e" || arg == "--execute") {
            options.is_execute = true;
            continue;
        }
        if (arg == "-v" || arg == "--verbose") {
            options.is_verbose = true;
            continue;
        }
        // No ++i here: these take no value, and skipping one swallowed
        // whatever flag happened to follow.
        if (arg == "-O0") {
            options.level = BackendOptimizationLevel::O0;
            continue;
        }
        if (arg == "-O3") {
            options.level = BackendOptimizationLevel::O3;
            continue;
        }
        if (arg == "-b" || arg == "--backend") {
            internal_assert(i + 1 < args.size());
            internal_assert(!target.has_value());
            target = string_to_backend(args[i + 1]);
            ++i;
            continue;
        }
        if (arg == "-p" || arg == "--pass") {
            internal_assert(i + 1 < args.size());
            options.passes.push_back(args[i + 1]);
            ++i;
            continue;
        }
        if (arg == "-o" || arg == "--output") {
            internal_assert(options.output_file.empty())
                << "already received output file: " << options.output_file;
            internal_assert(i + 1 < args.size());
            options.output_file = args[i + 1];
            ++i;
            continue;
        }
        if (arg == "--triple") {
            internal_assert(i + 1 < args.size());
            options.target_triple = args[i + 1];
            ++i;
            continue;
        }
        if (arg == "--no-heap") {
            options.no_heap = true;
            continue;
        }
        if (arg == "--ffp-contract") {
            options.ffp_contract = true;
            continue;
        }
        if (arg == "--dump-ssa-preschedule") {
            options.dump_ssa_preschedule = true;
            continue;
        }
        if (arg == "--dump-ssa-postschedule") {
            options.dump_ssa_postschedule = true;
            continue;
        }
        if (arg == "--mcpu") {
            internal_assert(i + 1 < args.size());
            options.target_cpu = args[i + 1];
            ++i;
            continue;
        }
        if (arg == "--gpu-arch") {
            internal_assert(i + 1 < args.size());
            options.gpu_arch = args[i + 1];
            ++i;
            continue;
        }
        if (arg == "--fast-math") {
            options.fast_math = true;
            continue;
        }
        if (arg == "--gpu-max-registers") {
            internal_assert(i + 1 < args.size());
            const int n = std::atoi(args[i + 1].c_str());
            internal_assert(n > 0 && n <= 255)
                << "--gpu-max-registers takes a count from 1 to 255, not `"
                << args[i + 1] << "`";
            options.gpu_max_registers = uint32_t(n);
            ++i;
            continue;
        }
        if (arg == "-i" || arg == "--input") {
            // Repeatable: the files make one program, in this order (see
            // CompilerOptions::input_files).
            internal_assert(i + 1 < args.size());
            options.input_files.push_back(args[i + 1]);
            ++i;
            continue;
        }

        internal_error << "unexpected argument: " << arg;
    }

    options.target = target.has_value() ? *target : BackendTarget::NONE;
    if (options.passes.empty()) {
        options.passes = {"default"};
    }
    return {options, false};
}

int run(const Flags &flags) {
    try {
        const auto &[options, display_help] = flags;

        if (display_help) {
            std::cout << command_help();
            return EXIT_SUCCESS;
        }
        verify_options(options);

        // Parse the input files into one program.
        ir::Program program = parser::parse(options.input_files);

        // Perform type inference.
        program = lower::infer_types(program);

        // Lower the program.
        lower::lower(program, options);

        // Execute the steps specified by the compiler options.
        return execute(program, options);
    } catch (const Error &e) {
        std::cerr << e.what();
        return EXIT_FAILURE;
    }
}

} // namespace bonsai::cli
