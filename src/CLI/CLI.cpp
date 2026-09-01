#include "CLI/CLI.h"

#include "Bonsai.h"
#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

using namespace bonsai;

int combine_files(std::string a, std::string b, std::string &combined) {
    std::ifstream in_a(a);
    if (!in_a) {
        std::cerr << "Error: Cannot open input file " << a << std::endl;
        return 1;
    }

    std::ifstream in_b(b);
    if (!in_b) {
        std::cerr << "Error: Cannot open input file " << b << std::endl;
        return 1;
    }

    std::filesystem::path path = std::filesystem::temp_directory_path();
    path /= "combined.bonsai";
    std::ofstream out(path.string(), std::ios::trunc);
    if (!out) {
        std::cerr << "Error: Cannot open temporary file " << path.string()
                  << " for writing" << std::endl;
        return 1;
    }

    out << in_a.rdbuf(); // Copy all of file `a` into the temporary file.
    out << in_b.rdbuf(); // Copy all of file `b` into the temporary file.

    combined = path.string();
    return 0;
}

// Returns a helpful message to outline the command line arguments for the
// Bonsai compiler.
std::string command_help() {
    std::stringstream s;
    s << "Bonsai Command Line:\n"
      << "-b   | --backend <backend>         | e.g., `-b llvm`\n"
      << "-p   | --pass <pass>               | e.g., `-p dce`\n"
      << "     | --up-to <pass>              | e.g., `--up-to lower-trees`\n"
      << "     | --target <triple>           | e.g., `--target "
         "x86_64-linux-gnu`"
      << "-e   | --execute,                  | e.g., `-e`\n"
      << "-i   | --input <input file name>   | e.g., `-i in.bonsai`\n"
      << "-l   | --layout <layout file name> | e.g., `-l pbrt.bonsai`\n"
      << "-o   | --output <output file name> | e.g., `-o out.bonsai`\n"
      << "-v   | --verbose                   | e.g., `-v`\n"
      << "-O<n>| n/a                         | e.g., `-O3`\n"
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
    case BackendTarget::CUDA: {
        codegen::to_cuda(program, options);
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
        if (arg == "--up-to") {
            options.up_to = args[i + 1];
            ++i;
            continue;
        }
        if (arg == "--target") {
            internal_assert(i + 1 < args.size());
            options.target_triple = args[i + 1];
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
        if (arg == "-i" || arg == "--input") {
            internal_assert(options.input_file.empty())
                << "already received input file: " << options.input_file;
            internal_assert(i + 1 < args.size());
            options.input_file = args[i + 1];
            ++i;
            continue;
        }
        if (arg == "-l" || arg == "--layout") {
            internal_assert(options.layout_file.empty())
                << "already received layout file: " << options.layout_file;
            internal_assert(i + 1 < args.size());
            options.layout_file = args[i + 1];
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

        // Parse the input file.
        std::string input = options.input_file;
        if (!options.layout_file.empty()) {
            combine_files(options.input_file, options.layout_file, input);
        }
        // A program may leave a type to be inferred -- `x = f(y)` where `f`
        // declares no return type -- so the IR the parser builds is only
        // partially typed. infer_types turns enforcement back on once it has
        // resolved them.
        ir::global_disable_type_enforcement();
        ir::Program program = parser::parse(input);

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
