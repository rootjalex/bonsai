#include "CLI/CLI.h"

#include "Bonsai.h"

#include <fstream>
#include <iostream>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

namespace {

using namespace bonsai;

// Returns a helpful message to outline the command line arguments for the
// Bonsai compiler.
std::string command_help() {
    std::stringstream s;
    s << "Bonsai Command Line:\n"
      << "-b|--backend <backend>         | e.g., `-b llvm`\n"
      << "-p|--pass <pass>               | e.g., `-p dce`\n"
      << "-e|--execute,                  | e.g., `-e`\n"
      << "-i|--input <input file name>   | e.g., `-i in.bonsai`\n"
      << "-o|--output <output file name> | e.g., `-o out.bonsai`\n"
      << "-h|--help";
    return s.str();
}

// Executes the Bonsai `program` with the provide compiler `options`. Upon
// success, returns zero.
int execute(const ir::Program &program, const CompilerOptions &options) {
    switch (options.target) {
    case BackendTarget::NONE: {
        internal_assert(!options.is_execute);
        if (options.output_file.empty()) {
            program.dump(std::cout);
            return EXIT_SUCCESS;
        }
        std::ofstream os(options.output_file);
        internal_assert(os.is_open())
            << "failed to open: " << options.output_file;
        program.dump(os);
        return EXIT_SUCCESS;
    }
    case BackendTarget::ASM: {
        internal_assert(!options.is_execute);
        CodeGen_LLVM codegen;
        codegen::to_asm(options.output_file, program, &codegen);
        return EXIT_SUCCESS;
    }
    case BackendTarget::LLVM: {
        CodeGen_LLVM codegen;
        if (options.is_execute) {
            codegen::jit(program, &codegen);
            return EXIT_SUCCESS;
        }
        std::unique_ptr<llvm::Module> module = codegen.compile_program(program);
        if (options.output_file.empty()) {
            module->print(llvm::outs(), /*AAW=*/nullptr);
            return EXIT_SUCCESS;
        }
        auto os = make_raw_fd_ostream(options.output_file);
        module->print(*os, /*AAW=*/nullptr);
        return EXIT_SUCCESS;
    }
    }
    internal_error << "Unknown backend";
    return EXIT_FAILURE;
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
            internal_assert(options.input_file.empty())
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
        ir::Program program = parser::parse(options.input_file);

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
