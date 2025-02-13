#include "Bonsai.h"

#include <fstream>
#include <iostream>

#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/FileSystem.h>

namespace {

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

bonsai::CompilerOptions parse_cli(int argc, char *argv[]) {
    bonsai::CompilerOptions options;
    std::optional<bonsai::BackendTarget> target;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-e" || arg == "--execute") {
            options.is_execute = true;
            continue;
        }
        if (arg == "-b" || arg == "--backend") {
            bonsai::internal_assert(i + 1 < argc);
            bonsai::internal_assert(!target.has_value());
            target = bonsai::string_to_backend(argv[i + 1]);
            ++i;
            continue;
        }
        if (arg == "-p" || arg == "--pass") {
            bonsai::internal_assert(i + 1 < argc);
            options.passes.push_back(argv[i + 1]);
            ++i;
            continue;
        }
        if (arg == "-o" || arg == "--output") {
            bonsai::internal_assert(options.input_file.empty())
                << "already received output file: " << options.output_file;
            bonsai::internal_assert(i + 1 < argc);
            options.output_file = argv[i + 1];
            ++i;
            continue;
        }
        if (arg == "-i" || arg == "--input") {
            bonsai::internal_assert(options.input_file.empty())
                << "already received input file: " << options.input_file;
            bonsai::internal_assert(i + 1 < argc);
            options.input_file = argv[i + 1];
            ++i;
            continue;
        }

        bonsai::internal_error << "unexpected argument: " << arg;
    }

    options.target = target.has_value() ? *target : bonsai::BackendTarget::NONE;
    if (options.passes.empty()) {
        options.passes = {"default"};
    }
    return options;
}

// Executes the Bonsai `program` with the provide compiler `options`. Upon
// success, returns zero.
int execute(const bonsai::ir::Program &program,
            const bonsai::CompilerOptions &options) {
    switch (options.target) {
    case bonsai::BackendTarget::NONE: {
        bonsai::internal_assert(!options.is_execute);
        if (options.output_file.empty()) {
            program.dump(std::cout);
            return 0;
        }
        std::ofstream os(options.output_file);
        bonsai::internal_assert(os.is_open())
            << "failed to open: " << options.output_file;
        program.dump(os);
        return 0;
    }
    case bonsai::BackendTarget::ASM: {
        bonsai::internal_assert(!options.is_execute);
        bonsai::CodeGen_LLVM codegen;
        bonsai::codegen::to_asm(options.output_file, program, &codegen);
        return 0;
    }
    case bonsai::BackendTarget::LLVM: {
        bonsai::CodeGen_LLVM codegen;
        if (options.is_execute) {
            bonsai::codegen::jit(program, &codegen);
            return 0;
        }
        std::unique_ptr<llvm::Module> module = codegen.compile_program(program);
        if (options.output_file.empty()) {
            module->print(llvm::outs(), /*AAW=*/nullptr);
            return 0;
        }
        auto os = bonsai::make_raw_fd_ostream(options.output_file);
        module->print(*os, /*AAW=*/nullptr);
        return 0;
    }
    }
}

} // namespace

int main(int argc, char *argv[]) {
    // If any argument is --help, then return early.
    auto *end = argv + argc;
    if (auto it = std::find_if(argv, end,
                               [](const char *argument) {
                                   std::string a(argument);
                                   return a == "-h" || a == "--help";
                               });
        it != end) {
        std::cout << command_help();
        return 0;
    }
    const bonsai::CompilerOptions options = parse_cli(argc, argv);
    bonsai::verify_options(options);

    // Parse the input file.
    bonsai::ir::Program program = bonsai::parser::parse(options.input_file);

    // Perform type inference.
    program = bonsai::lower::infer_types(program);

    // Lower the program.
    bonsai::lower::lower(program, options);

    // Execute the steps specified by the compiler options.
    return execute(program, options);
}
