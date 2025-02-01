#include "Bonsai.h"
#include <fstream>
#include <iostream>

#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/FileSystem.h>

struct ParsedOptions {
    bool jit = false;
    bool gen_asm = false;
    bool gen_llvm = false;
    std::string output_filename;
    std::string input_filename;
};

ParsedOptions parse_cli(int argc, char *argv[]) {
    if (argc < 2) {
        // TODO: include --help equivalent.
        bonsai::internal_error << "Usage: " << argv[0] << " <input_file>\n";
    }

    // TODO: probably use a library for this.

    ParsedOptions options;

    int arg = 1;
    while (arg < argc) {
        const std::string _arg = argv[arg];
        if (_arg == "-run") {
            options.jit = true;
            arg++;
        } else if (_arg == "-asm") {
            options.gen_asm = true;
            if (arg + 1 < argc) {
                options.output_filename = argv[arg + 1];
                ++arg;
            }
            ++arg;
        } else if (_arg == "-llvm") {
            options.gen_llvm = true;
            if (arg + 1 < argc) {
                options.output_filename = argv[arg + 1];
                ++arg;
            }
            ++arg;
        } else if (_arg == "-o") {
            bonsai::internal_assert(arg + 1 < argc)
                << "-o flag requires output file to follow.";
            options.output_filename = argv[arg + 1];
            arg += 2;
        } else {
            // TODO: support more options.
            // Must be input file.
            bonsai::internal_assert(options.input_filename.empty())
                << "Multiple input files detected, already have: "
                << options.input_filename << " and received " << _arg;
            options.input_filename = _arg;
            arg++;
        }
    }
    bonsai::internal_assert(!options.input_filename.empty())
        << "Failed to parse input file name.";
    return options;
}

int main(int argc, char *argv[]) {
    ParsedOptions options = parse_cli(argc, argv);

    // Parse the input file
    bonsai::ir::Program program = bonsai::parser::parse(options.input_filename);

    // Perform type inference.
    program = bonsai::lower::infer_types(program);

    // Perform canoncalization.
    program = bonsai::lower::canonicalize(program);

    // TODO:
    // Lower spatial queries
    // Perform first round of scheduling.
    // Lower data structures.
    // Perform second round of scheduling + bit data lowering.
    // Perform final code generation

    if (!(options.jit || options.gen_asm || options.gen_llvm)) {
        // Just dump to output filename.
        // Create an output file stream
        if (options.output_filename.empty()) {
            program.dump(std::cout);
            return 0;
        }
        std::ofstream os(options.output_filename);
        bonsai::internal_assert(os.is_open())
            << "Could not open output file: " << options.output_filename;
        program.dump(os);
        return 0;
    }

    // TODO: in the above case, should we be dumping the LLVM codegen?
    // TODO: compile to object / header files.

    // TODO: for the below cases, remove the duplicate work being done.

    if (options.jit) {
        bonsai::CodeGen_LLVM codegen;
        bonsai::codegen::jit(program, &codegen);
    }

    if (options.gen_asm) {
        bonsai::CodeGen_LLVM codegen;
        bonsai::codegen::to_asm(options.output_filename, program, &codegen);
    }

    if (options.gen_llvm) {
        bonsai::CodeGen_LLVM codegen;
        auto _module = codegen.compile_program(program);
        if (options.output_filename.empty()) {
            _module->print(llvm::outs(), /*AAW=*/nullptr);
            return 0;
        }
        auto fd_os = bonsai::make_raw_fd_ostream(options.output_filename);
        _module->print(*fd_os, /*AAW=*/nullptr);
    }

    return 0;
}