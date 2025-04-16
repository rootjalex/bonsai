#include "CodeGen/CPP.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "CompilerOptions.h"
#include "Error.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "IR/Visitor.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace bonsai {
namespace codegen {

namespace {
class TypeEmitter : public ir::Visitor {
  public:
    TypeEmitter(std::stringstream &ss, int64_t indent_level)
        : ss(ss), indent_level(indent_level) {}
    void visit(const ir::Struct_t *type) override {
        ss << "struct" << ' ' << type->name << ' ' << '{' << '\n';
        for (const auto &[name, child] : type->fields) {
            increment_indent();
            ss << indent();
            if (const auto *inner = child.as<ir::Vector_t>()) {
                inner->etype.accept(this);
                ss << ' ' << name;
                ss << '[' << inner->lanes << ']';
            } else if (const auto *inner = child.as<ir::Struct_t>()) {
                ss << inner->name << ' ' << name;
            } else {
                child.accept(this);
                ss << ' ' << name;
            }

            ss << ';' << '\n';
            decrement_indent();
        }
        ss << indent() << '}' << ';' << '\n';
    }
    void visit(const ir::Int_t *type) override {
        ss << "int" << type->bits << '_' << 't';
    }

    void visit(const ir::UInt_t *type) override {
        ss << "uint" << type->bits << '_' << 't';
    }

    void visit(const ir::Float_t *type) override {
        switch (type->bits()) {
        case 64:
            ss << "double";
            break;
        case 32:
            ss << "float";
            break;
        default:
            internal_error << "unimplemented: e" << type->exponent << "m"
                           << type->mantissa;
        }
    }

    void visit(const ir::Vector_t *type) override {
        // Needs to be handled earlier since we emit before and after the
        // argument name.
    }

  private:
    std::string indent() { return std::string(indent_level, ' '); }
    void increment_indent() { indent_level += 4; }
    void decrement_indent() { indent_level -= 4; }
    std::stringstream &ss;
    int64_t indent_level;
};

class BonsaiToCpp {
  public:
    std::string create_header(const ir::Program &program) {
        emit_prologue();
        emit_program(program);
        emit_epilogue();
        return ss.str();
    }

  private:
    std::string indent() { return std::string(indent_level, ' '); }
    int64_t indent_level = 0;
    std::stringstream ss;

    // TODO(cgyurgyik): How are non-standard types handled? Depends whether
    // they are custom types or encoded / decoded w.r.t. standard types.
    void emit_type(const ir::Type &type) {
        TypeEmitter emit(ss, indent_level);
        type.accept(&emit);
    }

    void emit_signature_type(const ir::Type &type) {
        if (const auto *struct_type = type.as<ir::Struct_t>()) {
            ss << struct_type->name;
            return;
        }
        emit_type(type);
    }

    void emit_func(const ir::Function &func) {
        emit_signature_type(func.ret_type);
        ss << ' ' << func.name;
        ss << '(';
        for (int i = 0, e = func.args.size(); i < e; ++i) {
            const ir::Function::Argument &arg = func.args[i];
            if (const auto *vector_type = arg.type.as<ir::Vector_t>()) {
                emit_signature_type(vector_type->etype);
                ss << ' ' << arg.name;
                ss << '[' << vector_type->lanes << ']';
            } else {
                emit_signature_type(arg.type);
                ss << ' ' << arg.name;
            }
            if (i + 1 == e) {
                continue;
            }
            ss << ',' << ' ';
        }
        ss << ')' << ';' << '\n';
    }

    void emit_program(const ir::Program &program) {
        for (const auto &[_, type] : program.types) {
            ss << indent();
            emit_type(type);
        }
        ss << '\n';
        for (const auto &[_, func] : program.funcs) {
            ss << indent();
            emit_func(*func);
        }
    }

    void emit_prologue() {
        // Only include this header once during compilation.
        ss << "#pragma once";
        ss << '\n' << '\n';

        // Headers for C++ types.
        ss << "#include <array>" << '\n';   // array
        ss << "#include <cstdint>" << '\n'; // integer

        // Disable C++ name mangling.
        ss << '\n' << "extern \"C\"";
        ss << ' ' << '{' << '\n';
    }

    void emit_epilogue() {
        ss << '}';
        ss << '\n';
    }
};

} // namespace

void to_cpp(const ir::Program &program, const CompilerOptions &options) {
    // Compile the program to LLVM.
    CodeGen_LLVM codegen;
    std::unique_ptr<llvm::Module> module = codegen.compile_program(program);

    std::unique_ptr<llvm::TargetMachine> target_machine =
        codegen.make_target_machine(*module, /*to_object_file=*/true);
    internal_assert(target_machine);

    // Open the object file (`.o`). We produce an object file during a dry run
    // to ensure no issues occur when testing.
    std::string output_file =
        !options.output_file.empty()
            ? options.output_file
            : std::string(std::filesystem::temp_directory_path());
    std::error_code ec;
    llvm::raw_fd_ostream os(output_file + ".o", ec, llvm::sys::fs::OF_None);
    internal_assert(!ec) << ec.message();

    // AFAICT, the only way to lower LLVM IR to object files is through the
    // legacy pass manager.
    llvm::legacy::PassManager pass;
    internal_assert(!target_machine->addPassesToEmitFile(
        pass, os, nullptr, llvm::CodeGenFileType::ObjectFile));

    // Run the passes to generate the object file.
    pass.run(*module);
    os.flush();

    // Write C++ header file with struct and function declarations (`.h`).
    if (options.output_file.empty()) {
        // Mostly for dry-run / testing purposes.
        llvm::outs() << "// Bonsai Header" << '\n';
        llvm::outs() << BonsaiToCpp().create_header(program) << '\n';
        llvm::outs() << std::string(42, '-') << '\n';
        llvm::outs() << '\n' << "; LLVM Module" << '\n';
        module->print(llvm::outs(), nullptr);
    } else {
        std::ofstream file;
        file.open(output_file + ".h");
        file << BonsaiToCpp().create_header(program);
        file.close();
    }
}

} // namespace codegen
} // namespace bonsai
