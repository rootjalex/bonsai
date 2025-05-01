#include "CodeGen/CPP.h"

#include "CodeGen/CodeGen_LLVM.h"
#include "CompilerOptions.h"
#include "Error.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "IR/Visitor.h"
#include "Utils.h"

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
#include <set>
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
            if (handle_constant_sized_container(name, child)) {
            } else if (const auto *inner = child.as<ir::Struct_t>()) {
                ss << inner->name << ' ' << name;
            } else {
                child.accept(this);
                ss << ' ' << name;
            }

            ss << ';' << '\n';
            decrement_indent();
        }
        ss << indent() << '}';
        if (type->is_packed()) {
            constexpr std::string_view P = "__attribute__((packed))";
            ss << ' ' << P;
        }
        ss << ';' << '\n';
    }
    void visit(const ir::Array_t *type) override {
        // This case should have already been handled.
        internal_assert(!is_const(type->size));
        if (const auto *element = type->etype.as<ir::Struct_t>()) {
            ss << element->name;
        } else {
            type->etype.accept(this);
        }
        ss << "*";
    }
    void visit(const ir::Int_t *type) override {
        ss << "int" << type->bits << '_' << 't';
    }

    void visit(const ir::Void_t *type) override { ss << "void"; }

    void visit(const ir::UInt_t *type) override {
        ss << "uint" << type->bits << '_' << 't';
    }

    // TODO(cgyurgyik): use std::float when it is supported:
    // https://en.cppreference.com/w/cpp/types/floating-point
    void visit(const ir::Float_t *type) override {
        switch (type->bits()) {
        case 64:
            internal_assert(type->is_ieee754());
            ss << "double";
            break;
        case 32:
            internal_assert(type->is_ieee754());
            ss << "float";
            break;
        default:
            internal_error << "unimplemented: e" << type->exponent << "m"
                           << type->mantissa;
        }
    }

    void visit(const ir::Vector_t *type) override {
        internal_error
            << "Vector_t needs to be handled early since we emit "
               "before and after the argument name, i.e., `T name[N]`";
    }

    void visit(const ir::Ptr_t *type) override {
        if (const ir::Struct_t *struct_t = type->etype.as<ir::Struct_t>()) {
            ss << struct_t->name << "&";
        } else {
            type->etype.accept(this);
            ss << "&";
        }
    }

  private:
    // Prints the type for a constant sized array or vector, and returns true.
    // If this fails, returns false.
    bool handle_constant_sized_container(const std::string &name,
                                         ir::Type type) {
        if (const auto *t = type.as<ir::Vector_t>()) {
            t->etype.accept(this);
            ss << ' ' << name;
            ss << '[' << t->lanes << ']';
            return true;
        }
        if (const auto *t = type.as<ir::Array_t>()) {
            std::optional<uint64_t> size = get_constant_value(t->size);
            if (!size.has_value()) {
                return false;
            }
            t->etype.accept(this);
            ss << ' ' << name;
            ss << '[' << *size << ']';
            return true;
        }
        return false;
    }

    std::string indent() { return std::string(indent_level, ' '); }
    void increment_indent() { indent_level += 4; }
    void decrement_indent() { indent_level -= 4; }
    std::stringstream &ss;
    int64_t indent_level;
};

class BonsaiToCpp {
  public:
    // Creates the bonsai header with external functions and their respective
    // struct definitions.
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

    void emit_signature_type(const ir::Type &type, bool is_mutating = false,
                             bool is_return_type = false) {
        const auto *struct_type = type.as<ir::Struct_t>();
        if (struct_type == nullptr) {
            if (!is_mutating && !is_return_type) {
                ss << "const ";
            }
            emit_type(type);
            return;
        }
        // Mutable structs must be passed in as pointers.
        internal_assert(!is_mutating);
        if (is_return_type) {
            ss << struct_type->name;
            return;
        }
        ss << "const " << struct_type->name << "&";
    }

    void emit_func(const ir::Function &func) {
        emit_signature_type(func.ret_type, /*is_mutating=*/false,
                            /*is_return_type=*/true);
        ss << ' ' << func.name;
        ss << '(';
        for (int i = 0, e = func.args.size(); i < e; ++i) {
            const ir::Function::Argument &arg = func.args[i];
            if (const auto *vector_type = arg.type.as<ir::Vector_t>()) {
                emit_signature_type(vector_type->etype,
                                    /*is_mutating=*/arg.mutating);
                ss << ' ' << arg.name;
                ss << '[' << vector_type->lanes << ']';
            } else {
                emit_signature_type(arg.type, /*is_mutating=*/arg.mutating);
                ss << ' ' << arg.name;
            }
            if (i + 1 == e) {
                continue;
            }
            ss << ',' << ' ';
        }
        ss << ')' << ';' << '\n';
    }

    // Recursively acquire all *unique* types and inserted them into `types`.
    void get_struct_types(const ir::Type &type, std::set<ir::Type> &deduplicate,
                          std::vector<ir::Type> &types) {
        if (type.is<ir::Vector_t, ir::Array_t>()) {
            get_struct_types(type.element_of(), deduplicate, types);
            return;
        }
        if (const ir::Ptr_t *ptr_t = type.as<ir::Ptr_t>()) {
            get_struct_types(ptr_t->etype, deduplicate, types);
            return;
        }
        const auto *struct_type = type.as<ir::Struct_t>();
        if (struct_type == nullptr) {
            return;
        }
        for (const auto &[_, field_type] : struct_type->fields) {
            get_struct_types(field_type, deduplicate, types);
        }
        if (auto [_, inserted] = deduplicate.insert(type); inserted) {
            types.push_back(type);
        }
    }

    void emit_program(const ir::Program &program) {
        std::set<ir::Type> deduplicate;
        std::vector<ir::Type> exported_types;
        for (const auto &[_, func] : program.funcs) {
            if (!func->is_exported()) {
                continue;
            }
            for (const auto &arg_sig : func->argument_types()) {
                get_struct_types(arg_sig.type, deduplicate, exported_types);
            }
            if (ir::Type type = func->ret_type; type.is<ir::Struct_t>()) {
                get_struct_types(type, deduplicate, exported_types);
            }
        }
        for (const ir::Type &type : exported_types) {
            ss << indent();
            emit_type(type);
        }
        ss << '\n';
        for (const auto &[_, func] : program.funcs) {
            if (!func->is_exported()) {
                continue;
            }
            ss << indent();
            emit_func(*func);
        }
    }

    void emit_prologue() {
        // Only include this header once during compilation.
        ss << "#pragma once";
        ss << '\n' << '\n';

        // Headers for C++ types.
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
    std::unique_ptr<llvm::Module> module =
        codegen.compile_program(program, options);

    std::unique_ptr<llvm::TargetMachine> target_machine =
        codegen.make_target_machine(*module, options);
    internal_assert(target_machine);

    // Open the object file (`.o`). We produce an object file during a dry run
    // to ensure no issues occur when testing.
    if (options.output_file.empty()) {
        // Mostly for dry-run / testing purposes.
        llvm::outs() << "// Bonsai Header" << '\n';
        llvm::outs() << BonsaiToCpp().create_header(program) << '\n';
        llvm::outs() << std::string(42, '-') << '\n';
        llvm::outs() << '\n' << "; LLVM Module" << '\n';
        codegen.print_module(*module, llvm::outs(), /*redacted=*/true);
        return;
    }
    std::error_code ec;
    llvm::raw_fd_ostream os(options.output_file + ".o", ec,
                            llvm::sys::fs::OF_None);
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
    std::ofstream file;
    file.open(options.output_file + ".h");
    file << BonsaiToCpp().create_header(program);
    file.close();
}

} // namespace codegen
} // namespace bonsai
