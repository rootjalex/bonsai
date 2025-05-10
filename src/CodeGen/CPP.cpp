#include "CodeGen/CPP.h"

#include "CodeGen/CodeGen_LLVM.h"

#include "IR/Analysis.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "IR/Visitor.h"

#include "CompilerOptions.h"
#include "Error.h"
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

using namespace ir;

void emit_type(std::ostream &ss, Type type) {
    struct Emit : public Visitor {
        std::ostream &ss;

        Emit(std::ostream &ss) : ss(ss) {}

        void visit(const Void_t *node) override { ss << "void"; }

        void visit(const Int_t *node) override {
            ss << "int" << node->bits << "_t";
        }

        void visit(const UInt_t *node) override {
            ss << "uint" << node->bits << "_t";
        }

        // TODO: Index_t
        RESTRICT_VISITOR(Index_t);

        // TODO(cgyurgyik): use std::float when it is supported:
        // https://en.cppreference.com/w/cpp/types/floating-point
        void visit(const Float_t *type) override {
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

        void visit(const Bool_t *node) override { ss << "bool"; }

        void visit(const Ptr_t *node) override {
            node->etype.accept(this);
            ss << " *";
        }

        RESTRICT_VISITOR(Ref_t);

        void visit(const Vector_t *node) override {
            ss << "vec" << node->lanes << "_";
            // TODO: this breaks on vectors of ptrs.
            internal_assert(!contains<Ptr_t>(node->etype));
            node->etype.accept(this);
        }

        void visit(const Struct_t *node) override { ss << node->name; }

        RESTRICT_VISITOR(Tuple_t);

        void visit(const Array_t *node) override {
            internal_assert(!is_const(node->size));
            node->etype.accept(this);
            ss << " *";
        }

        RESTRICT_VISITOR(Option_t);
        RESTRICT_VISITOR(Set_t);
        RESTRICT_VISITOR(Function_t);
        RESTRICT_VISITOR(Generic_t);
        RESTRICT_VISITOR(BVH_t);
        RESTRICT_VISITOR(Rand_State_t);
    };

    Emit emitter(ss);
    type.accept(&emitter);
}

namespace {

void emit_const_var(std::stringstream &ss, const Expr &expr) {
    internal_assert(is_const(expr));

    // This *must* support printing any Expr considered is_const.
    struct EmitConstVar : public Visitor {
        std::stringstream &ss;

        EmitConstVar(std::stringstream &ss) : ss(ss) {}

        void visit(const Broadcast *node) override {
            ss << "{";
            for (size_t i = 0; i < node->lanes; i++) {
                if (i != 0) {
                    ss << ", ";
                }
                node->value.accept(this);
            }
            ss << "}";
        }

        void visit(const Build *node) override {
            emit_type(ss, node->type);
            ss << "{";
            for (size_t i = 0; i < node->values.size(); i++) {
                if (i != 0) {
                    ss << ", ";
                }
                node->values[i].accept(this);
            }
            ss << "}";
        }

        void visit(const IntImm *node) override { ss << node->value; }

        void visit(const UIntImm *node) override { ss << node->value; }

        void visit(const FloatImm *node) override {
            // Save current formatting flags and precision
            std::ios::fmtflags f = ss.flags();
            std::streamsize p = ss.precision();

            // Set full precision for float output
            ss << std::setprecision(std::numeric_limits<double>::digits10 + 1)
               << node->value;

            // Restore previous formatting
            ss.flags(f);
            ss.precision(p);
        }

        void visit(const BoolImm *node) override {
            ss << (node->value ? "true" : "false");
        }

        void visit(const Infinity *node) override {
            ss << "std::numeric_limits<";
            emit_type(ss, node->type);
            ss << ">::infinity()";
        }

        void visit(const VecImm *node) override {
            ss << "{";
            for (size_t i = 0; i < node->values.size(); i++) {
                if (i != 0) {
                    ss << ", ";
                }
                node->values[i].accept(this);
            }
            ss << "}";
        }
    };

    EmitConstVar emitter(ss);
    expr.accept(&emitter);
}

void emit_type_declaration(std::stringstream &ss, Type type) {
    auto indent = std::string(4, ' ');

    if (const Struct_t *struct_t = type.as<Struct_t>()) {
        ss << "struct" << ' ' << struct_t->name << ' ' << '{' << '\n';
        for (const auto &[name, child] : struct_t->fields) {
            ss << indent;
            if (const Array_t *array_t = child.as<Array_t>();
                array_t && is_const(array_t->size)) {
                emit_type(ss, array_t->etype);
                std::optional<uint64_t> size =
                    get_constant_value(array_t->size);
                internal_assert(size.has_value());
                ss << " " << name << "[" << *size << "]";
            } else {
                emit_type(ss, child);
                ss << " " << name;
            }
            if (const auto &it = struct_t->defaults.find(name);
                it != struct_t->defaults.cend()) {
                ss << " = ";
                emit_const_var(ss, it->second);
            }
            ss << ";\n";
        }
        ss << '}';
        if (struct_t->is_packed()) {
            ss << " __attribute__((packed))";
        }
        ss << ";\n";
        return;
    } else if (const Vector_t *vector_t = type.as<Vector_t>()) {
        ss << "typedef ";
        emit_type(ss, vector_t->etype);
        ss << " ";
        emit_type(ss, type); // get the name
        ss << " __attribute__((vector_size(";
        ss << vector_t->lanes * vector_t->etype.bytes() << ")));\n";
        return;
    }
    internal_error << "Can't emit type declaration for: " << type;
}

class BonsaiToCpp {
  public:
    // Creates the bonsai header with external functions and their respective
    // struct definitions.
    std::string create_header(const Program &program) {
        emit_prologue();
        emit_program(program);
        emit_epilogue();
        return ss.str();
    }

  private:
    std::string indent() { return std::string(indent_level, ' '); }
    int64_t indent_level = 0;
    std::stringstream ss;

    void emit_signature_type(const Type &type, bool is_mutating = false,
                             bool is_return_type = false) {
        internal_assert(!type.is<Struct_t>());
        if (!is_mutating && !is_return_type) {
            ss << "const ";
        }
        if (const Ptr_t *ptr_t = type.as<Ptr_t>()) {
            emit_type(ss, ptr_t->etype);
            ss << "&";
        } else {
            emit_type(ss, type);
        }
    }

    void emit_func(const Function &func) {
        emit_signature_type(func.ret_type, /*is_mutating=*/false,
                            /*is_return_type=*/true);
        ss << ' ' << func.name;
        ss << '(';
        for (int i = 0, e = func.args.size(); i < e; ++i) {
            const Function::Argument &arg = func.args[i];
            emit_signature_type(arg.type, /*is_mutating=*/arg.mutating);
            ss << ' ' << arg.name;
            if (i + 1 == e) {
                continue;
            }
            ss << ',' << ' ';
        }
        ss << ')' << ';' << '\n';
    }

    // Recursively acquire all *unique* types and inserted them into `types`.
    void get_declared_types(const Type &type, std::set<Type> &deduplicate,
                            std::vector<Type> &types) {
        if (const Vector_t *vector_t = type.as<Vector_t>()) {
            get_declared_types(vector_t->etype, deduplicate, types);
            if (auto [_, inserted] = deduplicate.insert(type); inserted) {
                types.push_back(type);
            }
            return;
        } else if (const Array_t *array_t = type.as<Array_t>()) {
            get_declared_types(array_t->etype, deduplicate, types);
            return;
        } else if (const Ptr_t *ptr_t = type.as<Ptr_t>()) {
            get_declared_types(ptr_t->etype, deduplicate, types);
            return;
        } else if (const Struct_t *struct_t = type.as<Struct_t>()) {
            for (const auto &[_, field_type] : struct_t->fields) {
                get_declared_types(field_type, deduplicate, types);
            }
            if (auto [_, inserted] = deduplicate.insert(type); inserted) {
                types.push_back(type);
            }
        }
    }

    void emit_program(const Program &program) {
        std::set<Type> deduplicate;
        std::vector<Type> exported_types;
        for (const auto &[_, func] : program.funcs) {
            if (!func->is_exported()) {
                continue;
            }
            get_declared_types(func->ret_type, deduplicate, exported_types);
            for (const auto &arg_sig : func->argument_types()) {
                get_declared_types(arg_sig.type, deduplicate, exported_types);
            }
        }
        for (const Type &type : exported_types) {
            ss << indent();
            emit_type_declaration(ss, type);
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
