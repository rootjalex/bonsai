#include "CodeGen/CPP.h"

#include "CodeGen/CodeGen_LLVM.h"

#include "IR/Analysis.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "IR/Visitor.h"
#include "Lower/TopologicalOrder.h"

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

        void visit(const String_t *node) override { ss << "std::string"; }

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
            node->etype.accept(this);
            ss << "*";
        }

        void visit(const Set_t *node) override {
            ss << "set<";
            node->etype.accept(this);
            ss << ">";
        }

        void visit(const BVH_t *node) override { ss << node->name; }

        RESTRICT_VISITOR(Option_t);
        // RESTRICT_VISITOR(Set_t);
        RESTRICT_VISITOR(Function_t);
        RESTRICT_VISITOR(Generic_t);
        // RESTRICT_VISITOR(BVH_t);
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

        void visit(const StringImm *node) override {
            print_string_imm(ss, node->value);
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
                array_t && array_t->size.defined() && is_const(array_t->size) &&
                // The buffer field of dynamic arrays should always be treated
                // as a pointers, since its capacity may be resized.
                !is_dynamic_array_struct_type(type)) {

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
        ss << "using ";
        emit_type(ss, type); // get the name
        ss << " = vector<";
        emit_type(ss, vector_t->etype);
        // ss << " __attribute__((vector_size(";
        ss << ", " << vector_t->lanes << ">;\n";
        return;
    } else if (const BVH_t *bvh_t = type.as<BVH_t>()) {
        for (const auto &node : bvh_t->nodes) {
            // Forward declare
            ss << "struct " << node.name() << ";\n";
        }
        ss << "using " << bvh_t->name << " = ";
        ss << "tree<";
        bool first = true;
        for (const auto &node : bvh_t->nodes) {
            if (!first) {
                ss << ", ";
            }
            first = false;
            ss << node.name();
        }
        ss << ">;\n";
        return;
    }
    internal_error << "Can't emit type declaration for: " << type;
}

class BonsaiToCpp : ir::Printer {
  public:
    BonsaiToCpp() : ir::Printer(ss, 1) {}

    // Creates the bonsai header with external functions and their respective
    // struct definitions.
    std::string create_header(const Program &program, bool allow_mangling) {
        emit_prologue(allow_mangling);
        emit_program(program);
        emit_epilogue(allow_mangling);
        return ss.str();
    }

    std::string create_source(const Program &program,
                              std::string header_name = "") {
        if (!header_name.empty()) {
            ss << "#include \"" << header_name << "\""
               << '\n'; // c++ runtime types.
        }
        emit_funcs(program.funcs);
        return ss.str();
    }

  private:
    std::stringstream ss;

    void emit_signature_type(const Type &type, bool is_mutating = false,
                             bool is_return_type = false) {
        // TODO: understand why this was here.
        // internal_assert(!type.is<Struct_t>()) << type;
        const bool is_const = !is_mutating && !is_return_type;
        if (is_const) {
            ss << "const ";
        }
        if (const Ptr_t *ptr_t = type.as<Ptr_t>()) {
            emit_type(ss, ptr_t->etype);
            ss << "&";
        } else {
            emit_type(ss, type);
            if (!is_return_type && should_be_ref(type)) {
                ss << "&";
            }
        }
    }

    bool should_be_ref(const Type &type) const {
        // TODO: finish
        return type.is<Set_t>() || type.is<BVH_t>();
    }

    void emit_func_decl(const Function &func) {
        emit_func_header(func);
        ss << ";\n";
    }

    void emit_func_header(const Function &func) {
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
        ss << ")";
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
        } else if (const BVH_t *bvh_t = type.as<BVH_t>()) {
            if (auto [_, inserted] = deduplicate.insert(type); inserted) {
                types.push_back(type);
            }
            for (const auto &node : bvh_t->nodes) {
                get_declared_types(node.struct_type, deduplicate, types);
                for (const auto &annot : node.annotations) {
                    if (const auto *vol = annot.as<Annotation::Volume>()) {
                        get_declared_types(vol->struct_type, deduplicate,
                                           types);
                    }
                }
            }
        }
    }

    void emit_program(const Program &program) {
        std::set<Type> deduplicate;
        std::vector<Type> exported_types;

        // Any generated structs might be used in code generated,
        // and therefore must be ommitted.
        for (const auto &[name, type] : program.types) {
            if (name.starts_with("_tree"))
                get_declared_types(type, deduplicate, exported_types);
        }

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
            ss << get_indent();
            emit_type_declaration(ss, type);
        }
        ss << '\n';
        for (const auto &[_, func] : program.funcs) {
            if (!func->is_exported()) {
                continue;
            }
            ss << get_indent();
            emit_func_decl(*func);
        }
    }

    void emit_prologue(bool allow_mangling) {
        // Only include this header once during compilation.
        ss << "#pragma once";
        ss << '\n' << '\n';

        // Headers for C++ types.
        ss << "#include <cstdint>" << '\n'; // integer
        ss << "#include \"runtime/bonsai_cpp.h\"\n\n";

        // Disable C++ name mangling.
        if (!allow_mangling) {
            ss << '\n' << "extern \"C\"";
            ss << ' ' << '{' << '\n';
        }
    }

    void emit_epilogue(bool allow_mangling) {
        if (!allow_mangling) {
            ss << '}';
            ss << '\n';
        }
    }

    void emit_funcs(const FuncMap &funcs) {
        const std::vector<std::string> topological_order =
            lower::func_topological_order(funcs,
                                          /*undef_calls=*/false);

        for (int i = 0, e = topological_order.size(); i < e; ++i) {
            const std::string &name = topological_order[i];
            const auto &it = funcs.find(name);
            internal_assert(it != funcs.end());
            const auto &func = it->second;

            ss << get_indent();
            emit_func_header(*func);
            ss << " {\n";
            increment();
            func->body.accept(this);
            decrement();
            ss << "}\n";
        }
    }

    // Exprs
    // void visit(const IntImm *) override;
    // void visit(const UIntImm *) override;
    void visit(const FloatImm *node) override {
        std::ios::fmtflags f = ss.flags();
        std::streamsize p = ss.precision();

        double val = node->value;

        // Ensure float is parsed correctly in C++ by forcing float literal
        // syntax Add 'f' suffix and a decimal point if needed
        if (std::isnan(val)) {
            ss << "NAN";
        } else if (std::isinf(val)) {
            ss << (val < 0 ? "-" : "") << "INFINITY";
        } else {
            ss << std::fixed << std::setprecision(8) << val << "f";
        }

        ss.flags(f);
        ss.precision(p);
    }
    // void visit(const BoolImm *) override;
    // void visit(const VecImm *) override;
    // void visit(const StringImm *) override;
    // void visit(const Infinity *) override;
    // void visit(const Var *) override;
    // void print(const BinOp::OpType &op);
    // void visit(const BinOp *) override;
    // void print(const UnOp::OpType &op);
    // void visit(const UnOp *) override;
    void visit(const Select *node) override {
        ss << "(";
        print(node->cond);
        ss << " ? ";
        print(node->tvalue);
        ss << " : ";
        print(node->fvalue);
        ss << ")";
    }

    void visit(const Cast *node) override {
        if (node->mode == Cast::Mode::Reinterpret) {
            ss << "reinterpret<";
            emit_type(ss, node->type);
            ss << ">(";
            print_no_parens(node->value);
            ss << ")";
            return;
        } else {
            ss << "(";
            emit_type(ss, node->type);
            ss << ")(";
            print_no_parens(node->value);
            ss << ")";
            return;
        }
        internal_error << "TODO: cast C++ codegen: " << Expr(node);
    }
    // void visit(const Broadcast *) override;
    // void print(const VectorReduce::OpType &op);
    // void visit(const VectorReduce *) override;
    // void visit(const VectorShuffle *) override;
    // void visit(const Ramp *) override;
    // void visit(const Extract *) override;
    // void visit(const Build *) override;
    void visit(const Access *node) override {
        if (node->type.is<Ref_t>()) {
            ss << "(*"; // deref
        }
        ir::Printer::visit(node);
        if (node->type.is<Ref_t>()) {
            ss << ")";
        }
    }
    void visit(const Unwrap *node) override {
        // TODO: be less hacky about this. relies on current Match lowering.
        print_no_parens(node->value);
        ss << "_" << node->type.as<Struct_t>()->name;
    }
    // void visit(const Intrinsic *) override;
    // void visit(const Generator *) override;
    void visit(const Lambda *node) override {
        ss << "[&](";
        for (size_t i = 0, e = node->args.size(); i < e; i++) {
            const TypedVar &tvar = node->args[i];
            emit_signature_type(tvar.type, false);
            ss << "& " << tvar.name;
            if (i + 1 == e) {
                continue;
            }
            ss << ", ";
        }
        ss << ") { return ";
        print_no_parens(node->value);
        ss << "; }";
    }
    // void visit(const GeomOp *) override;
    // void visit(const Call *) override;
    // void visit(const Instantiate *) override;
    // void visit(const PtrTo *) override;
    // void visit(const Deref *) override;
    // void visit(const AtomicAdd *) override;
    // Stmts
    // void visit(const CallStmt *) override;
    // void visit(const Print *) override;

    // needs to override for ending `;`
    // void visit(const Return *node) override;

    // void visit(const LetStmt *) override;
    // void visit(const IfElse *) override;
    // void visit(const DoWhile *) override;
    // void visit(const Sequence *) override;
    void visit(const Allocate *node) override {
        internal_assert(node->loc.base_type.is<Set_t>())
            << "TODO: C++ Allocate lowering: " << Stmt(node);
        ss << get_indent();
        emit_type(ss, node->loc.base_type);
        ss << " " << node->loc.base;
        internal_assert(!node->value.defined())
            << "TODO: C++ Allocate lowering: " << Stmt(node);
        ss << ";\n";
    }
    // void visit(const Free *) override;
    // void visit(const Store *) override;
    // void visit(const Accumulate *) override;
    // void visit(const Label *) override;
    // void visit(const RecLoop *) override;
    void visit(const Match *node) override {
        ss << get_indent();
        print(node->loc);
        ss << ".match(\n";
        increment();
        for (size_t i = 0, e = node->arms.size(); i < e; i++) {
            const auto &arm = node->arms[i];
            ss << get_indent() << "[&](const ";
            ss << arm.first.struct_type.as<Struct_t>()->name;
            ss << "& ";
            print(node->loc);
            ss << "_";
            ss << arm.first.struct_type.as<Struct_t>()->name;
            // e.g. (const Interior& tree_Interior) { . . . }
            ss << ") {\n";
            increment();
            print(arm.second);
            decrement();
            ss << get_indent() << "}";
            if (i != (e - 1)) {
                ss << ",\n";
            } else {
                ss << "\n";
            }
        }
        decrement();
        ss << get_indent() << ");\n";
    }
    // void visit(const Yield *) override;
    // void visit(const Iterate *) override;
    // void visit(const Scan *) override;
    // void visit(const YieldFrom *) override;
    void visit(const ForAll *node) override {
        ss << get_indent();
        ss << "for (";
        emit_type(ss, node->slice.begin.type());
        ss << " " << node->index << " = ";
        print_no_parens(node->slice.begin);
        ss << "; " << node->index << " < ";
        print_no_parens(node->slice.end);
        ss << "; " << node->index << " += ";
        print_no_parens(node->slice.stride);
        ss << ") {\n";
        increment();
        print(node->body);
        decrement();
        ss << get_indent() << "}\n";
    }
    // void visit(const ForEach *) override;
    // void visit(const Continue *) override;
    // void visit(const Launch *) override;
    void visit(const Append *node) override {
        ss << get_indent();
        print(node->loc);
        ss << ".push_back(";
        print_no_parens(node->value);
        // if (node->value.type().is<Array_t>() &&
        // !node->loc.type.is<Array_t>()) {
        //     ss << ", ";
        //     print_no_parens(node->value.type().as<Array_t>()->size);
        // }
        ss << ");\n";
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
        llvm::outs() << BonsaiToCpp().create_header(program,
                                                    /*allow_mangling=*/false)
                     << '\n';
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
    file << BonsaiToCpp().create_header(program,
                                        /*allow_mangling=*/false);
    file.close();
}

void to_cppx(const ir::Program &program, const CompilerOptions &options) {
    // Compile the program to C++.

    if (options.output_file.empty()) {
        // Mostly for dry-run / testing purposes.
        llvm::outs() << "// Bonsai Header" << '\n';
        llvm::outs() << BonsaiToCpp().create_header(program,
                                                    /* allow_mangling */ true)
                     << '\n';
        llvm::outs() << std::string(42, '-') << '\n';
        llvm::outs() << BonsaiToCpp().create_source(program) << '\n';
        return;
    }

    // Write C++ header file with struct and function declarations (`.h`).
    std::ofstream h_file;
    h_file.open(options.output_file + ".h");
    h_file << BonsaiToCpp().create_header(program,
                                          /* allow_mangling */ true);
    h_file.close();

    // Write C++ source file with struct and function declarations (`.cpp`).
    std::ofstream src_file;
    src_file.open(options.output_file + ".cpp");
    src_file << BonsaiToCpp().create_source(program,
                                            options.output_file + ".h");
    src_file.close();
}

} // namespace codegen
} // namespace bonsai
