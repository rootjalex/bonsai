#include "CodeGen/CodeGen_CUDA.h"

#include "CodeGen/CPP.h"
#include "IR/Analysis.h"
#include "IR/Expr.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <fstream>
#include <sstream>

namespace bonsai {
namespace codegen {
void to_cuda(const ir::Program &program, const CompilerOptions &options) {
    if (options.output_file.empty()) {
        CodeGen_CUDA codegen(std::cout);
        codegen.print(program);
        return;
    }
    std::ofstream os(options.output_file);
    internal_assert(os.is_open()) << "failed to open: " << options.output_file;
    CodeGen_CUDA codegen(os);
    codegen.print(program);
    os.close();
}
} // namespace codegen

using namespace ir;

namespace {

// Returns the appropriate prefix for builtin CUDA vector types.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
std::string vector_prefix(Type element_type) {
    if (element_type.is<Int_t, UInt_t>()) {
        const bool is_unsigned = element_type.is<UInt_t>();
        switch (element_type.bits()) {
        case 64:
            return std::string(is_unsigned ? "u" : "") + "longlong";
        case 32:
            return std::string(is_unsigned ? "u" : "") + "int";
        case 16:
            return std::string(is_unsigned ? "u" : "") + "short";
        case 8:
            return std::string(is_unsigned ? "u" : "") + "char";
        default:
            break;
        }
    }
    if (const auto *float_t = element_type.as<Float_t>();
        float_t && float_t->is_ieee754()) {
        switch (float_t->bits()) {
        case 64:
            return "double";
        case 32:
            return "float";
        case 16:
            return "half";
        default:
            break;
        }
    }
    internal_error << "[unimplemented] vector prefix for element type: "
                   << element_type;
}

} // namespace

void CodeGen_CUDA::visit(const Int_t *node) { codegen::emit_type(os, node); }

void CodeGen_CUDA::visit(const UInt_t *node) { codegen::emit_type(os, node); }

void CodeGen_CUDA::visit(const Float_t *node) {
    if (node->is_ieee754()) {
        switch (node->bits()) {
        case 16:
            // Assumes <cuda_fp16.h> is included somewhere
            os << "__half";
            return;
        case 32:
            os << "float";
            return;
        case 64:
            os << "double";
            return;
        default:
            break;
        }
    }
    internal_error << "[unimplemented] float type codegen on CUDA: "
                   << Type(node);
}

void CodeGen_CUDA::visit(const Vector_t *node) {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
    internal_assert(1 <= node->lanes && node->lanes <= 4)
        << "[unimplemented] vector size: " << node->lanes
        << " in CUDA codegen, " << Type(node);
    os << vector_prefix(node->etype) << node->lanes;
}

void CodeGen_CUDA::visit(const VecImm *node) {
    internal_error << "[unimplemented] VecImm CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Infinity *node) {
    // TODO(cgyurgyik): Assumes implementation-defined version of infinity.
    // Requires #include <cmath>
    os << "INFINITY";
}

void CodeGen_CUDA::visit(const Cast *node) {
    // TODO(cgyurgyik): Is this what cast really means in Bonsai?
    os << '(';
    node->type.accept(this);
    os << ')';
    node->value.accept(this);
}

void CodeGen_CUDA::visit(const Broadcast *node) {
    os << "make" << '_' << vector_prefix(node->value.type()) << node->lanes;
    os << '(';
    for (int i = 0, e = node->lanes; i < e; ++i) {
        node->value.accept(this);
        if (i + 1 == e) {
            continue;
        }
        os << ',';
    }
    os << ')';
}

void CodeGen_CUDA::visit(const VectorReduce *node) {
    internal_error << "[unimplemented] VectorReduce CUDA codegen: "
                   << Expr(node);
}

void CodeGen_CUDA::visit(const VectorShuffle *node) {
    internal_error << "[unimplemented] VectorShuffle CUDA codegen: "
                   << Expr(node);
}

void CodeGen_CUDA::visit(const Ramp *node) {
    internal_error << "[unimplemented] Ramp CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Build *node) {
    node->type.accept(this);
    os << '{';
    for (size_t i = 0, n = node->values.size(); i < n; i++) {
        if (i != 0) {
            os << ',' << ' ';
        }
        print_no_parens(node->values[i]);
    }
    os << '}';
}

void CodeGen_CUDA::visit(const Select *node) {
    open();
    node->cond.accept(this);
    os << " ? ";
    node->tvalue.accept(this);
    os << " : ";
    node->fvalue.accept(this);
    close();
}

void CodeGen_CUDA::visit(const ir::Intrinsic *node) {
    internal_error << "[unimplemented] Intrinsic CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const ir::LetStmt *node) {
    os << get_indent() << "const" << ' ';
    node->loc.type.accept(this);
    os << ' ' << node->loc.base << ' ' << '=' << ' ';
    node->value.accept(this);
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const ir::Return *node) {
    os << get_indent() << "return";
    if (ir::Expr value = node->value; value.defined()) {
        os << ' ';
        value.accept(this);
    }
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const ir::CallStmt *node) {
    internal_error << "[unimplemented] CallStmt CUDA codegen: " << Stmt(node);
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const Print *node) {
    // TODO(cgyurgyik): CUDA enables printing through `printf`. I imagine
    // (though have not verified) this is going to use the same format
    // specifiers as C printf, so we can just refactor the LLVM version and use
    // it here.
    internal_error << "[unimplemented] Print CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const IfElse *node) {
    os << get_indent() << "if" << ' ' << '(';
    print_no_parens(node->cond);
    os << ')' << ' ' << '{' << '\n';
    increment();
    node->then_body.accept(this);
    decrement();
    os << get_indent() << '}';
    if (node->else_body.defined()) {
        os << ' ' << "else" << ' ' << '{' << '\n';
        increment();
        node->else_body.accept(this);
        decrement();
        os << get_indent() << "}";
    }
    os << "\n";
}

void CodeGen_CUDA::visit(const DoWhile *node) {
    os << get_indent() << "do" << ' ' << '{';
    node->body.accept(this);
    os << get_indent() << '}' << ' ' << "while" << ' ' << '(';
    os << get_indent() << "} while (";
    print_no_parens(node->cond);
    os << ')' << '\n';
}

void CodeGen_CUDA::visit(const Assign *node) {
    // TODO(ajr): if this is a launched kernel, this cannot be an array
    // allocation. Otherwise, this should probably cuda malloc for arrays.
}

void CodeGen_CUDA::visit(const Accumulate *node) {
    internal_error << "[unimplemented] Accumulate CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const Label *node) {
    internal_error << "[unimplemented] Label CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const ForAll *node) {
    internal_error << "[unimplemented] ForAll CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const Continue *node) {
    internal_error << "[unimplemented] Continue CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const Launch *node) {
    internal_error << "[unimplemented] Launch CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::print(const Program &program) {
    int i = 0, e = program.funcs.size();
    for (const auto &[_, func] : program.funcs) {
        if (func == nullptr) {
            os << get_indent() << "[BONSAI NULL FUNCTION]" << '\n' << '\n';
            continue;
        }
        print(*func);
        os << '\n';
        if (i++ + 1 == e) {
            continue;
        }
        os << '\n';
    }
}
void CodeGen_CUDA::print(const Function &function) {
    os << get_indent();
    function.ret_type.accept(this);
    os << ' ' << function.name << '(';
    for (int i = 0, e = function.args.size(); i < e; ++i) {
        const Function::Argument &arg = function.args[i];
        arg.type.accept(this);
        os << ' ' << arg.name;
        if (ir::Expr value = arg.default_value; value.defined()) {
            os << '=';
            value.accept(this);
        }
        if (i + 1 == e) {
            continue;
        }
        os << ',' << ' ';
    }
    os << ')' << ' ' << '{' << '\n';
    increment();
    function.body.accept(this);
    decrement();
    os << get_indent() << '}';
}

} //  namespace bonsai
