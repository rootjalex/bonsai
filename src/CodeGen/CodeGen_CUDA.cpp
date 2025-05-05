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
}
} // namespace codegen

using namespace ir;

void CodeGen_CUDA::visit(const ir::LetStmt *node) {
    os << get_indent();
    codegen::emit_type(os, node->loc.type);
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
    codegen::emit_type(os, function.ret_type);
    os << ' ' << function.name << '(';
    for (int i = 0, e = function.args.size(); i < e; ++i) {
        const Function::Argument &arg = function.args[i];
        codegen::emit_type(os, arg.type);
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
    increment_indent();
    function.body.accept(this);
    decrement_indent();
    os << get_indent() << '}';
}

} //  namespace bonsai
