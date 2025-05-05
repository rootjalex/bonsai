#include "CodeGen/CodeGen_CUDA.h"

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

void to_cuda(const Program &program, const CompilerOptions &options) {
    std::ofstream file;

    if (!options.output_file.empty()) {
        file.open(options.output_file);
    }

    CodeGen_CUDA codegen(options.output_file.empty() ? std::cout : file);
    codegen.print(program);

    if (file.is_open()) {
        file.close();
    }
}

} // namespace codegen

using namespace ir;

void CodeGen_CUDA::print(const Program &program) {}

void CodeGen_CUDA::print(const Function &function) {}

void CodeGen_CUDA::visit(const Void_t *) {
    if (!declare) {
        os << "void";
    }
}
void CodeGen_CUDA::visit(const Int_t *node) {
    if (!declare) {
        os << "int" << node->bits << "_t";
    }
}

void CodeGen_CUDA::visit(const UInt_t *node) {
    if (!declare) {
        os << "uint" << node->bits << "_t";
    }
}

void CodeGen_CUDA::visit(const Index_t *) {
    if (!declare) {
        os << "int";
    }
}

void CodeGen_CUDA::visit(const Float_t *node) {
    if (!declare) {
        switch (node->bits()) {
        case 16:
            os << "__half"; // Assumes <cuda_fp16.h> is included somewhere
            return;
        case 32:
            os << "float";
            return;
        case 64:
            os << "double";
            return;
        default:
            internal_error << "[unimplemented] float type codegen on CUDA: "
                           << Type(node);
        }
    }
}

void CodeGen_CUDA::visit(const Bool_t *) {
    if (!declare) {
        os << "bool";
    }
}

void CodeGen_CUDA::visit(const Ptr_t *node) {
    if (!declare) {
        print(node->etype);
        os << " *";
    }
}

void CodeGen_CUDA::visit(const Ref_t *node) {
    internal_error << "Ref_t in CodeGen_CUDA: " << Type(node);
}

void CodeGen_CUDA::visit(const Vector_t *node) {
    if (!declare) {
        internal_assert(node->lanes == 2 || node->lanes == 3 ||
                        node->lanes == 4)
            << "[unimplemented] vector size larger than 4 in CUDA codegen: "
            << Type(node);
        const Type &elem = node->etype;

        std::string prefix;
        if (const auto *i = elem.as<Int_t>()) {
            internal_assert(i->bits == 32)
                << "[unsupported] int vector bit width: " << i->bits;
            prefix = "int";
        } else if (const auto *u = elem.as<UInt_t>()) {
            internal_assert(u->bits == 32)
                << "[unsupported] uint vector bit width: " << u->bits;
            prefix = "uint";
        } else if (const auto *f = elem.as<Float_t>()) {
            switch (f->bits()) {
            case 16:
                prefix = "half";
                break;
            case 32:
                prefix = "float";
                break;
            case 64:
                prefix = "double";
                break;
            default:
                internal_error << "[unsupported] float vector bit width: "
                               << Type(node);
            }
        } else {
            internal_error
                << "[unsupported] vector element type in CUDA codegen: "
                << Type(node);
        }

        os << prefix << node->lanes;
    }
}

void CodeGen_CUDA::visit(const Array_t *node) {
    if (!declare) {
        internal_assert(!is_const(node->size))
            << "TODO: constant sized array_t in CUDA codegen: " << Type(node);
        print(node->etype);
        // TODO: is this right? just print like pointer.
        os << " *";
    }
}

void CodeGen_CUDA::visit(const Struct_t *node) {
    if (!declare) {
        os << node->name;
        return;
    }
    os << get_indent();
    os << "struct " << node->name << "{\n";
    // TODO: alignment or packing?
    ScopedValue<bool> _(declare, false);
    indent++;
    for (const auto &field : node->fields) {
        // TODO: handle constant-sized arrays?
        os << get_indent();
        print(field.type);
        os << " " << field.name;
        if (const auto &iter = node->defaults.find(field.name);
            iter != node->defaults.cend()) {
            os << " = ";
            print_no_parens(iter->second);
        }
        os << ";\n";
    }
    indent--;
    os << get_indent() << "}\n";
}

void CodeGen_CUDA::visit(const VecImm *node) {
    internal_error << "[unimplemented] VecImm CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Infinity *node) {
    internal_error << "[unimplemented] Infinity CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Select *node) {
    open();
    print(node->cond);
    os << " ? ";
    print(node->tvalue);
    os << " : ";
    print(node->fvalue);
    close();
}

void CodeGen_CUDA::visit(const Cast *node) {
    internal_error << "[unimplemented] Cast CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Broadcast *node) {
    internal_error << "[unimplemented] Broadcast CUDA codegen: " << Expr(node);
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
    ScopedValue<bool> _(declare, false);
    print(node->type);
    os << "{";
    for (size_t i = 0, n = node->values.size(); i < n; i++) {
        if (i != 0) {
            os << ", ";
        }
        print_no_parens(node->values[i]);
    }
    os << "}";
}

void CodeGen_CUDA::visit(const Intrinsic *node) {
    // TODO: do we need special type-label handling?
    Printer::visit(node);
}

void CodeGen_CUDA::visit(const Print *node) {
    internal_error << "[unimplemented] Print CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const LetStmt *node) {
    internal_assert(node->loc.accesses.empty());
    os << get_indent() << "const ";
    print(node->loc.base_type);
    os << " " << node->loc.base << " = ";
    print_no_parens(node->value);
    os << ";\n";
}

void CodeGen_CUDA::visit(const IfElse *node) {
    os << get_indent() << "if (";
    print_no_parens(node->cond);
    os << ") {\n";
    indent++;
    print(node->then_body);
    indent--;
    os << get_indent() << "}";
    if (node->else_body.defined()) {
        os << " else {\n";
        indent++;
        print(node->else_body);
        indent--;
        os << get_indent() << "}";
    }
    os << "\n";
}

void CodeGen_CUDA::visit(const DoWhile *node) {
    os << get_indent() << "do {";
    print(node->body);
    os << get_indent() << "} while (";
    print_no_parens(node->cond);
    os << ")\n";
}

void CodeGen_CUDA::visit(const Assign *node) {
    // TODO: if is_launch is set, this cannot be an array allocation.
    // if is_launch is false, this should probably cuda malloc for arrays.
}

void CodeGen_CUDA::visit(const Accumulate *) {}

void CodeGen_CUDA::visit(const Label *) {}

void CodeGen_CUDA::visit(const ForAll *) {}

void CodeGen_CUDA::visit(const Continue *) {}

void CodeGen_CUDA::visit(const Launch *) {}

} //  namespace bonsai
