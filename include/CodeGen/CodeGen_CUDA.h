#pragma once

/** \file
 *
 * Defines the base-class for all architecture-specific code
 * generators that use llvm.
 */

#include "CompilerOptions.h"
#include "IR/Frame.h"
#include "IR/Function.h"
#include "IR/Program.h"
#include "IR/Visitor.h"

#include <memory>

namespace bonsai {
namespace codegen {

// Generates CUDA from a bonsai program with the respective compiler options.
// If an output file is provided, then the emitted LLVM IR is written there.
// Otherwise it is printed to standard I/O.
void to_cuda(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen

struct CodeGen_CUDA : public ir::Printer {
    CodeGen_CUDA(std::ostream &_os) : ir::Printer(_os) {}

    void print(const Program &program) override;
    void print(const Function &function) override;
    using ir::Printer::print;
    // Types
    bool declare = false; // if true, print struct declaration, else name.
    void visit(const ir::Void_t *) override;
    void visit(const ir::Int_t *) override;
    void visit(const ir::UInt_t *) override;
    void visit(const ir::Index_t *) override;
    void visit(const ir::Float_t *) override;
    void visit(const ir::Bool_t *) override;
    void visit(const ir::Ptr_t *) override;
    void visit(const ir::Ref_t *) override;
    void visit(const ir::Vector_t *) override;
    void visit(const ir::Array_t *) override;
    void visit(const ir::Struct_t *) override;
    RESTRICT_VISITOR(ir::Tuple_t);
    RESTRICT_VISITOR(ir::Function_t);
    RESTRICT_VISITOR(ir::Option_t);
    RESTRICT_VISITOR(ir::Set_t);
    RESTRICT_VISITOR(ir::Generic_t);
    RESTRICT_VISITOR(ir::BVH_t);
    // Interfaces
    RESTRICT_VISITOR(ir::IEmpty);
    RESTRICT_VISITOR(ir::IFloat);
    RESTRICT_VISITOR(ir::IVector);
    // Expressions
    // Default behavior of Imms is fine.
    void visit(const ir::VecImm *) override;
    void visit(const ir::Infinity *) override;
    // Default behavior of Var / BinOp / Unop is fine
    void visit(const ir::Select *) override;
    void visit(const ir::Cast *) override;
    void visit(const ir::Broadcast *) override;
    void visit(const ir::VectorReduce *) override;
    void visit(const ir::VectorShuffle *) override;
    void visit(const ir::Ramp *) override;
    void visit(const ir::Build *) override;
    // Default of Access and Extract are fine
    RESTRICT_VISITOR(ir::Unwrap);
    void visit(const ir::Intrinsic *) override;
    RESTRICT_VISITOR(ir::Generator);
    RESTRICT_VISITOR(ir::Lambda);
    RESTRICT_VISITOR(ir::GeomOp);
    RESTRICT_VISITOR(ir::SetOp);
    // Default of Call is fine
    RESTRICT_VISITOR(ir::Instantiate);
    // Default of PtrTo and Deref are fine
    // Stmts
    // Default of CallStmt and Return are fine.
    void visit(const ir::CallStmt *) override;
    void visit(const ir::Print *) override;
    void visit(const ir::Return *) override;
    void visit(const ir::LetStmt *) override;
    void visit(const ir::IfElse *) override;
    void visit(const ir::DoWhile *) override;
    // default behavior is fine.
    // void visit(const ir::Sequence *) override;
    void visit(const ir::Assign *) override;
    void visit(const ir::Accumulate *) override;
    void visit(const ir::Label *) override;
    RESTRICT_VISITOR(ir::RecLoop);
    RESTRICT_VISITOR(ir::YieldFrom);
    RESTRICT_VISITOR(ir::Match);
    RESTRICT_VISITOR(ir::Yield);
    RESTRICT_VISITOR(ir::Scan);
    void visit(const ir::ForAll *) override;
    RESTRICT_VISITOR(ir::ForEach);
    void visit(const ir::Continue *) override;
    void visit(const ir::Launch *) override;

    bool in_launch = false;
};

} //  namespace bonsai
