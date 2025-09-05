#pragma once

#include "CompilerOptions.h"
#include "IR/Frame.h"
#include "IR/Function.h"
#include "IR/Printer.h"
#include "IR/Program.h"
#include "IR/Visitor.h"
#include "Scope.h"

#include <memory>

namespace bonsai {
namespace codegen {

// Generates CUDA from a bonsai program with the respective compiler options.
// If an output file is provided, then the emitted CUDA program is written
// there. Otherwise it is printed to standard I/O.
void to_cuda(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen

class CodeGen_CUDA : public ir::Printer {
  public:
    explicit CodeGen_CUDA(std::ostream &os) : ir::Printer(os), os(os) {}

    void print(const ir::Program &program);
    void print(const ir::Function &function);

    // Types
    void visit(const ir::Float_t *) override;
    void visit(const ir::Vector_t *) override;
    void visit(const ir::Int_t *) override;
    void visit(const ir::UInt_t *) override;
    void visit(const ir::Struct_t *) override;
    void visit(const ir::Array_t *) override;
    void visit(const ir::Ptr_t *) override;

    RESTRICT_VISITOR(ir::String_t);
    RESTRICT_VISITOR(ir::Tuple_t);
    RESTRICT_VISITOR(ir::Function_t);
    RESTRICT_VISITOR(ir::Option_t);
    RESTRICT_VISITOR(ir::Set_t);
    RESTRICT_VISITOR(ir::Generic_t);
    RESTRICT_VISITOR(ir::BVH_t);
    void visit(const ir::Rand_State_t *) override;
    // TODO(cgyurgyik): CUDA supports a std::vector variant through their thrust
    // library. Potentially sufficient for our use case?
    RESTRICT_VISITOR(ir::DynArray_t);

    // Interfaces
    RESTRICT_VISITOR(ir::IEmpty);
    RESTRICT_VISITOR(ir::IFloat);
    RESTRICT_VISITOR(ir::IVector);
    // Expressions
    void visit(const ir::FloatImm *) override;
    void visit(const ir::VecImm *) override;
    void visit(const ir::StringImm *) override;
    void visit(const ir::Infinity *) override;
    void visit(const ir::Deref *) override;
    RESTRICT_VISITOR(ir::AtomicAdd); // TODO
    void visit(const ir::Select *) override;
    void visit(const ir::Cast *) override;
    void visit(const ir::Broadcast *) override;
    void visit(const ir::VectorReduce *) override;
    void visit(const ir::VectorShuffle *) override;
    void visit(const ir::Ramp *) override;
    void visit(const ir::Build *) override;
    void visit(const ir::Intrinsic *) override;
    void visit(const ir::Access *) override;
    void visit(const ir::Extract *) override;
    RESTRICT_VISITOR(ir::Unwrap);
    RESTRICT_VISITOR(ir::Generator);
    RESTRICT_VISITOR(ir::Lambda);
    RESTRICT_VISITOR(ir::GeomOp);
    RESTRICT_VISITOR(ir::SetOp);
    RESTRICT_VISITOR(ir::Instantiate);
    // Statements
    void visit(const ir::CallStmt *) override;
    void visit(const ir::Print *) override;
    void visit(const ir::Return *) override;
    void visit(const ir::LetStmt *) override;
    void visit(const ir::IfElse *) override;
    void visit(const ir::DoWhile *) override;
    void visit(const ir::Free *) override;
    void visit(const ir::Allocate *) override;
    void visit(const ir::Store *) override;
    void visit(const ir::Accumulate *) override;
    void visit(const ir::Label *) override;
    void visit(const ir::ForAll *) override;
    void visit(const ir::Continue *) override;
    void visit(const ir::Launch *) override;
    RESTRICT_VISITOR(ir::Append); // TODO
    RESTRICT_VISITOR(ir::ForEach);
    RESTRICT_VISITOR(ir::RecLoop);
    RESTRICT_VISITOR(ir::YieldFrom);
    RESTRICT_VISITOR(ir::Match);
    RESTRICT_VISITOR(ir::Yield);
    RESTRICT_VISITOR(ir::Iterate);
    RESTRICT_VISITOR(ir::Scan);

  private:
    // Whether we are printing type declarations, e.g.,
    // `struct P { int x; int y; int z; }` versus `P`
    bool is_declaration = false;
    // Whether the next function definition exists on device. This is necessary
    // for correct usage of rand, which is different for __host__ and
    // __device__.
    bool on_device = false;
    // We need to also device allocate inner children, a concept that is
    // (currently) foreign to bonsai so is done here. This list tracks all the
    // device allocated children of a struct.
    // TODO(cgyurgyik): it might be cleaner to do this struct allocation
    // / deallocation traversal in the Parallelize pass.
    std::vector<ir::TypedVar> device_allocated;
    //  The stream that is printed to.
    std::ostream &os;

    // Necessary prologue code.
    void emit_prologue();
    // Performs rng setup for CUDA, using cuRAND.
    void setup_kernel_rng(const ir::Function &function);
    // Emits necessary malloc & host->device code for a struct.
    void emit_to_device(const ir::Allocate *);
    void emit_to_device(std::string base, ir::Type, ir::Expr,
                        std::optional<ir::Expr> = {});
    void emit_to_device(std::string base, const ir::Struct_t *, ir::Expr);
    void emit_to_device(std::string base, const ir::Array_t *, ir::Expr,
                        std::optional<ir::Expr> = {});
};

} //  namespace bonsai
