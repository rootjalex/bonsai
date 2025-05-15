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
#include "LLVMIncl.h"
#include "Scope.h"

#include <memory>

namespace bonsai {
namespace codegen {

// Generates LLVM IR from a bonsai program with the respective compiler options.
// If an output file is provided, then the emitted LLVM IR is written there.
// Otherwise it is printed to standard I/O.
void to_llvm(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen

struct CodeGen_LLVM : public ir::Visitor {
    CodeGen_LLVM();

    /** Takes a bonsai Program and compiles it to an llvm Module. */
    virtual std::unique_ptr<llvm::Module>
    compile_program(const ir::Program &program, const CompilerOptions &options);

    std::unique_ptr<llvm::LLVMContext> steal_context() {
        return std::move(context);
    }

    // Creates a target machine and updates the module's backend and data
    // layout.
    std::unique_ptr<llvm::TargetMachine>
    make_target_machine(llvm::Module &module, const CompilerOptions &options);

    // Print the LLVM module. If `redacted` is true, we don't print the target
    // triple or data layout.
    void print_module(llvm::Module &module, llvm::raw_ostream &os,
                      bool redacted = false);

  protected:
    /** Initialize internal llvm state for the enabled targets. */
    static void init_llvm();
    /** Grab all the context specific internal state. */
    virtual void init_context();
    /** Initialize the CodeGen_LLVM internal state to compile a fresh
     * module. This allows reuse of one CodeGen_LLVM object to compiled
     * multiple related modules (e.g. multiple device kernels). */
    virtual void init_module();

    virtual void optimize_module(llvm::TargetMachine &tm,
                                 const CompilerOptions &options);

    llvm::Function *declare_function(const ir::Function &func);
    void compile_function(const ir::Function &func, llvm::Function *function);
    llvm::Value *codegen_expr(const ir::Expr &expr);
    std::vector<llvm::Value *> codegen_exprs(const std::vector<ir::Expr> exprs);
    void codegen_stmt(const ir::Stmt &stmt);
    llvm::Type *codegen_type(const ir::Type &type);
    llvm::Function *codegen_func_ptr(const ir::Expr &expr);
    llvm::Value *codegen_write_loc(const ir::WriteLoc &loc);

    llvm::Value *codegen_buffer_pointer(const std::string &buffer,
                                        const ir::Type &type,
                                        const ir::Expr &idx);
    llvm::Value *codegen_buffer_pointer(const std::string &buffer,
                                        const ir::Type &type, llvm::Value *idx);
    void add_tbaa_metadata(llvm::Instruction *inst, const std::string &buffer,
                           const ir::Expr &index);

    void declare_struct_types(const std::vector<const ir::Struct_t *> structs);

    /** Get a unique name for the actual block of memory that an
     * allocate node uses. Used so that alias analysis understands
     * when multiple Allocate nodes shared the same memory. */
    virtual std::string get_allocation_name(const std::string &n) { return n; }

    // Generates a short-circuiting if else.
    void codegen_short_circuit(ir::Expr cond, llvm::BasicBlock *true_bb,
                               llvm::BasicBlock *false_bb);
    // Inserts a branch only if the block does not already have a terminator
    // (e.g. a ret or br)
    void codegen_branch(llvm::BasicBlock *bb);

    // Types
    virtual void visit(const ir::Void_t *) override;
    virtual void visit(const ir::Int_t *) override;
    virtual void visit(const ir::UInt_t *) override;
    virtual void visit(const ir::Index_t *) override;
    virtual void visit(const ir::Float_t *) override;
    virtual void visit(const ir::Bool_t *) override;
    virtual void visit(const ir::Ptr_t *) override;
    virtual void visit(const ir::Ref_t *) override;
    virtual void visit(const ir::Vector_t *) override;
    virtual void visit(const ir::Array_t *) override;
    virtual void visit(const ir::Struct_t *) override;
    virtual void visit(const ir::Tuple_t *) override;
    virtual void visit(const ir::Function_t *) override;
    RESTRICT_VISITOR(ir::Option_t);
    RESTRICT_VISITOR(ir::Set_t);
    RESTRICT_VISITOR(ir::Generic_t);
    RESTRICT_VISITOR(ir::BVH_t);
    virtual void visit(const ir::Rand_State_t *) override;
    RESTRICT_VISITOR(ir::Queue_t); // TODO
    // Interfaces
    RESTRICT_VISITOR(ir::IEmpty);
    RESTRICT_VISITOR(ir::IFloat);
    RESTRICT_VISITOR(ir::IVector);
    // Expressions
    virtual void visit(const ir::IntImm *) override;
    virtual void visit(const ir::UIntImm *) override;
    virtual void visit(const ir::FloatImm *) override;
    virtual void visit(const ir::BoolImm *) override;
    virtual void visit(const ir::VecImm *) override;
    virtual void visit(const ir::Infinity *) override;
    virtual void visit(const ir::Var *) override;
    virtual void visit(const ir::BinOp *) override;
    virtual void visit(const ir::UnOp *) override;
    virtual void visit(const ir::Select *) override;
    virtual void visit(const ir::Cast *) override;
    virtual void visit(const ir::Broadcast *) override;
    virtual void visit(const ir::VectorReduce *) override;
    virtual void visit(const ir::VectorShuffle *) override;
    virtual void visit(const ir::Ramp *) override;
    virtual void visit(const ir::Extract *) override;
    virtual void visit(const ir::Build *) override;
    virtual void visit(const ir::Access *) override;
    virtual void visit(const ir::Unwrap *) override;
    virtual void visit(const ir::Intrinsic *) override;
    RESTRICT_VISITOR(ir::Generator);
    virtual void visit(const ir::Lambda *) override;
    virtual void visit(const ir::GeomOp *) override;
    virtual void visit(const ir::SetOp *) override;
    virtual void visit(const ir::Call *) override;
    virtual void visit(const ir::Instantiate *) override;
    virtual void visit(const ir::PtrTo *) override;
    virtual void visit(const ir::Deref *) override;
    // Stmts
    virtual void visit(const ir::CallStmt *) override;
    virtual void visit(const ir::Print *) override;
    virtual void visit(const ir::Return *) override;
    virtual void visit(const ir::LetStmt *) override;
    virtual void visit(const ir::IfElse *) override;
    virtual void visit(const ir::DoWhile *) override;
    // default behavior is fine.
    // virtual void visit(const ir::Sequence *) override;
    virtual void visit(const ir::Allocate *) override;
    virtual void visit(const ir::Store *) override;
    virtual void visit(const ir::Accumulate *) override;
    virtual void visit(const ir::Label *) override;
    // TODO(cgyurgyik): support deallocation.
    RESTRICT_VISITOR(ir::Free);
    RESTRICT_VISITOR(ir::RecLoop);
    RESTRICT_VISITOR(ir::YieldFrom);
    RESTRICT_VISITOR(ir::Match);
    RESTRICT_VISITOR(ir::Yield);
    RESTRICT_VISITOR(ir::Iterate);
    RESTRICT_VISITOR(ir::Scan);
    virtual void visit(const ir::ForAll *) override;
    RESTRICT_VISITOR(ir::ForEach);
    virtual void visit(const ir::Continue *) override;
    virtual void visit(const ir::Launch *) override;
    RESTRICT_VISITOR(ir::QueueWrite);

  private:
    llvm::FunctionType *get_function_type(const ir::Type &type);
    // Recursively creates IR that will print the given expression. This
    // performs exactly one call to C's `printf` with the string `to_print` and
    // the arguments `args`.
    void print_helper(const ir::Expr &expr, std::vector<llvm::Value *> &args,
                      std::string &to_print, uint32_t indent_level = 0);

    // Local state for codegen() impls.
    llvm::Value *value = nullptr;
    llvm::Type *type = nullptr;
    llvm::Function *current_function = nullptr;
    // Used to compile `continue`
    std::vector<llvm::BasicBlock *> latch_blocks;
    // TODO(ajr): will need this for `break` statements.
    // std::vector<llvm::BasicBlock *> escape_blocks;

    // Global LLVM state
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::Module> module;
    std::unique_ptr<llvm::IRBuilder<>> builder;
    llvm::MDNode *very_likely_branch = nullptr;
    // Scope<llvm::Value *> scope;
    ir::MapStack<std::string, llvm::Value *> frames;
    std::map<std::string, llvm::StructType *> struct_types;

    /** Some useful llvm types */
    // @{
    llvm::Type *void_t, *i1_t, *i8_t, *i16_t, *i32_t, *i64_t, *f16_t, *f32_t,
        *f64_t;
    // llvm::StructType *halide_buffer_t_type,
    //     *type_t_type,
    //     *dimension_t_type,
    //     *metadata_t_type,
    //     *argument_t_type,
    //     *scalar_value_t_type,
    //     *device_interface_t_type,
    //     *pseudostack_slot_t_type,
    //     *semaphore_t_type;

    // @}

    llvm::Value *create_aligned_load(llvm::Type *etype, llvm::Value *ptr,
                                     const std::string &name);
    llvm::Value *create_alloca_at_entry(llvm::Type *etype,
                                        const std::string &name,
                                        llvm::Value *size = nullptr);
    llvm::Value *create_malloc(llvm::Type *etype, llvm::Value *size,
                               bool zero_initialize, const std::string &name);

    virtual int native_vector_bits() const {
        // TODO(ajr): override for other targets.
        return 128; // ARM Neon
    }

    bool is_llvm_const_one(llvm::Value *value) const {
        if (auto *constInt = llvm::dyn_cast<llvm::ConstantInt>(value)) {
            return constInt->isOne();
        }
        return false;
    }

    // Used to uniquely label forall loop codegen.
    uint64_t forall_loop_id = 0;
    // Memory type to perform Build<Array_t>s in.
    ir::Allocate::Memory allocate_memory = ir::Allocate::Memory::Heap;
};

std::unique_ptr<llvm::raw_fd_ostream>
make_raw_fd_ostream(const std::string &filename);

} //  namespace bonsai
