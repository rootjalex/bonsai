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

    // Generate a function straight from its SSA form, instead of from the
    // statements the relooper rebuilds out of it (see ir::Program::ssa_funcs).
    //
    // Only the control flow is handled here: an SSA block becomes a basic
    // block and a block argument becomes a phi, which is what LLVM wanted in
    // the first place. Everything inside a block is turned into the same
    // ir::Expr the relooper would have built and handed to codegen_expr, so
    // there is one lowering of an add or a gather, not two.
    void compile_function(const ir::ssa::Function &func,
                          llvm::Function *function);
    struct SSALowering;
    friend struct SSALowering;
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

    /** Give a union the body that stands in for it: the member needing the
     * strictest alignment, then enough bytes to reach the size of the largest.
     * Answers false if a member has no size yet, which is how a union reached
     * while struct bodies are still being built gets left for later. */
    bool set_union_body(const ir::Union_t *node, llvm::StructType *made);

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
    RESTRICT_VISITOR(ir::String_t);
    virtual void visit(const ir::Ptr_t *) override;
    virtual void visit(const ir::Ref_t *) override;
    virtual void visit(const ir::Vector_t *) override;
    virtual void visit(const ir::Array_t *) override;
    virtual void visit(const ir::Struct_t *) override;
    virtual void visit(const ir::Union_t *) override;
    virtual void visit(const ir::Function_t *) override;
    virtual void visit(const ir::Rand_State_t *) override;
    // These should have been lowered already.
    RESTRICT_VISITOR(ir::DynArray_t);
    RESTRICT_VISITOR(ir::Option_t);
    RESTRICT_VISITOR(ir::Tuple_t);
    RESTRICT_VISITOR(ir::Set_t);
    RESTRICT_VISITOR(ir::Generic_t);
    RESTRICT_VISITOR(ir::BVH_t);
    RESTRICT_VISITOR(ir::ADT_t);
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
    virtual void visit(const ir::StringImm *) override;
    virtual void visit(const ir::Extrema *) override;
    virtual void visit(const ir::SizeOf *) override;
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
    // LowerADTs turns this into a Build of the storage the layout chose.
    RESTRICT_VISITOR(ir::Construct);
    virtual void visit(const ir::UnionOf *) override;
    virtual void visit(const ir::Access *) override;
    virtual void visit(const ir::Unwrap *) override;
    virtual void visit(const ir::Intrinsic *) override;
    RESTRICT_VISITOR(ir::Generator);
    virtual void visit(const ir::Lambda *) override;
    virtual void visit(const ir::GeomOp *) override;
    virtual void visit(const ir::SetOp *) override;
    virtual void visit(const ir::AggOp *) override;
    virtual void visit(const ir::Call *) override;
    virtual void visit(const ir::Instantiate *) override;
    virtual void visit(const ir::PtrTo *) override;
    virtual void visit(const ir::Deref *) override;
    virtual void visit(const ir::AtomicAdd *) override;
    // Stmts
    virtual void visit(const ir::CallStmt *) override;
    virtual void visit(const ir::Print *) override;
    virtual void visit(const ir::Return *) override;
    virtual void visit(const ir::LetStmt *) override;
    virtual void visit(const ir::IfElse *) override;
    virtual void visit(const ir::DoWhile *) override;
    virtual void visit(const ir::While *) override;
    // default behavior is fine.
    // virtual void visit(const ir::Sequence *) override;
    virtual void visit(const ir::Allocate *) override;
    virtual void visit(const ir::Store *) override;
    virtual void visit(const ir::Accumulate *) override;
    virtual void visit(const ir::Label *) override;
    virtual void visit(const ir::Append *) override;
    // TODO(cgyurgyik): support deallocation.
    RESTRICT_VISITOR(ir::Free);
    RESTRICT_VISITOR(ir::RecLoop);
    RESTRICT_VISITOR(ir::YieldFrom);
    RESTRICT_VISITOR(ir::Match);
    // LowerADTs turns this into a test of the tag per arm.
    RESTRICT_VISITOR(ir::MatchVariant);
    RESTRICT_VISITOR(ir::Yield);
    RESTRICT_VISITOR(ir::Iterate);
    RESTRICT_VISITOR(ir::Scan);
    virtual void visit(const ir::ForAll *) override;
    RESTRICT_VISITOR(ir::ForEach);
    virtual void visit(const ir::ParFor *) override;
    virtual void visit(const ir::Continue *) override;
    virtual void visit(const ir::Launch *) override;

  private:
    llvm::FunctionType *get_function_type(const ir::Type &type);
    // Recursively creates IR that will print the given expression. This
    // performs exactly one call to C's `printf` with the string `to_print` and
    // the arguments `args`.
    void print_helper(const ir::Expr &expr, std::vector<llvm::Value *> &args,
                      std::string &to_print, uint32_t indent_level = 0);

    // Names the temporary print_helper binds an array to before indexing it,
    // so that nested arrays do not collide (see the Array_t case there).
    uint64_t array_print_counter = 0;

    // Allocates memory for the dynamic array type in Bonsai.
    void allocate_dynamic_array_type(const ir::Allocate *);

    // Ensures the buffer has the capacity for one more element (indicated by
    // `index`), and otherwise growing the buffer.
    void ensure_capacity(ir::Expr ptr, llvm::Value *index,
                         llvm::Value *dynamic_array,
                         const ir::Struct_t *struct_t,
                         llvm::Type *llvm_struct_t, llvm::Value *size_ptr,
                         llvm::Value *capacity_ptr, llvm::Value *mutex,
                         llvm::Type *elt_ty, const std::string &base_n);

    llvm::FunctionCallee get_pthread_lock();
    llvm::FunctionCallee get_pthread_unlock();
    llvm::FunctionCallee get_pthread_init();

    // A call to the libm function `name`, for the maths LLVM has no intrinsic
    // for. Single-precision goes to the `f`-suffixed entry point, as C's
    // overloads do. A vector argument is taken a lane at a time, because libm
    // is scalar; the intrinsics LLVM does have would have been legalised into
    // the same calls.
    llvm::Value *codegen_libm_call(const std::string &name,
                                   const ir::Intrinsic *node);

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
    // The platform being generated for. Not read off the module, which is
    // left without a triple for the LLVM backend so that its output does not
    // depend on the host (see make_target_machine).
    std::string target_triple;
    // Made before anything is generated, so that generated code can ask about
    // the target: how a parallel loop is spelled, and how a struct is laid
    // out for the type-based aliasing below.
    std::unique_ptr<llvm::TargetMachine> target_machine;

    //===------------------------------------------------------------------===//
    // Type-based alias analysis
    //===------------------------------------------------------------------===//
    //
    // Two accesses at unrelated types cannot be to the same memory. This IR
    // reads the same bytes at more than one type in exactly one place -- a
    // tree layout, whose node is read as one variant or another depending on
    // a tag inside it -- and those are left out.
    //
    // Saying so is worth a lot around calls: a traversal that takes the thing
    // it reads and the thing it writes as separate pointers otherwise has to
    // assume a write to one may have changed the other, and re-reads it on
    // every iteration.
    llvm::MDNode *tbaa_root = nullptr;
    std::map<std::string, llvm::MDNode *> tbaa_types;

    // The node describing `type`, or null when nothing can be said about it.
    llvm::MDNode *tbaa_type_node(const ir::Type &type);
    // Tags an access to a whole object of `type`. Sub-object accesses are
    // left untagged, which means "may alias anything" and is always safe.
    void add_tbaa(llvm::Instruction *inst, const ir::Type &type);
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

    llvm::LoadInst *create_aligned_load(llvm::Type *etype, llvm::Value *ptr,
                                        const std::string &name);
    // A per-lane memory operation: `base` addresses the container, `index`
    // holds one index per lane. A Ramp of stride one is contiguous and
    // becomes an ordinary vector load/store; anything else is a
    // gather/scatter over a vector of addresses.
    // `mask`, when defined, is a boolean vector disabling the lanes that must
    // not touch memory.
    llvm::Value *create_vector_load(llvm::Type *etype, llvm::Value *base,
                                    const ir::Expr &index, uint32_t lanes,
                                    const ir::Expr &mask,
                                    const std::string &name);
    void create_vector_store(llvm::Value *value, llvm::Type *etype,
                             llvm::Value *base, const ir::Expr &index,
                             uint32_t lanes, const ir::Expr &mask);
    llvm::Value *create_alloca_at_entry(llvm::Type *etype,
                                        const std::string &name,
                                        llvm::Value *size = nullptr);
    llvm::Value *materialize_for_address(llvm::Value *pointee,
                                         const std::string &name);
    // Built on demand rather than up front like the structs, since a union is
    // named by the type that holds it rather than declared on its own.
    std::map<std::string, llvm::StructType *> union_types;
    // The unions reached before their members had a size, waiting for one.
    std::vector<const ir::Union_t *> pending_unions;
    // One counted loop, shared by every statement that is one: `ForAll`, and
    // a `ParFor` that no schedule placed on any hardware.
    void codegen_counted_loop(const std::string &index,
                              const ir::Expr &begin_expr,
                              const ir::Expr &end_expr,
                              const ir::Expr &stride_expr,
                              const ir::Stmt &body);
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
    // `--no-heap`: refuse to emit a heap allocation. See CompilerOptions.
    bool no_heap = false;
};

std::unique_ptr<llvm::raw_fd_ostream>
make_raw_fd_ostream(const std::string &filename);

} //  namespace bonsai
