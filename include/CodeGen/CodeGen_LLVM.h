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

#include <llvm/Analysis/TargetLibraryInfo.h>

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <set>

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

    // The library information the optimizer and the backend work from: libm,
    // and -- when following the host -- the vector math of the host's
    // libmvec, mapped onto LLVM's vector intrinsics so that
    // `llvm.sin.v8f32` becomes one call of `_ZGVdN8v_sinf` rather than eight
    // of `sinf` (LLVM's ReplaceWithVeclib pass does the replacing). Given to
    // every pass manager that runs over the module, so the two agree.
    llvm::TargetLibraryInfoImpl target_library_info(const llvm::Triple &triple);

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
    virtual void visit(const ir::ElementRef_t *) override;
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
    virtual void visit(const ir::Shuffle *) override;
    virtual void visit(const ir::Ramp *) override;
    virtual void visit(const ir::Extract *) override;
    virtual void visit(const ir::Build *) override;
    // LowerADTs turns this into a Build of the storage the layout chose.
    RESTRICT_VISITOR(ir::Construct);
    virtual void visit(const ir::UnionOf *) override;
    virtual void visit(const ir::Access *) override;
    virtual void visit(const ir::Unwrap *) override;
    // LowerADTs turns this into a branch per variant.
    RESTRICT_VISITOR(ir::MatchExpr);
    virtual void visit(const ir::Intrinsic *) override;
    RESTRICT_VISITOR(ir::Generator);
    virtual void visit(const ir::Lambda *) override;
    virtual void visit(const ir::GeomOp *) override;
    virtual void visit(const ir::SetOp *) override;
    virtual void visit(const ir::AggOp *) override;
    virtual void visit(const ir::Call *) override;
    virtual void visit(const ir::Instantiate *) override;
    virtual void visit(const ir::PtrTo *) override;
    virtual void visit(const ir::RefTo *) override;
    virtual void visit(const ir::Deref *) override;
    virtual void visit(const ir::AtomicAdd *) override;
    // Stmts
    virtual void visit(const ir::CallStmt *) override;
    virtual void visit(const ir::MultiRecurse *) override;
    virtual void visit(const ir::Print *) override;
    virtual void visit(const ir::Return *) override;
    virtual void visit(const ir::LetStmt *) override;
    virtual void visit(const ir::IfElse *) override;
    virtual void visit(const ir::SwitchStmt *) override;
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

    // The aggregate a function returns through a hidden pointer argument
    // instead of in registers, or null when it returns directly.
    //
    // A first-class aggregate return is legalised by handing each leaf its own
    // register: on x86-64 that is XMM0 and XMM1 for floating-point and vector
    // leaves and EAX, EDX and ECX for integer ones, and the third
    // floating-point leaf goes on the *x87 stack* -- an `fstps` on the way out
    // and an `flds` on the way in, four billion of them in a render of the
    // pavilion, and the FP scheduler stalling around them. C++ never sees
    // this because its ABI returns anything over two eightbytes through a
    // pointer the caller provides. So does this, for exactly the aggregates
    // the return registers cannot hold; the ones they can -- `option[i32]`, a
    // pair of floats -- keep the register return, which is cheaper than a
    // store and a load.
    llvm::Type *indirect_return_type(const ir::Type &ret_type);

    // Between a vector's packed storage (an array aggregate) and the vector
    // the program computes with; see ir::Vector_t::packed.
    llvm::Value *unpack_vector(llvm::Value *packed, const ir::Vector_t *type);
    llvm::Value *pack_vector(llvm::Value *vector, const ir::Vector_t *type);

    //===------------------------------------------------------------------===//
    // Shuffles
    //===------------------------------------------------------------------===//
    //
    // How an ir::Shuffle is lowered, following Halide's CodeGen_LLVM (see
    // ir::Shuffle). The interleave and its inverse are the ones that take
    // thought: LLVM's own lowering of a wide interleave built from two-way
    // shuffles is poor, so they are decomposed here, after Catanzaro, Keller
    // and Garland, "A Decomposition for In-place Matrix Transposition",
    // PPoPP 2014, and Halide PR 8925 ("Better vector interleaves"). A target
    // with a better answer for particular shapes -- x86's unpck and
    // immediate-controlled shuffles -- overrides these.

    // Lanes `indices` picks out of `a` then `b`; a negative index is a lane
    // whose value does not matter. The two may differ in width.
    llvm::Value *shuffle_vectors(llvm::Value *a, llvm::Value *b,
                                 const std::vector<int> &indices);
    llvm::Value *shuffle_vectors(llvm::Value *a,
                                 const std::vector<int> &indices);
    // Lanes [start, start + size) of `v`; lanes past its end do not matter.
    llvm::Value *slice_vector(llvm::Value *v, int start, int size);
    // The vectors end to end.
    llvm::Value *concat_vectors(const std::vector<llvm::Value *> &vs);
    // a0 b0 c0 a1 b1 c1 .. from a, b, c of one width.
    virtual llvm::Value *interleave_vectors(const std::vector<llvm::Value *> &vs);
    // The inverse: `count` vectors, the k-th holding lanes k, k + count, ..
    virtual std::vector<llvm::Value *> deinterleave_vector(llvm::Value *v,
                                                           int count);
    // Keeps LLVM from re-fusing shuffles on either side of it into a worse
    // sequence; see Halide's optimization_fence.
    llvm::Value *optimization_fence(llvm::Value *v);
    // A scalar as a one-lane vector, so that it can be shuffled.
    llvm::Value *as_vector(llvm::Value *v);

    //===------------------------------------------------------------------===//
    // Unions, word by word
    //===------------------------------------------------------------------===//
    //
    // A vectorized union is the union's 32-bit words, each one gang vector
    // (see ir::widen). A member's field lies at the same offset in every
    // lane, so reading a member of every lane at once is reading its fields
    // out of the words -- a bitcast for a word-sized field, a shift for a
    // narrower one, two words for a wider -- and making every lane's union
    // from one member value per lane is putting them in. Nothing is
    // transposed; the shuffle network above is for the per-lane
    // reinterpretation of one aggregate as another (see the Cast visitor),
    // which the same unit-wise helpers do at whatever unit divides both.

    // `v`, whose storage is the size of `as`, read as `as`: through a stack
    // slot, which is how a value of one aggregate type is read at another.
    llvm::Value *reinterpret_via_memory(llvm::Value *v, llvm::Type *as);
    // The unit the per-lane reinterpretation of an aggregate works in, in
    // bytes: the largest that divides every field's size and offset.
    uint64_t union_unit(const ir::Type &member, uint64_t offset,
                        uint64_t so_far);
    // Every lane's `member` value, held gang-wide as `wide` (of type
    // ir::widen(member)), spread into unit vectors at byte offset `offset`:
    // `slots[u]` is the vector of every lane's unit `u`. A field narrower than
    // a unit is shifted into its place in the unit it shares. With a unit of
    // four this is how a member goes into a widened union's words (see
    // ir::widen), and no transpose is involved: a field is a word, or a part
    // of one, or two.
    void scatter_units(const ir::Type &member, llvm::Value *wide,
                       uint64_t offset, uint64_t unit,
                       std::vector<llvm::Value *> &slots);
    // The inverse: every lane's `member`, gang-wide, from the unit vectors.
    llvm::Value *gather_units(const ir::Type &member, const ir::Type &wide_t,
                              uint64_t offset, uint64_t unit,
                              const std::vector<llvm::Value *> &slots,
                              uint32_t lanes);
    // That the IR's count of a widened union's words (ir::layout_bytes)
    // agrees with this target's layout of the union.
    void check_union_words(const ir::Union_t &as_union,
                           const ir::Struct_t &words);

    // The high N bits of the 2N-bit product of two N-bit integers (the mulhi
    // intrinsic), signed or unsigned as `is_signed` says, for a scalar or a
    // vector of them.
    llvm::Value *multiply_high(llvm::Value *a, llvm::Value *b, bool is_signed,
                               const std::string &name);
    // The multiplier a division by `d` becomes a multiply-high by (the
    // div_multiplier intrinsic; see SSA/InvariantDivision.h), for a scalar
    // or a vector of divisors. Defined for every `d`, zero included.
    llvm::Value *division_multiplier(llvm::Value *d, bool is_signed,
                                     const std::string &name);

    // The address of one element per lane of an array: `base` plus each
    // lane's index, scaled, in 32-bit addressing (ISPC's model).
    llvm::Value *element_addresses(llvm::Type *element, llvm::Value *base,
                                   llvm::Value *indices,
                                   const std::string &name);
    // One value of `element` per lane, each at `base` plus that lane's 32-bit
    // byte offset in `offsets` plus `disp`, gathered field by field into the
    // gang-wide form `wide_t` (ir::widen(element)); `mask` says which lanes
    // load, or is null for all of them.
    llvm::Value *gather_elements(const ir::Type &element, const ir::Type &wide_t,
                                 llvm::Value *base, llvm::Value *offsets,
                                 uint64_t disp, uint32_t lanes,
                                 llvm::Value *mask, const std::string &name);

    // An alloca at the top of the current function's entry block: where a
    // callee returning through a hidden pointer puts its result. In the entry
    // block so that it is one slot however often the call runs, and so that
    // SROA can take it apart once the callee is inlined.
    llvm::AllocaInst *create_entry_alloca(llvm::Type *type,
                                          const std::string &name);

    // A call, with the hidden return pointer threaded through when the
    // callee returns through one; yields the returned value either way.
    llvm::Value *emit_call(llvm::FunctionCallee callee, llvm::Type *sret_type,
                           std::vector<llvm::Value *> args,
                           const std::string &name = "");
    // The same, reading whether there is a hidden pointer off the callee's
    // own declaration.
    llvm::Value *emit_call(llvm::Function *callee,
                           std::vector<llvm::Value *> args,
                           const std::string &name = "");

    // The hidden return pointer of the function being compiled, or null when
    // it returns in registers.
    llvm::Value *current_sret = nullptr;
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
    // overloads do. A vector argument goes to the host libmvec's vector
    // entry point when it has one (see probe_host_vector_math), and a lane
    // at a time otherwise, because libm is scalar.
    llvm::Value *codegen_libm_call(const std::string &name,
                                   const ir::Intrinsic *node);

    // What the host's libmvec provides, found by asking it. Every symbol
    // here exists in this machine's libmvec and is of an ISA level this
    // machine runs.
    std::set<std::string> host_vector_math;
    void probe_host_vector_math();
    // The libmvec entry point for `name` on `lanes` lanes of `bits`-bit
    // floats with `arity` vector arguments, if the host has one: the Vector
    // Function ABI name, `_ZGVdN8v_sinf` for sinf on eight lanes for AVX2.
    std::optional<std::string> vector_math_symbol(const std::string &name,
                                                  uint32_t lanes, unsigned bits,
                                                  unsigned arity) const;
    // The strings a TargetLibraryInfoImpl's vector descriptions refer to,
    // which have to outlive it (they are StringRefs there).
    std::list<std::string> vector_math_names;

    // Local state for codegen() impls.
    llvm::Value *value = nullptr;
    llvm::Type *type = nullptr;
    llvm::Function *current_function = nullptr;
    // Set while a function is being lowered straight from SSA -- the vectorized
    // ones, and the parfor kernels they contain, whose stacks are realigned and
    // whose aggregate locals are over-aligned to match the wide vector stores
    // LLVM may initialize them with (see create_alloca_at_entry). Off for the
    // functions the relooper still hands over, so their allocas -- and the
    // golden IR that pins them -- are unchanged.
    bool lowering_from_ssa = false;

    // Whether the target machine is the host's, features and all, rather
    // than one `--mcpu` or `--triple` named. Following the host is what turns
    // on `prefer-vector-width=256` (see optimize_module).
    bool follows_host = false;
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
    // A predicated store through one shared address, `dest`. What is stored
    // decides what that means: a gang-wide vector, or an aggregate of them,
    // is memory laid out with a slot per lane -- a `mut` local every lane has
    // its own copy of -- and each lane's slot is written under its own bit of
    // the mask; a scalar is memory the lanes share, and is written once if
    // any lane is on.
    void create_masked_store_at(llvm::Value *value, llvm::Value *dest,
                                llvm::Value *mask);
    // A select that also takes a vector condition over two aggregates of
    // gang-wide fields, choosing field by field.
    llvm::Value *create_select(llvm::Value *cond, llvm::Value *tvalue,
                               llvm::Value *fvalue);
    // Emits `body` under a branch taken if any lane of `mask` is on: what a
    // gang does with a side effect on memory the lanes share, which is made
    // once, provided some lane would have made it.
    void emit_if_any_lane(llvm::Value *mask, const std::function<void()> &body);
    // `body`, under a branch on the scalar `cond`.
    void emit_if(llvm::Value *cond, const std::function<void()> &body);
    // The indivisible read-modify-write an accumulate of `op` on values of
    // type `value_t` is, or an error for one there is no such instruction for.
    llvm::AtomicRMWInst::BinOp atomic_rmw_op(ir::Accumulate::OpType op,
                                             const ir::Type &value_t);
    // One indivisible read-modify-write per lane that is on: lane k's value
    // into lane k's place. `ptrs` is one address per lane (a scatter) or the
    // one address of memory laid out per lane, in which case lane k's place
    // is element k of the vector there. LLVM has no masked atomic scatter, so
    // this is a branch and an atomicrmw per lane.
    void emit_atomic_lanes(ir::Accumulate::OpType op, const ir::Type &value_t,
                           llvm::Value *ptrs, llvm::Value *values,
                           llvm::Value *mask);
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
