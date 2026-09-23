#pragma once

/** \file
 *
 * Defines the base class for every code generator that goes through LLVM.
 *
 * The division follows Halide's CodeGen_LLVM and its per-target subclasses.
 * CodeGen_LLVM holds the lowering that is the same on every machine: what a
 * type, an expression, a block graph and a statement become in LLVM IR, how a
 * gather is spelled in LLVM's generic form, how an aggregate is read at
 * another type, how a division by an invariant becomes multiplies. Anything
 * that is a fact about one machine is a virtual method with a target-agnostic
 * default, and the subclass for that machine overrides it:
 *
 *   - CodeGen_X86 (the host): the return-register budget that decides which
 *     aggregates come back through a hidden pointer, the register width a
 *     gang fills and a stack slot is aligned to, the machine's own gather
 *     instructions, a vector of integers divided through doubles, and glibc's
 *     libmvec for a vector's transcendental functions.
 *   - CodeGen_GPU_Host<CodeGen_CPU> (the host of a program with loops on
 *     the GPU), layered over whichever CPU generator the machine has, as
 *     Halide's CodeGen_GPU_Host is: it owns the device generator, hands it
 *     the bound loops as kernels, embeds the PTX and emits the launches.
 *   - CodeGen_PTX (the device): a kernel's entry and its thread indices,
 *     libdevice for maths, vprintf for print, and no thread runtime at all.
 *
 * This class knows nothing of devices. What it does know about a parfor a
 * schedule bound to hardware is the one binding every CPU host answers the
 * same way -- CPUThread is a call into the parallel runtime, as Halide's
 * base handles a parallel loop -- and for the rest it has a virtual the
 * GPU host and the device each answer for themselves.
 */

#include "CompilerOptions.h"
#include "IR/Frame.h"
#include "IR/Function.h"
#include "IR/Program.h"
#include "IR/Visitor.h"
#include "LLVMIncl.h"
#include "SSA/SSA.h"
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

struct CodeGen_LLVM;

// The code generator for the machine `options` name, or the host's when they
// name none: the target-agnostic CodeGen_LLVM, or the subclass that knows the
// target's own instructions where LLVM's generic form loses something
// (CodeGen_X86); and, for a program whose schedule binds a loop to the GPU,
// the GPU host layered over that (CodeGen_GPU_Host), which is what Halide
// does for a target with a GPU feature. Every backend that generates LLVM
// goes through this, so that they all agree on what a program compiles to.
std::unique_ptr<CodeGen_LLVM> make_llvm_codegen(const ir::Program &program,
                                                const CompilerOptions &options);

struct CodeGen_LLVM : public ir::Visitor {
    CodeGen_LLVM();
    virtual ~CodeGen_LLVM() = default;

    /** Takes a bonsai Program and compiles it to an llvm Module. */
    virtual std::unique_ptr<llvm::Module>
    compile_program(const ir::Program &program, const CompilerOptions &options);

    std::unique_ptr<llvm::LLVMContext> steal_context() {
        return std::move(context);
    }

    // A parfor a schedule bound to a hardware resource, as the code generator
    // that has to place it sees it: the loop, the block its body starts at
    // (whose arguments are the loop's index first and then the values the
    // body reads from outside -- the closure, which the SSA form has already
    // computed), and three things the lowering of the enclosing function
    // lends it. `operand` reads an SSA value of the enclosing function as an
    // expression of whatever already holds it; `bind` names a value for the
    // body's blocks; `emit_body` emits the body's blocks into the current
    // function starting at the given block, with a `Yield` in them branching
    // to the second block, or returning when it is null. Everything the body
    // needs from the parent must be bound before emit_body is called.
    struct BoundLoop {
        const ir::ssa::Function &func;
        const ir::ssa::Terminator::ParFor &loop;
        const ir::ssa::Block &body_head;
        std::function<ir::Expr(const std::shared_ptr<ir::ssa::Value> &)> operand;
        std::function<void(const std::string &, llvm::Value *)> bind;
        std::function<void(llvm::BasicBlock *, llvm::BasicBlock *)> emit_body;
    };

    // Creates a target machine and updates the module's backend and data
    // layout. The machine `options` name, or the host's when they name none;
    // a device code generator overrides this with its own machine.
    virtual std::unique_ptr<llvm::TargetMachine>
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

    // Passes of the backend's own, registered with the pipeline's extension
    // points before optimize_module builds it: none here; the PTX
    // generator's marking of device memory.
    virtual void register_backend_passes(llvm::PassBuilder &) {}
    // What `--fast-math` puts on every float operation the builder makes:
    // all of LLVM's flags here, which is clang's -ffast-math; nothing on the
    // device, whose fast math is nvcc's and is chosen another way (see
    // CompilerOptions::fast_math and CodeGen_PTX).
    virtual llvm::FastMathFlags fast_math_flags();
    virtual void optimize_module(llvm::TargetMachine &tm,
                                 const CompilerOptions &options);
    // Called by compile_program once every function is generated and before
    // the module is optimized: where a subclass finishes whatever it built
    // alongside the functions. The GPU host finishes its device module here
    // and embeds the PTX the launches refer to.
    virtual void end_functions() {}

    llvm::Function *declare_function(const ir::Function &func);
    void compile_function(const ir::Function &func, llvm::Function *function);
    // The random generator's state for a function that seeds it (see
    // Lower/Random.h): seeded from C's rand(), held in a stack slot bound
    // under lower::rng_state_name, at the entry of the function being
    // compiled.
    void emit_rng_setup();
    // The generator's state for one iteration of a loop bound to hardware
    // threads, at the entry of the iteration's body. The state is
    // thread-specific -- Lower/Random.cpp's rule, and what the parallel
    // runtime and the GPU both need, since the function's own state is one
    // stack slot that every thread would otherwise share, and on the device
    // is not even addressable -- so each iteration makes its own, seeded
    // from its `index` as Halide seeds random_int from the loop variables:
    // rng32(index) for the first lane and the lanes after it in order, and,
    // when the loop is inside another bound loop whose state is
    // `outer_state`, rng32(outer's first lane + index), so that the threads
    // of two blocks with the same thread number draw different numbers.
    void emit_rng_setup(llvm::Value *index, llvm::Value *outer_state);
    // Both of the above: `state`, a vector of one seed per lane, into a stack
    // slot bound under lower::rng_state_name -- in the innermost frame,
    // shadowing an enclosing function's state, when `shadows`; as a new
    // name otherwise.
    void bind_rng_state(llvm::Value *state, bool shadows);
    // Which of a bound loop body's arguments after its index a launch hands
    // over: all of them but the generator's state, which is made afresh by
    // emit_rng_setup(index) in the body's prologue instead. Positions into
    // `body_head.args`; position i is passed the loop's body.args[i - 1].
    static std::vector<size_t> launch_captures(const ir::ssa::Block &body_head);
    static bool seeds_rng(const ir::ssa::Block &body_head);

    //===------------------------------------------------------------------===//
    // Buffers: an exported function's arrays
    //===------------------------------------------------------------------===//
    //
    // An exported function's array parameters arrive as `bonsai_buffer`
    // descriptors (runtime/bonsai_buffer.h): where the bytes are on each side
    // and which copy is current. The function's prologue asks for each array
    // it uses on the host (`bonsai_buffer_require`), which is a flag test
    // when the array is already there and a copy when it is not, and marks
    // the ones it may write as dirty on the host; a launch asks for the ones
    // its kernel reads on the device and marks them dirty there after (see
    // CodeGen_GPU_Host::emit_gpu_launch). The program's own calls never see a
    // descriptor: they go to the function's internal twin (see
    // Lower/ReturnToOutParameter.h).

    // What the prologue learnt about one buffer-bearing parameter of the
    // exported function being compiled: its descriptor, whether the
    // function may write it, and whether any host-side code of the function
    // reads or writes it (as against only a kernel).
    //
    // Two kinds of parameter carry buffers. An array is one descriptor. A
    // layout struct -- a tree's `_tree_layout`, a struct the layout language
    // built whose fields are counts and arrays -- arrives with each of its
    // array fields a descriptor (`layout` names the struct), and is read
    // into the form the program uses, with each array's pointer for the side
    // in question, by unwrap_layout.
    struct ExportedBuffer {
        llvm::Value *descriptor = nullptr;
        ir::Type type;
        bool mutating = false;
        bool host_used = false;
        const ir::Struct_t *layout = nullptr;
    };
    // By parameter name, for the function being compiled.
    std::map<std::string, ExportedBuffer> exported_buffers;
    // `bonsai_buffer_require(descriptor, side)`: the pointer for that side.
    llvm::Value *buffer_require(llvm::Value *descriptor, bool device);
    // `bonsai_buffer_mark_dirty(descriptor, side)`.
    void buffer_mark_dirty(llvm::Value *descriptor, bool device);
    // A layout struct as the driver handed it over, read into a struct of the
    // program's form in a stack slot of the current function: every count as
    // it is, every array field's descriptor asked for on `device`'s side,
    // and, if `mark_dirty`, marked dirty there (the function may write it).
    // `boundary` points at the driver's struct.
    llvm::Value *unwrap_layout(llvm::Value *boundary, const ir::Struct_t *layout,
                               bool device, bool mark_dirty,
                               const std::string &name);
    // Every array field's descriptor of the layout at `boundary`.
    std::vector<llvm::Value *> layout_descriptors(llvm::Value *boundary,
                                                  const ir::Struct_t *layout);

  public:
    // Per exported function, the side each buffer of a call is needed on, in
    // parameter order -- an array parameter one entry, a layout struct one
    // per array field: BONSAI_HOST (1), BONSAI_DEVICE (2), or both. What the
    // generated header prints as `<function>_sides`, for a driver that stages
    // its buffers before a timed call (bonsai_buffer_stage_all).
    const std::map<std::string, std::vector<uint8_t>> &exported_sides() const {
        return sides_of_exported;
    }
    // The struct a parameter of `type` (a struct, or a pointer to one) is a
    // layout of -- one with a dynamically sized array among its fields, as
    // the layout language builds them -- or null. What the C boundary hands
    // over with a buffer descriptor per array field.
    static const ir::Struct_t *layout_struct_of(const ir::Type &type);
    // The number of array fields of `layout`: how many descriptors it holds.
    static size_t layout_buffer_count(const ir::Struct_t *layout);

  protected:
    std::map<std::string, std::vector<uint8_t>> sides_of_exported;

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

    // The body of `loop`, a parfor of `func`, as the whole of `kernel`: the
    // blocks the body reaches become the kernel's blocks, with `prologue`
    // called first to bind the body's arguments -- its index, from wherever
    // the kernel gets one, and its captures, from wherever the kernel was
    // handed them. This is how a parfor bound to threads becomes a function
    // the thread runtime calls, and how one bound to the GPU becomes a
    // kernel in the device module (CodeGen_PTX::add_kernel).
    void compile_kernel_body(const ir::ssa::Function &func,
                             const ir::ssa::Terminator::ParFor &loop,
                             llvm::Function *kernel,
                             const std::function<void(BoundLoop &)> &prologue);

    // What a parfor the schedule bound to hardware becomes here. A parfor
    // bound to CPUThread is the body as a function and one call into the
    // parallel runtime (runtime/bonsai_parallel.h), on every CPU host alike;
    // any other binding is an error, since this generator has no such
    // hardware. CodeGen_GPU_Host overrides this for the GPU bindings with a
    // kernel and its launch, and CodeGen_PTX, being the device, with what a
    // bound loop is inside a kernel.
    virtual void emit_bound_parfor(BoundLoop &loop);
    void emit_cpu_parfor(BoundLoop &loop);
    // How many iterations a loop has, `(end - begin + stride - 1) / stride`,
    // in the loop's own index type.
    static ir::Expr trip_count(const ir::Expr &begin, const ir::Expr &end,
                               const ir::Expr &stride);

    // The condition under which this thread performs a side effect -- a
    // store, an accumulate, a print -- or null when every thread does. Null
    // everywhere but inside a GPU kernel at the block level: the body of a
    // loop bound to GPUBlock is run by every thread of the block, since a
    // block is only its threads, and an effect outside the thread loop is
    // meant once per block, so one thread makes it (see CodeGen_PTX).
    virtual llvm::Value *effects_once_guard() { return nullptr; }
    // Whether effects_once_guard would answer with a guard, without
    // emitting one: for refusing what cannot go under a guard.
    virtual bool effects_run_once() const { return false; }
    // A call made where effects_once_guard applies: its callee must have no
    // effects, or its result would exist in one thread only. Checked by the
    // device; nothing to check on a host.
    virtual void check_block_level_call(const std::string &callee) {}
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
    RESTRICT_VISITOR(ir::String_t);
    virtual void visit(const ir::Ptr_t *) override;
    virtual void visit(const ir::Ref_t *) override;
    virtual void visit(const ir::ElementRef_t *) override;
    virtual void visit(const ir::Vector_t *) override;
    virtual void visit(const ir::Array_t *) override;
    virtual void visit(const ir::Struct_t *) override;
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
    virtual void visit(const ir::Undef *) override;
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

    // Protected rather than private: a target's subclass (CodeGen_X86) emits
    // through the same builder, types and target machine.
  protected:
    llvm::FunctionType *get_function_type(const ir::Type &type);

    // The aggregate a function returns through a hidden pointer argument
    // instead of in registers, or null when it returns directly.
    //
    // Which aggregates a target can return in registers is that target's
    // fact -- x86-64 has a budget of return registers and puts anything over
    // it on the x87 stack (see CodeGen_X86) -- so the target-agnostic answer
    // is that every value is returned directly, which is what LLVM's own
    // legalisation makes correct on every target, and a target whose
    // legalisation is *slow* for some aggregates says which.
    virtual llvm::Type *indirect_return_type(const ir::Type &ret_type) {
        return nullptr;
    }

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
    // Per-lane aggregates, unit by unit
    //===------------------------------------------------------------------===//
    //
    // One aggregate per lane read at another type -- a tree's node, gathered
    // as a struct of gang-wide fields, read as the struct of the arm its tag
    // says it is (see the Cast visitor). Per lane it is that lane's bytes at
    // the other type, so each lane's fields are spread into vectors of
    // fixed-size units as the source lays them out and gathered back as the
    // destination does. A field lies at the same offset in every lane, so a
    // unit-sized field is a bitcast of one unit vector, a narrower one a
    // shift out of the unit it shares, and a wider one several units
    // interleaved; the shuffle network above does the interleaving.

    // `v`, whose storage is the size of `as`, read as `as`: through a stack
    // slot, which is how a value of one aggregate type is read at another.
    llvm::Value *reinterpret_via_memory(llvm::Value *v, llvm::Type *as);
    // The unit the per-lane reinterpretation of an aggregate works in, in
    // bytes: the largest that divides every field's size and offset.
    uint64_t common_unit(const ir::Type &member, uint64_t offset,
                        uint64_t so_far);
    // Every lane's `member` value, held gang-wide as `wide` (of type
    // ir::widen(member)), spread into unit vectors at byte offset `offset`:
    // `slots[u]` is the vector of every lane's unit `u`. A field narrower than
    // a unit is shifted into its place in the unit it shares.
    void scatter_units(const ir::Type &member, llvm::Value *wide,
                       uint64_t offset, uint64_t unit,
                       std::vector<llvm::Value *> &slots);
    // The inverse: every lane's `member`, gang-wide, from the unit vectors.
    llvm::Value *gather_units(const ir::Type &member, const ir::Type &wide_t,
                              uint64_t offset, uint64_t unit,
                              const std::vector<llvm::Value *> &slots,
                              uint32_t lanes);

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
    // A vector of integer lanes divided by another, in whatever form the
    // target divides a vector fastest, or nothing to leave it the integer
    // division LLVM legalises. Nothing here: a machine that divides a vector
    // of integers slowly says so itself (see CodeGen_X86, which divides
    // through doubles), and a GPU lane is a scalar and never asks.
    virtual llvm::Value *vector_int_division(llvm::Value *a, llvm::Value *b,
                                             bool is_signed, bool remainder) {
        return nullptr;
    }

    // The address of one element per lane of an array: `base` plus each
    // lane's index, scaled, in 32-bit addressing (ISPC's model).
    llvm::Value *element_addresses(llvm::Type *element, llvm::Value *base,
                                   llvm::Value *indices,
                                   const std::string &name);
    // The whole element a gather reads fields out of: how many bytes each
    // lane's element occupies and how its start is aligned. What decides
    // whether a byte or halfword field can be read as part of the aligned
    // word that holds it (see gather_elements).
    struct GatheredElement {
        uint64_t bytes;
        uint64_t align;
    };
    // One value of `element` per lane, each at `base` plus that lane's 32-bit
    // byte offset in `offsets` plus `disp`, gathered field by field into the
    // gang-wide form `wide_t` (ir::widen(element)); `mask` says which lanes
    // load, or is null for all of them. `whole` is the outermost element the
    // offsets point at, which the recursion carries down unchanged.
    llvm::Value *gather_elements(const ir::Type &element, const ir::Type &wide_t,
                                 llvm::Value *base, llvm::Value *offsets,
                                 uint64_t disp, const GatheredElement &whole,
                                 uint32_t lanes, llvm::Value *mask,
                                 const std::string &name);
  protected:
    // One element per lane at `base + disp + indices[lane] * scale`, for a
    // scale the machine's addressing has (1, 2, 4 or 8) or any other: the
    // gather every per-lane read comes down to, whether of a field of each
    // lane's element (`gather_words`, scale one) or of an element of an
    // array (`create_vector_load`, the element's size). `align` is what the
    // addresses are known to be aligned to. Virtual because this is the one
    // place a target's own gather instruction is worth naming (see
    // CodeGen_X86); this emits LLVM's masked gather through a vector of
    // addresses, which every target can lower.
    virtual llvm::Value *gather_indexed(llvm::Type *elem_llvm, llvm::Value *base,
                                        llvm::Value *indices, uint64_t scale,
                                        uint64_t disp, uint64_t align,
                                        uint32_t lanes, llvm::Value *mask,
                                        const std::string &name);
    // One 32-bit word per lane at `base` plus that lane's byte offset in
    // `offsets` (32-bit) plus `disp`: gather_indexed at a scale of one.
    llvm::Value *gather_words(llvm::Type *elem_llvm, llvm::Value *base,
                              llvm::Value *offsets, uint64_t disp,
                              uint64_t align, uint32_t lanes, llvm::Value *mask,
                              const std::string &name);
    // Whether the machine being compiled for has `feature` -- "avx2",
    // "avx512vl" -- as the target machine made for the module says.
    bool has_feature(const std::string &feature) const;

    // A sub-word element -- a byte or a halfword -- from each lane's 32-bit
    // byte offset into an array of them, read as part of the aligned word
    // that holds it. `total_bytes` is the array's size in bytes, which the
    // caller has established is at least a word, so that the word can be
    // kept inside the array at its end.
    llvm::Value *gather_sub_word_elements(llvm::Type *etype, llvm::Value *base,
                                          llvm::Value *offsets,
                                          llvm::Value *total_bytes,
                                          uint32_t lanes, llvm::Value *mask,
                                          const std::string &name);

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

    // The mutex a dynamic array grows under: pthread's, on a host with a C
    // runtime. A device has no threads to lock against in this sense, and
    // its code generator says so where these are reached.
    virtual llvm::FunctionCallee get_pthread_lock();
    virtual llvm::FunctionCallee get_pthread_unlock();
    virtual llvm::FunctionCallee get_pthread_init();

    // The C runtime's printf, called with `format` and `args` as C's variadic
    // call passes them (the promotions already applied by print_helper). A
    // device has vprintf, which takes the arguments packed in memory, and
    // overrides this.
    virtual llvm::Value *emit_printf(llvm::Value *format,
                                     const std::vector<llvm::Value *> &args);

    // A libm-family intrinsic -- sin, exp, pow, atan2 -- for a target with a
    // maths library of its own, or nothing to leave it to LLVM's intrinsics
    // and libm, which is what a CPU host has. CodeGen_PTX answers with
    // libdevice, since NVPTX cannot lower `llvm.sin.f32` by itself.
    virtual llvm::Value *codegen_math_call(const std::string &name,
                                           const ir::Intrinsic *node) {
        return nullptr;
    }
    // `tex_sample_grad_2d`: the texture units' sample, which only a device
    // generator has (CodeGen_PTX). Here an error: the program reaches the
    // intrinsic through a function a schedule bound to TextureUnit, and a
    // schedule that does so for code this generator runs has bound it to
    // hardware this machine lacks.
    virtual llvm::Value *codegen_texture_sample(const ir::Intrinsic *node);

    // A call to the libm function `name`, for the maths LLVM has no intrinsic
    // for. Single-precision goes to the `f`-suffixed entry point, as C's
    // overloads do. A vector argument goes to the host libmvec's vector
    // entry point when it has one (see vector_math_symbol), and a lane
    // at a time otherwise, because libm is scalar.
    llvm::Value *codegen_libm_call(const std::string &name,
                                   const ir::Intrinsic *node);
    // A call to libm's `name` on scalar arguments -- `sinf` for floats,
    // `sin` for doubles -- for a lane the vector maths hands back (see
    // CodeGen/VectorMath.h).
    llvm::Value *scalar_math_call(const std::string &name,
                                  llvm::ArrayRef<llvm::Value *> args);

    // Whatever a target has to find out about the machine it is following
    // before code is generated for it; called by make_target_machine when
    // the target is the host's own. CodeGen_X86 asks glibc's libmvec what
    // it provides; nothing is known to be true of every host.
    virtual void probe_host_libraries() {}
    // The vector maths library's entry point for `name` on `lanes` lanes of
    // `bits`-bit floats with `arity` vector arguments, if the target has
    // one. Nothing, on a target without such a library; see CodeGen_X86 for
    // libmvec's `_ZGVdN8v_sinf`.
    virtual std::optional<std::string>
    vector_math_symbol(const std::string &name, uint32_t lanes, unsigned bits,
                       unsigned arity) const {
        return std::nullopt;
    }
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
    // `length`, when known, is how many elements the array holds: what lets a
    // byte or halfword element be gathered as part of a whole word without
    // reading past the array's end (see gather_sub_word_elements).
    llvm::Value *create_vector_load(llvm::Type *etype, llvm::Value *base,
                                    const ir::Expr &index, uint32_t lanes,
                                    const ir::Expr &mask,
                                    const std::string &name,
                                    llvm::Value *length = nullptr);
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
    // A compacting store: `dest` is one address, the first of a run of
    // slots, and the lanes `mask` has on write their values -- a gang-wide
    // vector, or a uniform value every lane writes a copy of -- into
    // consecutive slots from it, in lane order (LLVM's masked.compressstore,
    // `vpcompressd` to memory). What a gang's push writes each field of its
    // entries with (see lower_pushes in SSA/Defer.cpp).
    void create_compress_store_at(llvm::Value *value, llvm::Value *dest,
                                  llvm::Value *mask);

    // A scatter: `ptrs` is one address per lane, at each of which memory
    // holds a `pointee`, and `value` is what the lanes write there -- a
    // gang-wide vector, or an aggregate of them where the pointee is one, or
    // a uniform value every lane writes a copy of. Under `mask` when one is
    // given, every lane otherwise. What a store through per-lane addresses
    // is: a queue's compacted push, a frame's write into a lane's own entry.
    void create_scatter_at(llvm::Value *value, llvm::Value *ptrs,
                           const ir::Type &pointee, llvm::Value *mask);
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
    // Who an atomic has to be indivisible with respect to: every agent in the
    // system, on a CPU, where the threads that share the memory are all
    // there is. A device says the device (see CodeGen_PTX): the host reads
    // what a kernel wrote only after the launch has completed, which is a
    // fence of its own, and a system-scope atomic on the GPU is the slower
    // instruction for a guarantee nothing there needs.
    virtual llvm::SyncScope::ID atomic_scope() { return llvm::SyncScope::System; }
    // The address an atomic read-modify-write is made on, given the address
    // the program computed: the same one here, and on a device one qualified
    // with the memory it is known to be in (CodeGen_PTX), which is what lets
    // LLVM pick the hardware's atomic over a compare-and-swap loop.
    virtual llvm::Value *atomic_address(llvm::Value *loc) { return loc; }
    // One lane of a vector at an index computed at run time (`v[axis]`):
    // LLVM's extractelement here, which the CPU has an instruction for; a
    // device generator may spell it another way (CodeGen_PTX, where the
    // instruction goes through local memory).
    // The symbol a function of the program is defined and called under. The
    // program's names carry `!` where a copy was made -- a variant's
    // `integrator_li!VolPath`, a stage's `step!after`, a split queue's
    // `hits!Some!Diffuse` -- which LLVM's x86 assembler quotes and PTX
    // cannot spell at all ("Symbol name with unsupported characters"), so a
    // backend with a narrower alphabet maps them (CodeGen_PTX). Every
    // definition and every lookup goes through this, so they agree.
    virtual std::string symbol_name(const std::string &name) const {
        return name;
    }

    virtual llvm::Value *extract_lane(llvm::Value *vec, llvm::Value *idx) {
        return builder->CreateExtractElement(vec, idx);
    }
    // A value reduced across the threads of a GPU block -- the fold of
    // every thread's `v` under `op`, in every thread (the block_reduce_*
    // intrinsics SSA/BlockAccumulates.h makes). `type` is the program's
    // type of `v`, for the sign an integer's min and max go by. Only a
    // device has blocks; CodeGen_PTX lowers it.
    virtual llvm::Value *block_reduce(ir::Intrinsic::OpType op,
                                      const ir::Type &type, llvm::Value *v);
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
    // One counted loop, shared by every statement that is one: `ForAll`, and
    // a `ParFor` that no schedule placed on any hardware.
    void codegen_counted_loop(const std::string &index,
                              const ir::Expr &begin_expr,
                              const ir::Expr &end_expr,
                              const ir::Expr &stride_expr,
                              const ir::Stmt &body);
    llvm::Value *create_malloc(llvm::Type *etype, llvm::Value *size,
                               bool zero_initialize, const std::string &name);

    // The width of the random number generator's state, in bits: four 32-bit
    // lanes on every target, so that a program's random stream does not
    // depend on the machine it runs on. Not the machine's register width,
    // which is vector_register_bits().
    virtual int native_vector_bits() const {
        // TODO(ajr): override for other targets.
        return 128; // ARM Neon
    }

    // The widest vector register the target has, in bits: what a stack slot
    // an aggregate lives in is aligned to, so that a store of a whole
    // register into it cannot fault. The target-agnostic answer is the
    // RNG's; a target with wider registers says so (see CodeGen_X86).
    virtual int vector_register_bits() const { return native_vector_bits(); }

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
