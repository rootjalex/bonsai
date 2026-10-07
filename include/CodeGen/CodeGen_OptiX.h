#pragma once

#include "CodeGen/CodeGen_PTX.h"

#include <map>
#include <string>
#include <vector>

namespace bonsai {

// The device side of a loop bound to OptixThread: CodeGen_PTX for a module
// OptiX runs rather than the CUDA driver. One module per such loop, since
// OptiX hands a module its launch parameters through one named variable
// (`params`, in constant memory) and the loop's captures are that variable's
// fields.
//
// What the module holds. The loop's body as the raygen program,
// `__raygen__<index>`: the launch index is the iteration, and the body's
// captures are read out of `params` in the prologue. And for every RTCore
// query the body reaches -- every `rt_trace` in it or in what it calls,
// which names its query by the type of the context it traces with,
// `_RTContext_<query>` (Lower/Trees.cpp) -- the query's programs, the
// functions marked `optix_program`: `__closesthit__<query>`,
// `__anyhit__<query>` when the query has one, `__intersection__<query>` and
// `__miss__<query>`, each an entry point whose parameters the prologue
// supplies: the context from the two payload words (a promoted context --
// one the payload registers carry whole, Lower/Trees.cpp -- crosses as its
// own words instead, and its programs take no context argument at all),
// everything else from
// `params` -- the slot found by following the operand of the query's
// `rt_trace` that carries that parameter (Lower/RTCoreReads.h) back to the
// capture it came from (find_read_slots). The queries are numbered
// alphabetically, and that number is the ray type the trace names -- the
// offset into the shader binding table's hit and miss records -- which
// runtime/bonsai_optix.h lays out in the same order from the module's
// `__miss__` entries.
//
// The `rt_*` intrinsics are the OptiX device API as its headers spell it,
// inline PTX calls to `_optix_*` symbols the OptiX compiler in the driver
// resolves (optix_device_impl.h): `_optix_trace_typed_32` with the context's
// address as the two payload words (or a promoted context's own words),
// `_optix_get_ray_tmax`, `_optix_get_triangle_barycentrics`,
// `_optix_get_world_ray_origin_*` and `_optix_get_world_ray_direction_*`,
// `_optix_read_primitive_idx`, `_optix_read_instance_id`,
// `_optix_get_sbt_data_ptr_64`, `_optix_report_intersection_0`,
// `_optix_ignore_intersection`, `_optix_get_payload`, `_optix_set_payload`
// and `_optix_get_launch_index_x`.
struct CodeGen_OptiX : public CodeGen_PTX {
    CodeGen_OptiX();

    // Adds the body of `loop`, a parfor of `host` bound to OptixThread, as
    // the module's raygen program, and the programs of every RTCore query
    // it reaches. `by_value` is per capture, as CodeGen_PTX::add_kernel
    // takes it; the Kernel's `param_bytes` are the launch parameters' slots
    // in order -- begin, stride, the captures -- which the host packs into
    // one struct and hands to bonsai_optix_launch.
    Kernel add_raygen(const ir::ssa::Function &host,
                      const ir::ssa::Terminator::ParFor &loop,
                      const std::vector<bool> &by_value);

    // CodeGen_PTX's begin and finish, without the packed pair-of-float
    // instructions (`f32x2`, Blackwell's) the backend otherwise makes of
    // vec2 arithmetic: OptiX 9.1's PTX front end, in driver 595, faults on
    // them, where ptxas takes them. The choice is the backend's option
    // `nvptx-no-f32x2`, read when the target machine is made -- what is
    // legal is fixed then -- and again as instructions are selected, so it
    // is set around both.
    void begin(const ir::Program &program, const CompilerOptions &options) override;
    void finish() override;

    // The widest payload use of any trace this module's raygen makes (see
    // Kernel::payload_values), for the module's `bonsai_optix_load` call;
    // meaningful once add_raygen has run.
    uint64_t payload_values_used() const { return payload_values; }

  protected:
    // The `rt_*` intrinsics, as the inline PTX calls above.
    llvm::Value *codegen_rt_intrinsic(const ir::Intrinsic *node) override;
    // An address rooted at a launch parameter is device memory: cast to the
    // global address space, so a float accumulate is `atom.add.f32` (see
    // CodeGen_PTX::atomic_address, which does this for a kernel's
    // arguments).
    llvm::Value *atomic_address(llvm::Value *loc) override;

  private:
    // An inline PTX call: `ret` (or void) from `text` with `constraints`
    // over `args`.
    llvm::Value *asm_call(llvm::Type *ret, const std::string &text,
                          const std::string &constraints,
                          const std::vector<llvm::Value *> &args,
                          bool side_effects);
    // Payload word `k`.
    llvm::Value *payload(unsigned k);
    // The RTCore queries `loop`'s body reaches, by name: the contexts the
    // `rt_trace`s in it and in every function it calls trace with.
    std::vector<std::string>
    queries_traced(const ir::ssa::Function &host,
                   const ir::ssa::Terminator::ParFor &loop);
    // Compiles the program `func` (its SSA `ssa`) as an entry point whose
    // parameters the prologue supplies.
    void add_program(const ir::Function &func, const ir::ssa::Function &ssa);
    // Which launch parameter each read of each query the body of `loop`
    // traces comes from (`read_slots`): every `rt_trace` in the body or in
    // what it calls carries its query's reads as its operands after the
    // seventh, and each is followed to a capture of the body -- to the
    // definition of one of `body_head`'s arguments at `captures`, through
    // the parameter of the function the trace is in and every call that
    // passes it. A struct the launch holds by value (`by_value`, per
    // capture) is read where the body loads it through the captured
    // address. A read that is no capture (computed inside the launch, or
    // an argument that varies along a run of calls) is an error, as is a
    // query traced with two different values of one read.
    void find_read_slots(const ir::ssa::Function &host,
                         const ir::ssa::Terminator::ParFor &loop,
                         const ir::ssa::Block &body_head,
                         const std::vector<size_t> &captures,
                         const std::vector<bool> &by_value);

    // The launch parameters: their struct, the variable, and the name of
    // each slot (begin, stride, then the captures).
    llvm::StructType *params_ty = nullptr;
    llvm::GlobalVariable *params = nullptr;
    std::vector<std::string> slot_names;
    std::vector<llvm::Type *> slot_types;
    // Per traced query, the slot of each read at its place in the query's
    // list (ir::Function::OptixProgram::reads).
    std::map<std::string, std::vector<int>> read_slots;
    // The queries this module holds programs for, alphabetical: the ray
    // types.
    std::vector<std::string> ray_types;
    // The widest payload use of any trace under this raygen: two (the
    // memory form's context address) or a promoted context's word count,
    // whichever is more. Found by find_read_slots; the launch carries it
    // to the runtime, whose pipeline compiles every module against it
    // (runtime/bonsai_optix.h, pipeline_options).
    unsigned payload_values = 2;
};

} // namespace bonsai
