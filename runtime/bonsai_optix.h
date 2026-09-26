#pragma once

#include <stdint.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "bonsai_cuda.h"

// The OptiX runtime a `f.bind(RTCore)` and `bind(i, OptixThread)` schedule
// lowers to: the ray tracing hardware, reached through the OptiX API the
// NVIDIA driver ships (`libnvoptix.so.1`), with the headers of the OptiX SDK
// -- 9.1.0, for shader execution reordering -- found where the compiler was
// told (`BONSAI_OPTIX_SDK`, see CMakeLists.txt) or where a driver's build
// puts them on its include path.
//
// Generated code makes one call per launch, `bonsai_optix_launch`: run the
// raygen program named from the PTX module `ptx` over `count` threads with
// `params` as the module's launch parameters, against the acceleration
// structure `traversable`. Everything about *how* -- the function table, the
// device context, modules, program groups, pipelines, shader binding tables
// -- is this file's, as runtime/bonsai_cuda.h is for the CUDA launches, and
// the two share the CUDA driver's context.
//
// The acceleration structure is the driver's to build, as the tree's nodes
// are in a CPU schedule (apps/pbrt/render_hook.cpp builds the BVH): a
// geometry acceleration structure from build inputs of triangles (a mesh
// each) and of bounding boxes (the shapes the hardware does not intersect
// itself, whose intersection program is the query's own `distmin`), and an
// instance acceleration structure over the geometries, one instance per
// placement. The handle it returns is what the tree's layout keeps in its
// `traversable` word (Lower/Layouts.cpp) and what `rt_trace` traces.
//
// What a hit tells the programs (Lower/Trees.cpp): each build input's
// shader binding table record carries `base`, where the input's elements
// start in the tree's storage, and the hit's primitive index counts from
// there; the instance's `id` is what `rt_instance_id` reads -- the compiler
// takes zero for the scene's own elements and one more than the index of the
// element holding the instance's tree otherwise. The driver lays the storage
// out so that each input's elements are one contiguous run.
//
// One pipeline per raygen program: its module, a program group for the
// raygen, and per ray type -- one per RTCore query of the *program*, in
// alphabetical order of the query's name, which is the order the compiler
// numbers them in (CodeGen_OptiX) -- a miss program group and two hit
// groups, one for the hardware's triangles and one for the boxes with the
// intersection program; a query this module has no programs for gets empty
// ones. The shader binding table is built per pipeline and scene: one
// hit-group record per build input per ray type, laid out geometry by
// geometry, input by input, type by type, so that an instance's `sbtOffset`
// is where its geometry's records start and the trace's stride is the
// number of ray types, as OptiX indexes them. That stride is baked into the
// instance acceleration structure, so it has to be the program's and not one
// module's: the program declares its ray types first, in its
// `bonsai_gpu_prepare` (bonsai_optix_ray_types), and a scene is built after.
//
// Header-only, like bonsai_cuda.h, so that a driver has the whole runtime by
// including the generated header; the compiler includes it once too
// (runtime/bonsai_optix.cpp) for the programs it runs itself. Without the
// SDK's headers on the include path the API is declared and every call
// stops with a message saying what to install.

#if __has_include(<optix_function_table.h>) && __has_include(<cuda.h>)
#include <optix_function_table.h>
#define BONSAI_OPTIX_SDK_FOUND 1
#endif

extern "C" {

// A build input of the hardware's triangles: one mesh, as the program keeps
// it -- three floats a vertex, tightly packed; three indices a triangle,
// relative to `positions`.
struct bonsai_optix_triangles {
    const float *positions;
    uint32_t vertex_count;
    const uint32_t *indices;
    uint32_t triangle_count;
    // Where the input's elements start in the tree's storage.
    uint32_t base;
};

// A build input of shapes the hardware does not intersect itself, by their
// bounding boxes: six floats each, the minimum corner then the maximum.
struct bonsai_optix_boxes {
    const float *boxes;
    uint32_t count;
    uint32_t base;
};

// Builds a geometry acceleration structure over the inputs; the id it returns
// names it to bonsai_optix_scene. Aborts with OptiX's reason on failure.
uint64_t bonsai_optix_geometry(const bonsai_optix_triangles *triangles,
                               int64_t triangle_inputs,
                               const bonsai_optix_boxes *boxes,
                               int64_t box_inputs);

// One placement of a geometry: OptiX's 3x4 row-major transform, the
// geometry's id, and the instance id the hit programs read.
struct bonsai_optix_instance {
    float transform[12];
    uint64_t geometry;
    uint32_t id;
};

// Builds the instance acceleration structure over `instances`, and returns
// its traversable handle: what the tree's layout holds, what `rt_trace`
// traces, and what a launch is given to build its shader binding table.
uint64_t bonsai_optix_scene(const bonsai_optix_instance *instances,
                            int64_t count);

// The program's RTCore queries, in the order the compiler numbers their ray
// types: what every module's traces and every scene's shader binding table
// offsets are laid out by. Declared once, before any scene is built or
// module loaded; what the generated `bonsai_gpu_prepare()` calls first.
void bonsai_optix_ray_types(const char *const *names, int64_t count);

// Compiles the PTX module ahead of its first launch, so that the compile is
// part of a driver's setup rather than of its first timed call. What the
// generated `bonsai_gpu_prepare()` calls.
void bonsai_optix_load(const char *ptx);

// Runs the raygen program `raygen` of `ptx` over `count` threads, with
// `param_bytes` of `params` as the module's launch parameters, against the
// scene `traversable` (zero when the program traces nothing), and returns
// when it is done. `slots[i]` is the address of the i-th parameter's slot
// within `params`, and `buffers` the host memory among them that travels to
// the device for the launch and back after it, with which slot holds its
// address -- as bonsai_cuda_launch takes them (runtime/bonsai_cuda.h). Any
// failure prints OptiX's reason and aborts, as the CUDA launch does.
void bonsai_optix_launch(const char *ptx, const char *raygen, int64_t count,
                         void *params, int64_t param_bytes, void **slots,
                         int64_t nslots, bonsai_cuda_buffer *buffers,
                         int64_t nbuffers, uint64_t traversable);
}

#ifdef BONSAI_OPTIX_SDK_FOUND

namespace bonsai_optix_detail {

[[noreturn]] inline void fail(const std::string &message) {
    std::fprintf(stderr, "bonsai_optix: %s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
}

// The function table `libnvoptix.so.1` fills for this SDK's ABI, and the
// device context, made once.
struct Api {
    OptixFunctionTable table = {};
    OptixDeviceContext context = nullptr;
    bool ok = false;
    std::string why;
};

inline void log_callback(unsigned int level, const char *tag,
                         const char *message, void *) {
    // Errors and warnings only; 3 and 4 are prints and verbose.
    if (level <= 2) {
        std::fprintf(stderr, "bonsai_optix: [%s] %s\n", tag, message);
    }
}

inline Api &api() {
    static Api a;
    static std::once_flag once;
    std::call_once(once, [] {
        void *lib = dlopen("libnvoptix.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (lib == nullptr) {
            a.why = "libnvoptix.so.1 could not be loaded (is the NVIDIA driver "
                    "installed, with its OptiX library?)";
            return;
        }
        void *symbol = dlsym(lib, "optixQueryFunctionTable");
        if (symbol == nullptr) {
            a.why = "libnvoptix.so.1 has no optixQueryFunctionTable";
            return;
        }
        auto *query = reinterpret_cast<OptixQueryFunctionTable_t *>(symbol);
        const OptixResult r =
            query(OPTIX_ABI_VERSION, 0, nullptr, nullptr, &a.table, sizeof(a.table));
        if (r != OPTIX_SUCCESS) {
            a.why = "the driver's OptiX does not provide ABI version " +
                    std::to_string(OPTIX_ABI_VERSION) +
                    " (result " + std::to_string(int(r)) +
                    "); a newer driver is needed for this SDK";
            return;
        }
        // The CUDA context the launches share (bonsai_cuda.h).
        bonsai_cuda_detail::Driver &d =
            bonsai_cuda_detail::ready("create the OptiX device context");
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = log_callback;
        options.logCallbackLevel = 2;
        const OptixResult made = a.table.optixDeviceContextCreate(
            reinterpret_cast<CUcontext>(d.context), &options, &a.context);
        if (made != OPTIX_SUCCESS) {
            a.why = std::string("optixDeviceContextCreate failed: ") +
                    a.table.optixGetErrorName(made);
            return;
        }
        a.ok = true;
    });
    return a;
}

inline Api &ready(const char *what) {
    Api &a = api();
    if (!a.ok) {
        fail(std::string("cannot ") + what + ": " + a.why);
    }
    bonsai_cuda_detail::ready(what);
    return a;
}

inline void check(Api &a, OptixResult r, const std::string &what) {
    if (r == OPTIX_SUCCESS) {
        return;
    }
    fail(what + " failed: " + a.table.optixGetErrorName(r) + " (" +
         a.table.optixGetErrorString(r) + ")");
}

// Device memory holding `bytes` of `host`; `bytes` rounded up so that a
// zero-length input still has an address.
inline CUdeviceptr upload(const void *host, size_t bytes) {
    void *device = bonsai_cuda_malloc(bytes == 0 ? 1 : bytes);
    if (bytes != 0) {
        bonsai_cuda_copy_to_device(device, host, bytes);
    }
    return reinterpret_cast<CUdeviceptr>(device);
}

// `n` rounded up to a multiple of `alignment`.
inline size_t round_up(size_t n, size_t alignment) {
    return (n + alignment - 1) / alignment * alignment;
}

// One build input of a geometry, as the shader binding table needs it:
// whether its primitives are the hardware's triangles or boxes with an
// intersection program, and where its elements start.
struct Input {
    bool triangles;
    uint32_t base;
};

struct Geometry {
    OptixTraversableHandle handle = 0;
    CUdeviceptr storage = 0;
    std::vector<Input> inputs;
};

struct Scene {
    OptixTraversableHandle handle = 0;
    CUdeviceptr storage = 0;
    // The geometries the instances name, each once, in the order their
    // records are laid out; `record_start[g]` is where geometry g's records
    // begin, in records per ray type.
    std::vector<uint64_t> geometries;
    std::vector<uint32_t> record_start;
    uint32_t records = 0; // per ray type
};

// A hit-group or miss record: OptiX's header, then the input's base.
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    uint32_t base;
    uint32_t pad[3];
};

// A compiled module and, per raygen program of it, a pipeline.
struct Module {
    OptixModule module = nullptr;
    // The program's RTCore queries, in the order the compiler numbers the
    // ray types (alphabetical by the query's name); which of them this
    // module holds programs for, and which of those have an any-hit program.
    std::vector<std::string> ray_types;
    std::vector<bool> has_programs;
    std::vector<bool> has_anyhit;
};

struct Pipeline {
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygen = nullptr;
    std::vector<OptixProgramGroup> miss;             // per ray type
    std::vector<OptixProgramGroup> hit_triangles;    // per ray type
    std::vector<OptixProgramGroup> hit_boxes;        // per ray type
    // The shader binding table per scene, and the launch parameters' device
    // copy.
    std::map<uint64_t, OptixShaderBindingTable> tables;
    CUdeviceptr params = 0;
    size_t param_capacity = 0;
};

struct State {
    std::mutex mutex;
    // The program's ray types, once declared (bonsai_optix_ray_types).
    bool declared = false;
    std::vector<std::string> ray_types;
    std::map<const char *, Module> modules;
    std::map<std::pair<const char *, std::string>, Pipeline> pipelines;
    std::vector<Geometry> geometries;
    std::map<uint64_t, Scene> scenes;
};

inline State &state() {
    static State s;
    return s;
}

// The `.entry` names of `ptx` that begin with `prefix`, in the order they
// appear.
inline std::vector<std::string> entries_with_prefix(const char *ptx,
                                                    const std::string &prefix) {
    std::vector<std::string> names;
    const std::string text(ptx);
    const std::string marker = ".entry " + prefix;
    size_t at = 0;
    while ((at = text.find(marker, at)) != std::string::npos) {
        const size_t start = at + std::string(".entry ").size();
        size_t end = start;
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) ||
                text[end] == '_' || text[end] == '$')) {
            end++;
        }
        names.push_back(text.substr(start, end - start));
        at = end;
    }
    return names;
}

// Builds an acceleration structure from `inputs`; returns its handle and
// leaves its storage in `storage`. The temporary buffer is released.
inline OptixTraversableHandle build(Api &a, const std::vector<OptixBuildInput> &inputs,
                                    CUdeviceptr *storage, const char *what) {
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes = {};
    check(a,
          a.table.optixAccelComputeMemoryUsage(a.context, &options, inputs.data(),
                                               unsigned(inputs.size()), &sizes),
          std::string("optixAccelComputeMemoryUsage for ") + what);
    void *temp = bonsai_cuda_malloc(round_up(sizes.tempSizeInBytes, 128) + 128);
    void *output =
        bonsai_cuda_malloc(round_up(sizes.outputSizeInBytes, 128) + 128);
    // The buffers aligned as OptiX asks (OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT).
    const CUdeviceptr temp_aligned =
        round_up(reinterpret_cast<CUdeviceptr>(temp), 128);
    const CUdeviceptr output_aligned =
        round_up(reinterpret_cast<CUdeviceptr>(output), 128);
    OptixTraversableHandle handle = 0;
    check(a,
          a.table.optixAccelBuild(a.context, /*stream=*/nullptr, &options,
                                  inputs.data(), unsigned(inputs.size()),
                                  temp_aligned, sizes.tempSizeInBytes,
                                  output_aligned, sizes.outputSizeInBytes,
                                  &handle, nullptr, 0),
          std::string("optixAccelBuild for ") + what);
    bonsai_cuda_detail::Driver &d = bonsai_cuda_detail::driver();
    bonsai_cuda_detail::check(d, d.cuCtxSynchronize(),
                              std::string("cuCtxSynchronize after building ") +
                                  what);
    bonsai_cuda_free(temp);
    *storage = reinterpret_cast<CUdeviceptr>(output);
    return handle;
}

// The pipeline's compile options: what every module and program group of a
// pipeline is compiled against, so they agree.
inline OptixPipelineCompileOptions pipeline_options() {
    OptixPipelineCompileOptions options = {};
    options.usesMotionBlur = 0;
    options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    // The context's address, as two words (see Lower/Trees.cpp).
    options.numPayloadValues = 2;
    // The triangles' barycentrics, which the hardware reports; the boxes'
    // intersection program reports none.
    options.numAttributeValues = 2;
    options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    options.pipelineLaunchParamsVariableName = "params";
    return options;
}

inline Module &module_of(Api &a, State &s, const char *ptx) {
    if (const auto it = s.modules.find(ptx); it != s.modules.end()) {
        return it->second;
    }
    Module m;
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    const OptixPipelineCompileOptions pipeline = pipeline_options();
    char log[4096];
    size_t log_size = sizeof(log);
    const OptixResult r = a.table.optixModuleCreate(
        a.context, &module_options, &pipeline, ptx, std::strlen(ptx), log,
        &log_size, &m.module);
    if (r != OPTIX_SUCCESS) {
        fail(std::string("optixModuleCreate failed: ") +
             a.table.optixGetErrorName(r) + "\n" + std::string(log, log_size));
    }
    // The queries this module holds programs for: one miss program each.
    std::vector<std::string> held;
    for (const std::string &miss : entries_with_prefix(ptx, "__miss__")) {
        held.push_back(miss.substr(std::string("__miss__").size()));
    }
    std::sort(held.begin(), held.end());
    // The ray types: the program's, as declared -- or, for a program run
    // without its declaration (one the compiler runs itself), this module's
    // own queries, which is the same list when the module traces them all.
    if (s.declared) {
        for (const std::string &query : held) {
            if (std::find(s.ray_types.begin(), s.ray_types.end(), query) ==
                s.ray_types.end()) {
                fail("the module holds programs for the query `" + query +
                     "`, which the program's declared ray types "
                     "(bonsai_optix_ray_types) do not name");
            }
        }
        m.ray_types = s.ray_types;
    } else {
        m.ray_types = held;
    }
    const std::vector<std::string> anyhits =
        entries_with_prefix(ptx, "__anyhit__");
    for (const std::string &query : m.ray_types) {
        m.has_programs.push_back(std::find(held.begin(), held.end(), query) !=
                                 held.end());
        m.has_anyhit.push_back(std::find(anyhits.begin(), anyhits.end(),
                                         "__anyhit__" + query) != anyhits.end());
    }
    return s.modules.emplace(ptx, std::move(m)).first->second;
}

inline Pipeline &pipeline_of(Api &a, State &s, const char *ptx,
                             const std::string &raygen) {
    const auto key = std::make_pair(ptx, raygen);
    if (const auto it = s.pipelines.find(key); it != s.pipelines.end()) {
        return it->second;
    }
    Module &m = module_of(a, s, ptx);
    Pipeline p;
    std::vector<OptixProgramGroupDesc> descs;
    std::vector<std::string> names; // kept alive for the descriptors
    names.reserve(1 + 3 * m.ray_types.size() * 2);
    const auto keep = [&](std::string n) -> const char * {
        names.push_back(std::move(n));
        return names.back().c_str();
    };
    {
        OptixProgramGroupDesc d = {};
        d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module = m.module;
        d.raygen.entryFunctionName = keep(raygen);
        descs.push_back(d);
    }
    if (m.ray_types.empty()) {
        // A raygen program that traces nothing: OptiX still wants a miss
        // record, so a miss program group with no program in it.
        OptixProgramGroupDesc miss = {};
        miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descs.push_back(miss);
    }
    for (size_t t = 0; t < m.ray_types.size(); t++) {
        const std::string &query = m.ray_types[t];
        // A query this module has no programs for -- one of the program's
        // this raygen never traces -- gets groups with no programs in them,
        // so that every module's records are laid out alike.
        OptixProgramGroupDesc miss = {};
        miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        OptixProgramGroupDesc hit = {};
        hit.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (m.has_programs[t]) {
            miss.miss.module = m.module;
            miss.miss.entryFunctionName = keep("__miss__" + query);
            hit.hitgroup.moduleCH = m.module;
            hit.hitgroup.entryFunctionNameCH = keep("__closesthit__" + query);
            if (m.has_anyhit[t]) {
                hit.hitgroup.moduleAH = m.module;
                hit.hitgroup.entryFunctionNameAH = keep("__anyhit__" + query);
            }
        }
        descs.push_back(miss);
        descs.push_back(hit); // the hardware's triangles
        if (m.has_programs[t]) {
            hit.hitgroup.moduleIS = m.module;
            hit.hitgroup.entryFunctionNameIS = keep("__intersection__" + query);
        }
        descs.push_back(hit); // the boxes
    }
    std::vector<OptixProgramGroup> groups(descs.size(), nullptr);
    OptixProgramGroupOptions group_options = {};
    char log[4096];
    size_t log_size = sizeof(log);
    const OptixResult made = a.table.optixProgramGroupCreate(
        a.context, descs.data(), unsigned(descs.size()), &group_options, log,
        &log_size, groups.data());
    if (made != OPTIX_SUCCESS) {
        fail(std::string("optixProgramGroupCreate failed: ") +
             a.table.optixGetErrorName(made) + "\n" +
             std::string(log, log_size));
    }
    p.raygen = groups[0];
    if (m.ray_types.empty()) {
        p.miss.push_back(groups[1]);
    }
    for (size_t t = 0; t < m.ray_types.size(); t++) {
        p.miss.push_back(groups[1 + 3 * t]);
        p.hit_triangles.push_back(groups[2 + 3 * t]);
        p.hit_boxes.push_back(groups[3 + 3 * t]);
    }
    const OptixPipelineCompileOptions compile = pipeline_options();
    OptixPipelineLinkOptions link = {};
    // A raygen program traces; the hit programs it runs do not.
    link.maxTraceDepth = 1;
    log_size = sizeof(log);
    const OptixResult linked = a.table.optixPipelineCreate(
        a.context, &compile, &link, groups.data(), unsigned(groups.size()), log,
        &log_size, &p.pipeline);
    if (linked != OPTIX_SUCCESS) {
        fail(std::string("optixPipelineCreate failed: ") +
             a.table.optixGetErrorName(linked) + "\n" +
             std::string(log, log_size));
    }
    // The stack, as optix_stack_size.h computes it for a trace depth of
    // one and no callables: the raygen's continuation stack plus the larger
    // of the closest-hit/miss programs' and the intersection plus any-hit
    // programs'.
    unsigned css_rg = 0, css_ms = 0, css_ch = 0, css_ah = 0, css_is = 0;
    for (size_t i = 0; i < groups.size(); i++) {
        OptixStackSizes sizes = {};
        check(a, a.table.optixProgramGroupGetStackSize(groups[i], &sizes, p.pipeline),
              "optixProgramGroupGetStackSize");
        css_rg = std::max(css_rg, sizes.cssRG);
        css_ms = std::max(css_ms, sizes.cssMS);
        css_ch = std::max(css_ch, sizes.cssCH);
        css_ah = std::max(css_ah, sizes.cssAH);
        css_is = std::max(css_is, sizes.cssIS);
    }
    const unsigned continuation =
        css_rg + std::max(std::max(css_ch, css_ms), css_is + css_ah);
    // Two levels: the instances, and the geometry under each.
    check(a,
          a.table.optixPipelineSetStackSize(p.pipeline, 0, 0, continuation, 2),
          "optixPipelineSetStackSize");
    return s.pipelines.emplace(key, std::move(p)).first->second;
}

// The shader binding table of `p` for `scene`: the raygen's record, the
// miss records per ray type, and the hit-group records geometry by
// geometry, input by input, ray type by ray type.
inline const OptixShaderBindingTable &table_of(Api &a, State &s, Module &m,
                                               Pipeline &p, uint64_t traversable) {
    if (const auto it = p.tables.find(traversable); it != p.tables.end()) {
        return it->second;
    }
    // No scene for a raygen program that traces nothing: no hit records.
    static const Scene nothing;
    const Scene *scene_ptr = &nothing;
    if (traversable != 0) {
        const auto found = s.scenes.find(traversable);
        if (found == s.scenes.end()) {
            fail("the launch names a traversable no bonsai_optix_scene built: " +
                 std::to_string(traversable));
        }
        scene_ptr = &found->second;
    }
    const Scene &scene = *scene_ptr;
    const size_t types = m.ray_types.size();
    std::vector<Record> records;
    // The raygen's record, then the misses: one per ray type, or the one
    // empty program's when there are none.
    records.emplace_back();
    check(a, a.table.optixSbtRecordPackHeader(p.raygen, records.back().header),
          "optixSbtRecordPackHeader(raygen)");
    for (const OptixProgramGroup miss : p.miss) {
        records.emplace_back();
        check(a, a.table.optixSbtRecordPackHeader(miss, records.back().header),
              "optixSbtRecordPackHeader(miss)");
    }
    const size_t first_hit = records.size();
    for (const uint64_t g : scene.geometries) {
        const Geometry &geometry = s.geometries.at(g);
        for (const Input &input : geometry.inputs) {
            for (size_t t = 0; t < types; t++) {
                records.emplace_back();
                records.back().base = input.base;
                check(a,
                      a.table.optixSbtRecordPackHeader(
                          input.triangles ? p.hit_triangles[t] : p.hit_boxes[t],
                          records.back().header),
                      "optixSbtRecordPackHeader(hit)");
            }
        }
    }
    const CUdeviceptr device = upload(records.data(), records.size() * sizeof(Record));
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = device;
    sbt.missRecordBase = device + sizeof(Record);
    sbt.missRecordStrideInBytes = sizeof(Record);
    sbt.missRecordCount = unsigned(p.miss.size());
    if (records.size() > first_hit) {
        sbt.hitgroupRecordBase = device + first_hit * sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = unsigned(records.size() - first_hit);
    }
    return p.tables.emplace(traversable, sbt).first->second;
}

} // namespace bonsai_optix_detail

extern "C" {

__attribute__((used)) inline uint64_t
bonsai_optix_geometry(const bonsai_optix_triangles *triangles,
                      int64_t triangle_inputs, const bonsai_optix_boxes *boxes,
                      int64_t box_inputs) {
    using namespace bonsai_optix_detail;
    Api &a = ready("build a geometry acceleration structure");
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Geometry geometry;
    std::vector<OptixBuildInput> inputs;
    std::vector<CUdeviceptr> uploaded;
    // Every input's device address, at a stable place the descriptors point
    // into.
    std::vector<CUdeviceptr> vertex_buffers(static_cast<size_t>(triangle_inputs));
    std::vector<CUdeviceptr> box_buffers(static_cast<size_t>(box_inputs));
    static const unsigned flags = OPTIX_GEOMETRY_FLAG_NONE;
    for (int64_t i = 0; i < triangle_inputs; i++) {
        const bonsai_optix_triangles &t = triangles[i];
        vertex_buffers[i] = upload(t.positions, size_t(t.vertex_count) * 3 * sizeof(float));
        const CUdeviceptr index_buffer =
            upload(t.indices, size_t(t.triangle_count) * 3 * sizeof(uint32_t));
        uploaded.push_back(vertex_buffers[i]);
        uploaded.push_back(index_buffer);
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexBuffers = &vertex_buffers[i];
        input.triangleArray.numVertices = t.vertex_count;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
        input.triangleArray.indexBuffer = index_buffer;
        input.triangleArray.numIndexTriplets = t.triangle_count;
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
        input.triangleArray.flags = &flags;
        input.triangleArray.numSbtRecords = 1;
        inputs.push_back(input);
        geometry.inputs.push_back(Input{true, t.base});
    }
    for (int64_t i = 0; i < box_inputs; i++) {
        const bonsai_optix_boxes &b = boxes[i];
        box_buffers[i] = upload(b.boxes, size_t(b.count) * 6 * sizeof(float));
        uploaded.push_back(box_buffers[i]);
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers = &box_buffers[i];
        input.customPrimitiveArray.numPrimitives = b.count;
        input.customPrimitiveArray.strideInBytes = 6 * sizeof(float);
        input.customPrimitiveArray.flags = &flags;
        input.customPrimitiveArray.numSbtRecords = 1;
        inputs.push_back(input);
        geometry.inputs.push_back(Input{false, b.base});
    }
    if (inputs.empty()) {
        fail("a geometry acceleration structure needs at least one input");
    }
    geometry.handle = build(a, inputs, &geometry.storage, "a geometry");
    // The structure is self-contained: the inputs' copies are done with.
    for (const CUdeviceptr ptr : uploaded) {
        bonsai_cuda_free(reinterpret_cast<void *>(ptr));
    }
    s.geometries.push_back(std::move(geometry));
    return uint64_t(s.geometries.size() - 1);
}

__attribute__((used)) inline uint64_t
bonsai_optix_scene(const bonsai_optix_instance *instances, int64_t count) {
    using namespace bonsai_optix_detail;
    Api &a = ready("build an instance acceleration structure");
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (count <= 0) {
        fail("a scene needs at least one instance");
    }
    Scene scene;
    // The stride of the records: the program's ray types, which have to be
    // declared by now since the offsets below are built in.
    if (!s.declared) {
        fail("a scene is built before the program declared its ray types: "
             "call the program's bonsai_gpu_prepare() first, since the shader "
             "binding table's stride -- how many ray types a build input's "
             "records span -- is the program's and is baked into the "
             "instance acceleration structure");
    }
    const uint32_t stride = uint32_t(std::max<size_t>(1, s.ray_types.size()));
    // The geometries, each once, and where each one's records start, in
    // records per ray type.
    std::map<uint64_t, uint32_t> start_of;
    for (int64_t i = 0; i < count; i++) {
        const uint64_t g = instances[i].geometry;
        if (g >= s.geometries.size()) {
            fail("instance " + std::to_string(i) + " names geometry " +
                 std::to_string(g) + ", which bonsai_optix_geometry did not build");
        }
        if (start_of.contains(g)) {
            continue;
        }
        start_of[g] = scene.records;
        scene.geometries.push_back(g);
        scene.record_start.push_back(scene.records);
        scene.records += uint32_t(s.geometries[g].inputs.size());
    }
    std::vector<OptixInstance> optix_instances(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; i++) {
        OptixInstance &out = optix_instances[i];
        out = {};
        std::memcpy(out.transform, instances[i].transform, sizeof(out.transform));
        out.instanceId = instances[i].id;
        out.visibilityMask = 255;
        out.flags = OPTIX_INSTANCE_FLAG_NONE;
        out.traversableHandle = s.geometries[instances[i].geometry].handle;
        // In records: the geometry's first input's, times the ray types a
        // build input's records span (the trace's stride).
        out.sbtOffset = start_of.at(instances[i].geometry) * stride;
    }
    const CUdeviceptr device =
        upload(optix_instances.data(), optix_instances.size() * sizeof(OptixInstance));
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = device;
    input.instanceArray.numInstances = unsigned(count);
    scene.handle = build(a, {input}, &scene.storage, "the scene");
    bonsai_cuda_free(reinterpret_cast<void *>(device));
    const uint64_t handle = scene.handle;
    s.scenes[handle] = std::move(scene);
    return handle;
}

__attribute__((used)) inline void
bonsai_optix_ray_types(const char *const *names, int64_t count) {
    using namespace bonsai_optix_detail;
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::vector<std::string> declared(names, names + count);
    if (s.declared) {
        if (declared != s.ray_types) {
            fail("the program's ray types were declared twice, differently");
        }
        return;
    }
    if (!s.modules.empty() || !s.scenes.empty()) {
        fail("the program's ray types are declared after a module was loaded "
             "or a scene built; call bonsai_gpu_prepare() before either");
    }
    s.ray_types = std::move(declared);
    s.declared = true;
}

__attribute__((used)) inline void bonsai_optix_load(const char *ptx) {
    using namespace bonsai_optix_detail;
    Api &a = ready("compile the OptiX module");
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Module &m = module_of(a, s, ptx);
    // And every raygen's pipeline, so that a first launch compiles nothing.
    for (const std::string &raygen : entries_with_prefix(ptx, "__raygen__")) {
        pipeline_of(a, s, ptx, raygen);
    }
    (void)m;
}

__attribute__((used)) inline void
bonsai_optix_launch(const char *ptx, const char *raygen, int64_t count,
                    void *params, int64_t param_bytes, void **slots,
                    int64_t nslots, bonsai_cuda_buffer *buffers,
                    int64_t nbuffers, uint64_t traversable) {
    using namespace bonsai_optix_detail;
    Api &a = ready((std::string("launch `") + raygen + "`").c_str());
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (count <= 0) {
        return;
    }
    Module &m = module_of(a, s, ptx);
    Pipeline &p = pipeline_of(a, s, ptx, raygen);
    const OptixShaderBindingTable &sbt = table_of(a, s, m, p, traversable);
    // The buffers: to the device, and their slots repointed, as the CUDA
    // launch does it.
    std::vector<void *> device_memory(size_t(nbuffers), nullptr);
    for (int64_t b = 0; b < nbuffers; b++) {
        const bonsai_cuda_buffer &buffer = buffers[b];
        if (buffer.param >= uint64_t(nslots)) {
            fail("buffer " + std::to_string(b) + " of `" + raygen +
                 "` names parameter " + std::to_string(buffer.param) + " of " +
                 std::to_string(nslots));
        }
        device_memory[b] = bonsai_cuda_malloc(buffer.bytes == 0 ? 1 : buffer.bytes);
        if (buffer.bytes != 0) {
            bonsai_cuda_copy_to_device(device_memory[b], buffer.host, buffer.bytes);
        }
        *static_cast<void **>(slots[buffer.param]) = device_memory[b];
    }
    // The launch parameters, on the device: one buffer per pipeline, grown
    // as needed.
    const size_t bytes = size_t(param_bytes);
    if (p.param_capacity < bytes) {
        if (p.params != 0) {
            bonsai_cuda_free(reinterpret_cast<void *>(p.params));
        }
        p.params = reinterpret_cast<CUdeviceptr>(
            bonsai_cuda_malloc(bytes == 0 ? 1 : bytes));
        p.param_capacity = bytes;
    }
    if (bytes != 0) {
        bonsai_cuda_copy_to_device(reinterpret_cast<void *>(p.params), params, bytes);
    }
    check(a,
          a.table.optixLaunch(p.pipeline, /*stream=*/nullptr, p.params, bytes,
                              &sbt, unsigned(count), 1, 1),
          std::string("optixLaunch(") + raygen + ")");
    bonsai_cuda_detail::Driver &d = bonsai_cuda_detail::driver();
    bonsai_cuda_detail::check(d, d.cuCtxSynchronize(),
                              "cuCtxSynchronize after " + std::string(raygen));
    // Back, and the slots as they were.
    for (int64_t b = 0; b < nbuffers; b++) {
        const bonsai_cuda_buffer &buffer = buffers[b];
        if (buffer.bytes != 0) {
            bonsai_cuda_copy_to_host(buffer.host, device_memory[b], buffer.bytes);
        }
        bonsai_cuda_free(device_memory[b]);
        *static_cast<void **>(slots[buffer.param]) = buffer.host;
    }
}

} // extern "C"

#else // !BONSAI_OPTIX_SDK_FOUND

// Without the SDK's headers nothing here can be compiled; a program that
// reaches OptiX all the same stops with what to install.
namespace bonsai_optix_detail {
[[noreturn]] inline void missing(const char *what) {
    std::fprintf(stderr,
                 "bonsai_optix: cannot %s: this runtime was compiled without "
                 "the OptiX SDK's headers (optix_function_table.h) or CUDA's "
                 "(cuda.h). Put the SDK's include directory on the include "
                 "path -- BONSAI_OPTIX_SDK for the compiler's build -- and "
                 "CUDA's, and rebuild.\n",
                 what);
    std::fflush(stderr);
    std::abort();
}
} // namespace bonsai_optix_detail

extern "C" {
__attribute__((used)) inline uint64_t
bonsai_optix_geometry(const bonsai_optix_triangles *, int64_t,
                      const bonsai_optix_boxes *, int64_t) {
    bonsai_optix_detail::missing("build a geometry acceleration structure");
}
__attribute__((used)) inline uint64_t
bonsai_optix_scene(const bonsai_optix_instance *, int64_t) {
    bonsai_optix_detail::missing("build an instance acceleration structure");
}
__attribute__((used)) inline void bonsai_optix_ray_types(const char *const *,
                                                         int64_t) {
    bonsai_optix_detail::missing("declare the program's ray types");
}
__attribute__((used)) inline void bonsai_optix_load(const char *) {
    bonsai_optix_detail::missing("compile an OptiX module");
}
__attribute__((used)) inline void
bonsai_optix_launch(const char *, const char *, int64_t, void *, int64_t,
                    void **, int64_t, bonsai_cuda_buffer *, int64_t, uint64_t) {
    bonsai_optix_detail::missing("launch a raygen program");
}
} // extern "C"

#endif // BONSAI_OPTIX_SDK_FOUND
