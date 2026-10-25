// FCPW's GPU as the reference: its compute shaders over the tree it
// transfers to the card (fcpw/gpu/bvh.cs.slang: rayIntersection,
// closestPoint, closestSilhouettePoint; the traversals bvh.slang's), run as
// its GPUScene runs them (fcpw_gpu.inl) but with the clock where a
// comparison of traversals wants it. GPUScene's batch calls allocate the
// query buffer and upload it, dispatch, wait, and read the answers back,
// all in one call; here the upload (GPURun*Query::allocate) and the read
// (read) are the driver's, off the clock, and what is timed is the dispatch
// and the wait for it (fcpw::runShader, FCPW's own: one command encoder,
// the pipeline and the resources bound, ceil(n / 256) groups of 256 threads
// -- GPUScene's nThreadsPerGroup -- submitted and waited on the host), as
// on the program's side the clock is around the kernel's launch and wait
// with its buffers staged on the device beforehand. Built with
// FCPW_USE_GPU against slang-rhi and the Slang release it fetched
// (apps/wosx/build_slang_rhi.sh), as apps/wosx's GPU comparison is; the
// driver that includes this is rtq_hook.cpp under RTQ_GPU.
//
// The setup is GPUScene::transferToGPU's, step for step: the context on
// the CUDA backend with FCPW's gpu directory as the search path and the
// tree's kind as the one macro (FCPW_BVH_TYPE: the triangle BVH, or the
// triangle SNCH when the scene has silhouettes), its two Slang modules
// loaded, the entry points' programs, the tree's buffers allocated from the
// scene's data (GPUBvhBuffers::allocate: the nodes, the primitives and,
// for the SNCH, the silhouette edges, written by FCPW's own extractors,
// the very rows apps/rtq/fcpw_snch.h and fcpw_bvh.h write for the program),
// and the binder that hands them to a shader as `gBvh`.
#pragma once

#include <fcpw/fcpw.h>
#include <fcpw/fcpw_gpu.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace rtq_gpu {

struct FcpwGpu {
    fcpw::GPUContext context;
    fcpw::GPUBvhBuffers bvh;
    fcpw::ComputeShader ray_shader, closest_shader, silhouette_shader;
    std::function<void(const fcpw::ComputeShader &, const fcpw::ShaderCursor &)> bind_bvh;
    bool has_silhouettes = false;

    // GPUScene(fcpw_dir) and transferToGPU(scene, "cuda"), for a scene of one
    // triangle object, built scalar (the GPU takes the binary tree alone).
    void setup(fcpw::Scene<3> &scene, const std::string &fcpw_dir) {
        fcpw::SceneData<3> *data = scene.getSceneData();
        has_silhouettes = data->silhouetteEdgeObjects.size() > 0;
        const std::string gpu_dir = fcpw_dir + "/include/fcpw/gpu";
        context.searchPaths = {gpu_dir};
        context.macros = {
            {"FCPW_BVH_TYPE", std::to_string(has_silhouettes ? FCPW_TRIANGLE_SNCH : FCPW_TRIANGLE_BVH)}};
        context.initDevice(fcpw::parseDeviceBackend("cuda"));

        std::vector<fcpw::GPUModule> libraries(1);
        libraries[0].load(context, gpu_dir + "/fcpw.slang");
        fcpw::GPUModule main_module;
        main_module.load(context, gpu_dir + "/bvh.cs.slang");
        ray_shader.loadProgram(context, main_module, libraries, "rayIntersection");
        closest_shader.loadProgram(context, main_module, libraries, "closestPoint");
        if (has_silhouettes) {
            silhouette_shader.loadProgram(context, main_module, libraries, "closestSilhouettePoint");
        }

        bvh.template allocate<3>(context, data, /*allocatePrimitiveData=*/true,
                                 /*allocateSilhouetteData=*/has_silhouettes,
                                 /*allocateNodeData=*/true, /*allocateRefitData=*/false);
        bind_bvh = [this](const fcpw::ComputeShader &shader, const fcpw::ShaderCursor &cursor) {
            fcpw::ComPtr<fcpw::IShaderObject> object =
                shader.createShaderObject(context, bvh.getReflectionType());
            fcpw::ShaderCursor bvh_cursor(object);
            bvh.setResources(bvh_cursor, /*printLogs=*/false);
            cursor.getPath("gBvh").setObject(object);
        };
    }

    // One dispatch of a shader over `n` queries whose buffers `run` holds,
    // and the wait for it: what the clock is around.
    template <typename Run>
    void dispatch(const fcpw::ComputeShader &shader, const Run &run, uint32_t n) {
        const uint32_t groups = (n + 255) / 256;
        fcpw::runShader<Run>(context, shader, run, bind_bvh, groups, 1, /*printLogs=*/false);
    }
};

} // namespace rtq_gpu
