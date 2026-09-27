#include "rtcore-alpha.h"

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

// Used in rtcore-alpha.bonsai: two triangles across the x axis -- one with
// an alpha cutout at x = 0 and an opaque one at x = 5 -- each its own build
// input of one acceleration structure, and eight rays along +x.
//
// What the driver owes the compiled programs (Lower/Trees.cpp): each build
// input's `base` is where its elements start in the tree's storage, so that
// `base + primitive index` names the element; a tree over the scene's own
// elements is an instance with id zero; and each input says whether the
// queries' any-hit programs can reject a hit on its elements, which is the
// compiler's word on the queries' filters -- `trace_anyhit_matters` and
// `trace_any_anyhit_matters`, exported for the driver to ask -- so that the
// hardware skips the programs over the opaque triangle.
namespace {

// A triangle as a build input of the hardware's: three floats a vertex, in
// order.
struct Input {
    std::vector<float> positions;
    std::vector<uint32_t> indices = {0, 1, 2};
    bonsai_optix_triangles input;
};

Input input_of(const Tri &t, uint32_t base) {
    Input in;
    for (const float3 &p : {t.tri.p0, t.tri.p1, t.tri.p2}) {
        in.positions.push_back(p[0]);
        in.positions.push_back(p[1]);
        in.positions.push_back(p[2]);
    }
    in.input.positions = in.positions.data();
    in.input.vertex_count = 3;
    in.input.indices = in.indices.data();
    in.input.triangle_count = 1;
    in.input.base = base;
    in.input.any_hit = trace_anyhit_matters(t) || trace_any_anyhit_matters(t);
    return in;
}

} // namespace

int main() {
    // Both triangles span y in [-1, 1] at z = 0 and narrow to a point at
    // z = 2. The front one keeps a hit only where the ray's y is below 0.5;
    // the back one is opaque.
    std::vector<Tri> tris = {
        Tri{Triangle{float3{0.0f, -2.0f, -2.0f}, float3{0.0f, 2.0f, -2.0f},
                     float3{0.0f, 0.0f, 2.0f}},
            0.5f},
        Tri{Triangle{float3{5.0f, -2.0f, -2.0f}, float3{5.0f, 2.0f, -2.0f},
                     float3{5.0f, 0.0f, 2.0f}},
            -1.0f},
    };

    // The program's modules first: the shader binding table's stride, which
    // the scene's offsets are built with, is the program's ray types.
    bonsai_gpu_prepare();

    // What the compiler says of each triangle: whether a query's any-hit
    // program can reject a hit on it. The cutout's can, the opaque one's
    // cannot.
    const char *const which[2] = {"cutout", "opaque"};
    for (size_t i = 0; i < tris.size(); i++) {
        std::cout << which[i] << ": any-hit matters to trace "
                  << (trace_anyhit_matters(tris[i]) ? "yes" : "no")
                  << ", to trace_any "
                  << (trace_any_anyhit_matters(tris[i]) ? "yes" : "no") << '\n';
    }

    // One structure over two inputs, each based where its element is.
    Input front = input_of(tris[0], 0);
    Input back = input_of(tris[1], 1);
    const bonsai_optix_triangles inputs[2] = {front.input, back.input};
    const uint64_t geometry = bonsai_optix_geometry(inputs, 2, nullptr, 0);
    bonsai_optix_instance top = {};
    for (int row = 0; row < 3; row++) {
        top.transform[4 * row + row] = 1.0f;
    }
    top.geometry = geometry;
    top.id = 0;
    const uint64_t traversable = bonsai_optix_scene(&top, 1);

    // The rays: along +x from x = -10.
    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };
    const char *const names[8] = {"below the cutoff ", "through the cutout",
                                  "off both         ", "well below       ",
                                  "at the cutoff    ", "cutout, low z    ",
                                  "off the front    ", "far off          "};
    std::vector<Ray> rays = {ray(0.0f, 0.0f),  ray(0.8f, 0.0f),  ray(0.8f, 1.9f),
                             ray(-0.8f, 0.0f), ray(0.5f, 0.0f),  ray(0.9f, -1.5f),
                             ray(1.5f, 0.0f),  ray(0.0f, 10.0f)};
    const int32_t n = int32_t(rays.size());
    std::vector<float> hit_x(n, -2.0f);
    std::vector<int32_t> hit_any(n, -2);

    // The layout's arrays reach the program as buffer descriptors; the tree
    // is the hardware's handle and its elements, nothing else.
    bonsai_buffer elems_buffer =
        bonsai_buffer_wrap(tris.data(), tris.size() * sizeof(tris[0]));
    _tree_layout0 scene;
    scene.traversable = traversable;
    scene.n = uint32_t(tris.size());
    scene.elems = &elems_buffer;

    bonsai_buffer rays_buffer =
        bonsai_buffer_wrap(rays.data(), rays.size() * sizeof(rays[0]));
    bonsai_buffer x_buffer =
        bonsai_buffer_wrap(hit_x.data(), hit_x.size() * sizeof(hit_x[0]));
    bonsai_buffer any_buffer =
        bonsai_buffer_wrap(hit_any.data(), hit_any.size() * sizeof(hit_any[0]));
    bonsai_buffer *buffers[] = {&rays_buffer, &x_buffer, &any_buffer,
                                &elems_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(trace_all_sides));
    bonsai_buffer_stage_all(buffers, trace_all_sides, sizeof(trace_all_sides));
    bonsai_buffer_implicit_copies(0);

    trace_all(n, &rays_buffer, &x_buffer, &any_buffer, scene);

    bonsai_buffer_stage(&x_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&any_buffer, BONSAI_HOST);
    std::cout << std::fixed << std::setprecision(1);
    for (int32_t i = 0; i < n; i++) {
        std::cout << names[i] << ": ";
        if (hit_x[i] < 0.0f) {
            std::cout << "miss";
        } else {
            std::cout << "hit x=" << hit_x[i];
        }
        std::cout << " any=" << (hit_any[i] == 1 ? "yes" : "no") << '\n';
    }
    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
}
