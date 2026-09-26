#include "rtcore-tlas.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// Used in rtcore-tlas.bonsai: the scene of correctness/cpp/main_mixed_tlas.cpp
// -- two objects, three instances, three standalone triangles -- with the
// acceleration structures built by the ray tracing hardware's runtime rather
// than the tree the driver builds there (runtime/bonsai_optix.h), and the
// eight rays traced by one launch. The answers are main_mixed_tlas.cpp's.
//
// What the driver owes the compiled programs (Lower/Trees.cpp): each build
// input's `base` is where its elements start in the tree's storage, so that
// `base + primitive index` names the element; the instance holding an
// object's tree has id one more than its element's index, and the scene's own
// triangles are in an instance with id zero. So the standalone triangles are
// the last three elements, in the order their build input lists them, and
// each object's triangles are a run of `tris`.
namespace {

float3 apply(const Transform &t, const float3 &p) {
    return t.m0 * p[0] + t.m1 * p[1] + t.m2 * p[2] + t.m3;
}

// A rotation given by the images of the three basis vectors, and a
// translation, with the inverse stored beside it.
Transform rigid(const float3 &c0, const float3 &c1, const float3 &c2,
                const float3 &t) {
    Transform out;
    out.m0 = c0;
    out.m1 = c1;
    out.m2 = c2;
    out.m3 = t;
    out.i0 = float3{c0[0], c1[0], c2[0]};
    out.i1 = float3{c0[1], c1[1], c2[1]};
    out.i2 = float3{c0[2], c1[2], c2[2]};
    out.i3 = float3{-(c0[0] * t[0] + c0[1] * t[1] + c0[2] * t[2]),
                    -(c1[0] * t[0] + c1[1] * t[1] + c1[2] * t[2]),
                    -(c2[0] * t[0] + c2[1] * t[1] + c2[2] * t[2])};
    return out;
}

Transform translate(const float3 &t) {
    return rigid(float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f},
                 float3{0.0f, 0.0f, 1.0f}, t);
}

// A quarter turn about x: y goes to z and z goes to -y.
Transform rotate_x_quarter(const float3 &t) {
    return rigid(float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f},
                 float3{0.0f, -1.0f, 0.0f}, t);
}

// OptiX's instance transform: three rows of four, the columns being the
// images of the basis vectors and the translation.
void optix_transform(const Transform &t, float out[12]) {
    for (int row = 0; row < 3; row++) {
        out[4 * row + 0] = t.m0[row];
        out[4 * row + 1] = t.m1[row];
        out[4 * row + 2] = t.m2[row];
        out[4 * row + 3] = t.m3[row];
    }
}

// A run of triangles as a build input of the hardware's: three floats a
// vertex, three vertices a triangle, indices in order.
struct Input {
    std::vector<float> positions;
    std::vector<uint32_t> indices;
    bonsai_optix_triangles input;
};

Input input_of(const std::vector<Triangle> &tris, uint32_t first, uint32_t count,
               uint32_t base) {
    Input in;
    for (uint32_t i = first; i < first + count; i++) {
        for (const float3 &p : {tris[i].p0, tris[i].p1, tris[i].p2}) {
            in.indices.push_back(uint32_t(in.positions.size() / 3));
            in.positions.push_back(p[0]);
            in.positions.push_back(p[1]);
            in.positions.push_back(p[2]);
        }
    }
    in.input.positions = in.positions.data();
    in.input.vertex_count = uint32_t(in.positions.size() / 3);
    in.input.indices = in.indices.data();
    in.input.triangle_count = count;
    in.input.base = base;
    return in;
}

} // namespace

int main() {
    // Object A: two triangles in the x = 0 plane, one in each z lane.
    // Object B: one small triangle around the object-space origin.
    std::vector<Triangle> tris = {
        Triangle{float3{0.0f, -1.0f, -1.5f}, float3{0.0f, 1.0f, -1.5f},
                 float3{0.0f, 0.0f, -0.5f}},
        Triangle{float3{0.0f, -1.0f, 0.5f}, float3{0.0f, 1.0f, 0.5f},
                 float3{0.0f, 0.0f, 1.5f}},
        Triangle{float3{0.0f, -0.25f, -0.25f}, float3{0.0f, 0.25f, -0.25f},
                 float3{0.0f, 0.0f, 0.25f}},
    };
    // The standalone triangles, in world space already: in lane z = -1 ahead
    // of both placements of A, in lane z = +1 behind both, alone in lane
    // z = 30.
    const std::vector<Triangle> solo = {
        Triangle{float3{-3.0f, -1.0f, -1.5f}, float3{-3.0f, 1.0f, -1.5f},
                 float3{-3.0f, 0.0f, -0.5f}},
        Triangle{float3{8.0f, -1.0f, 0.5f}, float3{8.0f, 1.0f, 0.5f},
                 float3{8.0f, 0.0f, 1.5f}},
        Triangle{float3{2.0f, -1.0f, 29.5f}, float3{2.0f, 1.0f, 29.5f},
                 float3{2.0f, 0.0f, 30.5f}}};

    // The program's modules first: the shader binding table's stride, which
    // the scene's offsets are built with, is the program's ray types.
    bonsai_gpu_prepare();

    // The objects' geometry: object A is tris[0..2), object B tris[2..3),
    // each input's base where its run starts in `tris`. The standalone
    // triangles are one geometry too, based where the Solo elements start.
    Input a = input_of(tris, 0, 2, 0);
    Input b = input_of(tris, 2, 1, 2);
    Input s = input_of(solo, 0, 3, /*base=*/3);
    const uint64_t geometry_a = bonsai_optix_geometry(&a.input, 1, nullptr, 0);
    const uint64_t geometry_b = bonsai_optix_geometry(&b.input, 1, nullptr, 0);
    const uint64_t geometry_solo = bonsai_optix_geometry(&s.input, 1, nullptr, 0);

    // The elements: the three instances first, then the three standalone
    // triangles, as the geometries' bases say. Under `layout Prim =
    // tagged_index` an element is a handle into the arm's pool.
    std::vector<Solo> solo_pool(3);
    uint64_t solo_fill = 0;
    std::vector<Inst> inst_pool(3);
    uint64_t inst_fill = 0;
    std::vector<uint64_t> elems;
    std::vector<bonsai_optix_instance> instances;
    const auto place = [&](const Transform &t, uint64_t geometry) {
        // The tree the instance holds has no nodes here: the hardware's
        // structure stands in for it, so its root is nothing.
        elems.push_back(Prim_Inst(t, /*blas=*/0, inst_pool.data(), &inst_fill));
        bonsai_optix_instance inst = {};
        optix_transform(t, inst.transform);
        inst.geometry = geometry;
        inst.id = uint32_t(elems.size()); // one more than the element's index
        instances.push_back(inst);
    };
    // Two placements of object A, one behind the other in both lanes, and
    // object B turned a quarter turn about x.
    place(translate(float3{0.0f, 0.0f, 0.0f}), geometry_a);
    place(translate(float3{5.0f, 0.0f, 0.0f}), geometry_a);
    place(rotate_x_quarter(float3{0.0f, 0.0f, 20.0f}), geometry_b);
    for (const Triangle &t : solo) {
        elems.push_back(Prim_Solo(t, solo_pool.data(), &solo_fill));
    }
    assert(solo_fill == solo_pool.size());
    assert(inst_fill == inst_pool.size());
    {
        // The scene's own triangles: an instance at the identity, with the
        // id the programs take for the top level.
        bonsai_optix_instance top = {};
        optix_transform(translate(float3{0.0f, 0.0f, 0.0f}), top.transform);
        top.geometry = geometry_solo;
        top.id = 0;
        instances.push_back(top);
    }
    const uint64_t traversable =
        bonsai_optix_scene(instances.data(), int64_t(instances.size()));

    // The rays: along +x from x = -10.
    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };
    const std::array<const char *, 8> names = {
        "solo in front ", "inst in front ", "solo alone    ", "turned        ",
        "unturned      ", "wrong object  ", "empty lane    ", "past the top  "};
    std::vector<Ray> rays = {ray(0.0f, -1.0f), ray(0.0f, 1.0f),  ray(0.0f, 30.0f),
                             ray(0.2f, 20.0f), ray(0.0f, 20.2f), ray(0.8f, 20.0f),
                             ray(0.0f, 40.0f), ray(5.0f, 0.0f)};
    const int32_t n = int32_t(rays.size());
    std::vector<float3> hit_p0(n);
    std::vector<int32_t> hit_kind(n, -2), hit_any(n, -2);

    // The layout's arrays reach the program as buffer descriptors; the tree
    // has no nodes, only the hardware's handle and its elements.
    std::vector<_tree_layout1> no_blas_nodes;
    std::vector<_tree_layout4> no_tlas_nodes;
    bonsai_buffer tris_buffer =
        bonsai_buffer_wrap(tris.data(), tris.size() * sizeof(tris[0]));
    bonsai_buffer blas_buffer = bonsai_buffer_wrap(no_blas_nodes.data(), 0);
    bonsai_buffer elems_buffer =
        bonsai_buffer_wrap(elems.data(), elems.size() * sizeof(elems[0]));
    bonsai_buffer tlas_buffer = bonsai_buffer_wrap(no_tlas_nodes.data(), 0);
    _tree_layout0 scene;
    scene.traversable = traversable;
    scene.tCount = uint32_t(tris.size());
    scene.tris = &tris_buffer;
    scene.bCount = 0;
    scene.group0_bnode = &blas_buffer;
    scene.pCount = uint32_t(elems.size());
    scene.elems = &elems_buffer;
    scene.nCount = 0;
    scene.group1_index = &tlas_buffer;

    bonsai_buffer rays_buffer =
        bonsai_buffer_wrap(rays.data(), rays.size() * sizeof(rays[0]));
    bonsai_buffer p0_buffer =
        bonsai_buffer_wrap(hit_p0.data(), hit_p0.size() * sizeof(hit_p0[0]));
    bonsai_buffer kind_buffer =
        bonsai_buffer_wrap(hit_kind.data(), hit_kind.size() * sizeof(int32_t));
    bonsai_buffer any_buffer =
        bonsai_buffer_wrap(hit_any.data(), hit_any.size() * sizeof(int32_t));
    bonsai_buffer solo_buffer = bonsai_buffer_wrap(
        solo_pool.data(), solo_pool.size() * sizeof(solo_pool[0]));
    bonsai_buffer inst_buffer = bonsai_buffer_wrap(
        inst_pool.data(), inst_pool.size() * sizeof(inst_pool[0]));
    // The externs in the order they were declared, `near` first: it is read
    // by the hit programs alone and reached `trace_all` last, and the
    // signature has to put it where the declaration did.
    float near[1] = {0.001f};
    bonsai_buffer near_buffer = bonsai_buffer_wrap(near, sizeof(near));
    bonsai_buffer *buffers[] = {&rays_buffer, &p0_buffer,   &kind_buffer,
                                &any_buffer,  &near_buffer, &tris_buffer,
                                &blas_buffer, &elems_buffer, &tlas_buffer,
                                &solo_buffer, &inst_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(trace_all_sides));
    bonsai_buffer_stage_all(buffers, trace_all_sides, sizeof(trace_all_sides));
    bonsai_buffer_implicit_copies(0);

    trace_all(n, &rays_buffer, &p0_buffer, &kind_buffer, &any_buffer,
              &near_buffer, scene, &solo_buffer, &inst_buffer);

    bonsai_buffer_stage(&p0_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&kind_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&any_buffer, BONSAI_HOST);
    std::cout << std::fixed << std::setprecision(2);
    for (int32_t i = 0; i < n; i++) {
        std::cout << names[i] << ": ";
        if (hit_kind[i] < 0) {
            std::cout << "miss";
        } else {
            std::cout << "hit " << (hit_kind[i] == 1 ? "instance" : "solo")
                      << " tri=(" << hit_p0[i][0] << ", " << hit_p0[i][1] << ", "
                      << hit_p0[i][2] << ")";
        }
        std::cout << " any=" << (hit_any[i] == 1 ? "yes" : "no") << '\n';
    }
    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
}
