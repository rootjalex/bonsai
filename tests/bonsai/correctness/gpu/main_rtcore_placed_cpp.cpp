// Used in rtcore-placed.bonsai: main_rtcore_tlas_cpp.cpp's scene with
// object A made of two meshes, one triangle each, built as ONE structure
// of two build inputs -- pbrt's buildBVHForTriangles over a definition's
// shapes -- and placed twice, each placement's records naming its own
// `Placed` primitives consecutively from the first
// (bonsai_optix_instance::holder_per_input). Object B stays an instance
// holding a tree of one triangle. The three standalone triangles and the
// eight rays are rtcore-tlas's, and so are the answers, with the hits that
// went through object A now naming the mesh they hit.
#include "rtcore-placed.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

float3 apply(const Transform &t, const float3 &p) {
    return t.m0 * p[0] + t.m1 * p[1] + t.m2 * p[2] + t.m3;
}

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

void optix_transform(const Transform &t, float out[12]) {
    for (int row = 0; row < 3; row++) {
        out[4 * row + 0] = t.m0[row];
        out[4 * row + 1] = t.m1[row];
        out[4 * row + 2] = t.m2[row];
        out[4 * row + 3] = t.m3[row];
    }
}

// A run of triangles as one build input: three floats a vertex, three
// vertices a triangle, indices in order, based where the run starts in
// `tris`. The any-hit programs matter for every triangle here, as in
// rtcore-tlas (the filter's `distmin > near` is no triangle's to settle).
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
    in.input.any_hit = false;
    for (uint32_t i = first; i < first + count; i++) {
        in.input.any_hit = in.input.any_hit || trace_anyhit_matters(tris[i]) ||
                           trace_any_anyhit_matters(tris[i]);
    }
    assert(in.input.any_hit);
    return in;
}

} // namespace

int main() {
    // Object A: two meshes of one triangle each in the x = 0 plane, one per
    // z lane -- tris[0] and tris[1]. Object B: one small triangle around
    // the object-space origin, tris[2].
    std::vector<Triangle> tris = {
        Triangle{float3{0.0f, -1.0f, -1.5f}, float3{0.0f, 1.0f, -1.5f},
                 float3{0.0f, 0.0f, -0.5f}},
        Triangle{float3{0.0f, -1.0f, 0.5f}, float3{0.0f, 1.0f, 0.5f},
                 float3{0.0f, 0.0f, 1.5f}},
        Triangle{float3{0.0f, -0.25f, -0.25f}, float3{0.0f, 0.25f, -0.25f},
                 float3{0.0f, 0.0f, 0.25f}},
    };
    const std::vector<Triangle> solo = {
        Triangle{float3{-3.0f, -1.0f, -1.5f}, float3{-3.0f, 1.0f, -1.5f},
                 float3{-3.0f, 0.0f, -0.5f}},
        Triangle{float3{8.0f, -1.0f, 0.5f}, float3{8.0f, 1.0f, 0.5f},
                 float3{8.0f, 0.0f, 1.5f}},
        Triangle{float3{2.0f, -1.0f, 29.5f}, float3{2.0f, 1.0f, 29.5f},
                 float3{2.0f, 0.0f, 30.5f}}};

    bonsai_gpu_prepare();

    // Definition A's structure: two build inputs, one per mesh, each based
    // where its mesh's run of `tris` starts -- built once for both of its
    // placements. B's: one input. The standalone triangles: one input based
    // where the Solo elements start in `elems`.
    Input a0 = input_of(tris, 0, 1, 0);
    Input a1 = input_of(tris, 1, 1, 1);
    Input b = input_of(tris, 2, 1, 2);
    Input s = input_of(solo, 0, 3, /*base=*/5);
    const bonsai_optix_triangles a_inputs[2] = {a0.input, a1.input};
    const uint64_t geometry_a = bonsai_optix_geometry(a_inputs, 2, nullptr, 0);
    const uint64_t geometry_b = bonsai_optix_geometry(&b.input, 1, nullptr, 0);
    const uint64_t geometry_solo = bonsai_optix_geometry(&s.input, 1, nullptr, 0);

    // The elements: A's two placements as two Placed primitives each (mesh
    // 0 then mesh 1, the definition's order), then B's instance, then the
    // three standalone triangles. Under `layout Prim = tagged_index` an
    // element is a handle into the arm's pool.
    std::vector<Solo> solo_pool(3);
    uint64_t solo_fill = 0;
    std::vector<Placed> placed_pool(4);
    uint64_t placed_fill = 0;
    std::vector<Inst> inst_pool(1);
    uint64_t inst_fill = 0;
    std::vector<uint64_t> elems;
    std::vector<bonsai_optix_instance> instances;
    // One placement of definition A: a Placed primitive per mesh, and one
    // OptixInstance whose records name them from the first on.
    const auto place_a = [&](const Transform &t) {
        const uint32_t first = uint32_t(elems.size());
        elems.push_back(Prim_Placed(t, 0u, /*tris=*/0u, placed_pool.data(), &placed_fill));
        elems.push_back(Prim_Placed(t, 1u, /*tris=*/1u, placed_pool.data(), &placed_fill));
        bonsai_optix_instance inst = {};
        optix_transform(t, inst.transform);
        inst.geometry = geometry_a;
        inst.holder = first;
        inst.holder_per_input = 1;
        instances.push_back(inst);
    };
    place_a(translate(float3{0.0f, 0.0f, 0.0f}));
    place_a(translate(float3{5.0f, 0.0f, 0.0f}));
    {
        // B turned a quarter turn about x, an instance over its tree: the
        // field stores where B's run of `tris` starts.
        const Transform t = rotate_x_quarter(float3{0.0f, 0.0f, 20.0f});
        elems.push_back(Prim_Inst(t, /*blas=*/2u, inst_pool.data(), &inst_fill));
        bonsai_optix_instance inst = {};
        optix_transform(t, inst.transform);
        inst.geometry = geometry_b;
        inst.holder = uint32_t(elems.size() - 1);
        instances.push_back(inst);
    }
    for (const Triangle &t : solo) {
        elems.push_back(Prim_Solo(t, solo_pool.data(), &solo_fill));
    }
    assert(solo_fill == solo_pool.size());
    assert(placed_fill == placed_pool.size());
    assert(inst_fill == inst_pool.size());
    {
        bonsai_optix_instance top = {};
        optix_transform(translate(float3{0.0f, 0.0f, 0.0f}), top.transform);
        top.geometry = geometry_solo;
        top.holder = BONSAI_OPTIX_OWN;
        instances.push_back(top);
    }
    const uint64_t traversable =
        bonsai_optix_scene(instances.data(), int64_t(instances.size()));

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };
    const std::array<const char *, 8> names = {
        "solo in front ", "placed in front", "solo alone    ", "turned        ",
        "unturned      ", "wrong object  ", "empty lane    ", "past the top  "};
    std::vector<Ray> rays = {ray(0.0f, -1.0f), ray(0.0f, 1.0f),  ray(0.0f, 30.0f),
                             ray(0.2f, 20.0f), ray(0.0f, 20.2f), ray(0.8f, 20.0f),
                             ray(0.0f, 40.0f), ray(5.0f, 0.0f)};
    const int32_t n = int32_t(rays.size());
    std::vector<float3> hit_p0(n);
    std::vector<int32_t> hit_kind(n, -2), hit_mesh(n, -2), hit_any(n, -2);

    bonsai_buffer tris_buffer =
        bonsai_buffer_wrap(tris.data(), tris.size() * sizeof(tris[0]));
    bonsai_buffer elems_buffer =
        bonsai_buffer_wrap(elems.data(), elems.size() * sizeof(elems[0]));
    _tree_layout0 scene;
    scene.traversable = traversable;
    scene.tCount = uint32_t(tris.size());
    scene.tris = &tris_buffer;
    scene.pCount = uint32_t(elems.size());
    scene.elems = &elems_buffer;
    bonsai_buffer rays_buffer =
        bonsai_buffer_wrap(rays.data(), rays.size() * sizeof(rays[0]));
    bonsai_buffer p0_buffer =
        bonsai_buffer_wrap(hit_p0.data(), hit_p0.size() * sizeof(hit_p0[0]));
    bonsai_buffer kind_buffer =
        bonsai_buffer_wrap(hit_kind.data(), hit_kind.size() * sizeof(int32_t));
    bonsai_buffer mesh_buffer =
        bonsai_buffer_wrap(hit_mesh.data(), hit_mesh.size() * sizeof(int32_t));
    bonsai_buffer any_buffer =
        bonsai_buffer_wrap(hit_any.data(), hit_any.size() * sizeof(int32_t));
    bonsai_buffer solo_buffer = bonsai_buffer_wrap(
        solo_pool.data(), solo_pool.size() * sizeof(solo_pool[0]));
    bonsai_buffer placed_buffer = bonsai_buffer_wrap(
        placed_pool.data(), placed_pool.size() * sizeof(placed_pool[0]));
    bonsai_buffer inst_buffer = bonsai_buffer_wrap(
        inst_pool.data(), inst_pool.size() * sizeof(inst_pool[0]));
    float near[1] = {0.001f};
    bonsai_buffer near_buffer = bonsai_buffer_wrap(near, sizeof(near));
    // The externs in the order they were declared, the pools in the arms'
    // order (Solo, Placed, Inst).
    bonsai_buffer *buffers[] = {&rays_buffer,  &p0_buffer,     &kind_buffer,
                                &mesh_buffer,  &any_buffer,    &near_buffer,
                                &tris_buffer,  &elems_buffer,  &solo_buffer,
                                &placed_buffer, &inst_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(trace_all_sides));
    bonsai_buffer_stage_all(buffers, trace_all_sides, sizeof(trace_all_sides));
    bonsai_buffer_implicit_copies(0);
    trace_all(n, &rays_buffer, &p0_buffer, &kind_buffer, &mesh_buffer, &any_buffer,
              &near_buffer, scene, &solo_buffer, &placed_buffer, &inst_buffer);
    bonsai_buffer_stage(&p0_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&kind_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&mesh_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&any_buffer, BONSAI_HOST);

    std::cout << std::fixed << std::setprecision(2);
    for (int32_t i = 0; i < n; i++) {
        std::cout << names[i] << ": ";
        if (hit_kind[i] < 0) {
            std::cout << "miss";
        } else {
            const char *kinds[3] = {"solo", "instance", "placed"};
            std::cout << "hit " << kinds[hit_kind[i]];
            if (hit_kind[i] == 2) {
                std::cout << " mesh=" << hit_mesh[i];
            }
            std::cout << " tri=(" << hit_p0[i][0] << ", " << hit_p0[i][1] << ", "
                      << hit_p0[i][2] << ")";
        }
        std::cout << " any=" << (hit_any[i] == 1 ? "yes" : "no") << '\n';
    }
    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
}
