// The driver of rtcore-holder-reference.bonsai: two triangle meshes and a
// standalone triangle on the RT cores, the meshes' triangles stored nowhere.
// Each mesh is one build input built from the mesh's own positions and
// indices, based at 0 -- the hit's primitive index is the triangle's index
// in the mesh -- with the mesh primitive as the input's holder, as
// apps/pbrt/render_hook.cpp builds the scene's own meshes; the layout
// computes the triangle from (holder, index) and the driver uploads no
// row per triangle. Five rays along +x, each with its expected answer.
#include "rtcore-holder-reference.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

// pbrt: a TriangleMesh's buffers, every mesh's together, with where each
// mesh's begin.
struct Meshes {
    std::vector<uint32_t> first_index, first_vertex;
    std::vector<uint32_t> indices;
    std::vector<float3> positions;
    std::vector<uint32_t> counts;

    // A mesh of `tris.size() / 3` triangles over its own vertices.
    uint32_t add(const std::vector<float3> &vertices, const std::vector<uint32_t> &tris) {
        first_index.push_back(uint32_t(indices.size()));
        first_vertex.push_back(uint32_t(positions.size()));
        indices.insert(indices.end(), tris.begin(), tris.end());
        positions.insert(positions.end(), vertices.begin(), vertices.end());
        counts.push_back(uint32_t(tris.size() / 3));
        return uint32_t(counts.size() - 1);
    }
};

// A mesh as one build input: its own buffers, based at 0, the mesh
// primitive its holder, and the input's hit-group record: the holder's
// fields as the program's _RTRecord lays them out, which the hit programs
// read instead of loading the mesh primitive (pbrt's TriangleMeshRecord).
// The program's bonsai_rt_record_fill writes it from the element; the
// runtime takes the bytes after the record's own two words, base and
// holder (bonsai_optix_record). The hardware takes three floats a vertex,
// tightly packed, where the program's vec3f is padded to four, so the
// positions are repacked. The any-hit programs matter for every triangle
// here (the filter's `distmin > near` is no triangle's to settle).
struct Input {
    std::vector<float> positions;
    _RTRecord record = {};
    bonsai_optix_triangles input;
};

// The filler takes, beside the tree and the Mesh pool, the arrays the hit
// programs read at the record's mesh -- the mesh's first index and first
// vertex -- whose values the record carries (SSA/RecordHoist.h), so that a
// hit goes from the record to the mesh's indices in one load.
void fill_record(Input &in, uint32_t holder, const _tree_layout0 &prims,
                 const Mesh *mesh_pool, const Meshes &meshes) {
    bonsai_rt_record_fill(holder, &in.record, prims, mesh_pool, meshes.first_index.data(),
                          meshes.first_vertex.data());
    in.input.record = reinterpret_cast<const unsigned char *>(&in.record) +
                      2 * sizeof(uint32_t);
}

Input input_of(const Meshes &m, uint32_t mesh, uint32_t holder) {
    Input in;
    uint32_t vertex_count = 0;
    for (uint32_t j = 0; j < 3 * m.counts[mesh]; j++) {
        vertex_count = std::max(vertex_count, m.indices[m.first_index[mesh] + j] + 1);
    }
    for (uint32_t v = 0; v < vertex_count; v++) {
        const float3 &p = m.positions[m.first_vertex[mesh] + v];
        in.positions.push_back(p[0]);
        in.positions.push_back(p[1]);
        in.positions.push_back(p[2]);
    }
    in.input.positions = in.positions.data();
    in.input.vertex_count = vertex_count;
    in.input.indices = m.indices.data() + m.first_index[mesh];
    in.input.triangle_count = m.counts[mesh];
    in.input.base = 0;
    in.input.any_hit = true;
    in.input.holder = holder;
    return in;
}

} // namespace

int main() {
    // Mesh 0: two triangles in the x = 0 plane, one per z lane. Mesh 1: one
    // triangle in the x = 2 plane at z = 20. A standalone triangle in the
    // x = -3 plane in front of mesh 0's first triangle.
    Meshes meshes;
    const uint32_t mesh0 = meshes.add(
        {float3{0.0f, -1.0f, -1.5f}, float3{0.0f, 1.0f, -1.5f}, float3{0.0f, 0.0f, -0.5f},
         float3{0.0f, -1.0f, 0.5f}, float3{0.0f, 1.0f, 0.5f}, float3{0.0f, 0.0f, 1.5f}},
        {0, 1, 2, 3, 4, 5});
    const uint32_t mesh1 = meshes.add(
        {float3{2.0f, -1.0f, 19.5f}, float3{2.0f, 1.0f, 19.5f}, float3{2.0f, 0.0f, 20.5f}},
        {0, 1, 2});
    const Triangle solo = Triangle{float3{-3.0f, -1.0f, -1.5f}, float3{-3.0f, 1.0f, -1.5f},
                                   float3{-3.0f, 0.0f, -0.5f}};

    bonsai_gpu_prepare();

    // The elements: the standalone triangle, then the two meshes. Under
    // `layout Prim = tagged_index` an element is a handle into the arm's
    // pool. A mesh's `tris` field stores nothing -- its triangles are
    // computed from the mesh and the index -- so the stored Mesh is its
    // mesh index alone, and the constructor takes no `tris`.
    std::vector<Solo> solo_pool(1);
    uint64_t solo_fill = 0;
    std::vector<Mesh> mesh_pool(2);
    uint64_t mesh_fill = 0;
    static_assert(sizeof(Mesh) == sizeof(uint32_t));
    std::vector<uint64_t> elems;
    elems.push_back(Prim_Solo(solo, solo_pool.data(), &solo_fill));
    const uint32_t holder0 = uint32_t(elems.size());
    elems.push_back(Prim_Mesh(mesh0, mesh_pool.data(), &mesh_fill));
    const uint32_t holder1 = uint32_t(elems.size());
    elems.push_back(Prim_Mesh(mesh1, mesh_pool.data(), &mesh_fill));
    assert(solo_fill == solo_pool.size());
    assert(mesh_fill == mesh_pool.size());

    // The tree's storage: the elements, and the traversable once the
    // structures are built.
    bonsai_buffer elems_buffer =
        bonsai_buffer_wrap(elems.data(), elems.size() * sizeof(elems[0]));
    _tree_layout0 scene;
    scene.pCount = uint32_t(elems.size());
    scene.elems = &elems_buffer;

    // The structures: the standalone triangle, one input based where the
    // Solo element is in `elems`; the meshes, one input each with the mesh
    // primitive as holder and its record filled from that primitive; both
    // placed once by the identity, the meshes' inputs' own holders
    // applying.
    std::vector<float> solo_positions;
    std::vector<uint32_t> solo_indices;
    for (const float3 &p : {solo.p0, solo.p1, solo.p2}) {
        solo_indices.push_back(uint32_t(solo_positions.size() / 3));
        solo_positions.push_back(p[0]);
        solo_positions.push_back(p[1]);
        solo_positions.push_back(p[2]);
    }
    bonsai_optix_triangles solo_input = {};
    solo_input.positions = solo_positions.data();
    solo_input.vertex_count = 3;
    solo_input.indices = solo_indices.data();
    solo_input.triangle_count = 1;
    solo_input.base = 0;
    solo_input.any_hit = true;
    solo_input.holder = BONSAI_OPTIX_OWN;
    const uint64_t geometry_solo = bonsai_optix_geometry(&solo_input, 1, nullptr, 0);
    Input m0 = input_of(meshes, mesh0, holder0);
    Input m1 = input_of(meshes, mesh1, holder1);
    fill_record(m0, holder0, scene, mesh_pool.data(), meshes);
    fill_record(m1, holder1, scene, mesh_pool.data(), meshes);
    const bonsai_optix_triangles mesh_inputs[2] = {m0.input, m1.input};
    const uint64_t geometry_meshes = bonsai_optix_geometry(mesh_inputs, 2, nullptr, 0);
    std::vector<bonsai_optix_instance> instances;
    for (const uint64_t g : {geometry_solo, geometry_meshes}) {
        bonsai_optix_instance inst = {};
        const float identity[12] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
        std::copy(identity, identity + 12, inst.transform);
        inst.geometry = g;
        inst.holder = BONSAI_OPTIX_OWN;
        instances.push_back(inst);
    }
    const uint64_t traversable =
        bonsai_optix_scene(instances.data(), int64_t(instances.size()));

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const auto ray = [&](float y, float z) {
        return Ray{float3{-10.0f, y, z}, forward};
    };
    const std::array<const char *, 5> names = {"solo in front ", "mesh 0 tri 1  ",
                                               "mesh 1        ", "empty lane    ",
                                               "between       "};
    std::vector<Ray> rays = {ray(0.0f, -1.0f), ray(0.0f, 1.0f), ray(0.0f, 20.0f),
                             ray(0.0f, 40.0f), ray(0.9f, -1.0f)};
    const int32_t n = int32_t(rays.size());
    std::vector<float3> hit_p0(n);
    std::vector<int32_t> hit_mesh(n, -3), hit_any(n, -3);

    scene.traversable = traversable;
    bonsai_buffer first_index_buffer = bonsai_buffer_wrap(
        meshes.first_index.data(), meshes.first_index.size() * sizeof(uint32_t));
    bonsai_buffer first_vertex_buffer = bonsai_buffer_wrap(
        meshes.first_vertex.data(), meshes.first_vertex.size() * sizeof(uint32_t));
    bonsai_buffer indices_buffer = bonsai_buffer_wrap(
        meshes.indices.data(), meshes.indices.size() * sizeof(uint32_t));
    bonsai_buffer positions_buffer = bonsai_buffer_wrap(
        meshes.positions.data(), meshes.positions.size() * sizeof(float3));
    bonsai_buffer rays_buffer =
        bonsai_buffer_wrap(rays.data(), rays.size() * sizeof(rays[0]));
    bonsai_buffer p0_buffer =
        bonsai_buffer_wrap(hit_p0.data(), hit_p0.size() * sizeof(hit_p0[0]));
    bonsai_buffer mesh_buffer =
        bonsai_buffer_wrap(hit_mesh.data(), hit_mesh.size() * sizeof(int32_t));
    bonsai_buffer any_buffer =
        bonsai_buffer_wrap(hit_any.data(), hit_any.size() * sizeof(int32_t));
    bonsai_buffer solo_buffer = bonsai_buffer_wrap(
        solo_pool.data(), solo_pool.size() * sizeof(solo_pool[0]));
    bonsai_buffer mesh_pool_buffer = bonsai_buffer_wrap(
        mesh_pool.data(), mesh_pool.size() * sizeof(mesh_pool[0]));
    float near[1] = {0.001f};
    bonsai_buffer near_buffer = bonsai_buffer_wrap(near, sizeof(near));
    // The externs in the order they were declared, the pools in the arms'
    // order (Solo, Mesh).
    bonsai_buffer *buffers[] = {&rays_buffer,        &p0_buffer,     &mesh_buffer,
                                &any_buffer,         &first_index_buffer,
                                &first_vertex_buffer, &indices_buffer,
                                &positions_buffer,   &near_buffer,   &elems_buffer,
                                &solo_buffer,        &mesh_pool_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(trace_all_sides));
    bonsai_buffer_stage_all(buffers, trace_all_sides, sizeof(trace_all_sides));
    bonsai_buffer_implicit_copies(0);
    trace_all(n, &rays_buffer, &p0_buffer, &mesh_buffer, &any_buffer,
              &first_index_buffer, &first_vertex_buffer, &indices_buffer,
              &positions_buffer, &near_buffer, scene, &solo_buffer, &mesh_pool_buffer);
    bonsai_buffer_stage(&p0_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&mesh_buffer, BONSAI_HOST);
    bonsai_buffer_stage(&any_buffer, BONSAI_HOST);

    std::cout << std::fixed << std::setprecision(2);
    for (int32_t i = 0; i < n; i++) {
        std::cout << names[i] << ": ";
        if (hit_mesh[i] == -2) {
            std::cout << "miss";
        } else {
            std::cout << "hit mesh=" << hit_mesh[i] << " tri=(" << hit_p0[i][0] << ", "
                      << hit_p0[i][1] << ", " << hit_p0[i][2] << ")";
        }
        std::cout << " any=" << (hit_any[i] == 1 ? "yes" : "no") << '\n';
    }
    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
}
