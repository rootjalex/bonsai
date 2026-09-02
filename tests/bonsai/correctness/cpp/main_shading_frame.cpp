#include "shading-frame.h"

#include <cstdio>
#include <vector>

// Used in shading-frame.bonsai.
//
// The three shapes of mesh apps/pbrt can load, built the way its driver builds
// them: one pool of positions, one of normals, one of texture coordinates, and
// a mesh naming where its own run of each begins. A `loopsubdiv` shape brings
// per-vertex normals and no texture coordinates, so the tangent comes from
// pbrt's default (0,0), (1,0), (1,1) parameterization and is then made
// perpendicular to the interpolated normal. A `trianglemesh` with `uv` brings
// the coordinates and no normals, so the shading geometry is the geometry. The
// third has both, with the orientation reversed -- which pbrt applies to the
// vertex normals when it builds the mesh, so they are negated here too.
//
// The vertices are the same four every time, appended once per mesh, because a
// mesh owns its run of the pool. Two triangles share an edge; only the second
// is hit by these rays, so it is the only one asked about.
namespace {

const float3 kP[4] = {{-36.876f, 26.033f, -137.748f},
                      {-37.039f, 21.507f, -137.128f},
                      {-35.466f, -2.075f, -129.337f},
                      {-30.348f, -8.431f, -132.687f}};

// Not unit vectors, and not agreeing with the face they sit on: a subdivision
// surface's limit normals are neither.
const float3 kN[4] = {{0.31f, 0.42f, 0.85f},
                      {0.09f, 0.55f, 0.83f},
                      {-0.22f, 0.61f, 0.76f},
                      {0.40f, -0.13f, 0.91f}};

const float2 kUV[4] = {{0.f, 0.f}, {0.7f, 0.1f}, {0.2f, 0.9f}, {1.f, 1.f}};

const uint32_t kIndices[6] = {0, 1, 2, 2, 1, 3};

} // namespace

int main() {
    std::vector<TriangleMesh> meshes;
    std::vector<uint32_t> indices;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;

    const auto add_mesh = [&](bool with_normals, bool with_uv, bool reverse) {
        TriangleMesh m;
        m.first_index = uint32_t(indices.size());
        m.first_vertex = uint32_t(positions.size());
        m.first_normal = uint32_t(normals.size());
        m.first_uv = uint32_t(uvs.size());
        m.has_normals = with_normals;
        m.has_uv = with_uv;
        m.flip = reverse;
        for (uint32_t i : kIndices) {
            indices.push_back(i);
        }
        for (int i = 0; i < 4; i++) {
            positions.push_back(kP[i]);
            if (with_normals) {
                normals.push_back(reverse ? -kN[i] : kN[i]);
            }
            if (with_uv) {
                uvs.push_back(kUV[i]);
            }
        }
        meshes.push_back(m);
    };
    add_mesh(/*with_normals=*/true, /*with_uv=*/false, /*reverse=*/false);
    add_mesh(/*with_normals=*/false, /*with_uv=*/true, /*reverse=*/false);
    add_mesh(/*with_normals=*/true, /*with_uv=*/true, /*reverse=*/true);

    // Two directions through the triangle, far enough apart that a wrong
    // interpolation would not agree at both. Normalized by pbrt before it fired
    // them, so these are the directions `--print-shading` reports.
    const float3 kRays[2] = {{-0.155709431f, -0.578349292f, -0.800791323f},
                             {-0.138501793f, -0.639239013f, -0.756432831f}};

    for (uint32_t mesh = 0; mesh < meshes.size(); mesh++) {
        for (const float3 &d : kRays) {
            float out[22] = {};
            if (!probe(mesh, /*tri=*/1, d, out, meshes.data(), indices.data(),
                       positions.data(), normals.data(), uvs.data())) {
                // Not a case any of these rays takes, and printing something
                // keeps a miss from silently shifting every line below it.
                printf("-1\n");
                continue;
            }
            for (const float v : out) {
                printf("%f\n", v);
            }
        }
    }
    return 0;
}
