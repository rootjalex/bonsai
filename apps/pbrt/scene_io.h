#pragma once

// The flat scene that scene_dump.cpp writes and render_hook.cpp reads.
//
// This is a transport format, not a scene description: it exists because
// PBRT's parser and the bonsai renderer cannot be linked into one program (see
// the comment at the top of scene_dump.cpp), and it carries exactly what the
// renderer needs and nothing else. Nobody writes one of these by hand -- the
// .pbrt file remains the only authored description of a scene.
//
// Written and read by the same struct definitions on the same machine, so the
// layout is whatever the compiler picks. The magic and version are here to
// catch a stale file rather than to make the format portable.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace bonsai_scene {

constexpr char Magic[8] = {'b', 'o', 'n', 's', 'a', 'i', 's', 'c'};
constexpr uint32_t Version = 1;

enum ShapeTag : uint32_t {
    Sphere = 0,
    Triangle = 1,
};

// One shape, in render space. A sphere uses p0 as its centre and radius;
// a triangle uses the three points and ignores radius. The union that the
// renderer's own Shape type is remains the renderer's business -- the tags
// here are this file's, and the driver maps them across by calling the
// generated constructors.
struct Shape {
    uint32_t tag;
    float radius;
    float p0[3];
    float p1[3];
    float p2[3];
};

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t shape_count;
};

// Followed by two 4x4 row-major float matrices -- camera_from_raster then
// render_from_camera -- and then shape_count Shapes.

constexpr size_t MatrixFloats = 32;

inline bool read(const char *path, Header &header, std::vector<float> &matrices,
                 std::vector<Shape> &shapes) {
    FILE *in = fopen(path, "rb");
    if (!in) {
        return false;
    }
    bool ok = fread(&header, sizeof(header), 1, in) == 1 &&
              memcmp(header.magic, Magic, sizeof(header.magic)) == 0 &&
              header.version == Version;
    if (ok) {
        matrices.resize(MatrixFloats);
        shapes.resize(header.shape_count);
        ok = fread(matrices.data(), sizeof(float), MatrixFloats, in) ==
                 MatrixFloats &&
             fread(shapes.data(), sizeof(Shape), header.shape_count, in) ==
                 header.shape_count;
    }
    fclose(in);
    return ok;
}

} // namespace bonsai_scene
