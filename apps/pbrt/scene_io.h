#pragma once

// The flat scene that scene_dump.cpp writes and render_hook.cpp reads.
//
// This is a transport format, not a scene description. It exists because PBRT's
// parser and the bonsai renderer cannot be linked into one program (see the
// comment at the top of scene_dump.cpp), so the two halves hand off through a
// file. Nobody writes one by hand -- the .pbrt file remains the only authored
// description of a scene, and this is derived from it every run.
//
// Text, so that a scene which renders wrong can be read rather than hex-dumped,
// and so that the file needs no magic number or version to guard it: a stale or
// truncated one fails when a field does not parse. Parsing is slower than
// blitting structs would be, which does not matter -- loading a scene is
// preprocessing, and is outside what either renderer counts as render time.
//
// Floats are written with nine significant digits, which is exactly what it
// takes to read a float back unchanged. Anything less would quietly move
// geometry, and the whole point of this app is that it does not.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace bonsai_scene {

enum ShapeTag : uint32_t {
    Sphere = 0,
    Triangle = 1,
};

// One shape, in render space. A sphere uses p0 as its centre and radius; a
// triangle uses the three points and ignores radius. The renderer's own Shape
// is a variant type and stays its business: these tags are this file's, and the
// driver maps across by calling the generated constructors.
struct Shape {
    uint32_t tag;
    float radius;
    float p0[3];
    float p1[3];
    float p2[3];
};

// One BVH node, in PBRT's LinearBVHNode shape.
//
// Present only when the scene was dumped with --pbrt-tree, in which case it is
// PBRT's own tree -- the nodes its BVHAggregate built and flattened -- and the
// shapes are in the order its leaves expect, so a leaf's `offset` indexes them
// directly. Without a tree the driver builds its own, which is the general
// case: PBRT can only hand over a tree of the shape PBRT builds, so a schedule
// wanting a wider arity or a different bounding volume has to build its own.
struct Node {
    float low[3];
    float high[3];
    // Leaf: the first of n_prims shapes. Interior: the second child, relative
    // to this node, the first being the next node along.
    uint32_t offset;
    uint16_t n_prims; // 0 for an interior node.
    uint16_t axis;
};

struct Scene {
    uint32_t width = 0;
    uint32_t height = 0;
    // camera_from_raster then render_from_camera, each 4x4 in row order.
    float matrices[32] = {};
    std::vector<Shape> shapes;
    std::vector<Node> nodes;
};

namespace detail {

inline void put(std::ofstream &out, const float *v, int n) {
    char buf[32];
    for (int i = 0; i < n; i++) {
        snprintf(buf, sizeof(buf), " %.9g", double(v[i]));
        out << buf;
    }
}

} // namespace detail

inline bool write(const char *path, const Scene &scene) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "resolution " << scene.width << ' ' << scene.height << '\n';
    out << "camera_from_raster";
    detail::put(out, scene.matrices, 16);
    out << "\nrender_from_camera";
    detail::put(out, scene.matrices + 16, 16);
    out << '\n';

    out << "shapes " << scene.shapes.size() << '\n';
    for (const Shape &s : scene.shapes) {
        if (s.tag == ShapeTag::Sphere) {
            out << "  sphere";
            detail::put(out, s.p0, 3);
            detail::put(out, &s.radius, 1);
        } else {
            out << "  tri";
            detail::put(out, s.p0, 3);
            detail::put(out, s.p1, 3);
            detail::put(out, s.p2, 3);
        }
        out << '\n';
    }

    out << "nodes " << scene.nodes.size() << '\n';
    for (const Node &n : scene.nodes) {
        out << (n.n_prims == 0 ? "  interior" : "  leaf");
        detail::put(out, n.low, 3);
        detail::put(out, n.high, 3);
        if (n.n_prims == 0) {
            out << " axis " << n.axis << " right " << n.offset;
        } else {
            out << " first " << n.offset << " count " << n.n_prims;
        }
        out << '\n';
    }
    return bool(out);
}

inline bool read(const char *path, Scene &scene) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string word;

    auto floats = [&](float *v, int n) {
        for (int i = 0; i < n; i++) {
            in >> v[i];
        }
    };

    if (!(in >> word) || word != "resolution") {
        return false;
    }
    in >> scene.width >> scene.height;

    if (!(in >> word) || word != "camera_from_raster") {
        return false;
    }
    floats(scene.matrices, 16);
    if (!(in >> word) || word != "render_from_camera") {
        return false;
    }
    floats(scene.matrices + 16, 16);

    size_t count = 0;
    if (!(in >> word) || word != "shapes") {
        return false;
    }
    in >> count;
    scene.shapes.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word)) {
            return false;
        }
        Shape s = {};
        if (word == "sphere") {
            s.tag = ShapeTag::Sphere;
            floats(s.p0, 3);
            floats(&s.radius, 1);
        } else if (word == "tri") {
            s.tag = ShapeTag::Triangle;
            floats(s.p0, 3);
            floats(s.p1, 3);
            floats(s.p2, 3);
        } else {
            return false;
        }
        scene.shapes.push_back(s);
    }

    if (!(in >> word) || word != "nodes") {
        return false;
    }
    in >> count;
    scene.nodes.clear();
    for (size_t i = 0; i < count; i++) {
        if (!(in >> word)) {
            return false;
        }
        Node n = {};
        const bool interior = word == "interior";
        if (!interior && word != "leaf") {
            return false;
        }
        floats(n.low, 3);
        floats(n.high, 3);
        std::string a, b;
        uint32_t x = 0, y = 0;
        in >> a >> x >> b >> y;
        if (interior) {
            if (a != "axis" || b != "right") {
                return false;
            }
            n.axis = uint16_t(x);
            n.offset = y;
            n.n_prims = 0;
        } else {
            if (a != "first" || b != "count") {
                return false;
            }
            n.offset = x;
            n.n_prims = uint16_t(y);
        }
        scene.nodes.push_back(n);
    }

    return bool(in);
}

} // namespace bonsai_scene
