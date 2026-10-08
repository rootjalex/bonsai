// A triangle mesh read from a PLY file, for the drivers that compare a
// program against a reference on pbrt's meshes: apps/rtq's (rtq_hook.cpp)
// and apps/wosx's (wosx_hook.cpp). A small vector, the mesh as positions
// and index triples with its box, and the reader: a PLY file, plain or
// gzipped (zlib reads both), binary in either byte order, the vertex
// positions and the faces triangulated as fans, which is how pbrt's scenes
// store their meshes and how pbrt triangulates them.
#pragma once

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace rtq {

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(Vec3 a) {
    const float l = length(a);
    return l > 0 ? a * (1.0f / l) : a;
}
inline Vec3 vmin(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}
inline Vec3 vmax(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<uint32_t> indices; // three per triangle
    size_t triangles() const { return indices.size() / 3; }
    Vec3 lo{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()},
        hi{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
           -std::numeric_limits<float>::infinity()};
};

struct PlyReader {
    gzFile file = nullptr;
    bool big_endian = false;

    explicit PlyReader(const std::string &path) {
        file = gzopen(path.c_str(), "rb");
        if (file == nullptr) {
            std::cerr << "cannot open " << path << '\n';
            std::exit(1);
        }
    }
    ~PlyReader() {
        if (file != nullptr) {
            gzclose(file);
        }
    }
    PlyReader(const PlyReader &) = delete;
    PlyReader &operator=(const PlyReader &) = delete;

    std::string line() {
        char buffer[4096];
        if (gzgets(file, buffer, sizeof buffer) == nullptr) {
            std::cerr << "unexpected end of PLY header\n";
            std::exit(1);
        }
        std::string s(buffer);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        return s;
    }

    void read_bytes(void *out, size_t n) {
        if (gzread(file, out, unsigned(n)) != int(n)) {
            std::cerr << "unexpected end of PLY data\n";
            std::exit(1);
        }
    }

    template <typename T>
    T read() {
        T v;
        read_bytes(&v, sizeof v);
        if (big_endian) {
            char *p = reinterpret_cast<char *>(&v);
            std::reverse(p, p + sizeof v);
        }
        return v;
    }

    // One scalar of the named PLY type, as a double.
    double read_scalar(const std::string &type) {
        if (type == "float" || type == "float32") {
            return read<float>();
        } else if (type == "double" || type == "float64") {
            return read<double>();
        } else if (type == "uchar" || type == "uint8") {
            return read<uint8_t>();
        } else if (type == "char" || type == "int8") {
            return read<int8_t>();
        } else if (type == "ushort" || type == "uint16") {
            return read<uint16_t>();
        } else if (type == "short" || type == "int16") {
            return read<int16_t>();
        } else if (type == "uint" || type == "uint32") {
            return read<uint32_t>();
        } else if (type == "int" || type == "int32") {
            return read<int32_t>();
        }
        std::cerr << "PLY property of unknown type " << type << '\n';
        std::exit(1);
    }
};

inline Mesh load_ply(const std::string &path) {
    PlyReader in(path);
    if (in.line() != "ply") {
        std::cerr << path << " is not a PLY file\n";
        std::exit(1);
    }
    struct Property {
        std::string type;      // or the list's count type
        std::string item_type; // for a list
        std::string name;
        bool list = false;
    };
    struct Element {
        std::string name;
        size_t count = 0;
        std::vector<Property> properties;
    };
    std::vector<Element> elements;
    for (std::string l = in.line(); l != "end_header"; l = in.line()) {
        if (l.rfind("format ", 0) == 0) {
            if (l.find("binary_big_endian") != std::string::npos) {
                in.big_endian = true;
            } else if (l.find("binary_little_endian") == std::string::npos) {
                std::cerr << "only binary PLY is read: " << l << '\n';
                std::exit(1);
            }
        } else if (l.rfind("element ", 0) == 0) {
            Element e;
            char name[256];
            unsigned long long count = 0;
            if (std::sscanf(l.c_str(), "element %255s %llu", name, &count) != 2) {
                std::cerr << "bad PLY element line: " << l << '\n';
                std::exit(1);
            }
            e.name = name;
            e.count = size_t(count);
            elements.push_back(e);
        } else if (l.rfind("property ", 0) == 0) {
            if (elements.empty()) {
                std::cerr << "PLY property before any element\n";
                std::exit(1);
            }
            Property p;
            char a[64], b[64], c[64];
            if (std::sscanf(l.c_str(), "property list %63s %63s %63s", a, b, c) == 3) {
                p.list = true;
                p.type = a;
                p.item_type = b;
                p.name = c;
            } else if (std::sscanf(l.c_str(), "property %63s %63s", a, b) == 2) {
                p.type = a;
                p.name = b;
            } else {
                std::cerr << "bad PLY property line: " << l << '\n';
                std::exit(1);
            }
            elements.back().properties.push_back(p);
        }
    }

    Mesh mesh;
    for (const Element &e : elements) {
        if (e.name == "vertex") {
            int ix = -1, iy = -1, iz = -1;
            for (size_t k = 0; k < e.properties.size(); k++) {
                if (e.properties[k].name == "x") ix = int(k);
                if (e.properties[k].name == "y") iy = int(k);
                if (e.properties[k].name == "z") iz = int(k);
            }
            if (ix < 0 || iy < 0 || iz < 0) {
                std::cerr << "PLY vertices without x, y, z\n";
                std::exit(1);
            }
            mesh.vertices.reserve(e.count);
            for (size_t i = 0; i < e.count; i++) {
                Vec3 v;
                for (size_t k = 0; k < e.properties.size(); k++) {
                    const Property &p = e.properties[k];
                    if (p.list) {
                        const size_t n = size_t(in.read_scalar(p.type));
                        for (size_t j = 0; j < n; j++) {
                            in.read_scalar(p.item_type);
                        }
                        continue;
                    }
                    const double value = in.read_scalar(p.type);
                    if (int(k) == ix) v.x = float(value);
                    if (int(k) == iy) v.y = float(value);
                    if (int(k) == iz) v.z = float(value);
                }
                mesh.vertices.push_back(v);
                mesh.lo = vmin(mesh.lo, v);
                mesh.hi = vmax(mesh.hi, v);
            }
        } else if (e.name == "face") {
            for (size_t i = 0; i < e.count; i++) {
                for (const Property &p : e.properties) {
                    if (!p.list) {
                        in.read_scalar(p.type);
                        continue;
                    }
                    const size_t n = size_t(in.read_scalar(p.type));
                    std::vector<uint32_t> corners(n);
                    for (size_t j = 0; j < n; j++) {
                        corners[j] = uint32_t(in.read_scalar(p.item_type));
                    }
                    if (p.name != "vertex_indices" && p.name != "vertex_index") {
                        continue;
                    }
                    // A polygon as a fan, which is how pbrt triangulates.
                    for (size_t j = 1; j + 1 < n; j++) {
                        mesh.indices.push_back(corners[0]);
                        mesh.indices.push_back(corners[j]);
                        mesh.indices.push_back(corners[j + 1]);
                    }
                }
            }
        } else {
            // Something else (edges, say): read past it.
            for (size_t i = 0; i < e.count; i++) {
                for (const Property &p : e.properties) {
                    if (!p.list) {
                        in.read_scalar(p.type);
                        continue;
                    }
                    const size_t n = size_t(in.read_scalar(p.type));
                    for (size_t j = 0; j < n; j++) {
                        in.read_scalar(p.item_type);
                    }
                }
            }
        }
    }
    for (uint32_t index : mesh.indices) {
        if (index >= mesh.vertices.size()) {
            std::cerr << "PLY face names vertex " << index << " of " << mesh.vertices.size()
                      << '\n';
            std::exit(1);
        }
    }
    return mesh;
}

} // namespace rtq
