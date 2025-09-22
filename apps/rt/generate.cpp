#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct vec3_float {
    float x, y, z;
    vec3_float(std::initializer_list<float> list) {
        auto it = list.begin();
        x = *it++;
        y = *it++;
        z = *it;
    }
    vec3_float() : x(0), y(0), z(0) {}
};

struct Triangle {
    vec3_float p0, p1, p2;
};

struct Ray {
    vec3_float o; // origin
    vec3_float d; // direction
};

// TODO(cgyurgyik): duplicated from main.cpp
std::vector<Triangle> load_obj(const std::string &object) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai")) {
            break;
        }
        current_path = current_path.parent_path();
    }

    std::string object_path = "apps/data/" + object + ".obj";
    std::string material_path = "apps/data/" + object;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string _, err;
    bool result = tinyobj::LoadObj(&attrib, &shapes, &materials, &_, &err,
                                   object_path.c_str(), material_path.c_str());
    if (!err.empty()) {
        std::cerr << "error: " << err << std::endl;
    }
    if (!result) {
        std::cerr << "failed to load " << object_path << std::endl;
        return {};
    }

    std::vector<Triangle> triangles;
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;

        // Loop over faces (triangles)
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv =
                shapes[s]
                    .mesh.num_face_vertices[f]; // Should be 3 for triangles

            if (fv == 3) {
                Triangle tri;

                // Get vertices
                for (int v = 0; v < 3; v++) {
                    tinyobj::index_t idx =
                        shapes[s].mesh.indices[index_offset + v];

                    float x = attrib.vertices[3 * idx.vertex_index + 0];
                    float y = attrib.vertices[3 * idx.vertex_index + 1];
                    float z = attrib.vertices[3 * idx.vertex_index + 2];

                    if (v == 0)
                        tri.p0 = {x, y, z};
                    else if (v == 1)
                        tri.p1 = {x, y, z};
                    else
                        tri.p2 = {x, y, z};
                }

                triangles.push_back(tri);
            }

            index_offset += fv;
        }
    }
    return triangles;
}

bool save_rays_binary(const std::vector<Ray> &rays, const std::string &object,
                      const std::string &path) {
    std::string file_name = path + "/" + object + ".rays";
    std::ofstream file(file_name, std::ios::binary);
    if (!file) {
        std::cerr << "Error: could not open file " << file_name
                  << " for writing\n";
        return false;
    }

    size_t count = rays.size();
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));

    for (const Ray &ray : rays) {
        file.write(reinterpret_cast<const char *>(&ray.o.x), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.o.y), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.o.z), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.d.x), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.d.y), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.d.z), sizeof(float));
    }

    file.close();
    return true;
}

inline float random_float(std::mt19937 &generator, float min = -1.0,
                          float max = 1.0) {
    std::uniform_real_distribution<float> urd(min, max);
    return urd(generator);
}

vec3_float random_unit_vector(std::mt19937 &generator) {
    float x = random_float(generator);
    float y = random_float(generator);
    float z = random_float(generator);

    // Normalize the vector
    float length = std::sqrt(x * x + y * y + z * z);
    if (length > 0) {
        x /= length;
        y /= length;
        z /= length;
    }

    return vec3_float({x, y, z});
}

struct BoundingBox {
    vec3_float min, max;
};

BoundingBox compute_bounding_box(const std::vector<Triangle> &triangles) {
    if (triangles.empty())
        return {};

    BoundingBox bbox;
    bbox.min = bbox.max = triangles[0].p0;

    for (const auto &tri : triangles) {
        // Check all three vertices
        vec3_float vertices[3] = {tri.p0, tri.p1, tri.p2};
        for (int i = 0; i < 3; i++) {
            bbox.min.x = std::min(bbox.min.x, vertices[i].x);
            bbox.min.y = std::min(bbox.min.y, vertices[i].y);
            bbox.min.z = std::min(bbox.min.z, vertices[i].z);
            bbox.max.x = std::max(bbox.max.x, vertices[i].x);
            bbox.max.y = std::max(bbox.max.y, vertices[i].y);
            bbox.max.z = std::max(bbox.max.z, vertices[i].z);
        }
    }
    return bbox;
}

// Ray-triangle intersection using Möller-Trumbore algorithm
bool ray_triangle_intersect(const Ray &ray, const Triangle &tri, float &t) {
    constexpr float EPSILON = 0.0000001f;

    vec3_float edge1, edge2, h, s, q;
    float a, f, u, v;

    edge1.x = tri.p1.x - tri.p0.x;
    edge1.y = tri.p1.y - tri.p0.y;
    edge1.z = tri.p1.z - tri.p0.z;

    edge2.x = tri.p2.x - tri.p0.x;
    edge2.y = tri.p2.y - tri.p0.y;
    edge2.z = tri.p2.z - tri.p0.z;

    h.x = ray.d.y * edge2.z - ray.d.z * edge2.y;
    h.y = ray.d.z * edge2.x - ray.d.x * edge2.z;
    h.z = ray.d.x * edge2.y - ray.d.y * edge2.x;

    a = edge1.x * h.x + edge1.y * h.y + edge1.z * h.z;

    if (a > -EPSILON && a < EPSILON)
        return false; // Ray is parallel to triangle

    f = 1.0f / a;

    s.x = ray.o.x - tri.p0.x;
    s.y = ray.o.y - tri.p0.y;
    s.z = ray.o.z - tri.p0.z;

    u = f * (s.x * h.x + s.y * h.y + s.z * h.z);

    if (u < 0.0f || u > 1.0f)
        return false;

    q.x = s.y * edge1.z - s.z * edge1.y;
    q.y = s.z * edge1.x - s.x * edge1.z;
    q.z = s.x * edge1.y - s.y * edge1.x;

    v = f * (ray.d.x * q.x + ray.d.y * q.y + ray.d.z * q.z);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

    return t > EPSILON;
}

bool ray_intersects_mesh(const Ray &ray,
                         const std::vector<Triangle> &triangles) {
    float t;
    for (const Triangle &triangle : triangles) {
        if (ray_triangle_intersect(ray, triangle, t)) {
            return true;
        }
    }
    return false;
}

std::vector<Ray> generate_rays(const std::vector<Triangle> &triangles,
                               int count, float hit_ratio = 0.75f) {
    std::vector<Ray> rays;
    rays.reserve(count);

    std::random_device rd;
    std::mt19937 generator(rd());

    const BoundingBox bbox = compute_bounding_box(triangles);

    const int target_hits = static_cast<int>(count * hit_ratio);
    const int target_misses = count - target_hits;

    int hits_generated = 0;
    int misses_generated = 0;
    while (hits_generated < target_hits || misses_generated < target_misses) {
        Ray candidate_ray;
        candidate_ray.o = vec3_float(
            {random_float(generator, bbox.min.x - 10.0f, bbox.max.x + 10.0f),
             random_float(generator, bbox.min.y - 10.0f, bbox.max.y + 10.0f),
             random_float(generator, bbox.min.z - 10.0f, bbox.max.z + 10.0f)});
        candidate_ray.d = random_unit_vector(generator);

        const bool hit_object = ray_intersects_mesh(candidate_ray, triangles);
        if (hit_object && hits_generated < target_hits) {
            rays.push_back(candidate_ray);
            ++hits_generated;
            continue;
        }
        if (!hit_object && misses_generated < target_misses) {
            rays.push_back(candidate_ray);
            ++misses_generated;
            continue;
        }
    }
    std::shuffle(rays.begin(), rays.end(), generator);
    return rays;
}

vec3_float sample_triangle_point(const Triangle &tri, std::mt19937 &generator) {
    float r1 = random_float(generator, 0.0f, 1.0f);
    float r2 = random_float(generator, 0.0f, 1.0f);

    // Ensure point is inside triangle using barycentric coordinates
    if (r1 + r2 > 1.0f) {
        r1 = 1.0f - r1;
        r2 = 1.0f - r2;
    }

    float r3 = 1.0f - r1 - r2;

    return vec3_float({r1 * tri.p0.x + r2 * tri.p1.x + r3 * tri.p2.x,
                       r1 * tri.p0.y + r2 * tri.p1.y + r3 * tri.p2.y,
                       r1 * tri.p0.z + r2 * tri.p1.z + r3 * tri.p2.z});
}

int main(int argc, char *argv[]) {
    assert(argc == 5);
    std::string object = argv[1];
    std::string path = argv[2];
    int64_t ray_count = std::atoi(argv[3]);
    float hit_ratio = std::stof(argv[4]);
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());
    std::vector<Ray> rays = generate_rays(triangles, ray_count, hit_ratio);
    save_rays_binary(rays, object, path);
}
