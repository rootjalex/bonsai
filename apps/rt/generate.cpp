#include <algorithm>
#include <assert.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

std::string after_token(const std::string &input, char token) {
    size_t pos = input.find(token);
    if (pos == std::string::npos || pos + 1 >= input.size()) {
        return "";
    }
    return input.substr(pos + 1);
}

struct float3 {
    float x, y, z;
    float3(std::initializer_list<float> list) {
        auto it = list.begin();
        x = *it++;
        y = *it++;
        z = *it;
    }
    float3() : x(0), y(0), z(0) {}

    float &operator[](int i) { return (&x)[i]; }

    const float &operator[](int i) const { return (&x)[i]; }

    float3 operator+(const float3 &other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    float3 operator-(const float3 &other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    float3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    float dot(const float3 &other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    float3 cross(const float3 &other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z,
                x * other.y - y * other.x};
    }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    float3 normalize() const {
        float len = length();
        return len > 0 ? (*this) * (1.0f / len) : float3();
    }
};

struct Triangle {
    float3 p0, p1, p2;

    float3 normal() const { return (p1 - p0).cross(p2 - p0).normalize(); }

    float area() const { return (p1 - p0).cross(p2 - p0).length() * 0.5f; }
};

struct Ray {
    float3 o;
    float3 d;
};

struct HitRecord {
    float t;
    float3 point;
    float3 normal;
    int triangle_index;
};

struct BoundingBox {
    float3 min, max;

    BoundingBox() {}
    BoundingBox(const float3 &min_pt, const float3 &max_pt)
        : min(min_pt), max(max_pt) {}

    BoundingBox expand(const BoundingBox &other) const {
        return BoundingBox(
            {std::min(min.x, other.min.x), std::min(min.y, other.min.y),
             std::min(min.z, other.min.z)},
            {std::max(max.x, other.max.x), std::max(max.y, other.max.y),
             std::max(max.z, other.max.z)});
    }

    float3 center() const {
        return {(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f,
                (min.z + max.z) * 0.5f};
    }

    float surface_area() const {
        float dx = max.x - min.x;
        float dy = max.y - min.y;
        float dz = max.z - min.z;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }
};

struct BVHNode {
    BoundingBox bbox;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    std::vector<int> triangle_indices;
    bool is_leaf;

    BVHNode() : is_leaf(false) {}
};

bool ray_triangle_intersect(const Ray &ray, const Triangle &tri, float &t) {
    constexpr float EPSILON = 0.0000001f;

    float3 edge1, edge2, h, s, q;
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
        return false;

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

bool ray_triangle_intersect_full(const Ray &ray, const Triangle &tri,
                                 HitRecord &hit) {
    constexpr float EPSILON = 0.0000001f;

    float3 edge1, edge2, h, s, q;
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
        return false;

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

    hit.t = f * (edge2.x * q.x + edge2.y * q.y + edge2.z * q.z);

    if (hit.t > EPSILON) {
        hit.point = ray.o + ray.d * hit.t;
        hit.normal = tri.normal();
        return true;
    }

    return false;
}

class BVH {
  private:
    std::unique_ptr<BVHNode> root;
    const std::vector<Triangle> *triangles;

    BoundingBox compute_triangle_bbox(const Triangle &tri) const {
        BoundingBox bbox;
        bbox.min = bbox.max = tri.p0;

        bbox.min.x = std::min({bbox.min.x, tri.p1.x, tri.p2.x});
        bbox.min.y = std::min({bbox.min.y, tri.p1.y, tri.p2.y});
        bbox.min.z = std::min({bbox.min.z, tri.p1.z, tri.p2.z});

        bbox.max.x = std::max({bbox.max.x, tri.p1.x, tri.p2.x});
        bbox.max.y = std::max({bbox.max.y, tri.p1.y, tri.p2.y});
        bbox.max.z = std::max({bbox.max.z, tri.p1.z, tri.p2.z});

        return bbox;
    }

    BoundingBox compute_bbox(const std::vector<int> &indices) const {
        if (indices.empty())
            return BoundingBox();

        BoundingBox bbox = compute_triangle_bbox((*triangles)[indices[0]]);
        for (size_t i = 1; i < indices.size(); i++) {
            bbox = bbox.expand(compute_triangle_bbox((*triangles)[indices[i]]));
        }
        return bbox;
    }

    std::unique_ptr<BVHNode> build_recursive(std::vector<int> indices) {
        auto node = std::make_unique<BVHNode>();
        node->bbox = compute_bbox(indices);

        if (indices.size() <= 8) {
            node->is_leaf = true;
            node->triangle_indices = std::move(indices);
            return node;
        }

        float3 bbox_size = node->bbox.max - node->bbox.min;
        int split_axis = 0;
        if (bbox_size.y > bbox_size.x)
            split_axis = 1;
        if (bbox_size.z > bbox_size[split_axis])
            split_axis = 2;

        float split_pos =
            (node->bbox.min[split_axis] + node->bbox.max[split_axis]) * 0.5f;

        auto partition_point = std::partition(
            indices.begin(), indices.end(),
            [this, split_axis, split_pos](int idx) {
                float3 center =
                    compute_triangle_bbox((*triangles)[idx]).center();
                return center[split_axis] < split_pos;
            });

        size_t left_count = partition_point - indices.begin();
        if (left_count == 0 || left_count == indices.size()) {
            left_count = indices.size() / 2;
        }

        std::vector<int> left_indices(indices.begin(),
                                      indices.begin() + left_count);
        std::vector<int> right_indices(indices.begin() + left_count,
                                       indices.end());

        node->left = build_recursive(std::move(left_indices));
        node->right = build_recursive(std::move(right_indices));

        return node;
    }

    bool ray_bbox_intersect(const Ray &ray, const BoundingBox &bbox) const {
        float inv_dx = 1.0f / ray.d.x;
        float inv_dy = 1.0f / ray.d.y;
        float inv_dz = 1.0f / ray.d.z;

        float t1 = (bbox.min.x - ray.o.x) * inv_dx;
        float t2 = (bbox.max.x - ray.o.x) * inv_dx;
        if (t1 > t2)
            std::swap(t1, t2);

        float t3 = (bbox.min.y - ray.o.y) * inv_dy;
        float t4 = (bbox.max.y - ray.o.y) * inv_dy;
        if (t3 > t4)
            std::swap(t3, t4);

        float t5 = (bbox.min.z - ray.o.z) * inv_dz;
        float t6 = (bbox.max.z - ray.o.z) * inv_dz;
        if (t5 > t6)
            std::swap(t5, t6);

        float tmin = std::max({t1, t3, t5});
        float tmax = std::min({t2, t4, t6});

        return tmax >= 0 && tmin <= tmax;
    }

    bool intersect_recursive(const Ray &ray, const BVHNode *node) const {
        if (!ray_bbox_intersect(ray, node->bbox)) {
            return false;
        }

        if (node->is_leaf) {
            float t;
            for (int idx : node->triangle_indices) {
                if (ray_triangle_intersect(ray, (*triangles)[idx], t)) {
                    return true;
                }
            }
            return false;
        }

        return intersect_recursive(ray, node->left.get()) ||
               intersect_recursive(ray, node->right.get());
    }

    bool intersect_full_recursive(const Ray &ray, const BVHNode *node,
                                  HitRecord &closest_hit) const {
        if (!ray_bbox_intersect(ray, node->bbox)) {
            return false;
        }

        if (node->is_leaf) {
            bool hit_anything = false;
            HitRecord temp_hit;
            for (int idx : node->triangle_indices) {
                if (ray_triangle_intersect_full(ray, (*triangles)[idx],
                                                temp_hit)) {
                    if (!hit_anything || temp_hit.t < closest_hit.t) {
                        closest_hit = temp_hit;
                        closest_hit.triangle_index = idx;
                        hit_anything = true;
                    }
                }
            }
            return hit_anything;
        }

        HitRecord left_hit, right_hit;
        left_hit.t = right_hit.t = std::numeric_limits<float>::max();

        bool hit_left =
            intersect_full_recursive(ray, node->left.get(), left_hit);
        bool hit_right =
            intersect_full_recursive(ray, node->right.get(), right_hit);

        if (hit_left && hit_right) {
            closest_hit = (left_hit.t < right_hit.t) ? left_hit : right_hit;
            return true;
        } else if (hit_left) {
            closest_hit = left_hit;
            return true;
        } else if (hit_right) {
            closest_hit = right_hit;
            return true;
        }

        return false;
    }

  public:
    void build(const std::vector<Triangle> &tris) {
        triangles = &tris;
        std::vector<int> indices(tris.size());
        std::iota(indices.begin(), indices.end(), 0);
        root = build_recursive(std::move(indices));
    }

    bool intersect(const Ray &ray) const {
        return root ? intersect_recursive(ray, root.get()) : false;
    }

    bool intersect_full(const Ray &ray, HitRecord &hit) const {
        if (!root)
            return false;
        hit.t = std::numeric_limits<float>::max();
        return intersect_full_recursive(ray, root.get(), hit);
    }
};

std::vector<Triangle> load_obj(const std::string &object) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai")) {
            break;
        }
        current_path = current_path.parent_path();
    }

    std::string object_path = "apps/rt/data/" + object + ".obj";
    std::string material_path = "apps/rt/data/" + object;

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
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;

        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            if (fv == 3) {
                Triangle tri;

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
                      const std::string &ray_count, const std::string &ray_type,
                      const std::string &path) {
    std::string file_name =
        path + "/" + object + "_" + ray_count + "_" + ray_type + ".rays";
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

struct Camera {
    float3 position;
    float3 look_at;
    float3 up;
    float fov;
    float aspect_ratio;

    float3 forward;
    float3 right;
    float3 camera_up;
    float viewport_height;
    float viewport_width;

    void compute_basis() {
        forward = (look_at - position).normalize();
        right = forward.cross(up).normalize();
        camera_up = right.cross(forward);

        viewport_height = 2.0f * std::tan(fov * 0.5f * 3.14159265f / 180.0f);
        viewport_width = viewport_height * aspect_ratio;
    }

    Ray generate_ray(float u, float v) const {
        float x = (u - 0.5f) * viewport_width;
        float y = (v - 0.5f) * viewport_height;

        Ray ray;
        ray.o = position;
        ray.d = (forward + right * x + camera_up * y).normalize();
        return ray;
    }
};

BoundingBox compute_bounding_box(const std::vector<Triangle> &triangles) {
    if (triangles.empty())
        return {};

    BoundingBox bbox;
    bbox.min = bbox.max = triangles[0].p0;

    for (const auto &tri : triangles) {
        float3 vertices[3] = {tri.p0, tri.p1, tri.p2};
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

Camera setup_camera(const BoundingBox &bbox) {
    Camera cam;

    cam.look_at = bbox.center();

    float3 bbox_size = bbox.max - bbox.min;
    float max_dim = std::max({bbox_size.x, bbox_size.y, bbox_size.z});
    float distance = max_dim * 0.6f;

    cam.position =
        cam.look_at + float3{distance * 0.7f, distance * 0.5f, distance * 0.7f};

    cam.up = {0.0f, 1.0f, 0.0f};
    cam.fov = 25.0f;
    cam.aspect_ratio = 16.0f / 9.0f;

    cam.compute_basis();

    return cam;
}

inline float random_float(std::mt19937 &generator, float min = 0.0f,
                          float max = 1.0f) {
    std::uniform_real_distribution<float> urd(min, max);
    return urd(generator);
}

float3 random_unit_vector(std::mt19937 &generator) {
    float x = random_float(generator, -1.0f, 1.0f);
    float y = random_float(generator, -1.0f, 1.0f);
    float z = random_float(generator, -1.0f, 1.0f);

    float length = std::sqrt(x * x + y * y + z * z);
    if (length > 0) {
        x /= length;
        y /= length;
        z /= length;
    }

    return float3({x, y, z});
}

float3 random_hemisphere_direction(const float3 &normal,
                                   std::mt19937 &generator) {
    float3 random_dir = random_unit_vector(generator);

    // Flip if pointing in wrong hemisphere
    if (random_dir.dot(normal) < 0) {
        random_dir = random_dir * -1.0f;
    }

    return random_dir;
}

std::vector<Ray> generate_camera_rays(const std::vector<Triangle> &triangles,
                                      int count, const BVH &bvh) {

    const BoundingBox bbox = compute_bounding_box(triangles);
    float3 bbox_size = bbox.max - bbox.min;
    float max_dim = std::max({bbox_size.x, bbox_size.y, bbox_size.z});

    struct AspectConfig {
        float aspect_ratio;
        std::string name;
    };

    std::vector<AspectConfig> aspects = {
        {16.0f / 9.0f, "landscape 16:9"},
        {9.0f / 16.0f, "portrait 9:16"},
        {4.0f / 3.0f, "landscape 4:3"},
        {3.0f / 4.0f, "portrait 3:4"},
    };

    Camera best_overall_camera;
    std::vector<Ray> best_overall_rays;
    float best_overall_hit_rate = 0.0f;
    int best_overall_hits = 0;
    std::string best_aspect_name;

    for (const auto &aspect : aspects) {
        std::cerr << "\ntrying " << aspect.name << " aspect ratio..."
                  << std::endl;

        int width = static_cast<int>(std::sqrt(count * aspect.aspect_ratio));
        int height = static_cast<int>(width / aspect.aspect_ratio);

        while (width * height < count) {
            if (width * (height + 1) <= count) {
                height++;
            } else {
                width++;
            }
        }

        std::cerr << "image size: " << width << "x" << height << " ("
                  << (width * height) << " rays)" << std::endl;

        std::cerr << "generating rays for " << width << "x" << height
                  << " image (" << (width * height) << " rays)..." << std::endl;

        // Try different camera configurations to maximize hit rate.
        struct CameraConfig {
            float3 position_offset;
            float fov;
            std::string description;
        };

        std::vector<CameraConfig> configs = {
            {{max_dim * 0.7f, max_dim * 0.5f, max_dim * 0.7f},
             25.0f,
             "angled view"},
            // Wide FOV configurations for tall objects, e.g., Lucy.
            {{0.0f, 0.0f, max_dim * 1.0f}, 60.0f, "close front 60°"},
            {{0.0f, 0.0f, max_dim * 0.9f}, 70.0f, "very close 70°"},
            {{0.0f, 0.0f, max_dim * 0.85f}, 75.0f, "ultra close 75°"},
            {{max_dim * 0.3f, 0.0f, max_dim * 0.9f}, 65.0f, "slight angle 65°"},
            {{max_dim * 0.4f, 0.0f, max_dim * 1.0f}, 60.0f, "medium angle 60°"},
            {{0.0f, max_dim * 0.1f, max_dim * 0.9f},
             68.0f,
             "slightly elevated 68°"},
            {{max_dim * 0.2f, 0.0f, max_dim * 0.95f},
             65.0f,
             "shallow angle 65°"},
        };

        Camera best_camera;
        std::vector<Ray> best_rays;
        float best_hit_rate = 0.0f;

        for (size_t cfg_idx = 0; cfg_idx < configs.size(); cfg_idx++) {
            const auto &config = configs[cfg_idx];

            Camera camera;
            camera.look_at = bbox.center();
            camera.position = camera.look_at + config.position_offset;
            camera.up = {0.0f, 1.0f, 0.0f};
            camera.fov = config.fov;
            camera.aspect_ratio = aspect.aspect_ratio;
            camera.compute_basis();

            std::vector<Ray> rays;
            rays.reserve(width * height);
            int hits = 0;

            // Generate rays in scanline order (coherent).
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    float u =
                        static_cast<float>(i) / static_cast<float>(width - 1);
                    float v =
                        static_cast<float>(j) / static_cast<float>(height - 1);

                    Ray ray = camera.generate_ray(u, v);
                    rays.push_back(ray);
                    if (bvh.intersect(ray)) {
                        hits++;
                    }
                }
            }

            float hit_rate = static_cast<float>(hits) / rays.size();

            std::cerr << "config " << (cfg_idx + 1) << " ("
                      << config.description << "): " << hits << " hits ("
                      << (100.0f * hit_rate) << "%)" << std::endl;

            if (hit_rate > best_hit_rate) {
                best_hit_rate = hit_rate;
                best_camera = camera;
                best_rays = rays;
            }

            if (hit_rate > best_overall_hit_rate) {
                best_overall_hit_rate = hit_rate;
                best_overall_hits = hits;
                best_overall_camera = camera;
                best_overall_rays = std::move(rays);
                best_aspect_name = aspect.name;
                if (hit_rate > 0.75) {
                    break; // ...good enough for a government job.
                }
            }
        }

    } // end aspect ratio loop

    std::cerr << "camera ray statistics: " << best_overall_hits << " hits ("
              << (100.0f * best_overall_hit_rate) << "%), "
              << (best_overall_rays.size() - best_overall_hits) << " misses"
              << std::endl;

    return best_overall_rays;
}

std::vector<Ray> generate_secondary_rays(const std::vector<Triangle> &triangles,
                                         int count, const BVH &bvh) {

    std::cerr
        << "generating secondary rays by tracing optimized camera rays first..."
        << std::endl;
    int camera_ray_count = count * 2;
    std::vector<Ray> camera_rays =
        generate_camera_rays(triangles, camera_ray_count, bvh);

    std::cerr << "tracing camera rays to find hit points..." << std::endl;

    std::vector<HitRecord> hit_points;
    hit_points.reserve(count);

    for (const Ray &ray : camera_rays) {
        HitRecord hit;
        if (bvh.intersect_full(ray, hit)) {
            hit_points.push_back(hit);
            if (hit_points.size() >= (size_t)count) {
                break;
            }
        }
    }

    std::cerr << "found " << hit_points.size()
              << " hit points, generating secondary rays..." << std::endl;

    if (hit_points.empty()) {
        std::cerr
            << "Error: no hit points found, cannot generate secondary rays"
            << std::endl;
        return {};
    }
    std::vector<Ray> secondary_rays;
    secondary_rays.reserve(count);

    std::random_device rd;
    std::mt19937 generator(rd());

    for (int i = 0; i < count; i++) {
        int hit_idx = i % hit_points.size();
        const HitRecord &hit = hit_points[hit_idx];

        // Generate a random direction in the hemisphere around the normal.
        float3 bounce_dir = random_hemisphere_direction(hit.normal, generator);

        Ray secondary_ray;
        // Offset origin slightly along normal to avoid self-intersection.
        secondary_ray.o = hit.point + hit.normal * 0.001f;
        secondary_ray.d = bounce_dir;

        secondary_rays.push_back(secondary_ray);
    }

    int hits = 0;
    for (const Ray &ray : secondary_rays) {
        if (bvh.intersect(ray)) {
            hits++;
        }
    }

    std::cerr << "secondary rays stats: " << hits << " hits ("
              << (100.0f * hits / secondary_rays.size()) << "%), "
              << (secondary_rays.size() - hits) << " misses" << std::endl;

    return secondary_rays;
}

int main(int argc, char *argv[]) {
    assert(argc == 5);
    std::string object = argv[1];
    std::string path = argv[2];
    const char *ray_count = argv[3];
    std::string ray_type = argv[4]; // "camera" or "secondary"

    const int64_t count = std::atoi(ray_count);
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());

    std::cerr << "building " << object << " bvh..." << std::endl;
    BVH bvh;
    bvh.build(triangles);
    std::cerr << "..." << object << " bvh complete" << std::endl;

    if (ray_type == "camera") {
        std::vector<Ray> rays = generate_camera_rays(triangles, count, bvh);
        save_rays_binary(rays, object, ray_count, ray_type, path);
        return 0;
    }
    if (ray_type == "secondary") {
        std::vector<Ray> rays = generate_secondary_rays(triangles, count, bvh);
        save_rays_binary(rays, object, ray_count, ray_type, path);
        return 0;
    }

    std::cerr
        << "Error: the ray type must be 'camera' or 'secondary', received: '"
        << ray_type << "'" << std::endl;
    return 1;
}
