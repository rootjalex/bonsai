#include "helpers.h"

#include "rt.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<Ray> load_rays_binary(const std::string &filename,
                                  int64_t ray_count) {
    std::vector<Ray> rays;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename
                  << " for reading\n";
        return rays;
    }

    // Read number of rays
    size_t count;
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (ray_count > count) {
        std::cerr << "the requested ray count: " << ray_count
                  << " is greater than the total ray count: " << count
                  << " You need to re-generate the rays.";
    }
    assert(ray_count <= count);

    rays.reserve(ray_count);

    // Read ray data
    for (size_t i = 0; i < ray_count; ++i) {
        Ray ray;
        file.read(reinterpret_cast<char *>(&ray.o.x), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o.y), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o.z), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d.x), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d.y), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d.z), sizeof(float));
        rays.push_back(ray);
    }

    file.close();
    return rays;
}

std::pair<float3, float3> compute_aabb(uint32_t low, uint32_t high,
                                       const std::vector<Triangle> &triangles) {
    Triangle tri = triangles[low];
    float3 aabb_min = tri.p0;
    float3 aabb_max = tri.p0;
    for (uint32_t i = low; i < high; ++i) {
        Triangle t = triangles[i];
        for (float3 v : {t.p0, t.p1, t.p2}) {
            aabb_min = min(aabb_min, v);
            aabb_max = max(aabb_max, v);
        }
    }
    return {aabb_min, aabb_max};
}

float surface_area(const float3 &min, const float3 &max) {
    float3 extent = max - min;
    return 2.0f *
           (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
}

float3 triangle_centroid(const Triangle &tri) {
    return (tri.p0 + tri.p1 + tri.p2) * (1.0f / 3.0f);
}

std::pair<float3, float3> triangle_bounds(const Triangle &tri) {
    float3 min_ = min(min(tri.p0, tri.p1), tri.p2);
    float3 max_ = max(max(tri.p0, tri.p1), tri.p2);
    return {min_, max_};
}

BVH *build_canonical_tree_8_sah(std::vector<Triangle> &triangles,
                                int max_prims_per_leaf = 8,
                                int max_tree_depth = 64, int num_bins = 32,
                                float traversal_cost = 1.0f,
                                float intersection_cost = 1.5f) {

    struct Split {
        int axis;
        float positions[7]; // 7 split positions for 8-way
        float cost;
    };

    constexpr auto MAX = std::numeric_limits<float>::max();

    struct Bin {
        float3 min = float3{MAX, MAX, MAX};
        float3 max = float3{-MAX, -MAX, -MAX};
        uint32_t count = 0;
    };

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;
        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        // Create leaf if below threshold
        if (count <= max_prims_per_leaf || depth >= max_tree_depth - 1) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + low, triangles.begin() + high, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Compute centroid bounds
        float3 centroid_min = triangle_centroid(triangles[low]);
        float3 centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        // Find best 8-way split
        Split best_split;
        best_split.cost = MAX;
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        // Try splitting along each axis
        for (int axis = 0; axis < 3; ++axis) {
            float extent = (axis == 0   ? centroid_max.x - centroid_min.x
                            : axis == 1 ? centroid_max.y - centroid_min.y
                                        : centroid_max.z - centroid_min.z);
            if (extent < 1e-6f)
                continue;

            // Simple approach: divide into 8 equal parts
            float split_positions[7];
            float axis_min = (axis == 0   ? centroid_min.x
                              : axis == 1 ? centroid_min.y
                                          : centroid_min.z);
            for (int i = 0; i < 7; ++i) {
                split_positions[i] = axis_min + (i + 1) * extent / 8.0f;
            }

            // Evaluate this 8-way split
            std::vector<float3> group_mins(8, float3{MAX, MAX, MAX});
            std::vector<float3> group_maxs(8, float3{-MAX, -MAX, -MAX});
            std::vector<uint32_t> group_counts(8, 0);

            // Assign triangles to groups and compute bounds
            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0 ? c.x : axis == 1 ? c.y : c.z);
                int group = 7; // defaults to last group
                for (int j = 0; j < 7; ++j) {
                    if (c_axis < split_positions[j]) {
                        group = j;
                        break;
                    }
                }

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                group_mins[group] = min(group_mins[group], tri_min);
                group_maxs[group] = max(group_maxs[group], tri_max);
                group_counts[group]++;
            }

            // Calculate SAH cost for this split
            float split_cost = traversal_cost * 7; // 7 internal traversal steps
            for (int i = 0; i < 8; ++i) {
                if (group_counts[i] > 0) {
                    float area = surface_area(group_mins[i], group_maxs[i]);
                    split_cost += (area / parent_area) * intersection_cost *
                                  group_counts[i];
                }
            }

            if (split_cost < best_split.cost) {
                best_split.axis = axis;
                for (int i = 0; i < 7; ++i)
                    best_split.positions[i] = split_positions[i];
                best_split.cost = split_cost;
            }
        }

        // Check if splitting is worth it
        if (best_split.cost >= leaf_cost) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + low, triangles.begin() + high, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Partition triangles into 8 groups
        std::vector<std::vector<uint32_t>> groups(8);
        for (uint32_t i = low; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            float c_axis = (best_split.axis == 0   ? c.x
                            : best_split.axis == 1 ? c.y
                                                   : c.z);
            int group = 0;
            for (int j = 0; j < 7; ++j) {
                if (c_axis >= best_split.positions[j]) {
                    group = j + 1;
                } else {
                    break;
                }
            }
            groups[group].push_back(i);
        }

        // Reorder triangles based on groups
        std::vector<Triangle> temp_triangles;
        temp_triangles.reserve(count);
        std::vector<uint32_t> group_starts(9);
        group_starts[0] = low;

        for (int g = 0; g < 8; ++g) {
            for (uint32_t idx : groups[g]) {
                temp_triangles.push_back(triangles[idx]);
            }
            group_starts[g + 1] = group_starts[g] + groups[g].size();
        }

        std::copy(temp_triangles.begin(), temp_triangles.end(),
                  triangles.begin() + low);
        Interior *node = new Interior();
        for (int i = 0; i < 8; ++i) {
            if (group_starts[i] < group_starts[i + 1]) {
                node->children[i] =
                    partition(group_starts[i], group_starts[i + 1], depth + 1);
                auto [child_min, child_max] = compute_aabb(
                    group_starts[i], group_starts[i + 1], triangles);
                node->lo[i] = child_min;
                node->hi[i] = child_max;
            } else {
                // empty child
                node->children[i] = nullptr;
                node->lo[i] = float3{MAX, MAX, MAX};
                node->hi[i] = float3{-MAX, -MAX, -MAX};
            }
        }

        return new BVH(*node);
    };

    return partition(0, triangles.size(), 0);
}

void free_canonical_tree_8(BVH *node) {
    if (node == nullptr) {
        return;
    }
    if (std::holds_alternative<Interior>(*node)) {
        Interior &interior = std::get<Interior>(*node);
        free_canonical_tree_8(interior.children[0]);
        free_canonical_tree_8(interior.children[1]);
        free_canonical_tree_8(interior.children[2]);
        free_canonical_tree_8(interior.children[3]);
        free_canonical_tree_8(interior.children[4]);
        free_canonical_tree_8(interior.children[5]);
        free_canonical_tree_8(interior.children[6]);
        free_canonical_tree_8(interior.children[7]);
        free(&interior);
        return;
    }

    if (std::holds_alternative<Leaf>(*node)) {
        Leaf &leaf = std::get<Leaf>(*node);
        free(leaf.data);
        free(&leaf);
        return;
    }

    assert(false && "unexpected");
}

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

void run_test(const std::string &object) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());
    std::cout << "loaded triangles\n";

    BVH *canonical_tree = build_canonical_tree_8_sah(triangles);
    std::cout << "built canonical tree\n";

    Triangles tree = build_triangles(canonical_tree);
    std::cout << "built specialized tree\n";
    free_canonical_tree_8(canonical_tree);
    std::cout << "freed canonical tree\n";

    std::vector<int64_t> ray_counts = {
        1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20,
    };
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/cpu/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" +
                               std::to_string(75) + ".rays";
        Ray *rays = nullptr;
        {
            std::vector<Ray> r = load_rays_binary(ray_file, ray_count);
            assert(!r.empty());
            rays = reinterpret_cast<Ray *>(malloc(sizeof(Ray) * ray_count));
            std::copy(r.begin(), r.end(), rays);
            r.clear();
        }
        std::cout << "built rays\n";
        auto trace_begin = clock::now();
        cuda::std::optional<Triangle> *hits = chrt(ray_count, rays, &tree);
        auto trace_end = clock::now();
        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "trace time       : " << trace_time << " ms\n";
    }
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string object_file = argv[1];
    run_test(object_file);
    return 0;
}
