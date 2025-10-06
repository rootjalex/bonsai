#include "bonsai_cpp.h"
#include "rt.h"
#include "util.h"
#include <omp.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace {

// (do not touch)
// AUTO-GENERATED canonical_tree
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

BVH *build_canonical_tree_2_ms(std::vector<Triangle> &triangles,
                               int max_prims_per_leaf, int max_tree_depth) {
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        uint32_t count = high - low;

        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        if (count < max_prims_per_leaf) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (int i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            assert(depth != 0);
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        float3 extent = aabb_max - aabb_min;
        int axis = 0;
        float max_extent = extent.x;
        if (extent.y > max_extent) {
            axis = 1;
            max_extent = extent.y;
        }
        if (extent.z > max_extent) {
            axis = 2;
        }

        // Partition around midpoint along axis.
        auto mid_it = triangles.begin() + low + count / 2;
        std::nth_element(
            triangles.begin() + low, mid_it, triangles.begin() + high,
            [&](const Triangle &a, const Triangle &b) {
                float ca = (axis == 0)   ? (a.p0.x + a.p1.x + a.p2.x)
                           : (axis == 1) ? (a.p0.y + a.p1.y + a.p2.y)
                                         : (a.p0.z + a.p1.z + a.p2.z);
                float cb = (axis == 0)   ? (b.p0.x + b.p1.x + b.p2.x)
                           : (axis == 1) ? (b.p0.y + b.p1.y + b.p2.y)
                                         : (b.p0.z + b.p1.z + b.p2.z);
                return ca < cb;
            });

        const uint32_t mid = low + count / 2;
        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        return new BVH(Interior{
            .low = aabb_min,
            .high = aabb_max,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), /*depth=*/0);
}

BVH *build_canonical_tree_2_sah(std::vector<Triangle> &triangles,
                                int max_prims_per_leaf, int max_tree_depth,
                                float traversal_cost = 1.0f,
                                float intersection_cost = 15.0f) {
    struct Split {
        int axis;
        float position;
        float cost;
    };

    constexpr auto MAX = std::numeric_limits<float>::max();

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;
        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        if (count < max_prims_per_leaf || depth >= max_tree_depth - 1) {
            assert(count > 0);
            assert(count < max_prims_per_leaf);
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            assert(depth != 0);
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Compute centroid bounds for splitting.
        float3 centroid_min = triangle_centroid(triangles[low]);
        float3 centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        // Find best split using SAH.
        Split best_split = {-1, 0.0f, MAX};
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        // Try splitting along each axis.
        for (int axis = 0; axis < 3; ++axis) {
            float extent = (axis == 0)   ? centroid_max.x - centroid_min.x
                           : (axis == 1) ? centroid_max.y - centroid_min.y
                                         : centroid_max.z - centroid_min.z;
            if (extent < 1e-6f)
                continue; // Skip degenerate axis.

            // Simple approach: split at midpoint
            float axis_min = (axis == 0)   ? centroid_min.x
                             : (axis == 1) ? centroid_min.y
                                           : centroid_min.z;
            float split_position = axis_min + extent / 2.0f;

            // Evaluate this split
            float3 left_min = float3{MAX, MAX, MAX};
            float3 left_max = float3{-MAX, -MAX, -MAX};
            uint32_t left_count = 0;
            float3 right_min = float3{MAX, MAX, MAX};
            float3 right_max = float3{-MAX, -MAX, -MAX};
            uint32_t right_count = 0;

            // Assign triangles to left/right and compute bounds
            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);

                if (c_axis < split_position) {
                    left_min = min(left_min, tri_min);
                    left_max = max(left_max, tri_max);
                    left_count++;
                } else {
                    right_min = min(right_min, tri_min);
                    right_max = max(right_max, tri_max);
                    right_count++;
                }
            }

            // Skip if all triangles end up on one side
            if (left_count == 0 || right_count == 0)
                continue;

            // Calculate SAH cost for this split
            float left_area = surface_area(left_min, left_max);
            float right_area = surface_area(right_min, right_max);

            float cost =
                traversal_cost +
                (left_area / parent_area) * intersection_cost * left_count +
                (right_area / parent_area) * intersection_cost * right_count;

            if (cost < best_split.cost) {
                best_split.axis = axis;
                best_split.position = split_position;
                best_split.cost = cost;
            }
        }

        if (best_split.axis == -1 || best_split.cost >= leaf_cost) {
            if (count > max_prims_per_leaf) {
                uint32_t mid = low + count / 2;
                std::nth_element(
                    triangles.begin() + low, triangles.begin() + mid,
                    triangles.begin() + high,
                    [&](const Triangle &a, const Triangle &b) {
                        float3 ca = triangle_centroid(a);
                        float3 cb = triangle_centroid(b);
                        float3 extent = aabb_max - aabb_min;
                        int axis = (extent.x > extent.y && extent.x > extent.z)
                                       ? 0
                                   : (extent.y > extent.z) ? 1
                                                           : 2;
                        float ca_val = (axis == 0)   ? ca.x
                                       : (axis == 1) ? ca.y
                                                     : ca.z;
                        float cb_val = (axis == 0)   ? cb.x
                                       : (axis == 1) ? cb.y
                                                     : cb.z;
                        return ca_val < cb_val;
                    });

                BVH *left = partition(low, mid, depth + 1);
                BVH *right = partition(mid, high, depth + 1);
                return new BVH(Interior{
                    .low = aabb_min,
                    .high = aabb_max,
                    .left = left,
                    .right = right,
                });
            }
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Partition triangles based on best split.
        auto mid_it =
            std::partition(triangles.begin() + low, triangles.begin() + high,
                           [&](const Triangle &tri) {
                               float3 c = triangle_centroid(tri);
                               float c_axis = (best_split.axis == 0)   ? c.x
                                              : (best_split.axis == 1) ? c.y
                                                                       : c.z;
                               return c_axis < best_split.position;
                           });

        uint32_t mid = std::distance(triangles.begin(), mid_it);

        // Handle edge case where all triangles end up on one side.
        if (mid == low || mid == high) {
            // Fall back to median split.
            mid = low + count / 2;
            std::nth_element(triangles.begin() + low, triangles.begin() + mid,
                             triangles.begin() + high,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_axis = (best_split.axis == 0) ? ca.x
                                                 : (best_split.axis == 1)
                                                     ? ca.y
                                                     : ca.z;
                                 float cb_axis = (best_split.axis == 0) ? cb.x
                                                 : (best_split.axis == 1)
                                                     ? cb.y
                                                     : cb.z;
                                 return ca_axis < cb_axis;
                             });
        }

        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        return new BVH(Interior{
            .low = aabb_min,
            .high = aabb_max,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), /*depth=*/0);
}

void free_canonical_tree_2(BVH *node) {
    if (std::holds_alternative<Interior>(*node)) {
        Interior &interior = std::get<Interior>(*node);
        free_canonical_tree_2(interior.left);
        free_canonical_tree_2(interior.right);
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

enum class Heuristic {
    SurfaceArea = 0,
    MedianSplit = 1,
};

BVH *build_canonical_tree_2(std::vector<Triangle> &triangles,
                            Heuristic heuristic = Heuristic::SurfaceArea,
                            int max_prims_per_leaf = 15,
                            int max_tree_depth = 64) {
    switch (heuristic) {
    case Heuristic::SurfaceArea:
        return build_canonical_tree_2_sah(triangles, max_prims_per_leaf,
                                          max_tree_depth);
    case Heuristic::MedianSplit:
        return build_canonical_tree_2_ms(triangles, max_prims_per_leaf,
                                         max_tree_depth);
    }
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

void run(std::string object, std::string partition, bool is_single_threaded,
         std::vector<int64_t> ray_counts) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());

    Heuristic heuristic;
    if (partition == "sah") {
        heuristic = Heuristic::SurfaceArea;
    } else if (partition == "ms") {
        heuristic = Heuristic::MedianSplit;
    } else {
        std::cout << "unexpected construction partitioning strategy: "
                  << partition << std::endl;
        exit(1);
    }
    BVH *canonical_tree = build_canonical_tree_2(triangles, heuristic);

    Triangles tree = build_triangles(canonical_tree);
    free_canonical_tree_2(canonical_tree);

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" +
                               std::to_string(75) + ".rays";
        std::vector<Ray> rays = load_rays_binary(ray_file, ray_count);
        assert(!rays.empty());
        if (is_first_run) {
            for (int i = 0; i < std::max<size_t>(rays.size(), 512u); ++i)
                (void)trace(&rays[i], &tree); // warmup
            is_first_run = false;
        }
        size_t hit_count = 0;
        auto trace_begin = clock::now(), trace_end = clock::now();
        if (is_single_threaded) {
            std::vector<Triangle> hits;
            hits.reserve(rays.size());
            trace_begin = clock::now();
            for (int i = 0; i < rays.size(); ++i) {
                if (const std::optional<Triangle> t = trace(&rays[i], &tree)) {
                    hits.push_back(*t);
                }
            }
            trace_end = clock::now();

            hit_count = hits.size();
        } else {
            // parallel
            const size_t max_threads = omp_get_max_threads();
            std::vector<std::vector<Triangle>> hits_per_thread(max_threads);
            for (std::vector<Triangle> &v : hits_per_thread) {
                v.reserve(rays.size() / max_threads + 64);
            }
            trace_begin = clock::now();

#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                auto &hits = hits_per_thread[tid];

#pragma omp for schedule(dynamic, 64) nowait
                for (int i = 0; i < rays.size(); ++i) {
                    if (const std::optional<Triangle> t =
                            trace(&rays[i], &tree)) {
                        hits.push_back(*t);
                    }
                }
            }

            trace_end = clock::now();

            for (const std::vector<Triangle> &v : hits_per_thread) {
                hit_count += v.size();
            }
        }

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "hits             : " << hit_count << "\n";
        std::cout << "trace time       : " << trace_time << " ms\n";
    }
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc > 5);
    int i = 1;
    std::string object_file = argv[i++];
    std::string partition = argv[i++];
    std::string schedule = argv[i++];
    assert(schedule == "single-thread" || schedule == "parallel");
    const bool is_single_threaded = schedule == "single-thread";

    std::vector<int64_t> ray_counts;
    const int64_t size = std::atoi(argv[i++]);
    ray_counts.reserve(size);
    for (; i < 5 + size; ++i) {
        ray_counts.push_back(std::atoi(argv[i]));
    }
    run(object_file, partition, is_single_threaded, ray_counts);
    return 0;
}
