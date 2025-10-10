#include "bonsai_cpp.h"
#include "rt.h"
#include "util.h"
#include <omp.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace {

// (do not touch)
// AUTO-GENERATED canonical_tree
inline std::pair<float3, float3>
compute_aabb(uint32_t low, uint32_t high,
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

inline float surface_area(const float3 &min, const float3 &max) {
    float3 extent = max - min;
    return 2.0f *
           (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
}

inline float3 triangle_centroid(const Triangle &tri) {
    return (tri.p0 + tri.p1 + tri.p2) * (1.0f / 3.0f);
}

inline std::pair<float3, float3> triangle_bounds(const Triangle &tri) {
    float3 min_ = min(min(tri.p0, tri.p1), tri.p2);
    float3 max_ = max(max(tri.p0, tri.p1), tri.p2);
    return {min_, max_};
}

inline BVH *build_canonical_tree_8_sah(std::vector<Triangle> &triangles,
                                       int max_prims_per_leaf = 8,
                                       int max_tree_depth = 64,
                                       float traversal_cost = 1.0f,
                                       float intersection_cost = 15.0f) {

    struct Split {
        int axis;
        float positions[7]; // 7 split positions for 8-way
        float cost;
    };

    constexpr auto MAX = std::numeric_limits<float>::max();

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;
        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        // Create leaf if below threshold
        if (count < max_prims_per_leaf || depth >= max_tree_depth - 1) {
            assert(count > 0);
            assert(count < max_prims_per_leaf);
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

inline void free_canonical_tree_8(BVH *node) {
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

enum class Heuristic {
    SurfaceArea = 0,
    MedianSplit = 1,
};

BVH *build_canonical_tree_8(std::vector<Triangle> &triangles,
                            Heuristic heuristic = Heuristic::SurfaceArea) {
    switch (heuristic) {
    case Heuristic::SurfaceArea:
        return build_canonical_tree_8_sah(triangles);
    case Heuristic::MedianSplit:
        assert(false && "unimplemented");
    }
}

std::vector<Triangle> load_obj(const std::string &object) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai"))
            break;
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
    if (!err.empty())
        std::cerr << "error: " << err << std::endl;
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

void run(std::string object, std::string partition, bool is_single_threaded,
         std::vector<int64_t> ray_counts) {
    using clock = std::chrono::high_resolution_clock;

    // optional: pin main thread and raise priority
    pin_thread_to_core(0);
    set_high_priority();

    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());

    Heuristic heuristic;
    if (partition == "sah")
        heuristic = Heuristic::SurfaceArea;
    else if (partition == "ms")
        heuristic = Heuristic::MedianSplit;
    else {
        std::cerr << "unexpected construction partitioning strategy: "
                  << partition << std::endl;
        exit(1);
    }

    BVH *canonical_tree = build_canonical_tree_8(triangles, heuristic);
    Triangles tree = build_triangles(canonical_tree);
    free_canonical_tree_8(canonical_tree);

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
                if (std::optional<Triangle> t = trace(&rays[i], &tree)) {
                    hits.push_back(*t);
                }
            }
            trace_end = clock::now();
            hit_count = hits.size();
        } else {
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
                pin_thread_to_core(tid); // pin each thread to a core

#pragma omp for schedule(dynamic, 64) nowait
                for (int i = 0; i < rays.size(); ++i) {
                    if (const std::optional<Triangle> t =
                            trace(&rays[i], &tree)) {
                        hits.push_back(*t);
                    }
                }
            }

            trace_end = clock::now();
            for (const std::vector<Triangle> &v : hits_per_thread)
                hit_count += v.size();
        }

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "hits       : " << hit_count << "\n";
        std::cout << "trace time : " << trace_time << " ms\n";
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
    for (; i < 5 + size; ++i)
        ray_counts.push_back(std::atoi(argv[i]));

    run(object_file, partition, is_single_threaded, ray_counts);
    return 0;
}
