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
                                float intersection_cost = 15.0f,
                                int num_bins = 32) {

    struct Bin {
        float3 aabb_min = float3{std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max()};
        float3 aabb_max = float3{-std::numeric_limits<float>::max(),
                                 -std::numeric_limits<float>::max(),
                                 -std::numeric_limits<float>::max()};
        uint32_t count = 0;

        void extend(float3 bmin, float3 bmax) {
            aabb_min = min(aabb_min, bmin);
            aabb_max = max(aabb_max, bmax);
            count++;
        }

        Bin operator+(const Bin &other) const {
            Bin result;
            if (count > 0 && other.count > 0) {
                result.aabb_min = min(aabb_min, other.aabb_min);
                result.aabb_max = max(aabb_max, other.aabb_max);
            } else if (count > 0) {
                result.aabb_min = aabb_min;
                result.aabb_max = aabb_max;
            } else if (other.count > 0) {
                result.aabb_min = other.aabb_min;
                result.aabb_max = other.aabb_max;
            }
            result.count = count + other.count;
            return result;
        }
    };

    struct Split {
        int axis = -1;
        int bin_index = -1;
        float position = 0.0f;
        float sah_cost = std::numeric_limits<float>::max();
    };

    constexpr auto MAX = std::numeric_limits<float>::max();
    // https://github.com/RenderKit/embree/blob/1970895eb97a38ff67e7da97689a3f3c35fd705c/kernels/builders/bvh_builder_sah.h#L10
    constexpr int NUM_OBJECT_BINS = 32;

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t begin, uint32_t end, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = end - begin;

        auto [aabb_min, aabb_max] = compute_aabb(begin, end, triangles);
        if (count <= max_prims_per_leaf || depth >= max_tree_depth - 1) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[begin + i];
            }
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        float3 centroid_bounds_min = triangle_centroid(triangles[begin]);
        float3 centroid_bounds_max = centroid_bounds_min;

        for (uint32_t i = begin + 1; i < end; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_bounds_min = min(centroid_bounds_min, c);
            centroid_bounds_max = max(centroid_bounds_max, c);
        }

        // Find best split using Embree's exact binned SAH algorithm (to the
        // best of our ability).
        Split best_split;
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;
        for (int axis = 0; axis < 3; ++axis) {
            float extent =
                (axis == 0)   ? centroid_bounds_max.x - centroid_bounds_min.x
                : (axis == 1) ? centroid_bounds_max.y - centroid_bounds_min.y
                              : centroid_bounds_max.z - centroid_bounds_min.z;

            // Skip degenerate axis (all centroids in same place)
            if (extent < 1e-6f)
                continue;

            float centroid_min = (axis == 0)   ? centroid_bounds_min.x
                                 : (axis == 1) ? centroid_bounds_min.y
                                               : centroid_bounds_min.z;
            Bin bins[NUM_OBJECT_BINS];
            float bin_scale = NUM_OBJECT_BINS / extent;

            // 1. Assign primitives to bins by centroid
            for (uint32_t i = begin; i < end; ++i) {
                float3 centroid = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0)   ? centroid.x
                               : (axis == 1) ? centroid.y
                                             : centroid.z;
                int bin_idx =
                    static_cast<int>((c_axis - centroid_min) * bin_scale);
                bin_idx = std::min(bin_idx, NUM_OBJECT_BINS - 1);
                bin_idx = std::max(bin_idx, 0);

                // Extend bin with primitive bounds
                auto [prim_min, prim_max] = triangle_bounds(triangles[i]);
                bins[bin_idx].extend(prim_min, prim_max);
            }

            // 1. Build prefix sums (left sweep).
            Bin left_bins[NUM_OBJECT_BINS - 1];
            left_bins[0] = bins[0];
            for (int i = 1; i < NUM_OBJECT_BINS - 1; ++i) {
                left_bins[i] = left_bins[i - 1] + bins[i];
            }

            // 3. Build suffix sums (right sweep)
            Bin right_bins[NUM_OBJECT_BINS - 1];
            right_bins[NUM_OBJECT_BINS - 2] = bins[NUM_OBJECT_BINS - 1];
            for (int i = NUM_OBJECT_BINS - 3; i >= 0; --i) {
                right_bins[i] = bins[i + 1] + right_bins[i + 1];
            }

            // 4. Evaluate all split candidates. Split position i means: left
            // gets bins [0..i], right gets bins [i+1..N-1].
            for (int i = 0; i < NUM_OBJECT_BINS - 1; ++i) {
                uint32_t left_count = left_bins[i].count;
                uint32_t right_count = right_bins[i].count;

                // Skip invalid splits (all primitives on one side)
                if (left_count == 0 || right_count == 0)
                    continue;

                // Compute SAH cost
                float left_area =
                    surface_area(left_bins[i].aabb_min, left_bins[i].aabb_max);
                float right_area = surface_area(right_bins[i].aabb_min,
                                                right_bins[i].aabb_max);

                float sah =
                    traversal_cost +
                    (left_area / parent_area) * intersection_cost * left_count +
                    (right_area / parent_area) * intersection_cost *
                        right_count;

                if (sah < best_split.sah_cost) {
                    best_split.axis = axis;
                    best_split.bin_index = i;
                    best_split.sah_cost = sah;
                    // Split position is at between bin i and bin i+1.
                    best_split.position =
                        centroid_min + (i + 1) * (extent / NUM_OBJECT_BINS);
                }
            }
        }

        if (best_split.sah_cost >= leaf_cost) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[begin + i];
            }
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // 5. Partition (in place) primitives based on best split.
        auto mid_it =
            std::partition(triangles.begin() + begin, triangles.begin() + end,
                           [&](const Triangle &tri) {
                               float3 centroid = triangle_centroid(tri);
                               float c_axis =
                                   (best_split.axis == 0)   ? centroid.x
                                   : (best_split.axis == 1) ? centroid.y
                                                            : centroid.z;
                               return c_axis < best_split.position;
                           });

        uint32_t mid = std::distance(triangles.begin(), mid_it);
        if (mid == begin || mid == end) {
            // Fallback: split at median.
            mid = begin + count / 2;
            std::nth_element(triangles.begin() + begin, triangles.begin() + mid,
                             triangles.begin() + end,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_val = (best_split.axis == 0)   ? ca.x
                                                : (best_split.axis == 1) ? ca.y
                                                                         : ca.z;
                                 float cb_val = (best_split.axis == 0)   ? cb.x
                                                : (best_split.axis == 1) ? cb.y
                                                                         : cb.z;
                                 return ca_val < cb_val;
                             });
        }
        BVH *left = partition(begin, mid, depth + 1);
        BVH *right = partition(mid, end, depth + 1);

        return new BVH(Interior{
            .low = aabb_min,
            .high = aabb_max,
            .left = left,
            .right = right,
        });
    };

    return partition(0, triangles.size(), 0);
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
                            int max_prims_per_leaf = 32,
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
         std::vector<int64_t> ray_counts, std::string ray_type,
         std::string layout) {
    using clock = std::chrono::high_resolution_clock;

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

    // we have at most 4 bits for the snapped-grid extent quantization and can
    // add an additional value since we know this value will always be non-zero.
    // Otherwise, we use embree's default.
    const int32_t max_prims_per_leaf = layout.starts_with("eq") ? 15 : 32;
    BVH *canonical_tree =
        build_canonical_tree_2(triangles, heuristic, max_prims_per_leaf);

    Triangles tree = build_triangles(canonical_tree);
    free_canonical_tree_2(canonical_tree);

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
                               std::to_string(ray_count) + "_" + ray_type +
                               ".rays";
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
    assert(argc > 7);
    int i = 1;
    std::string object_file = argv[i++];
    std::string partition = argv[i++];
    std::string schedule = argv[i++];
    assert(schedule == "single-thread" || schedule == "parallel");
    const bool is_single_threaded = schedule == "single-thread";
    std::string ray_type = argv[i++];
    std::string layout = argv[i++];

    std::vector<int64_t> ray_counts;
    const int64_t size = std::atoi(argv[i++]);
    ray_counts.reserve(size);
    for (; i < 7 + size; ++i)
        ray_counts.push_back(std::atoi(argv[i]));

    run(object_file, partition, is_single_threaded, ray_counts, ray_type,
        layout);
    return 0;
}
