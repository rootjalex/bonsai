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

// (do not touch)
// AUTO-GENERATED canonical_tree
// CUDA
constexpr float gamma(int n) {
    constexpr float E = std::numeric_limits<float>::epsilon() * 0.5f;
    return (n * E) / (1.0f - n * E);
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
    float3 extent = aabb_max - aabb_min;
    float3 expansion = extent * gamma(3);
    aabb_max = aabb_max + expansion;
    aabb_min = aabb_min - expansion;
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
    float3 extent = max_ - min_;
    max_ = max_ + extent * gamma(3);
    return {min_, max_};
}

OBB compute_obb(uint32_t low, uint32_t high,
                const std::vector<Triangle> &tris) {
    float3 centroid = {0.0f, 0.0f, 0.0f};
    for (uint32_t i = low; i < high; ++i) {
        centroid = centroid + triangle_centroid(tris[i]);
    }
    centroid = centroid / static_cast<float>(high - low);

    float cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (uint32_t i = low; i < high; ++i) {
        float3 c = triangle_centroid(tris[i]) - centroid;
        cov[0][0] += c.x * c.x;
        cov[0][1] += c.x * c.y;
        cov[0][2] += c.x * c.z;
        cov[1][0] += c.y * c.x;
        cov[1][1] += c.y * c.y;
        cov[1][2] += c.y * c.z;
        cov[2][0] += c.z * c.x;
        cov[2][1] += c.z * c.y;
        cov[2][2] += c.z * c.z;
    }

    float3 v = {1.0f, 0.0f, 0.0f};
    for (int iter = 0; iter < 10; ++iter) {
        float3 v_new = {0.0f, 0.0f, 0.0f};
        v_new.x = cov[0][0] * v.x + cov[0][1] * v.y + cov[0][2] * v.z;
        v_new.y = cov[1][0] * v.x + cov[1][1] * v.y + cov[1][2] * v.z;
        v_new.z = cov[2][0] * v.x + cov[2][1] * v.y + cov[2][2] * v.z;

        float len =
            sqrt(v_new.x * v_new.x + v_new.y * v_new.y + v_new.z * v_new.z);
        if (len > 1e-6f) {
            v = v_new / len;
        }
    }

    float3 x_axis = v;
    float3 y_axis = {-v.y, v.x, 0.0f};
    float y_len = sqrt(y_axis.x * y_axis.x + y_axis.y * y_axis.y);
    if (y_len < 1e-6f) {
        y_axis = {0.0f, 1.0f, 0.0f};
    } else {
        y_axis = y_axis / y_len;
    }
    float3 z_axis = cross(x_axis, y_axis);
    float z_len =
        sqrt(z_axis.x * z_axis.x + z_axis.y * z_axis.y + z_axis.z * z_axis.z);
    if (z_len > 1e-6f) {
        z_axis = z_axis / z_len;
    }

    cuda::std::array<float4, 3> orientation;
    orientation[0] = {x_axis.x, x_axis.y, x_axis.z, 0.0f};
    orientation[1] = {y_axis.x, y_axis.y, y_axis.z, 0.0f};
    orientation[2] = {z_axis.x, z_axis.y, z_axis.z, 0.0f};

    constexpr auto MAX = std::numeric_limits<float>::max();
    float3 obb_min = {MAX, MAX, MAX};
    float3 obb_max = {-MAX, -MAX, -MAX};

    for (uint32_t i = low; i < high; ++i) {
        auto [tri_min, tri_max] = triangle_bounds(tris[i]);
        for (int corner = 0; corner < 8; ++corner) {
            float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                        (corner & 2) ? tri_max.y : tri_min.y,
                        (corner & 4) ? tri_max.z : tri_min.z};
            float3 p_obb = {dot(x_axis, p), dot(y_axis, p), dot(z_axis, p)};
            obb_min = min(obb_min, p_obb);
            obb_max = max(obb_max, p_obb);
        }
    }

    float3 extent = obb_max - obb_min;
    obb_max = obb_max + extent * gamma(3);
    obb_min = obb_min - extent * gamma(3);

    return OBB{obb_min, obb_max, orientation};
}

float compute_tightness(uint32_t low, uint32_t high,
                        const std::vector<Triangle> &tris) {
    auto [aabb_min, aabb_max] = compute_aabb(low, high, tris);
    float aabb_volume = (aabb_max.x - aabb_min.x) * (aabb_max.y - aabb_min.y) *
                        (aabb_max.z - aabb_min.z);

    if (aabb_volume < 1e-6f)
        return 1.0f;

    float3 centroid = {0.0f, 0.0f, 0.0f};
    for (uint32_t i = low; i < high; ++i) {
        centroid = centroid + triangle_centroid(tris[i]);
    }
    centroid = centroid / static_cast<float>(high - low);

    float cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (uint32_t i = low; i < high; ++i) {
        float3 c = triangle_centroid(tris[i]) - centroid;
        cov[0][0] += c.x * c.x;
        cov[0][1] += c.x * c.y;
        cov[0][2] += c.x * c.z;
        cov[1][0] += c.y * c.x;
        cov[1][1] += c.y * c.y;
        cov[1][2] += c.y * c.z;
        cov[2][0] += c.z * c.x;
        cov[2][1] += c.z * c.y;
        cov[2][2] += c.z * c.z;
    }

    float trace = cov[0][0] + cov[1][1] + cov[2][2];
    float obb_volume_estimate = std::pow(trace / 3.0f, 1.5f);

    return obb_volume_estimate / aabb_volume;
}

BVH *build_canonical_tree_8_mixed_sah(std::vector<Triangle> &triangles,
                                      int max_prims_per_leaf = 8,
                                      int max_tree_depth = 64,
                                      float traversal_cost_aabb = 1.0f,
                                      float traversal_cost_obb = 1.5f,
                                      float intersection_cost = 1.5f,
                                      int obb_depth_threshold = 3) {

    struct Split {
        int axis;
        float positions[7];
        float cost;
        bool use_obb;
    };

    constexpr auto MAX = std::numeric_limits<float>::max();
    constexpr float EPSILON = 1e-6f;

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = high - low;

        if (count <= max_prims_per_leaf || depth >= max_tree_depth - 1) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + low, triangles.begin() + high, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        bool consider_obb = (depth >= obb_depth_threshold);
        if (consider_obb) {
            float tightness = compute_tightness(low, high, triangles);
            consider_obb = (tightness < 0.7f);
        }

        float3 centroid_min = triangle_centroid(triangles[low]);
        float3 centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        Split best_split;
        best_split.cost = MAX;
        best_split.use_obb = false;

        for (int axis = 0; axis < 3; ++axis) {
            float centroid_min_val = (axis == 0)   ? centroid_min.x
                                     : (axis == 1) ? centroid_min.y
                                                   : centroid_min.z;
            float centroid_max_val = (axis == 0)   ? centroid_max.x
                                     : (axis == 1) ? centroid_max.y
                                                   : centroid_max.z;
            float extent = centroid_max_val - centroid_min_val;
            if (extent < EPSILON)
                continue;

            float split_positions[7];
            for (int i = 0; i < 7; ++i) {
                split_positions[i] = centroid_min_val + (i + 1) * extent / 8.0f;
            }

            std::vector<float3> group_mins(8, float3{MAX, MAX, MAX});
            std::vector<float3> group_maxs(8, float3{-MAX, -MAX, -MAX});
            std::vector<uint32_t> group_counts(8, 0);

            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_val = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;
                int group = 7;
                for (int j = 0; j < 7; ++j) {
                    if (c_val < split_positions[j]) {
                        group = j;
                        break;
                    }
                }

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                group_mins[group] = min(group_mins[group], tri_min);
                group_maxs[group] = max(group_maxs[group], tri_max);
                group_counts[group]++;
            }

            float split_cost = traversal_cost_aabb * 7;
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
                best_split.use_obb = false;
            }
        }

        if (consider_obb) {
            OBB obb = compute_obb(low, high, triangles);

            for (int axis = 0; axis < 3; ++axis) {
                float3 world_axis = {obb.orientation[axis].x,
                                     obb.orientation[axis].y,
                                     obb.orientation[axis].z};

                float min_proj = MAX, max_proj = -MAX;
                for (uint32_t i = low; i < high; ++i) {
                    float3 c = triangle_centroid(triangles[i]);
                    float proj = dot(c, world_axis);
                    min_proj = std::min(min_proj, proj);
                    max_proj = std::max(max_proj, proj);
                }

                float extent = max_proj - min_proj;
                if (extent < EPSILON)
                    continue;

                float split_positions[7];
                for (int i = 0; i < 7; ++i) {
                    split_positions[i] = min_proj + (i + 1) * extent / 8.0f;
                }

                std::vector<float3> group_mins(8, float3{MAX, MAX, MAX});
                std::vector<float3> group_maxs(8, float3{-MAX, -MAX, -MAX});
                std::vector<uint32_t> group_counts(8, 0);

                for (uint32_t i = low; i < high; ++i) {
                    float3 c = triangle_centroid(triangles[i]);
                    float proj = dot(c, world_axis);

                    int group = 7;
                    for (int j = 0; j < 7; ++j) {
                        if (proj < split_positions[j]) {
                            group = j;
                            break;
                        }
                    }

                    auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                    float3 obb_min = {MAX, MAX, MAX};
                    float3 obb_max = {-MAX, -MAX, -MAX};

                    for (int corner = 0; corner < 8; ++corner) {
                        float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                                    (corner & 2) ? tri_max.y : tri_min.y,
                                    (corner & 4) ? tri_max.z : tri_min.z};
                        float3 p_obb = {0.0f, 0.0f, 0.0f};
                        p_obb.x = dot(p, float3{obb.orientation[0].x,
                                                obb.orientation[0].y,
                                                obb.orientation[0].z});
                        p_obb.y = dot(p, float3{obb.orientation[1].x,
                                                obb.orientation[1].y,
                                                obb.orientation[1].z});
                        p_obb.z = dot(p, float3{obb.orientation[2].x,
                                                obb.orientation[2].y,
                                                obb.orientation[2].z});
                        obb_min = min(obb_min, p_obb);
                        obb_max = max(obb_max, p_obb);
                    }

                    group_mins[group] = min(group_mins[group], obb_min);
                    group_maxs[group] = max(group_maxs[group], obb_max);
                    group_counts[group]++;
                }

                float split_cost = traversal_cost_obb * 7;
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
                    best_split.use_obb = true;
                }
            }
        }

        if (best_split.cost >= leaf_cost) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + low, triangles.begin() + high, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        std::vector<std::vector<uint32_t>> groups(8);

        if (best_split.use_obb) {
            OBB obb = compute_obb(low, high, triangles);
            float3 world_axis = {obb.orientation[best_split.axis].x,
                                 obb.orientation[best_split.axis].y,
                                 obb.orientation[best_split.axis].z};

            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float proj = dot(c, world_axis);

                int group = 0;
                for (int j = 0; j < 7; ++j) {
                    if (proj >= best_split.positions[j]) {
                        group = j + 1;
                    } else {
                        break;
                    }
                }
                groups[group].push_back(i);
            }
        } else {
            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_val = (best_split.axis == 0)   ? c.x
                              : (best_split.axis == 1) ? c.y
                                                       : c.z;
                int group = 0;
                for (int j = 0; j < 7; ++j) {
                    if (c_val >= best_split.positions[j]) {
                        group = j + 1;
                    } else {
                        break;
                    }
                }
                groups[group].push_back(i);
            }
        }

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

        if (best_split.use_obb) {
            OBB parent_obb = compute_obb(low, high, triangles);
            OBBNode *node = new OBBNode();
            node->orientation = parent_obb.orientation;

            for (int i = 0; i < 8; ++i) {
                if (group_starts[i] < group_starts[i + 1]) {
                    node->obb_children[i] = partition(
                        group_starts[i], group_starts[i + 1], depth + 1);

                    float3 child_min = {MAX, MAX, MAX};
                    float3 child_max = {-MAX, -MAX, -MAX};

                    for (uint32_t j = group_starts[i]; j < group_starts[i + 1];
                         ++j) {
                        auto [tri_min, tri_max] = triangle_bounds(triangles[j]);
                        for (int corner = 0; corner < 8; ++corner) {
                            float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                                        (corner & 2) ? tri_max.y : tri_min.y,
                                        (corner & 4) ? tri_max.z : tri_min.z};
                            float3 axis_x = {parent_obb.orientation[0].x,
                                             parent_obb.orientation[0].y,
                                             parent_obb.orientation[0].z};
                            float3 axis_y = {parent_obb.orientation[1].x,
                                             parent_obb.orientation[1].y,
                                             parent_obb.orientation[1].z};
                            float3 axis_z = {parent_obb.orientation[2].x,
                                             parent_obb.orientation[2].y,
                                             parent_obb.orientation[2].z};
                            float3 p_obb = {dot(p, axis_x), dot(p, axis_y),
                                            dot(p, axis_z)};
                            child_min = min(child_min, p_obb);
                            child_max = max(child_max, p_obb);
                        }
                    }

                    float3 extent = child_max - child_min;
                    float3 expansion = extent * gamma(3);
                    child_max = child_max + expansion;
                    child_min = child_min - expansion;

                    node->obb_low[i] = child_min;
                    node->obb_high[i] = child_max;
                } else {
                    node->obb_children[i] = nullptr;
                    node->obb_low[i] = {MAX, MAX, MAX};
                    node->obb_high[i] = {-MAX, -MAX, -MAX};
                }
            }

            return new BVH(*node);
        } else {
            AABBNode *node = new AABBNode();

            for (int i = 0; i < 8; ++i) {
                if (group_starts[i] < group_starts[i + 1]) {
                    node->aabb_children[i] = partition(
                        group_starts[i], group_starts[i + 1], depth + 1);
                    auto [child_min, child_max] = compute_aabb(
                        group_starts[i], group_starts[i + 1], triangles);
                    node->aabb_low[i] = child_min;
                    node->aabb_high[i] = child_max;
                } else {
                    node->aabb_children[i] = nullptr;
                    node->aabb_low[i] = {MAX, MAX, MAX};
                    node->aabb_high[i] = {-MAX, -MAX, -MAX};
                }
            }

            return new BVH(*node);
        }
    };

    return partition(0, triangles.size(), 0);
}

void free_canonical_tree_8_mixed(BVH *node) {
    if (node == nullptr) {
        return;
    }
    if (std::holds_alternative<AABBNode>(*node)) {
        AABBNode &aabb = std::get<AABBNode>(*node);
        free_canonical_tree_8_mixed(aabb.aabb_children[0]);
        free_canonical_tree_8_mixed(aabb.aabb_children[1]);
        free_canonical_tree_8_mixed(aabb.aabb_children[2]);
        free_canonical_tree_8_mixed(aabb.aabb_children[3]);
        free_canonical_tree_8_mixed(aabb.aabb_children[4]);
        free_canonical_tree_8_mixed(aabb.aabb_children[5]);
        free_canonical_tree_8_mixed(aabb.aabb_children[6]);
        free_canonical_tree_8_mixed(aabb.aabb_children[7]);
        free(&aabb);
        return;
    }
    if (std::holds_alternative<OBBNode>(*node)) {
        OBBNode &obb = std::get<OBBNode>(*node);
        free_canonical_tree_8_mixed(obb.obb_children[0]);
        free_canonical_tree_8_mixed(obb.obb_children[1]);
        free_canonical_tree_8_mixed(obb.obb_children[2]);
        free_canonical_tree_8_mixed(obb.obb_children[3]);
        free_canonical_tree_8_mixed(obb.obb_children[4]);
        free_canonical_tree_8_mixed(obb.obb_children[5]);
        free_canonical_tree_8_mixed(obb.obb_children[6]);
        free_canonical_tree_8_mixed(obb.obb_children[7]);
        free(&obb);
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

BVH *build_canonical_tree_8_mixed(
    std::vector<Triangle> &triangles,
    Heuristic heuristic = Heuristic::SurfaceArea) {
    switch (heuristic) {
    case Heuristic::SurfaceArea:
        return build_canonical_tree_8_mixed_sah(triangles);
    case Heuristic::MedianSplit:
        assert(false && "unimplemented");
    }
}

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

void run(const std::string &object, const std::vector<int64_t> &ray_counts) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(object);
    assert(!triangles.empty());

    BVH *canonical_tree = build_canonical_tree_8_mixed(triangles);

    Triangles tree = build_triangles(canonical_tree);
    free_canonical_tree_8_mixed(canonical_tree);

    bool is_first_run = true;
    for (const int64_t ray_count : ray_counts) {
        std::cout << ray_count << std::endl;
        std::string ray_file = "apps/rt/rays/" + object + "_" +
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
        if (is_first_run) {
            (void)chrt(ray_count, rays, &tree); // warm-up run
            is_first_run = false;
        }

        auto trace_begin = clock::now();
        cuda::std::optional<Triangle> *hits = chrt(ray_count, rays, &tree);
        auto trace_end = clock::now();

        int64_t count = 0;
        for (int i = 0; i < ray_count; ++i) {
            if (hits[i].has_value()) {
                ++count;
            }
        }
        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "hits             : " << count << "\n";
        std::cout << "trace time       : " << trace_time << " ms\n";
    }
}

bool is_digit(std::string s) {
    for (char c : s) {
        if (std::isdigit(c)) {
            continue;
        }
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc > 3);
    std::string object_file = argv[1];
    std::vector<int64_t> ray_counts;
    assert(is_digit(argv[2]));
    const int64_t size = std::atoi(argv[2]);

    ray_counts.reserve(size);
    for (int i = 3; i < 3 + size; ++i) {
        assert(is_digit(argv[i]));
        ray_counts.push_back(std::atoi(argv[i]));
    }
    run(object_file, ray_counts);
    return 0;
}
