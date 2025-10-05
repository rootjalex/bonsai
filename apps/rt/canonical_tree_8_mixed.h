#pragma once

#include "cpu/rt.h"

#include <cstdint>
#include <limits>
#include <vector>

constexpr float gamma(int n) {
    constexpr float E = std::numeric_limits<float>::epsilon() * 0.5f;
    return (n * E) / (1.0f - n * E);
}

std::pair<vec3_float, vec3_float>
compute_aabb(uint32_t low, uint32_t high,
             const std::vector<Triangle> &triangles) {
    Triangle tri = triangles[low];
    vec3_float aabb_min = tri.p0;
    vec3_float aabb_max = tri.p0;
    for (uint32_t i = low; i < high; ++i) {
        Triangle t = triangles[i];
        for (vec3_float v : {t.p0, t.p1, t.p2}) {
            aabb_min = min(aabb_min, v);
            aabb_max = max(aabb_max, v);
        }
    }
    vec3_float extent = aabb_max - aabb_min;
    vec3_float expansion = extent * gamma(3);
    aabb_max = aabb_max + expansion;
    aabb_min = aabb_min - expansion;
    return {aabb_min, aabb_max};
}

float surface_area(const vec3_float &min, const vec3_float &max) {
    vec3_float extent = max - min;
    return 2.0f * (extent[0] * extent[1] + extent[0] * extent[2] +
                   extent[1] * extent[2]);
}

vec3_float triangle_centroid(const Triangle &tri) {
    return (tri.p0 + tri.p1 + tri.p2) * (1.0f / 3.0f);
}

std::pair<vec3_float, vec3_float> triangle_bounds(const Triangle &tri) {
    vec3_float min_ = min(min(tri.p0, tri.p1), tri.p2);
    vec3_float max_ = max(max(tri.p0, tri.p1), tri.p2);
    vec3_float extent = max_ - min_;
    max_ = max_ + extent * gamma(3);
    return {min_, max_};
}

OBB compute_obb(uint32_t low, uint32_t high,
                const std::vector<Triangle> &tris) {
    vec3_float centroid = {0.0f, 0.0f, 0.0f};
    for (uint32_t i = low; i < high; ++i) {
        centroid = centroid + triangle_centroid(tris[i]);
    }
    centroid = centroid / static_cast<float>(high - low);

    float cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (uint32_t i = low; i < high; ++i) {
        vec3_float c = triangle_centroid(tris[i]) - centroid;
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                cov[j][k] += c[j] * c[k];
            }
        }
    }

    vec3_float v = {1.0f, 0.0f, 0.0f};
    for (int iter = 0; iter < 10; ++iter) {
        vec3_float v_new = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                v_new[i] += cov[i][j] * v[j];
            }
        }
        float len = sqrt(v_new[0] * v_new[0] + v_new[1] * v_new[1] +
                         v_new[2] * v_new[2]);
        if (len > 1e-6f) {
            v = v_new / len;
        }
    }

    vec3_float x_axis = v;
    vec3_float y_axis = {-v[1], v[0], 0.0f};
    float y_len = sqrt(y_axis[0] * y_axis[0] + y_axis[1] * y_axis[1]);
    if (y_len < 1e-6f) {
        y_axis = {0.0f, 1.0f, 0.0f};
    } else {
        y_axis = y_axis / y_len;
    }
    vec3_float z_axis = cross(x_axis, y_axis);
    float z_len = sqrt(z_axis[0] * z_axis[0] + z_axis[1] * z_axis[1] +
                       z_axis[2] * z_axis[2]);
    if (z_len > 1e-6f) {
        z_axis = z_axis / z_len;
    }

    vec3_vec4_float orientation;
    orientation[0] = {x_axis[0], x_axis[1], x_axis[2], 0.0f};
    orientation[1] = {y_axis[0], y_axis[1], y_axis[2], 0.0f};
    orientation[2] = {z_axis[0], z_axis[1], z_axis[2], 0.0f};

    constexpr auto MAX = std::numeric_limits<float>::max();
    vec3_float obb_min = {MAX, MAX, MAX};
    vec3_float obb_max = {-MAX, -MAX, -MAX};

    for (uint32_t i = low; i < high; ++i) {
        auto [tri_min, tri_max] = triangle_bounds(tris[i]);
        for (int corner = 0; corner < 8; ++corner) {
            vec3_float p = {(corner & 1) ? tri_max[0] : tri_min[0],
                            (corner & 2) ? tri_max[1] : tri_min[1],
                            (corner & 4) ? tri_max[2] : tri_min[2]};
            vec3_float p_obb = {dot(x_axis, p), dot(y_axis, p), dot(z_axis, p)};
            obb_min = min(obb_min, p_obb);
            obb_max = max(obb_max, p_obb);
        }
    }

    vec3_float extent = obb_max - obb_min;
    obb_max = obb_max + extent * gamma(3);
    obb_min = obb_min - extent * gamma(3);

    return OBB{obb_min, obb_max, orientation};
};

float compute_tightness(uint32_t low, uint32_t high,
                        const std::vector<Triangle> &tris) {
    auto [aabb_min, aabb_max] = compute_aabb(low, high, tris);
    float aabb_volume = (aabb_max[0] - aabb_min[0]) *
                        (aabb_max[1] - aabb_min[1]) *
                        (aabb_max[2] - aabb_min[2]);

    if (aabb_volume < 1e-6f)
        return 1.0f;

    vec3_float centroid = {0.0f, 0.0f, 0.0f};
    for (uint32_t i = low; i < high; ++i) {
        centroid = centroid + triangle_centroid(tris[i]);
    }
    centroid = centroid / static_cast<float>(high - low);

    float cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (uint32_t i = low; i < high; ++i) {
        vec3_float c = triangle_centroid(tris[i]) - centroid;
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                cov[j][k] += c[j] * c[k];
            }
        }
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

        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        bool consider_obb = (depth >= obb_depth_threshold);
        if (consider_obb) {
            float tightness = compute_tightness(low, high, triangles);
            consider_obb = (tightness < 0.7f);
        }

        vec3_float centroid_min = triangle_centroid(triangles[low]);
        vec3_float centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            vec3_float c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        Split best_split;
        best_split.cost = MAX;
        best_split.use_obb = false;

        for (int axis = 0; axis < 3; ++axis) {
            float extent = centroid_max[axis] - centroid_min[axis];
            if (extent < EPSILON)
                continue;

            float split_positions[7];
            for (int i = 0; i < 7; ++i) {
                split_positions[i] =
                    centroid_min[axis] + (i + 1) * extent / 8.0f;
            }

            std::vector<vec3_float> group_mins(8, vec3_float{MAX, MAX, MAX});
            std::vector<vec3_float> group_maxs(8, vec3_float{-MAX, -MAX, -MAX});
            std::vector<uint32_t> group_counts(8, 0);

            for (uint32_t i = low; i < high; ++i) {
                vec3_float c = triangle_centroid(triangles[i]);
                int group = 7;
                for (int j = 0; j < 7; ++j) {
                    if (c[axis] < split_positions[j]) {
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
                vec3_float world_axis = {obb.orientation[axis][0],
                                         obb.orientation[axis][1],
                                         obb.orientation[axis][2]};

                float min_proj = MAX, max_proj = -MAX;
                for (uint32_t i = low; i < high; ++i) {
                    vec3_float c = triangle_centroid(triangles[i]);
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

                std::vector<vec3_float> group_mins(8,
                                                   vec3_float{MAX, MAX, MAX});
                std::vector<vec3_float> group_maxs(
                    8, vec3_float{-MAX, -MAX, -MAX});
                std::vector<uint32_t> group_counts(8, 0);

                for (uint32_t i = low; i < high; ++i) {
                    vec3_float c = triangle_centroid(triangles[i]);
                    float proj = dot(c, world_axis);

                    int group = 7;
                    for (int j = 0; j < 7; ++j) {
                        if (proj < split_positions[j]) {
                            group = j;
                            break;
                        }
                    }

                    auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                    vec3_float obb_min = {MAX, MAX, MAX};
                    vec3_float obb_max = {-MAX, -MAX, -MAX};

                    for (int corner = 0; corner < 8; ++corner) {
                        vec3_float p = {(corner & 1) ? tri_max[0] : tri_min[0],
                                        (corner & 2) ? tri_max[1] : tri_min[1],
                                        (corner & 4) ? tri_max[2] : tri_min[2]};
                        vec3_float p_obb;
                        for (int d = 0; d < 3; ++d) {
                            vec3_float axis_vec = {obb.orientation[d][0],
                                                   obb.orientation[d][1],
                                                   obb.orientation[d][2]};
                            p_obb[d] = dot(p, axis_vec);
                        }
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
            vec3_float world_axis = {obb.orientation[best_split.axis][0],
                                     obb.orientation[best_split.axis][1],
                                     obb.orientation[best_split.axis][2]};

            for (uint32_t i = low; i < high; ++i) {
                vec3_float c = triangle_centroid(triangles[i]);
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
                vec3_float c = triangle_centroid(triangles[i]);
                int group = 0;
                for (int j = 0; j < 7; ++j) {
                    if (c[best_split.axis] >= best_split.positions[j]) {
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

                    vec3_float child_min = {MAX, MAX, MAX};
                    vec3_float child_max = {-MAX, -MAX, -MAX};

                    for (uint32_t j = group_starts[i]; j < group_starts[i + 1];
                         ++j) {
                        auto [tri_min, tri_max] = triangle_bounds(triangles[j]);
                        for (int corner = 0; corner < 8; ++corner) {
                            vec3_float p = {
                                (corner & 1) ? tri_max[0] : tri_min[0],
                                (corner & 2) ? tri_max[1] : tri_min[1],
                                (corner & 4) ? tri_max[2] : tri_min[2]};
                            vec3_float p_obb;
                            for (int d = 0; d < 3; ++d) {
                                vec3_float axis = {
                                    parent_obb.orientation[d][0],
                                    parent_obb.orientation[d][1],
                                    parent_obb.orientation[d][2]};
                                p_obb[d] = dot(p, axis);
                            }
                            child_min = min(child_min, p_obb);
                            child_max = max(child_max, p_obb);
                        }
                    }

                    vec3_float extent = child_max - child_min;
                    vec3_float expansion = extent * gamma(3);
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