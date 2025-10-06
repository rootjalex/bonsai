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

#ifdef __CUDACC__
#include <cuda/std/array>
    using orientation_t = cuda::std::array<float4, 3>;
#else
    using orientation_t = float4x3;
#endif

    orientation_t orientation;
    orientation[0] = {x_axis.x, x_axis.y, x_axis.z, 0.0f};
    orientation[1] = {y_axis.x, y_axis.y, y_axis.z, 0.0f};
    orientation[2] = {z_axis.x, z_axis.y, z_axis.z, 0.0f};

    constexpr auto MAX = std::numeric_limits<float>::max();
    float3 obb_min = {MAX, MAX, MAX};
    float3 obb_max = {-MAX, -MAX, -MAX};

    for (uint32_t i = low; i < high; ++i) {
        const Triangle &tri = tris[i];

        // Extract OBB axes from the orientation matrix.
        float3 axis_x = {orientation[0].x, orientation[0].y, orientation[0].z};
        float3 axis_y = {orientation[1].x, orientation[1].y, orientation[1].z};
        float3 axis_z = {orientation[2].x, orientation[2].y, orientation[2].z};

        // Transform the 3 actual vertices to OBB space
        float3 v0_obb = {dot(tri.p0, axis_x), dot(tri.p0, axis_y),
                         dot(tri.p0, axis_z)};
        float3 v1_obb = {dot(tri.p1, axis_x), dot(tri.p1, axis_y),
                         dot(tri.p1, axis_z)};
        float3 v2_obb = {dot(tri.p2, axis_x), dot(tri.p2, axis_y),
                         dot(tri.p2, axis_z)};

        obb_min = min(obb_min, min(v0_obb, min(v1_obb, v2_obb)));
        obb_max = max(obb_max, max(v0_obb, max(v1_obb, v2_obb)));
    }

    float3 extent = obb_max - obb_min;
    obb_max = obb_max + extent * gamma(3);
    obb_min = obb_min - extent * gamma(3);

    return OBB{obb_min, obb_max, orientation};
}

struct AABBSplitResult {
    float cost;
    float positions[7];
    std::vector<float3> group_mins;
    std::vector<float3> group_maxs;
    std::vector<uint32_t> group_counts;
};

AABBSplitResult evaluate_aabb_split(int axis, uint32_t low, uint32_t high,
                                    const std::vector<Triangle> &triangles,
                                    float3 centroid_min, float3 centroid_max,
                                    float parent_area, float traversal_cost,
                                    float intersection_cost) {

    constexpr auto MAX = std::numeric_limits<float>::max();
    constexpr float EPSILON = 1e-6f;

    AABBSplitResult result;
    result.cost = MAX;

    float centroid_min_val = (axis == 0)   ? centroid_min.x
                             : (axis == 1) ? centroid_min.y
                                           : centroid_min.z;
    float centroid_max_val = (axis == 0)   ? centroid_max.x
                             : (axis == 1) ? centroid_max.y
                                           : centroid_max.z;
    float extent = centroid_max_val - centroid_min_val;

    if (extent < EPSILON)
        return result;

    // Divide into 8 equal parts
    for (int i = 0; i < 7; ++i) {
        result.positions[i] = centroid_min_val + (i + 1) * extent / 8.0f;
    }

    // Assign triangles to groups and compute bounds
    result.group_mins.assign(8, float3{MAX, MAX, MAX});
    result.group_maxs.assign(8, float3{-MAX, -MAX, -MAX});
    result.group_counts.assign(8, 0);

    for (uint32_t i = low; i < high; ++i) {
        float3 c = triangle_centroid(triangles[i]);
        float c_val = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;

        int group = 7;
        for (int j = 0; j < 7; ++j) {
            if (c_val < result.positions[j]) {
                group = j;
                break;
            }
        }

        const Triangle &tri = triangles[i];
        result.group_mins[group] =
            min(result.group_mins[group], min(tri.p0, min(tri.p1, tri.p2)));
        result.group_maxs[group] =
            max(result.group_maxs[group], max(tri.p0, max(tri.p1, tri.p2)));
        result.group_counts[group]++;
    }

    // Calculate SAH cost
    result.cost = traversal_cost * 7;
    for (int i = 0; i < 8; ++i) {
        if (result.group_counts[i] > 0) {
            float area =
                surface_area(result.group_mins[i], result.group_maxs[i]);
            result.cost += (area / parent_area) * intersection_cost *
                           result.group_counts[i];
        }
    }

    return result;
}

struct OBBSplitResult {
    float cost;
    float positions[7];
    std::vector<float3> group_mins;
    std::vector<float3> group_maxs;
    std::vector<uint32_t> group_counts;
};

OBBSplitResult evaluate_obb_split(int axis, uint32_t low, uint32_t high,
                                  const std::vector<Triangle> &triangles,
                                  const OBB &obb, float parent_area,
                                  float traversal_cost,
                                  float intersection_cost) {

    constexpr auto MAX = std::numeric_limits<float>::max();
    constexpr float EPSILON = 1e-6f;

    OBBSplitResult result;
    result.cost = MAX;

    float3 world_axis = {obb.orientation[axis].x, obb.orientation[axis].y,
                         obb.orientation[axis].z};

    // Project centroids onto OBB axis
    float min_proj = MAX, max_proj = -MAX;
    for (uint32_t i = low; i < high; ++i) {
        float3 c = triangle_centroid(triangles[i]);
        float proj = dot(c, world_axis);
        min_proj = std::min(min_proj, proj);
        max_proj = std::max(max_proj, proj);
    }

    float extent = max_proj - min_proj;
    if (extent < EPSILON)
        return result;

    // Divide into 8 equal parts along OBB axis
    for (int i = 0; i < 7; ++i) {
        result.positions[i] = min_proj + (i + 1) * extent / 8.0f;
    }

    // Assign triangles to groups and compute bounds in OBB space
    result.group_mins.assign(8, float3{MAX, MAX, MAX});
    result.group_maxs.assign(8, float3{-MAX, -MAX, -MAX});
    result.group_counts.assign(8, 0);

    float3 axis_x = {obb.orientation[0].x, obb.orientation[0].y,
                     obb.orientation[0].z};
    float3 axis_y = {obb.orientation[1].x, obb.orientation[1].y,
                     obb.orientation[1].z};
    float3 axis_z = {obb.orientation[2].x, obb.orientation[2].y,
                     obb.orientation[2].z};

    for (uint32_t i = low; i < high; ++i) {
        float3 c = triangle_centroid(triangles[i]);
        float proj = dot(c, world_axis);

        int group = 7;
        for (int j = 0; j < 7; ++j) {
            if (proj < result.positions[j]) {
                group = j;
                break;
            }
        }

        // Transform triangle AABB corners to OBB space
        auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
        float3 obb_min = {MAX, MAX, MAX};
        float3 obb_max = {-MAX, -MAX, -MAX};

        for (int corner = 0; corner < 8; ++corner) {
            float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                        (corner & 2) ? tri_max.y : tri_min.y,
                        (corner & 4) ? tri_max.z : tri_min.z};
            float3 p_obb = {dot(p, axis_x), dot(p, axis_y), dot(p, axis_z)};
            obb_min = min(obb_min, p_obb);
            obb_max = max(obb_max, p_obb);
        }

        result.group_mins[group] = min(result.group_mins[group], obb_min);
        result.group_maxs[group] = max(result.group_maxs[group], obb_max);
        result.group_counts[group]++;
    }

    // Calculate SAH cost
    result.cost = traversal_cost * 7;
    for (int i = 0; i < 8; ++i) {
        if (result.group_counts[i] > 0) {
            float area =
                surface_area(result.group_mins[i], result.group_maxs[i]);
            result.cost += (area / parent_area) * intersection_cost *
                           result.group_counts[i];
        }
    }

    return result;
}

struct Split {
    int axis;
    float positions[7];
    float cost;
    bool use_obb;
    OBB obb;
};

std::vector<std::vector<uint32_t>>
partition_triangles(uint32_t low, uint32_t high,
                    const std::vector<Triangle> &triangles,
                    const Split &split) {

    std::vector<std::vector<uint32_t>> groups(8);

    if (split.use_obb) {
        float3 world_axis = {split.obb.orientation[split.axis].x,
                             split.obb.orientation[split.axis].y,
                             split.obb.orientation[split.axis].z};

        for (uint32_t i = low; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            float proj = dot(c, world_axis);

            int group = 0;
            for (int j = 0; j < 7; ++j) {
                if (proj >= split.positions[j]) {
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
            float c_val = (split.axis == 0)   ? c.x
                          : (split.axis == 1) ? c.y
                                              : c.z;

            int group = 0;
            for (int j = 0; j < 7; ++j) {
                if (c_val >= split.positions[j]) {
                    group = j + 1;
                } else {
                    break;
                }
            }
            groups[group].push_back(i);
        }
    }

    return groups;
}

BVH *build_obb_node(
    const std::vector<std::vector<uint32_t>> &groups,
    const std::vector<uint32_t> &group_starts, std::vector<Triangle> &triangles,
    const OBB &obb, uint32_t depth,
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition) {

    constexpr auto MAX = std::numeric_limits<float>::max();

    OBBNode *node = new OBBNode();
    node->orientation = obb.orientation;

    float3 axis_x = {obb.orientation[0].x, obb.orientation[0].y,
                     obb.orientation[0].z};
    float3 axis_y = {obb.orientation[1].x, obb.orientation[1].y,
                     obb.orientation[1].z};
    float3 axis_z = {obb.orientation[2].x, obb.orientation[2].y,
                     obb.orientation[2].z};

    for (int i = 0; i < 8; ++i) {
        if (group_starts[i] < group_starts[i + 1]) {
            node->obb_children[i] =
                partition(group_starts[i], group_starts[i + 1], depth + 1);

            // Compute OBB bounds for child in parent OBB space
            float3 child_min = {MAX, MAX, MAX};
            float3 child_max = {-MAX, -MAX, -MAX};

            for (uint32_t j = group_starts[i]; j < group_starts[i + 1]; ++j) {
                auto [tri_min, tri_max] = triangle_bounds(triangles[j]);
                for (int corner = 0; corner < 8; ++corner) {
                    float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                                (corner & 2) ? tri_max.y : tri_min.y,
                                (corner & 4) ? tri_max.z : tri_min.z};
                    float3 p_obb = {dot(p, axis_x), dot(p, axis_y),
                                    dot(p, axis_z)};
                    child_min = min(child_min, p_obb);
                    child_max = max(child_max, p_obb);
                }
            }

            // Add small expansion for numerical robustness
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
}

BVH *build_aabb_node(
    const std::vector<std::vector<uint32_t>> &groups,
    const std::vector<uint32_t> &group_starts, std::vector<Triangle> &triangles,
    uint32_t depth,
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition) {

    constexpr auto MAX = std::numeric_limits<float>::max();

    AABBNode *node = new AABBNode();

    for (int i = 0; i < 8; ++i) {
        if (group_starts[i] < group_starts[i + 1]) {
            node->aabb_children[i] =
                partition(group_starts[i], group_starts[i + 1], depth + 1);
            auto [child_min, child_max] =
                compute_aabb(group_starts[i], group_starts[i + 1], triangles);
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

BVH *build_canonical_tree_8_mixed_sah(std::vector<Triangle> &triangles,
                                      int max_prims_per_leaf = 8,
                                      int max_tree_depth = 64,
                                      float traversal_cost = 1.0f,
                                      float intersection_cost = 15.0f,
                                      int obb_depth_threshold = 4) {

    struct Split {
        int axis;
        float positions[7];
        float cost;
        bool use_obb;
        OBB obb;
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
        best_split.use_obb = false;
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = intersection_cost * count;

        // Try AABB splits along each axis (identical to canonical_tree_8_sah)
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
                best_split.use_obb = false;
            }
        }

        // Try OBB-based splits if at sufficient depth
        if (depth >= obb_depth_threshold) {
            OBB obb = compute_obb(low, high, triangles);

            for (int axis = 0; axis < 3; ++axis) {
                float3 world_axis = {obb.orientation[axis].x,
                                     obb.orientation[axis].y,
                                     obb.orientation[axis].z};

                // Project centroids onto this OBB axis
                float min_proj = MAX, max_proj = -MAX;
                for (uint32_t i = low; i < high; ++i) {
                    float3 c = triangle_centroid(triangles[i]);
                    float proj = dot(c, world_axis);
                    min_proj = std::min(min_proj, proj);
                    max_proj = std::max(max_proj, proj);
                }

                float extent = max_proj - min_proj;
                if (extent < 1e-6f)
                    continue;

                // Divide into 8 equal parts along OBB axis
                float split_positions[7];
                for (int i = 0; i < 7; ++i) {
                    split_positions[i] = min_proj + (i + 1) * extent / 8.0f;
                }

                // Evaluate this 8-way OBB split
                std::vector<float3> group_mins(8, float3{MAX, MAX, MAX});
                std::vector<float3> group_maxs(8, float3{-MAX, -MAX, -MAX});
                std::vector<uint32_t> group_counts(8, 0);

                float3 axis_x = {obb.orientation[0].x, obb.orientation[0].y,
                                 obb.orientation[0].z};
                float3 axis_y = {obb.orientation[1].x, obb.orientation[1].y,
                                 obb.orientation[1].z};
                float3 axis_z = {obb.orientation[2].x, obb.orientation[2].y,
                                 obb.orientation[2].z};

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

                    // Transform triangle bounds to OBB space
                    auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                    float3 obb_min = {MAX, MAX, MAX};
                    float3 obb_max = {-MAX, -MAX, -MAX};

                    for (int corner = 0; corner < 8; ++corner) {
                        float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                                    (corner & 2) ? tri_max.y : tri_min.y,
                                    (corner & 4) ? tri_max.z : tri_min.z};
                        float3 p_obb = {dot(p, axis_x), dot(p, axis_y),
                                        dot(p, axis_z)};
                        obb_min = min(obb_min, p_obb);
                        obb_max = max(obb_max, p_obb);
                    }

                    group_mins[group] = min(group_mins[group], obb_min);
                    group_maxs[group] = max(group_maxs[group], obb_max);
                    group_counts[group]++;
                }

                // Calculate SAH cost for OBB split
                float split_cost = traversal_cost * 7;
                for (int i = 0; i < 8; ++i) {
                    if (group_counts[i] > 0) {
                        float area = surface_area(group_mins[i], group_maxs[i]);
                        split_cost += (area / parent_area) * intersection_cost *
                                      group_counts[i];
                    }
                }

                // Update best split if this OBB split is better
                if (split_cost < best_split.cost) {
                    best_split.axis = axis;
                    for (int i = 0; i < 7; ++i)
                        best_split.positions[i] = split_positions[i];
                    best_split.cost = split_cost;
                    best_split.use_obb = true;
                    best_split.obb = obb;
                }
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

        if (best_split.use_obb) {
            float3 world_axis = {best_split.obb.orientation[best_split.axis].x,
                                 best_split.obb.orientation[best_split.axis].y,
                                 best_split.obb.orientation[best_split.axis].z};

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
        }

        // Reorder triangles based on groups
        std::vector<Triangle> temporaries;
        temporaries.reserve(count);
        std::vector<uint32_t> group_starts(9);
        group_starts[0] = low;

        for (int g = 0; g < 8; ++g) {
            for (uint32_t idx : groups[g]) {
                temporaries.push_back(triangles[idx]);
            }
            group_starts[g + 1] = group_starts[g] + groups[g].size();
        }

        std::copy(temporaries.begin(), temporaries.end(),
                  triangles.begin() + low);

        // Build appropriate node type based on best split
        if (best_split.use_obb) {
            OBBNode *node = new OBBNode();
            node->orientation = best_split.obb.orientation;

            float3 axis_x = {best_split.obb.orientation[0].x,
                             best_split.obb.orientation[0].y,
                             best_split.obb.orientation[0].z};
            float3 axis_y = {best_split.obb.orientation[1].x,
                             best_split.obb.orientation[1].y,
                             best_split.obb.orientation[1].z};
            float3 axis_z = {best_split.obb.orientation[2].x,
                             best_split.obb.orientation[2].y,
                             best_split.obb.orientation[2].z};

            for (int i = 0; i < 8; ++i) {
                if (group_starts[i] < group_starts[i + 1]) {
                    node->obb_children[i] = partition(
                        group_starts[i], group_starts[i + 1], depth + 1);

                    // Compute OBB bounds for child in parent OBB space
                    float3 child_min = {MAX, MAX, MAX};
                    float3 child_max = {-MAX, -MAX, -MAX};

                    for (uint32_t j = group_starts[i]; j < group_starts[i + 1];
                         ++j) {
                        auto [tri_min, tri_max] = triangle_bounds(triangles[j]);
                        for (int corner = 0; corner < 8; ++corner) {
                            float3 p = {(corner & 1) ? tri_max.x : tri_min.x,
                                        (corner & 2) ? tri_max.y : tri_min.y,
                                        (corner & 4) ? tri_max.z : tri_min.z};
                            float3 p_obb = {dot(p, axis_x), dot(p, axis_y),
                                            dot(p, axis_z)};
                            child_min = min(child_min, p_obb);
                            child_max = max(child_max, p_obb);
                        }
                    }

                    // Add small expansion for numerical robustness
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
            // AABB node - identical to canonical_tree_8_sah
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
                    // empty child
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

BVH *build_canonical_tree_8_mixed(std::vector<Triangle> &triangles,
                                  Heuristic heuristic = Heuristic::SurfaceArea,
                                  int max_prims_per_leaf = 8,
                                  int max_tree_depth = 64) {
    switch (heuristic) {
    case Heuristic::SurfaceArea:
        return build_canonical_tree_8_mixed_sah(triangles, max_prims_per_leaf,
                                                max_tree_depth);
    case Heuristic::MedianSplit:
        assert(false && "unimplemented");
    }
}
