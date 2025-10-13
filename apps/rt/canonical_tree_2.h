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
            assert(static_cast<uint8_t>(count) > 0 && count <= 15);
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
                                float intersection_cost = 10.0f) {

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
            assert(static_cast<uint8_t>(count) > 0 &&
                   count <= max_prims_per_leaf);
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

        Split best_split;
        float parent_area = surface_area(aabb_min, aabb_max);
        float leaf_cost = (count <= max_prims_per_leaf)
                              ? intersection_cost * count
                              : std::numeric_limits<float>::max();

        for (int axis = 0; axis < 3; ++axis) {
            float extent =
                (axis == 0)   ? centroid_bounds_max.x - centroid_bounds_min.x
                : (axis == 1) ? centroid_bounds_max.y - centroid_bounds_min.y
                              : centroid_bounds_max.z - centroid_bounds_min.z;

            if (extent < 1e-6f)
                continue;

            float centroid_min = (axis == 0)   ? centroid_bounds_min.x
                                 : (axis == 1) ? centroid_bounds_min.y
                                               : centroid_bounds_min.z;
            Bin bins[NUM_OBJECT_BINS];
            float bin_scale = NUM_OBJECT_BINS / extent;

            for (uint32_t i = begin; i < end; ++i) {
                float3 centroid = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0)   ? centroid.x
                               : (axis == 1) ? centroid.y
                                             : centroid.z;
                int bin_idx =
                    static_cast<int>((c_axis - centroid_min) * bin_scale);
                bin_idx = std::min(bin_idx, NUM_OBJECT_BINS - 1);
                bin_idx = std::max(bin_idx, 0);

                auto [prim_min, prim_max] = triangle_bounds(triangles[i]);
                bins[bin_idx].extend(prim_min, prim_max);
            }

            Bin left_bins[NUM_OBJECT_BINS - 1];
            left_bins[0] = bins[0];
            for (int i = 1; i < NUM_OBJECT_BINS - 1; ++i) {
                left_bins[i] = left_bins[i - 1] + bins[i];
            }

            Bin right_bins[NUM_OBJECT_BINS - 1];
            right_bins[NUM_OBJECT_BINS - 2] = bins[NUM_OBJECT_BINS - 1];
            for (int i = NUM_OBJECT_BINS - 3; i >= 0; --i) {
                right_bins[i] = bins[i + 1] + right_bins[i + 1];
            }

            for (int i = 0; i < NUM_OBJECT_BINS - 1; ++i) {
                uint32_t left_count = left_bins[i].count;
                uint32_t right_count = right_bins[i].count;

                if (left_count == 0 || right_count == 0)
                    continue;

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
                    best_split.position =
                        centroid_min + (i + 1) * (extent / NUM_OBJECT_BINS);
                }
            }
        }

        if (best_split.axis == -1) {
            float3 extent = centroid_bounds_max - centroid_bounds_min;
            int longest_axis = 0;
            if (extent.y > extent.x && extent.y > extent.z)
                longest_axis = 1;
            else if (extent.z > extent.x && extent.z > extent.y)
                longest_axis = 2;

            uint32_t mid = begin + count / 2;
            std::nth_element(triangles.begin() + begin, triangles.begin() + mid,
                             triangles.begin() + end,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_val = (longest_axis == 0)   ? ca.x
                                                : (longest_axis == 1) ? ca.y
                                                                      : ca.z;
                                 float cb_val = (longest_axis == 0)   ? cb.x
                                                : (longest_axis == 1) ? cb.y
                                                                      : cb.z;
                                 return ca_val < cb_val;
                             });

            BVH *left = partition(begin, mid, depth + 1);
            BVH *right = partition(mid, end, depth + 1);

            return new BVH(Interior{
                .low = aabb_min,
                .high = aabb_max,
                .left = left,
                .right = right,
            });
        }

        if (best_split.sah_cost >= leaf_cost && count <= max_prims_per_leaf) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[begin + i];
            }
            assert(static_cast<uint8_t>(count) > 0 &&
                   count <= max_prims_per_leaf);
            return new BVH(Leaf{
                .low = aabb_min,
                .high = aabb_max,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

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
                            Heuristic heuristic, int max_prims_per_leaf,
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
