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

BVH *build_canonical_tree_2_sah(std::vector<Triangle> &triangles, int leaf_size,
                                int max_tree_depth, int nBuckets = 8) {
    constexpr auto MAX = std::numeric_limits<float>::max();

    struct BucketInfo {
        float3 box_min = float3{MAX, MAX, MAX};
        float3 box_max = float3{-MAX, -MAX, -MAX};
        int count = 0;
    };

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> BVH * {
        uint32_t count = high - low;
        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        // Create leaf only if at exact leaf_size or fewer, or max depth reached
        if (count <= leaf_size || depth >= max_tree_depth) {
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

        // Compute centroid bounds for splitting.
        float3 centroid_min = triangle_centroid(triangles[low]);
        float3 centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        // Find best split using binned SAH.
        int best_axis = -1;
        int best_bucket = -1;
        float best_cost = MAX;

        // Try splitting along each axis.
        for (int axis = 0; axis < 3; ++axis) {
            float centroid_extent =
                (axis == 0)   ? centroid_max.x - centroid_min.x
                : (axis == 1) ? centroid_max.y - centroid_min.y
                              : centroid_max.z - centroid_min.z;

            if (centroid_extent < 1e-6f)
                continue; // Skip degenerate axis.

            float centroid_min_axis = (axis == 0)   ? centroid_min.x
                                      : (axis == 1) ? centroid_min.y
                                                    : centroid_min.z;

            // Initialize buckets
            std::vector<BucketInfo> buckets(nBuckets);

            // Assign triangles to buckets
            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;

                int bucket_idx =
                    nBuckets * ((c_axis - centroid_min_axis) / centroid_extent);
                if (bucket_idx == nBuckets)
                    bucket_idx = nBuckets - 1;

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                buckets[bucket_idx].box_min =
                    min(buckets[bucket_idx].box_min, tri_min);
                buckets[bucket_idx].box_max =
                    max(buckets[bucket_idx].box_max, tri_max);
                buckets[bucket_idx].count++;
            }

            // Compute costs for each split candidate
            for (int split_bucket = 0; split_bucket < nBuckets - 1;
                 ++split_bucket) {
                float3 left_min = float3{MAX, MAX, MAX};
                float3 left_max = float3{-MAX, -MAX, -MAX};
                int left_count = 0;

                float3 right_min = float3{MAX, MAX, MAX};
                float3 right_max = float3{-MAX, -MAX, -MAX};
                int right_count = 0;

                // Accumulate left side
                for (int i = 0; i <= split_bucket; ++i) {
                    if (buckets[i].count > 0) {
                        left_min = min(left_min, buckets[i].box_min);
                        left_max = max(left_max, buckets[i].box_max);
                        left_count += buckets[i].count;
                    }
                }

                // Accumulate right side
                for (int i = split_bucket + 1; i < nBuckets; ++i) {
                    if (buckets[i].count > 0) {
                        right_min = min(right_min, buckets[i].box_min);
                        right_max = max(right_max, buckets[i].box_max);
                        right_count += buckets[i].count;
                    }
                }

                // Skip if one side is empty
                if (left_count == 0 || right_count == 0)
                    continue;

                // Calculate SAH cost (matching FCPW)
                float left_area = surface_area(left_min, left_max);
                float right_area = surface_area(right_min, right_max);
                float cost = left_count * left_area + right_count * right_area;

                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_bucket = split_bucket;
                }
            }
        }

        // If no valid split found, fall back to median split
        if (best_axis == -1) {
            uint32_t mid = low + count / 2;

            // Find longest axis for fallback
            float3 extent = aabb_max - aabb_min;
            int fallback_axis = (extent.x > extent.y && extent.x > extent.z) ? 0
                                : (extent.y > extent.z)                      ? 1
                                                        : 2;

            std::nth_element(triangles.begin() + low, triangles.begin() + mid,
                             triangles.begin() + high,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_axis = (fallback_axis == 0)   ? ca.x
                                                 : (fallback_axis == 1) ? ca.y
                                                                        : ca.z;
                                 float cb_axis = (fallback_axis == 0)   ? cb.x
                                                 : (fallback_axis == 1) ? cb.y
                                                                        : cb.z;
                                 return ca_axis < cb_axis;
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

        // Partition triangles based on best split
        float centroid_extent =
            (best_axis == 0)   ? centroid_max.x - centroid_min.x
            : (best_axis == 1) ? centroid_max.y - centroid_min.y
                               : centroid_max.z - centroid_min.z;
        float centroid_min_axis = (best_axis == 0)   ? centroid_min.x
                                  : (best_axis == 1) ? centroid_min.y
                                                     : centroid_min.z;

        auto mid_it = std::partition(
            triangles.begin() + low, triangles.begin() + high,
            [&](const Triangle &tri) {
                float3 c = triangle_centroid(tri);
                float c_axis = (best_axis == 0)   ? c.x
                               : (best_axis == 1) ? c.y
                                                  : c.z;
                int bucket_idx =
                    nBuckets * ((c_axis - centroid_min_axis) / centroid_extent);
                if (bucket_idx == nBuckets)
                    bucket_idx = nBuckets - 1;
                return bucket_idx <= best_bucket;
            });

        uint32_t mid = std::distance(triangles.begin(), mid_it);

        // Handle edge case where all triangles end up on one side
        if (mid == low || mid == high) {
            mid = low + count / 2;
            std::nth_element(triangles.begin() + low, triangles.begin() + mid,
                             triangles.begin() + high,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_axis = (best_axis == 0)   ? ca.x
                                                 : (best_axis == 1) ? ca.y
                                                                    : ca.z;
                                 float cb_axis = (best_axis == 0)   ? cb.x
                                                 : (best_axis == 1) ? cb.y
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
                            int max_prims_per_leaf = 4,
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
