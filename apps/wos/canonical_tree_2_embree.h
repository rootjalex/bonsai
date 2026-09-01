// An alternative reference builder for the binary tree, following Embree's
// construction. It offers both a median split and a surface-area heuristic,
// where canonical_tree_2.h (derived from FCPW) offers only the latter. The two
// define the same entry points, so a driver includes one or the other.

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
    return 2.0f * (extent[0] * extent[1] + extent[0] * extent[2] +
                   extent[1] * extent[2]);
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
        float max_extent = extent[0];
        if (extent[1] > max_extent) {
            axis = 1;
            max_extent = extent[1];
        }
        if (extent[2] > max_extent) {
            axis = 2;
        }

        // Partition around midpoint along axis.
        auto mid_it = triangles.begin() + low + count / 2;
        std::nth_element(
            triangles.begin() + low, mid_it, triangles.begin() + high,
            [&](const Triangle &a, const Triangle &b) {
                float ca = (axis == 0)   ? (a.p0[0] + a.p1[0] + a.p2[0])
                           : (axis == 1) ? (a.p0[1] + a.p1[1] + a.p2[1])
                                         : (a.p0[2] + a.p1[2] + a.p2[2]);
                float cb = (axis == 0)   ? (b.p0[0] + b.p1[0] + b.p2[0])
                           : (axis == 1) ? (b.p0[1] + b.p1[1] + b.p2[1])
                                         : (b.p0[2] + b.p1[2] + b.p2[2]);
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

// Based on FCPW's Bvh_SurfaceArea:
// https://github.com/rohan-sawhney/fcpw/blob/e36bc9b34af6088fb78ddbb6a93e26686779678a/include/fcpw/utilities/scene_data.h#L21
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

        float3 centroid_min = triangle_centroid(triangles[low]);
        float3 centroid_max = centroid_min;

        for (uint32_t i = low + 1; i < high; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_min = min(centroid_min, c);
            centroid_max = max(centroid_max, c);
        }

        int best_axis = -1;
        int best_bucket = -1;
        float best_cost = MAX;
        for (int axis = 0; axis < 3; ++axis) {
            float aabb_extent = (axis == 0)   ? aabb_max[0] - aabb_min[0]
                                : (axis == 1) ? aabb_max[1] - aabb_min[1]
                                              : aabb_max[2] - aabb_min[2];

            if (aabb_extent < 1e-6f)
                continue; // ...skip degenerate axis

            float aabb_min_axis = (axis == 0)   ? aabb_min[0]
                                  : (axis == 1) ? aabb_min[1]
                                                : aabb_min[2];

            float bucket_width = aabb_extent / nBuckets;
            std::vector<BucketInfo> buckets(nBuckets);

            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0) ? c[0] : (axis == 1) ? c[1] : c[2];

                int bucket_idx = (int)((c_axis - aabb_min_axis) / bucket_width);
                bucket_idx = std::clamp(bucket_idx, 0, nBuckets - 1);

                auto [tri_min, tri_max] = triangle_bounds(triangles[i]);
                buckets[bucket_idx].box_min =
                    min(buckets[bucket_idx].box_min, tri_min);
                buckets[bucket_idx].box_max =
                    max(buckets[bucket_idx].box_max, tri_max);
                buckets[bucket_idx].count++;
            }

            std::vector<BucketInfo> right_buckets(nBuckets);
            for (int i = nBuckets - 1; i >= 0; --i) {
                if (i == nBuckets - 1) {
                    right_buckets[i] = buckets[i];
                } else {
                    right_buckets[i].box_min =
                        min(right_buckets[i + 1].box_min, buckets[i].box_min);
                    right_buckets[i].box_max =
                        max(right_buckets[i + 1].box_max, buckets[i].box_max);
                    right_buckets[i].count =
                        right_buckets[i + 1].count + buckets[i].count;
                }
            }
            BucketInfo left_bucket;
            for (int split_bucket = 0; split_bucket < nBuckets - 1;
                 ++split_bucket) {
                if (buckets[split_bucket].count > 0) {
                    if (left_bucket.count == 0) {
                        left_bucket = buckets[split_bucket];
                    } else {
                        left_bucket.box_min = min(
                            left_bucket.box_min, buckets[split_bucket].box_min);
                        left_bucket.box_max = max(
                            left_bucket.box_max, buckets[split_bucket].box_max);
                        left_bucket.count += buckets[split_bucket].count;
                    }
                }

                const BucketInfo &right_bucket =
                    right_buckets[split_bucket + 1];
                if (left_bucket.count == 0 || right_bucket.count == 0)
                    continue;

                // https://github.com/rohan-sawhney/fcpw/blob/e36bc9b34af6088fb78ddbb6a93e26686779678a/include/fcpw/fcpw.inl#L826
                float left_area =
                    surface_area(left_bucket.box_min, left_bucket.box_max);
                float right_area =
                    surface_area(right_bucket.box_min, right_bucket.box_max);
                float cost = left_bucket.count * left_area +
                             right_bucket.count * right_area;
                if (cost <= best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_bucket = split_bucket;
                }
            }
        }

        if (best_axis == -1) {
            uint32_t mid = low + count / 2;
            float3 extent = aabb_max - aabb_min;
            int fallback_axis = (extent[0] > extent[1] && extent[0] > extent[2])
                                    ? 0
                                : (extent[1] > extent[2]) ? 1
                                                          : 2;

            std::nth_element(triangles.begin() + low, triangles.begin() + mid,
                             triangles.begin() + high,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_axis = (fallback_axis == 0)   ? ca[0]
                                                 : (fallback_axis == 1) ? ca[1]
                                                                        : ca[2];
                                 float cb_axis = (fallback_axis == 0)   ? cb[0]
                                                 : (fallback_axis == 1) ? cb[1]
                                                                        : cb[2];
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

        float aabb_extent = (best_axis == 0)   ? aabb_max[0] - aabb_min[0]
                            : (best_axis == 1) ? aabb_max[1] - aabb_min[1]
                                               : aabb_max[2] - aabb_min[2];
        float aabb_min_axis = (best_axis == 0)   ? aabb_min[0]
                              : (best_axis == 1) ? aabb_min[1]
                                                 : aabb_min[2];
        float bucket_width = aabb_extent / nBuckets;

        auto mid_it = std::partition(
            triangles.begin() + low, triangles.begin() + high,
            [&](const Triangle &tri) {
                float3 c = triangle_centroid(tri);
                float c_axis = (best_axis == 0)   ? c[0]
                               : (best_axis == 1) ? c[1]
                                                  : c[2];
                int bucket_idx = (int)((c_axis - aabb_min_axis) / bucket_width);
                bucket_idx = std::clamp(bucket_idx, 0, nBuckets - 1);
                return bucket_idx <= best_bucket;
            });

        uint32_t mid = std::distance(triangles.begin(), mid_it);
        if (mid == low || mid == high) {
            mid = low + count / 2;
            std::nth_element(triangles.begin() + low, triangles.begin() + mid,
                             triangles.begin() + high,
                             [&](const Triangle &a, const Triangle &b) {
                                 float3 ca = triangle_centroid(a);
                                 float3 cb = triangle_centroid(b);
                                 float ca_axis = (best_axis == 0)   ? ca[0]
                                                 : (best_axis == 1) ? ca[1]
                                                                    : ca[2];
                                 float cb_axis = (best_axis == 0)   ? cb[0]
                                                 : (best_axis == 1) ? cb[1]
                                                                    : cb[2];
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
                            int max_prims_per_leaf = 4, // same as FCPW default!
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
