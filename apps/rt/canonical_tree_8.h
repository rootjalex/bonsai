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

    constexpr auto MAX = std::numeric_limits<float>::max();
    constexpr int NUM_OBJECT_BINS = 32;
    constexpr int N = 8;

    struct BuildRecord {
        uint32_t begin;
        uint32_t end;
        float3 aabb_min;
        float3 aabb_max;
    };

    std::function<BVH *(BuildRecord, uint32_t)> build_recursive =
        [&](BuildRecord record, uint32_t depth) -> BVH * {
        assert(depth < max_tree_depth);
        uint32_t count = record.end - record.begin;

        if (count <= max_prims_per_leaf || depth >= max_tree_depth - 1) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            std::copy(triangles.begin() + record.begin,
                      triangles.begin() + record.end, data);
            return new BVH(Leaf{
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }
        float3 centroid_bounds_min = triangle_centroid(triangles[record.begin]);
        float3 centroid_bounds_max = centroid_bounds_min;

        for (uint32_t i = record.begin + 1; i < record.end; ++i) {
            float3 c = triangle_centroid(triangles[i]);
            centroid_bounds_min = min(centroid_bounds_min, c);
            centroid_bounds_max = max(centroid_bounds_max, c);
        }

        // Similar to embree, make multiple binary splits to create N children.
        std::vector<BuildRecord> children;
        children.push_back(record);

        // Split each partition binarily, up to log2(N) times
        for (int split_level = 0; split_level < 3 && children.size() < N;
             ++split_level) {
            std::vector<BuildRecord> next_children;

            for (const BuildRecord &child_rec : children) {
                uint32_t child_count = child_rec.end - child_rec.begin;
                if (child_count <= max_prims_per_leaf) {
                    // Child is too small, don't split it further.
                    next_children.push_back(child_rec);
                    continue;
                }
                struct BinarySplit {
                    int axis = -1;
                    float position = 0.0f;
                    float sah_cost = std::numeric_limits<float>::max();
                };

                BinarySplit best_split;
                float parent_area =
                    surface_area(child_rec.aabb_min, child_rec.aabb_max);
                float leaf_cost = intersection_cost * child_count;
                float3 local_centroid_min =
                    triangle_centroid(triangles[child_rec.begin]);
                float3 local_centroid_max = local_centroid_min;
                for (uint32_t i = child_rec.begin + 1; i < child_rec.end; ++i) {
                    float3 c = triangle_centroid(triangles[i]);
                    local_centroid_min = min(local_centroid_min, c);
                    local_centroid_max = max(local_centroid_max, c);
                }
                for (int axis = 0; axis < 3; ++axis) {
                    float extent =
                        (axis == 0)
                            ? local_centroid_max.x - local_centroid_min.x
                        : (axis == 1)
                            ? local_centroid_max.y - local_centroid_min.y
                            : local_centroid_max.z - local_centroid_min.z;

                    if (extent < 1e-6f)
                        continue;

                    float centroid_min = (axis == 0)   ? local_centroid_min.x
                                         : (axis == 1) ? local_centroid_min.y
                                                       : local_centroid_min.z;
                    Bin bins[NUM_OBJECT_BINS];
                    float bin_scale = NUM_OBJECT_BINS / extent;

                    for (uint32_t i = child_rec.begin; i < child_rec.end; ++i) {
                        float3 centroid = triangle_centroid(triangles[i]);
                        float c_axis = (axis == 0)   ? centroid.x
                                       : (axis == 1) ? centroid.y
                                                     : centroid.z;

                        int bin_idx = static_cast<int>((c_axis - centroid_min) *
                                                       bin_scale);
                        bin_idx = std::min(bin_idx, NUM_OBJECT_BINS - 1);
                        bin_idx = std::max(bin_idx, 0);

                        auto [prim_min, prim_max] =
                            triangle_bounds(triangles[i]);
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

                        float left_area = surface_area(left_bins[i].aabb_min,
                                                       left_bins[i].aabb_max);
                        float right_area = surface_area(right_bins[i].aabb_min,
                                                        right_bins[i].aabb_max);

                        float sah = traversal_cost +
                                    (left_area / parent_area) *
                                        intersection_cost * left_count +
                                    (right_area / parent_area) *
                                        intersection_cost * right_count;

                        if (sah < best_split.sah_cost) {
                            best_split.axis = axis;
                            best_split.sah_cost = sah;
                            best_split.position =
                                centroid_min +
                                (i + 1) * (extent / NUM_OBJECT_BINS);
                        }
                    }
                }
                if (best_split.sah_cost >= leaf_cost) {
                    next_children.push_back(child_rec);
                    continue;
                }
                auto mid_it = std::partition(
                    triangles.begin() + child_rec.begin,
                    triangles.begin() + child_rec.end,
                    [&](const Triangle &tri) {
                        float3 centroid = triangle_centroid(tri);
                        float c_axis = (best_split.axis == 0)   ? centroid.x
                                       : (best_split.axis == 1) ? centroid.y
                                                                : centroid.z;
                        return c_axis < best_split.position;
                    });

                uint32_t mid = std::distance(triangles.begin(), mid_it);
                if (mid == child_rec.begin || mid == child_rec.end) {
                    mid = child_rec.begin + child_count / 2;
                    std::nth_element(triangles.begin() + child_rec.begin,
                                     triangles.begin() + mid,
                                     triangles.begin() + child_rec.end,
                                     [&](const Triangle &a, const Triangle &b) {
                                         float3 ca = triangle_centroid(a);
                                         float3 cb = triangle_centroid(b);
                                         float ca_val =
                                             (best_split.axis == 0)   ? ca.x
                                             : (best_split.axis == 1) ? ca.y
                                                                      : ca.z;
                                         float cb_val =
                                             (best_split.axis == 0)   ? cb.x
                                             : (best_split.axis == 1) ? cb.y
                                                                      : cb.z;
                                         return ca_val < cb_val;
                                     });
                }
                auto [left_min, left_max] =
                    compute_aabb(child_rec.begin, mid, triangles);
                auto [right_min, right_max] =
                    compute_aabb(mid, child_rec.end, triangles);

                next_children.push_back(
                    BuildRecord{child_rec.begin, mid, left_min, left_max});
                next_children.push_back(
                    BuildRecord{mid, child_rec.end, right_min, right_max});
            }

            children = std::move(next_children);
            if (children.size() >= N)
                break;
        }
        Interior interior;
        for (size_t i = 0; i < N; ++i) {
            if (i < children.size()) {
                interior.children[i] = build_recursive(children[i], depth + 1);
                interior.lo[i] = children[i].aabb_min;
                interior.hi[i] = children[i].aabb_max;
            } else {
                interior.children[i] = nullptr;
                interior.lo[i] = float3{MAX, MAX, MAX};
                interior.hi[i] = float3{-MAX, -MAX, -MAX};
            }
        }

        return new BVH(interior);
    };
    const uint32_t high = static_cast<uint32_t>(triangles.size());
    auto [root_min, root_max] = compute_aabb(0, high, triangles);
    BuildRecord root{0, high, root_min, root_max};
    return build_recursive(root, 0);
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
