float3 triangle_centroid(const Triangle &tri) {
    return (tri.p0 + tri.p1 + tri.p2) * (1.0f / 3.0f);
}

constexpr float FCPW_EPSILON = std::numeric_limits<float>::epsilon();

// https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/core/bounding_volumes.h#L135

inline float surface_area(float3 box_min, float3 box_max) {
    float3 e = max(box_max - box_min, float3{1e-5f, 1e-5f, 1e-5f});
    float prod = e.x * e.y * e.z;
    return 2.0f * (prod / e.x + prod / e.y + prod / e.z);
}

inline std::pair<float3, float3>
compute_triangle_bbox_fcpw(const Triangle &tri) {
    // Start with invalid box
    float3 pMin = float3{std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
    float3 pMax = float3{std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest()};

    // expandToInclude each vertex (with epsilon like FCPW)
    float3 eps = float3{FCPW_EPSILON, FCPW_EPSILON, FCPW_EPSILON};
    for (int i = 0; i < 3; i++) {
        float3 v;
        if (i == 0)
            v = tri.p0;
        if (i == 1)
            v = tri.p1;
        if (i == 2)
            v = tri.p2;
        pMin = min(pMin, v - eps);
        pMax = max(pMax, v + eps);
    }

    return {pMin, pMax};
}

// Based on FCPW's BVH SAH implementation.
// https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L143
int64_t node_count = 0, leaf_count = 0; // Track counts for comparison.
BVH *build_canonical_tree_2_sah(std::vector<Triangle> &triangles, int leaf_size,
                                int max_tree_depth, int nBuckets = 8) {
    constexpr auto MAX = std::numeric_limits<float>::max();
    constexpr auto MIN = std::numeric_limits<float>::lowest();

    struct BoundingBox {
        float3 pMin = float3{MAX, MAX, MAX};
        float3 pMax = float3{MIN, MIN, MIN};

        void expandToInclude(const BoundingBox &b) {
            pMin = min(pMin, b.pMin);
            pMax = max(pMax, b.pMax);
        }

        void expandToInclude(float3 centroid) {
            float3 eps = float3{FCPW_EPSILON, FCPW_EPSILON, FCPW_EPSILON};
            pMin = min(pMin, centroid - eps);
            pMax = max(pMax, centroid + eps);
        }

        float3 extent() const { return pMax - pMin; }
    };

    // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L212
    std::vector<std::pair<float3, float3>> referenceBoxes(triangles.size());
    std::vector<float3> referenceCentroids(triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i) {
        referenceBoxes[i] = compute_triangle_bbox_fcpw(triangles[i]);
        referenceCentroids[i] = triangle_centroid(triangles[i]);
    }

    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t start, uint32_t end, uint32_t depth) -> BVH * {
        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L157
        node_count++;
        uint32_t nReferences = end - start;

        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L159
        BoundingBox bb, bc;
        for (uint32_t p = start; p < end; p++) {
            bb.pMin = min(bb.pMin, referenceBoxes[p].first);
            bb.pMax = max(bb.pMax, referenceBoxes[p].second);
            bc.expandToInclude(referenceCentroids[p]);
        }

        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L170
        if (nReferences <= leaf_size || depth >= max_tree_depth) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * nReferences));
            for (uint32_t i = 0; i < nReferences; ++i) {
                data[i] = triangles[start + i];
            }
            leaf_count++;
            return new BVH(Leaf{
                .low = bb.pMin,
                .high = bb.pMax,
                .nprims = static_cast<uint8_t>(nReferences),
                .data = data,
            });
        }

        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L200
        float3 extent = bb.extent();
        int splitDim = -1;
        float splitCoord = 0.0f;
        float splitCost = MAX;

        for (int dim = 0; dim < 3; dim++) {
            float extent_dim = extent[dim];
            if (extent_dim < 1e-6f)
                continue;

            float bucketWidth = extent_dim / nBuckets;

            // Bin references into buckets.
            std::vector<BoundingBox> buckets(nBuckets);
            std::vector<int> bucketCounts(nBuckets, 0);

            for (uint32_t p = start; p < end; p++) {
                float centroid_dim = referenceCentroids[p][dim];
                int bucketIndex =
                    (int)((centroid_dim - bb.pMin[dim]) / bucketWidth);
                bucketIndex = std::clamp(bucketIndex, 0, nBuckets - 1);

                buckets[bucketIndex].pMin =
                    min(buckets[bucketIndex].pMin, referenceBoxes[p].first);
                buckets[bucketIndex].pMax =
                    max(buckets[bucketIndex].pMax, referenceBoxes[p].second);
                bucketCounts[bucketIndex]++;
            }

            // Sweep right to left to build right bucket bounding boxes.
            std::vector<BoundingBox> rightBuckets(nBuckets);
            std::vector<int> rightBucketCounts(nBuckets, 0);
            BoundingBox boxRefRight;

            for (int b = nBuckets - 1; b > 0; b--) {
                boxRefRight.expandToInclude(buckets[b]);
                rightBuckets[b] = boxRefRight;
                rightBucketCounts[b] = bucketCounts[b];
                if (b != nBuckets - 1) {
                    rightBucketCounts[b] += rightBucketCounts[b + 1];
                }
            }

            // Evaluate bucket split costs.
            BoundingBox boxRefLeft;
            int nReferencesLeft = 0;

            for (int b = 1; b < nBuckets; b++) {
                boxRefLeft.expandToInclude(buckets[b - 1]);
                nReferencesLeft += bucketCounts[b - 1];
                if (nReferencesLeft > 0 && rightBucketCounts[b] > 0) {
                    float leftArea =
                        surface_area(boxRefLeft.pMin, boxRefLeft.pMax);
                    float rightArea = surface_area(rightBuckets[b].pMin,
                                                   rightBuckets[b].pMax);
                    float cost = nReferencesLeft * leftArea +
                                 rightBucketCounts[b] * rightArea;
                    if (cost < splitCost) {
                        splitCost = cost;
                        splitDim = dim;
                        splitCoord = bb.pMin[dim] + b * bucketWidth;
                    }
                }
            }
        }
        // https://github.com/rohan-sawhney/fcpw/blob/653798c6122674adffc70ddf36ec4480e5c26098/include/fcpw/aggregates/bvh.inl#L104
        if (splitDim == -1) {
            float3 centroid_extent = bc.extent();
            splitDim = (centroid_extent.x > centroid_extent.y &&
                        centroid_extent.x > centroid_extent.z)
                           ? 0
                       : (centroid_extent.y > centroid_extent.z) ? 1
                                                                 : 2;
            splitCoord = (bc.pMin[splitDim] + bc.pMax[splitDim]) * 0.5f;
        }

        uint32_t mid = start;
        for (uint32_t i = start; i < end; i++) {
            if (referenceCentroids[i][splitDim] < splitCoord) {
                std::swap(triangles[i], triangles[mid]);
                std::swap(referenceBoxes[i], referenceBoxes[mid]);
                std::swap(referenceCentroids[i], referenceCentroids[mid]);
                mid++;
            }
        }
        if (mid == start || mid == end) {
            mid = start + nReferences / 2;
        }

        // Recursively build left and right children
        BVH *left = partition(start, mid, depth + 1);
        BVH *right = partition(mid, end, depth + 1);

        return new BVH(Interior{
            .low = bb.pMin,
            .high = bb.pMax,
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
        delete node;
        return;
    }

    if (std::holds_alternative<Leaf>(*node)) {
        Leaf &leaf = std::get<Leaf>(*node);
        free(leaf.data);
        delete node;
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
    case Heuristic::SurfaceArea: {
        return build_canonical_tree_2_sah(triangles, max_prims_per_leaf,
                                          max_tree_depth);
    }
    case Heuristic::MedianSplit:
        assert(false && "unimplemented");
        return nullptr;
    }
}
