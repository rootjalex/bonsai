#include "runtime/bonsai_cpp.h"
#ifdef AJR_PROFILE
#include "rt_profile.h"
#else
#include "rt_gen.h"
#endif

#include <fcpw/fcpw.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>

#include "runtime/bonsai_benchmark.h"

constexpr int k = 7;
constexpr int m = 1;

// namespace {

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

std::vector<Ray> load_rays_binary(const std::string &filename,
                                  int64_t ray_count) {
    std::vector<Ray> rays;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename
                  << " for reading\n";
        return rays;
    }

    size_t count;
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (ray_count > count) {
        std::cout << "the requested ray count: " << ray_count
                  << " is greater than the total ray count: " << count
                  << " You need to re-generate the rays." << std::endl;
    }
    assert(ray_count <= count);

    rays.reserve(ray_count);
    for (size_t i = 0; i < ray_count; ++i) {
        Ray ray;
        file.read(reinterpret_cast<char *>(&ray.o[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[2]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[2]), sizeof(float));
        rays.push_back(ray);
    }

    file.close();
    return rays;
}

std::vector<fcpw::Ray<3>> convert_rays(const std::vector<Ray> &rays) {
    std::vector<fcpw::Ray<3>> fcpw_rays;
    fcpw_rays.reserve(rays.size());

    for (const auto r : rays) {
        fcpw::Vector<3> o = {r.o[0], r.o[1], r.o[2]};
        fcpw::Vector<3> d = {r.d[0], r.d[1], r.d[2]};
        fcpw_rays.emplace_back(o, d, r.tmax);
    }
    return fcpw_rays;
}

std::vector<Triangle> load_obj(const std::string &obj_dir,
                               const std::string &object) {
    /*
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai"))
            break;
        current_path = current_path.parent_path();
    }

    std::string object_path =
    "/Users/ajroot/projects/pldi-bonsai/apps/queries/rt/" + object + ".obj";
    std::string material_path =
    "/Users/ajroot/projects/pldi-bonsai/apps/queries/rt/" + object;
    */
    std::string object_path = obj_dir + "/" + object + ".obj";
    std::string material_path = obj_dir + "/" + object;

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

bool global_tree_built = false;
_tree_layout0 global_tree;

_tree_layout0 copy_tree(fcpw::Aggregate<3> *aggregate) {
    auto bvh_ptr =
        dynamic_cast<fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle> *>(
            aggregate);
    if (!bvh_ptr) {
        std::cerr << "Error: Aggregate is not a BVH<3>!\n";
        std::abort();
    }

    _tree_layout0 tree;
    tree.nCount = static_cast<uint32_t>(bvh_ptr->flatTree.size());
    if (tree.nCount > 0) {
        tree.group0_index = static_cast<_tree_layout1 *>(
            malloc(sizeof(_tree_layout1) * tree.nCount));
        if (!tree.group0_index) {
            std::cerr << "Malloc (nodes) failed" << std::endl;
            abort();
        }
        // copy nodes
        // TODO: could this be a memcpy?
        for (uint32_t i = 0; i < tree.nCount; ++i) {
            tree.group0_index[i].low[0] = bvh_ptr->flatTree[i].box.pMin[0];
            tree.group0_index[i].low[1] = bvh_ptr->flatTree[i].box.pMin[1];
            tree.group0_index[i].low[2] = bvh_ptr->flatTree[i].box.pMin[2];
            tree.group0_index[i].high[0] = bvh_ptr->flatTree[i].box.pMax[0];
            tree.group0_index[i].high[1] = bvh_ptr->flatTree[i].box.pMax[1];
            tree.group0_index[i].high[2] = bvh_ptr->flatTree[i].box.pMax[2];
            tree.group0_index[i].nPrims = bvh_ptr->flatTree[i].nReferences;
            tree.group0_index[i].offset =
                (tree.group0_index[i].nPrims == 0)
                    ? bvh_ptr->flatTree[i].secondChildOffset
                    : bvh_ptr->flatTree[i].referenceOffset;
        }
    } else {
        std::cerr << "Copying empty tree from FCPW " << std::endl;
        abort();
    }

    tree.pCount = static_cast<uint32_t>(bvh_ptr->primitives.size());
    if (tree.pCount > 0) {
        tree.prims =
            static_cast<Triangle *>(malloc(sizeof(Triangle) * tree.pCount));
        if (!tree.prims) {
            std::cerr << "Malloc (triangles) failed" << std::endl;
            abort();
        }
        for (uint32_t i = 0; i < tree.pCount; ++i) {
            const auto soup = bvh_ptr->primitives[i]->soup;
            tree.prims[i].p0.x =
                soup->positions[bvh_ptr->primitives[i]->indices[0]][0];
            tree.prims[i].p0.y =
                soup->positions[bvh_ptr->primitives[i]->indices[0]][1];
            tree.prims[i].p0.z =
                soup->positions[bvh_ptr->primitives[i]->indices[0]][2];
            tree.prims[i].p1.x =
                soup->positions[bvh_ptr->primitives[i]->indices[1]][0];
            tree.prims[i].p1.y =
                soup->positions[bvh_ptr->primitives[i]->indices[1]][1];
            tree.prims[i].p1.z =
                soup->positions[bvh_ptr->primitives[i]->indices[1]][2];
            tree.prims[i].p2.x =
                soup->positions[bvh_ptr->primitives[i]->indices[2]][0];
            tree.prims[i].p2.y =
                soup->positions[bvh_ptr->primitives[i]->indices[2]][1];
            tree.prims[i].p2.z =
                soup->positions[bvh_ptr->primitives[i]->indices[2]][2];
        }
    } else {
        std::cerr << "No triangles?" << std::endl;
        abort();
    }
    return tree;
}

using cgalKernel = CGAL::Simple_cartesian<float>;
using cgalPoint = cgalKernel::Point_3;
using cgalTriangle = cgalKernel::Triangle_3;
using cgalRay = cgalKernel::Ray_3;

struct MollerTrumboreIntersection {
    using Ray = cgalRay;
    using Triangle = cgalTriangle;
    using Point = cgalPoint;
    using Vector = cgalKernel::Vector_3;

    std::optional<Point> operator()(const Ray &ray, const Triangle &tri) const {
        const float EPS = 1e-6f;

        Point orig = ray.source();
        Vector dir = ray.to_vector();
        Point v0 = tri.vertex(0);
        Point v1 = tri.vertex(1);
        Point v2 = tri.vertex(2);

        Vector edge1 = v1 - v0;
        Vector edge2 = v2 - v0;
        Vector pvec = CGAL::cross_product(dir, edge2);
        float det = edge1 * pvec;

        if (fabs(det) < EPS)
            return std::nullopt;

        float invDet = 1.0f / det;
        Vector tvec = orig - v0;
        float u = (tvec * pvec) * invDet;
        if (u < 0.0f || u > 1.0f)
            return std::nullopt;

        Vector qvec = CGAL::cross_product(tvec, edge1);
        float v = (dir * qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return std::nullopt;

        float t = (edge2 * qvec) * invDet;
        if (t < 0.0f)
            return std::nullopt;

        return std::optional<Point>(orig + dir * t);
    }
};

// template <typename K>
// struct Intersection_traits<K, CGAL::Ray_3, typename K::Line_3> {
//     typedef typename std::variant<typename K::Segment_3, typename K::Point_3>
//         variant_type;
//     typedef typename std::optional<variant_type> result_type;
// };

// Then use standard traits with your custom primitive
// using CustomPrimitive =
//     Custom_triangle_primitive<std::vector<cgalTriangle>::iterator>;

using Primitive =
    CGAL::AABB_triangle_primitive_3<cgalKernel,
                                    std::vector<cgalTriangle>::iterator>;
using Traits = CGAL::AABB_traits_3<cgalKernel, Primitive>;
using Tree = CGAL::AABB_tree<Traits>;

/*
_tree_layout0 copy_tree(Tree aggregate) {
    if (aggregate.size() == 0) {
        std::cerr << "Error: Aggregate is empty!\n";
        std::abort();
    }
    _tree_layout0 tree;
    tree.nCount = static_cast<uint32_t>(aggregate.size());
    if (tree.nCount > 0) {
        tree.group0_index =
            static_cast<_tree_layout1*>(malloc(sizeof(_tree_layout1) *
tree.nCount)); if (!tree.group0_index) { std::cerr << "Malloc (nodes) failed" <<
std::endl; abort();
        }
        // copy nodes
        // TODO: could this be a memcpy?
        for (uint32_t i = 0; i < tree.nCount; ++i) {
            tree.group0_index[i].low[0] = bvh_ptr->flatTree[i].box.pMin[0];
            tree.group0_index[i].low[1] = bvh_ptr->flatTree[i].box.pMin[1];
            tree.group0_index[i].low[2] = bvh_ptr->flatTree[i].box.pMin[2];
            tree.group0_index[i].high[0] = bvh_ptr->flatTree[i].box.pMax[0];
            tree.group0_index[i].high[1] = bvh_ptr->flatTree[i].box.pMax[1];
            tree.group0_index[i].high[2] = bvh_ptr->flatTree[i].box.pMax[2];
            tree.group0_index[i].nPrims = bvh_ptr->flatTree[i].nReferences;
            tree.group0_index[i].offset = (tree.group0_index[i].nPrims == 0) ?
bvh_ptr->flatTree[i].secondChildOffset : bvh_ptr->flatTree[i].referenceOffset;
        }
    } else {
        std::cerr << "Copying empty tree from FCPW " << std::endl;
        abort();
    }

    tree.pCount = static_cast<uint32_t>(bvh_ptr->primitives.size());
    if (tree.pCount > 0) {
        tree.prims = static_cast<Triangle*>(malloc(sizeof(Triangle) *
tree.pCount)); if (!tree.prims) { std::cerr << "Malloc (triangles) failed" <<
std::endl; abort();
        }
        for (uint32_t i = 0; i < tree.pCount; ++i) {
            const auto soup = bvh_ptr->primitives[i]->soup;
            tree.prims[i].p0.x =
soup->positions[bvh_ptr->primitives[i]->indices[0]][0]; tree.prims[i].p0.y =
soup->positions[bvh_ptr->primitives[i]->indices[0]][1]; tree.prims[i].p0.z =
soup->positions[bvh_ptr->primitives[i]->indices[0]][2]; tree.prims[i].p1.x =
soup->positions[bvh_ptr->primitives[i]->indices[1]][0]; tree.prims[i].p1.y =
soup->positions[bvh_ptr->primitives[i]->indices[1]][1]; tree.prims[i].p1.z =
soup->positions[bvh_ptr->primitives[i]->indices[1]][2]; tree.prims[i].p2.x =
soup->positions[bvh_ptr->primitives[i]->indices[2]][0]; tree.prims[i].p2.y =
soup->positions[bvh_ptr->primitives[i]->indices[2]][1]; tree.prims[i].p2.z =
soup->positions[bvh_ptr->primitives[i]->indices[2]][2];
        }
    } else {
        std::cerr << "No triangles?" << std::endl;
        abort();
    }
    return tree;
}


_tree_layout0 copy_tree(const Tree &aggregate) {
    if (aggregate.size() == 0) {
        std::cerr << "Error: Aggregate is empty!\n";
        std::abort();
    }
    _tree_layout0 tree;
    tree.nCount = static_cast<uint32_t>(aggregate.m_nodes.size());
    tree.pCount = static_cast<uint32_t>(aggregate.size());
    if (tree.nCount == 0 || tree.pCount == 0) {
        std::cerr << "No triangles or no tree nodes" << std::endl;
        abort();
    }

    tree.group0_index = static_cast<_tree_layout1*>(calloc(tree.nCount,
sizeof(_tree_layout1)));
    // TODO: set tree.group0_index fully to 0
    tree.prims = static_cast<Triangle*>(calloc(tree.pCount, sizeof(Triangle)));

    uint32_t node_count = 0;
    uint32_t tri_count = 0;

    auto build_triangle = [&](const cgalTriangle &tri) -> uint32_t {
        if (tri_count >= tree.pCount) {
            std::cerr << "Wrote too many triangles!\n";
            abort();
        }
        tree.prims[tri_count].p0.x = tri.vertex(0).x();
        tree.prims[tri_count].p0.y = tri.vertex(0).y();
        tree.prims[tri_count].p0.z = tri.vertex(0).z();
        tree.prims[tri_count].p1.x = tri.vertex(1).x();
        tree.prims[tri_count].p1.y = tri.vertex(1).y();
        tree.prims[tri_count].p1.z = tri.vertex(1).z();
        tree.prims[tri_count].p2.x = tri.vertex(2).x();
        tree.prims[tri_count].p2.y = tri.vertex(2).y();
        tree.prims[tri_count].p2.z = tri.vertex(2).z();
        return tri_count++;
    };

    auto build_tri_range =
        [&](const Tree::Node &node) -> std::tuple<uint32_t, uint32_t> {
        const uint32_t offset = tri_count;
        (void)build_triangle(node.left_data().datum());
        (void)build_triangle(node.right_data().datum());
        return std::make_tuple(offset, 2);
    };

    std::function<uint32_t(const Tree::Node &, uint32_t)> recurse =
        [&](const Tree::Node &node, uint32_t nb_prims) -> uint32_t {
        if (node_count >= tree.nCount) {
            std::cerr << "Wrote too many nodes!\n";
            abort();
        }
        const uint32_t index = node_count++;
        tree.group0_index[index].low.x = node.bbox().xmin();
        tree.group0_index[index].low.y = node.bbox().ymin();
        tree.group0_index[index].low.z = node.bbox().zmin();
        tree.group0_index[index].high.x = node.bbox().xmax();
        tree.group0_index[index].high.y = node.bbox().ymax();
        tree.group0_index[index].high.z = node.bbox().zmax();
        switch (nb_prims) {
        case 2: {
            auto [offset, count] = build_tri_range(node);
            tree.group0_index[index].nPrims = count;
            tree.group0_index[index].offset = offset;
            return index;
        }
        case 3: {
            // Left triangle, right node, make a node of 3 for us.
            auto offset = build_triangle(node.left_data().datum());
            auto [_, count] = build_tri_range(node.right_child());
            tree.group0_index[index].nPrims = count + 1;
            tree.group0_index[index].offset = offset;
            return index;
        }
        default: {
            if (nb_prims < 4) {
                std::cerr << "UNEXPECTED: " << nb_prims << " primitives\n";
                abort();
            }
            // Both nodes.
            (void)recurse(node.left_child(), nb_prims / 2);
            auto right = recurse(node.right_child(), nb_prims - nb_prims / 2);
            tree.group0_index[index].nPrims = 0;
            tree.group0_index[index].offset = right - index;
            return index;
        }
        }
    };

    recurse(*aggregate.root_node(), aggregate.size());

    if (tri_count < tree.pCount) {
        std::cerr << "Too few triangles: " << tri_count << " stored but " <<
tree.pCount << "allocated\n"; abort();
    }

    if (node_count < tree.nCount) {
        for (uint32_t i = 0; i < node_count; i++) {
            const bool is_leaf = tree.group0_index[i].nPrims != 0;
            if (!is_leaf && (tree.group0_index[i].offset == 0 || (i +
tree.group0_index[i].offset) >= node_count)) { std::cerr << "Index: " << i << "
has a bad offset of: " << tree.group0_index[i].offset << "\n"; std::cerr <<
node_count << " stored and " << tree.nCount << " allocated.\n"; abort();
            }
        }
        for (uint32_t i = node_count; i < tree.nCount; i++) {
            if (tree.group0_index[i].offset != 0) {
                std::cerr << "Index: " << i << " has a nonzero offset of: " <<
tree.group0_index[i].offset << "\n"; std::cerr << node_count << " stored and "
<< tree.nCount << " allocated.\n"; abort();
            }
        }
        // std::cerr << "Too few nodes: " << node_count << " stored but " <<
tree.nCount << "allocated\n";
        // abort();
    }

    return tree;
}
    */

// TODO(ajr): you're going to need to update this so it builds the PBRT and not
// a canonical tree.
//
// 1:1 match with FCPW's Bvh_SurfaceArea:
// https://github.com/rohan-sawhney/fcpw/blob/e36bc9b34af6088fb78ddbb6a93e26686779678a/include/fcpw/utilities/scene_data.h#L21
_tree_layout0 build_triangles(std::vector<Triangle> &triangles, int leaf_size,
                              int max_tree_depth, int nBuckets = 8) {
    constexpr auto MAX = std::numeric_limits<float>::max();

    struct BucketInfo {
        float3 box_min = float3{MAX, MAX, MAX};
        float3 box_max = float3{-MAX, -MAX, -MAX};
        int count = 0;
    };

    std::vector<_tree_layout1> nodes;

#ifdef AJR_PROFILE
    size_t leaf_count = 0;
    size_t interior_count = 0;
#endif

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        uint32_t count = high - low;
        auto [aabb_min, aabb_max] = compute_aabb(low, high, triangles);

        uint32_t index = nodes.size();

        if (count <= leaf_size || depth >= max_tree_depth) {
            auto *data = (Triangle *)(malloc(sizeof(Triangle) * count));
            for (uint32_t i = 0; i < count; ++i) {
                data[i] = triangles[low + i];
            }
            nodes.push_back(_tree_layout1{
                .low = aabb_min,
                .high = aabb_max,
                .nPrims = static_cast<uint8_t>(count),
                .offset = low,
            });
#ifdef AJR_PROFILE
            leaf_count++;
#endif
            return index;
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
            float aabb_extent = (axis == 0)   ? aabb_max.x - aabb_min.x
                                : (axis == 1) ? aabb_max.y - aabb_min.y
                                              : aabb_max.z - aabb_min.z;

            if (aabb_extent < 1e-6f)
                continue; // ...skip degenerate axis

            float aabb_min_axis = (axis == 0)   ? aabb_min.x
                                  : (axis == 1) ? aabb_min.y
                                                : aabb_min.z;

            float bucket_width = aabb_extent / nBuckets;
            std::vector<BucketInfo> buckets(nBuckets);

            for (uint32_t i = low; i < high; ++i) {
                float3 c = triangle_centroid(triangles[i]);
                float c_axis = (axis == 0) ? c.x : (axis == 1) ? c.y : c.z;

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

            nodes.push_back(_tree_layout1{
                .low = aabb_min,
                .high = aabb_max,
                .nPrims = 0,
                .offset = 0, // for now
            });

            auto left = partition(low, mid, depth + 1);
            auto right = partition(mid, high, depth + 1);

            nodes[index].offset = right - index;
#ifdef AJR_PROFILE
            interior_count++;
#endif
            return index;
        }

        float aabb_extent = (best_axis == 0)   ? aabb_max.x - aabb_min.x
                            : (best_axis == 1) ? aabb_max.y - aabb_min.y
                                               : aabb_max.z - aabb_min.z;
        float aabb_min_axis = (best_axis == 0)   ? aabb_min.x
                              : (best_axis == 1) ? aabb_min.y
                                                 : aabb_min.z;
        float bucket_width = aabb_extent / nBuckets;

        auto mid_it = std::partition(
            triangles.begin() + low, triangles.begin() + high,
            [&](const Triangle &tri) {
                float3 c = triangle_centroid(tri);
                float c_axis = (best_axis == 0)   ? c.x
                               : (best_axis == 1) ? c.y
                                                  : c.z;
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
                                 float ca_axis = (best_axis == 0)   ? ca.x
                                                 : (best_axis == 1) ? ca.y
                                                                    : ca.z;
                                 float cb_axis = (best_axis == 0)   ? cb.x
                                                 : (best_axis == 1) ? cb.y
                                                                    : cb.z;
                                 return ca_axis < cb_axis;
                             });
        }

        nodes.push_back(_tree_layout1{
            .low = aabb_min,
            .high = aabb_max,
            .nPrims = 0,
            .offset = 0, // for now
        });

        auto left = partition(low, mid, depth + 1);
        auto right = partition(mid, high, depth + 1);

        nodes[index].offset = right - index;
#ifdef AJR_PROFILE
        interior_count++;
#endif
        return index;
    };

    partition(0, triangles.size(), /*depth=*/0);

#ifdef AJR_PROFILE
    std::cout << "Bonsai tree build: node_count     = " << nodes.size() << "\n";
    std::cout << "                   leaf_count     = " << leaf_count << "\n";
    std::cout << "                   interior_count = " << interior_count
              << "\n";
#endif

    _tree_layout0 tree;
    tree.nCount = static_cast<uint32_t>(nodes.size());
    if (tree.nCount > 0) {
        tree.group0_index = static_cast<_tree_layout1 *>(
            malloc(sizeof(_tree_layout1) * tree.nCount));
        if (!tree.group0_index) {
            std::cerr << "Malloc (nodes) failed" << std::endl;
            abort();
        }
        // copy nodes
        // TODO: could this be a memcpy?
        for (uint32_t i = 0; i < tree.nCount; ++i) {
            tree.group0_index[i] = nodes[i];
        }
    } else {
        std::cerr << "Built empty tree from: " << triangles.size() << std::endl;
        abort();
    }

    tree.pCount = static_cast<uint32_t>(triangles.size());
    if (tree.pCount > 0) {
        tree.prims =
            static_cast<Triangle *>(malloc(sizeof(Triangle) * tree.pCount));
        if (!tree.prims) {
            std::cerr << "Malloc (triangles) failed" << std::endl;
            abort();
        }
        for (uint32_t i = 0; i < tree.pCount; ++i) {
            tree.prims[i] = triangles[i];
        }
    } else {
        std::cerr << "No triangles? " << triangles.size() << std::endl;
        abort();
    }
    return tree;
}

// single-thread closest hit ray tracing.
// clang++ -std=c++20 -O3 -march=native -I. -Iruntime/CPP -o main.out main.cpp
// rt.cpp
void run_bonsai(const std::string &obj_dir, const std::string &object,
                const std::string &ray_type, const std::string &ray_dir,
                int low, int high) {
    using clock = std::chrono::high_resolution_clock;

    std::vector<Triangle> triangles = load_obj(obj_dir, object);
    assert(!triangles.empty());

    auto t0 = clock::now();
    auto tree =
        global_tree_built ? global_tree : build_triangles(triangles, 4, 64);
    auto t1 = clock::now();
    auto trace_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    // std::cout << "Bonsai build time : " << trace_time << " ms\n";

    bool timedout = false;

    bool is_first_run = true;
    for (int64_t c = low; c <= high; c++) {
        const int64_t ray_count = 1ll << c;
        const std::string ray_file = ray_dir + "/" + object + "_" +
                                     std::to_string(ray_count) + "_" +
                                     ray_type + ".rays";
        const std::vector<Ray> rays = load_rays_binary(ray_file, ray_count);
        assert(!rays.empty());

        if (is_first_run) {
            for (int i = 0; i < std::min<size_t>(rays.size(), 512u); ++i)
                (void)trace(rays[i], tree); // warmup
            is_first_run = false;
        }
        size_t hit_count = 0;
#ifdef AJR_PROFILE
        ajr_profiler_reset();

        auto trace_begin = clock::now(), trace_end = clock::now();

        // std::vector<Triangle> hits;
        // hits.reserve(rays.size());
        trace_begin = clock::now();
        for (int i = 0; i < rays.size(); ++i) {
            fcpw::Interaction<3> interact;
            if (std::optional<Triangle> t = trace(rays[i], tree)) {
                // hits.push_back(*t);
                hit_count++;
            }
        }
        trace_end = clock::now();
        // hit_count = hits.size();

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "  aabb hits tested = " << distmin_aabb_counter << "\n";
        std::cout << "  tri  hits tested = " << distmin_triangle_counter
                  << "\n";
        ajr_profiler_reset();
#else
        auto trace_time =
            benchmark_function(
                [&]() {
                    hit_count = 0;
                    for (int i = 0; i < rays.size(); ++i) {
                        if (std::optional<Triangle> t = trace(rays[i], tree)) {
                            // hits.push_back(*t);
                            hit_count++;
                        }
                    }
                },
                timedout, k, m) /
            (double)1e6;
#endif
        // std::cout << "hits       : " << hit_count << "\n";
        // std::cout << "trace time : " << trace_time << " ms\n";
        std::cout << "    (\"bonsai\", " << ray_count << ", " << trace_time
                  << ")," << std::endl;
        if (trace_time >= (timeout_sec * 1e3)) {
            // timeout
            return;
        }
#ifdef AJR_PROFILE
        ajr_profiler_reset();
#endif
    }
}

void run_fcpw(const std::string &obj_dir, const std::string &object,
              const std::string &ray_type, const std::string &ray_dir, int low,
              int high) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> triangles = load_obj(obj_dir, object);
    assert(!triangles.empty());

    std::vector<fcpw::Vector3> vertices1;
    std::vector<fcpw::Vector3i> indices1;

    vertices1.reserve(triangles.size() * 3);
    indices1.reserve(triangles.size());

    for (const Triangle &tri : triangles) {
        int idx0 = vertices1.size();
        vertices1.emplace_back(tri.p0[0], tri.p0[1], tri.p0[2]);
        int idx1 = vertices1.size();
        vertices1.emplace_back(tri.p1[0], tri.p1[1], tri.p1[2]);
        int idx2 = vertices1.size();
        vertices1.emplace_back(tri.p2[0], tri.p2[1], tri.p2[2]);
        indices1.emplace_back(idx0, idx1, idx2);
    }

    // ---- FCPW Setup ----
    auto t0 = clock::now();

    // Create FCPW scene for mesh
    fcpw::Scene<3> fcpw_scene;

    // Set object count
    fcpw_scene.setObjectCount(1);

    // Set vertices and triangles for object 0
    fcpw_scene.setObjectVertices(vertices1, 0);
    fcpw_scene.setObjectTriangles(indices1, 0);

    // Build the BVH
#ifdef AJR_PROFILE
    bool printStats = true;
#else
    bool printStats = false;
#endif
    bool reduceMemoryFootprint = true;
    bool vectorize = false;
    fcpw_scene.build(fcpw::AggregateType::Bvh_SurfaceArea, vectorize,
                     printStats, reduceMemoryFootprint);
    auto t1 = clock::now();
    auto build_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    // std::cout << "FCPW build time : " << trace_time << " ms\n";

    if (global_tree_built) {
        free(global_tree.group0_index);
        free(global_tree.prims);
    }
    global_tree = copy_tree(fcpw_scene.getSceneData()->aggregate.get());
    global_tree_built = true;

    bool is_first_run = true;

    bool timedout = false;

    for (int64_t c = low; c <= high; c++) {
        const int64_t ray_count = 1ll << c;
        const std::string ray_file = ray_dir + "/" + object + "_" +
                                     std::to_string(ray_count) + "_" +
                                     ray_type + ".rays";
        std::vector<Ray> brays = load_rays_binary(ray_file, ray_count);
        auto rays = convert_rays(brays);
        brays.clear();
        assert(!rays.empty());

        if (is_first_run) {
            for (int i = 0; i < std::min<size_t>(rays.size(), 512u); ++i) {
                fcpw::Interaction<3> interact;
                (void)fcpw_scene.intersect(rays[i], interact); // warmup
            }
            is_first_run = false;
        }
        size_t hit_count = 0;
#ifdef AJR_PROFILE
        ajr_profiler_reset();

        auto trace_begin = clock::now(), trace_end = clock::now();

        // std::vector<Triangle> hits;
        // hits.reserve(rays.size());
        trace_begin = clock::now();
        for (int i = 0; i < rays.size(); ++i) {
            fcpw::Interaction<3> interact;
            if (fcpw_scene.intersect(rays[i], interact)) {
                // hits.push_back(*t);
                hit_count++;
            }
        }
        trace_end = clock::now();
        // hit_count = hits.size();

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "  aabb hits tested = " << fcpw_aabb_counter << "\n";
        std::cout << "  tri  hits tested = " << fcpw_triangle_counter << "\n";
        ajr_profiler_reset();
#else
        auto trace_time =
            benchmark_function(
                [&]() {
                    hit_count = 0;
                    for (int i = 0; i < rays.size(); ++i) {
                        fcpw::Interaction<3> interact;
                        if (fcpw_scene.intersect(rays[i], interact)) {
                            // hits.push_back(*t);
                            hit_count++;
                        }
                    }
                },
                timedout, k, m) /
            (double)1e6;
#endif
        std::cout << "    (\"fcpw\", " << ray_count << ", " << trace_time
                  << ")," << std::endl;
        if (trace_time >= (timeout_sec * 1e3)) {
            // timeout
            return;
        }
        // std::cout << "hits       : " << hit_count << "\n";
        // std::cout << "trace time : " << trace_time << " ms\n";
    }
}

void run_cgal(const std::string &obj_dir, const std::string &object,
              const std::string &ray_type, const std::string &ray_dir, int low,
              int high) {
    // using cgalKernel = CGAL::Simple_cartesian<float>;
    // using cgalPoint = cgalKernel::Point_3;
    // using cgalTriangle = cgalKernel::Triangle_3;
    // using cgalRay = cgalKernel::Ray_3;

    // using Primitive = CGAL::AABB_triangle_primitive_3<cgalKernel,
    // std::vector<cgalTriangle>::iterator>; using Traits =
    // CGAL::AABB_traits_3<cgalKernel, Primitive>; using Tree =
    // CGAL::AABB_tree<Traits>;
    using clock = std::chrono::high_resolution_clock;
    std::vector<Triangle> tris = load_obj(obj_dir, object);
    assert(!tris.empty());

    std::vector<cgalTriangle> triangles;
    triangles.reserve(tris.size());
    for (const auto &t : tris) {
        triangles.emplace_back(cgalPoint(t.p0.x, t.p0.y, t.p0.z),
                               cgalPoint(t.p1.x, t.p1.y, t.p1.z),
                               cgalPoint(t.p2.x, t.p2.y, t.p2.z));
    }
    tris.clear();

    auto t0 = std::chrono::high_resolution_clock::now();
    Tree tree(triangles.begin(), triangles.end());
    tree.build();
    auto t1 = std::chrono::high_resolution_clock::now();

    // std::cout << "CGAL AABB tree build time (ms): "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
    //           t0).count() << "\n";

    // if (global_tree_built) {
    //     free(global_tree.group0_index);
    //     free(global_tree.prims);
    // }
    // global_tree = copy_tree(tree);
    // global_tree_built = true;

    bool timedout = false;

    bool is_first_run = true;
    for (int64_t c = low; c <= high; c++) {
        const int64_t ray_count = 1ll << c;
        const std::string ray_file = ray_dir + "/" + object + "_" +
                                     std::to_string(ray_count) + "_" +
                                     ray_type + ".rays";
        std::vector<Ray> bRays = load_rays_binary(ray_file, ray_count);
        assert(!bRays.empty());
        std::vector<cgalRay> rays;
        rays.reserve(bRays.size());

        for (const auto &r : bRays) {
            rays.emplace_back(cgalPoint(r.o.x, r.o.y, r.o.z),
                              cgalKernel::Direction_3(r.d.x, r.d.y, r.d.z));
        }
        bRays.clear();

        if (is_first_run) {
            for (int i = 0; i < std::min<size_t>(rays.size(), 512u); ++i)
                (void)tree.first_intersection(rays[i]); // warmup
            is_first_run = false;
        }
        size_t hit_count = 0;
#ifdef AJR_PROFILE
        ajr_profiler_reset();

        auto trace_begin = clock::now(), trace_end = clock::now();

        // std::vector<Triangle> hits;
        // hits.reserve(rays.size());
        trace_begin = clock::now();
        for (int i = 0; i < rays.size(); ++i) {
            fcpw::Interaction<3> interact;
            if (auto intersection = tree.first_intersection(rays[i])) {
                // hits.push_back(*t);
                hit_count++;
            }
        }
        trace_end = clock::now();
        // hit_count = hits.size();

        auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_begin)
                              .count();
        std::cout << "  aabb hits tested = " << cgal_aabb_counter << "\n";
        std::cout << "  tri  hits tested = " << cgal_triangle_counter << "\n";
        ajr_profiler_reset();
#else
        auto trace_time =
            benchmark_function(
                [&]() {
                    hit_count = 0;
                    for (int i = 0; i < rays.size(); ++i) {
                        if (auto intersection =
                                tree.first_intersection(rays[i])) {
                            // hits.push_back(*t);
                            hit_count++;
                        }
                    }
                },
                timedout, k, m) /
            (double)1e6;
#endif
        std::cout << "    (\"cgal\", " << ray_count << ", " << trace_time
                  << ")," << std::endl;
        if (trace_time >= (timeout_sec * 1e3)) {
            // timeout
            return;
        }
        // std::cout << "hits       : " << hit_count << "\n";
        // std::cout << "trace time : " << trace_time << " ms\n";
#ifdef AJR_PROFILE
        ajr_profiler_reset();
#endif
    }
}

void verify_bonsai_vs_fcpw(const std::vector<Ray> &rays,
                           const fcpw::Scene<3> &fcpw_scene,
                           const _tree_layout0 &bonsai_tree) {
    for (size_t i = 0; i < rays.size(); ++i) {
        const Ray &r = rays[i];

        // --- Run FCPW ---
        fcpw::Ray<3> fcpw_ray({r.o[0], r.o[1], r.o[2]},
                              {r.d[0], r.d[1], r.d[2]}, r.tmax);

        float fcpw_t = std::numeric_limits<float>::infinity();

        fcpw::Interaction<3> interact;
        bool fcpw_hit = false;
        if (fcpw_scene.intersect(fcpw_ray, interact)) {
            fcpw_t = interact.d;
            fcpw_hit = true;
        }

        // --- Run Bonsai ---
        float bonsai_t = std::numeric_limits<float>::infinity();
        bool bonsai_hit = false;
        if (std::optional<Triangle> t = trace(rays[i], bonsai_tree)) {
            bonsai_t = distmin_Ray_Triangle(rays[i], *t);
            bonsai_hit = true;
        }

        // --- Compare results ---
        bool both_miss = !fcpw_hit && !bonsai_hit;
        bool both_hit = fcpw_hit && bonsai_hit;
        bool mismatch = false;

        if (both_miss)
            continue;
        if (both_hit) {
            float diff = std::fabs(fcpw_t - bonsai_t);
            if (diff > 1e-4f)
                mismatch = true;
        } else {
            mismatch = true;
        }

        if (mismatch) {
            std::cerr << std::fixed << std::setprecision(6);
            std::cerr << "Ray mismatch at index " << i << "\n";
            std::cerr << "Origin: (" << r.o[0] << ", " << r.o[1] << ", "
                      << r.o[2] << ")\n";
            std::cerr << "Direction: (" << r.d[0] << ", " << r.d[1] << ", "
                      << r.d[2] << ")\n";
            std::cerr << "Inverse (FCPW): (" << fcpw_ray.invD[0] << ", "
                      << fcpw_ray.invD[1] << ", " << fcpw_ray.invD[2] << ")\n";
            std::cerr << "Inverse (Bons): (" << (float3{1.0f} / r.d)[0] << ", "
                      << (float3{1.0f} / r.d)[1] << ", "
                      << (float3{1.0f} / r.d)[2] << ")\n";
            std::cerr << "FCPW: " << (fcpw_hit ? "hit" : "miss")
                      << " t = " << (fcpw_hit ? std::to_string(fcpw_t) : "n/a")
                      << "\n";
            std::cerr << "Bonsai: " << (bonsai_hit ? "hit" : "miss") << " t = "
                      << (bonsai_hit ? std::to_string(bonsai_t) : "n/a")
                      << "\n";
            throw std::runtime_error("FCPW vs Bonsai mismatch on ray " +
                                     std::to_string(i));
        }
    }

    std::cout << "All " << rays.size()
              << " rays matched between FCPW and Bonsai.\n";
}

void verify(const std::string &obj_dir, const std::string &object,
            const std::string &ray_type, const std::string &ray_dir,
            const size_t ray_count) {
    using clock = std::chrono::high_resolution_clock;

    const std::string ray_file = ray_dir + "/" + object + "_" +
                                 std::to_string(ray_count) + "_" + ray_type +
                                 ".rays";
    const std::vector<Ray> rays = load_rays_binary(ray_file, ray_count);

    std::vector<Triangle> triangles = load_obj(obj_dir, object);
    assert(!triangles.empty());

    std::vector<fcpw::Vector3> vertices1;
    std::vector<fcpw::Vector3i> indices1;

    vertices1.reserve(triangles.size() * 3);
    indices1.reserve(triangles.size());

    for (const Triangle &tri : triangles) {
        int idx0 = vertices1.size();
        vertices1.emplace_back(tri.p0[0], tri.p0[1], tri.p0[2]);
        int idx1 = vertices1.size();
        vertices1.emplace_back(tri.p1[0], tri.p1[1], tri.p1[2]);
        int idx2 = vertices1.size();
        vertices1.emplace_back(tri.p2[0], tri.p2[1], tri.p2[2]);
        indices1.emplace_back(idx0, idx1, idx2);
    }

    // ---- FCPW Setup ----
    auto t0 = clock::now();

    // Create FCPW scene for mesh
    fcpw::Scene<3> fcpw_scene;

    // Set object count
    fcpw_scene.setObjectCount(1);

    // Set vertices and triangles for object 0
    fcpw_scene.setObjectVertices(vertices1, 0);
    fcpw_scene.setObjectTriangles(indices1, 0);

    // Build the BVH
    bool printStats = false;
    bool reduceMemoryFootprint = true;
    bool vectorize = false;
    fcpw_scene.build(fcpw::AggregateType::Bvh_SurfaceArea, vectorize,
                     printStats, reduceMemoryFootprint);
    auto t1 = clock::now();
    auto trace_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "FCPW build time : " << trace_time << " ms\n";

    auto tree = copy_tree(fcpw_scene.getSceneData()->aggregate.get());

    verify_bonsai_vs_fcpw(rays, fcpw_scene, tree);
}

// } // namespace

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./executable <obj dir> <object name> <ray_dir> "
                     "<ray_type> <log2(min_ray_count)> <log2(max_ray_count)>"
                  << std::endl;
        return 1;
    }
    std::string object_dir = argv[1];
    std::string object = argv[2];
    std::string ray_dir = argv[3];
    std::string ray_type = argv[4];
    int low = atoi(argv[5]);
    assert(low > 0);
    int high = atoi(argv[6]);
    assert(low < high);

    run_fcpw(object_dir, object, ray_type, ray_dir, low, high);

    run_bonsai(object_dir, object, ray_type, ray_dir, low, high);

    try {
        run_cgal(object_dir, object, ray_type, ray_dir, low, high);
    } catch (const CGAL::Precondition_exception &e) {
        std::cerr << "Caught CGAL precondition violation:\n"
                  << "  what(): " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Caught unexpected exception:\n"
                  << "  " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Caught unknown non-std exception from CGAL." << std::endl;
    }

    return 0;
}
