#include "solve_bonsai.cuh"
#include <fcpw/aggregates/bvh.h>

using fcpwTriBVH = fcpw::Bvh<3, fcpw::BvhNode<3>, fcpw::Triangle>;

_tree_layout0 convert_tree(const fcpwTriBVH *bvh) {
    constexpr uint64_t MAX_TREE_DEPTH = 64;

    _tree_layout0 tree;
    tree.pCount = bvh->primitives.size();
    if (tree.pCount > std::numeric_limits<uint16_t>::max()) {
        std::cerr << "(bonsai) Use larger index type for primitive offsets!\n";
        exit(-1);
    }

    auto build_triangle = [&](const uint64_t i) {
        const fcpw::Triangle *triangle = bvh->primitives[i];
        const fcpw::Vector3 &p0 =
            triangle->soup->positions[triangle->indices[0]];
        const fcpw::Vector3 &p1 =
            triangle->soup->positions[triangle->indices[1]];
        const fcpw::Vector3 &p2 =
            triangle->soup->positions[triangle->indices[2]];
        Triangle tri;
        tri.p0 = make_float3(p0(0), p0(1), p0(2));
        tri.p1 = make_float3(p1(0), p1(1), p1(2));
        tri.p2 = make_float3(p2(0), p2(1), p2(2));
        return tri;
    };

    // Build triangle list
    tree.prims = (Triangle *)malloc(sizeof(Triangle) * tree.pCount);
    for (size_t i = 0; i < tree.pCount; ++i) {
        tree.prims[i] = build_triangle(i);
    }

    tree.count = bvh->flatTree.size();
    tree.group0_index =
        (_tree_layout1 *)malloc(sizeof(_tree_layout1) * tree.count);

    for (size_t i = 0; i < tree.count; ++i) {
        const fcpw::BvhNode<3> &node = bvh->flatTree[i];
        const fcpw::Vector<3> &pMin = node.box.pMin;
        const fcpw::Vector<3> &pMax = node.box.pMax;
        uint32_t nPrimitives = node.nReferences;
        uint32_t offset =
            nPrimitives > 0 ? node.referenceOffset : node.secondChildOffset;

        tree.group0_index[i].low = make_float3(pMin(0), pMin(1), pMin(2));
        // tree.group0_index[i].pad0 = 0;
        tree.group0_index[i].high = make_float3(pMax(0), pMax(1), pMax(2));
        // TODO: set axis!
        // if (nPrimitives == 0) {
        //     // Look into children to select split axis.
        //     const fcpw::BvhNode<3>& left = bvh->flatTree[i + 1];
        //     const fcpw::BvhNode<3>& right = bvh->flatTree[i + offset];
        //     auto abs_diff = [&](const size_t dim) {
        //         return abs(max(left.box.pMax(dim) - right.box.pMin(dim),
        //         left.box.pMin(dim) - right.box.pMax(dim)));
        //     };
        //     const float x_diff = abs_diff(0);
        //     const float y_diff = abs_diff(1);
        //     const float z_diff = abs_diff(2);

        // } else {
        //     tree.group0_index[i].axis = 0;
        // }
        tree.group0_index[i].axis = 0;

        if (nPrimitives > std::numeric_limits<uint16_t>::max()) {
            std::cerr
                << "(bonsai) Use larger index type for primitive count!\n";
            exit(-1);
        }
        tree.group0_index[i].nPrims = nPrimitives;

        if (offset > std::numeric_limits<uint16_t>::max()) {
            std::cerr << "(bonsai) Use larger index type for offset type!\n";
            exit(-1);
        }
        *reinterpret_cast<uint16_t *>(&tree.group0_index[i].split0on_nPrims) =
            offset;
    }
    return tree;
}