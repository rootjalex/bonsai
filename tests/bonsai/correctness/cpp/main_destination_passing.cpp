// Used in destination_passing.bonsai.
//
// The scene and tree of main_bvh4_tiled_elements_where.cpp -- walls facing
// the x axis in lanes along z, a tree split in x, decoy lanes with ids of
// -1 -- queried two ways: one query at a time through `trace` and
// `closest`, whose record comes back inside the value they return, and as a
// batch through `trace_all` and `closest_all`, which write each query's
// record into `hits[i]` -- the slot the pass replaces with `hits[i]` itself
// (SSA/DestinationPassing.h). Every batch record must equal its single
// query's, field by field, and both must name the wall the query has to
// answer.
#include "destination_passing.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr int kWalls = 8;
constexpr int kLanes = 4;
constexpr float kWallSpacing = 10.0f;
constexpr float kLaneSpacing = 5.0f;
constexpr uint32_t kNoId = 0xffffffffu;

Triangle wall_at(float x, int lane) {
    const float z = kLaneSpacing * float(lane);
    Triangle t;
    t.p0 = float3{x, -1.0f, z - 1.0f};
    t.p1 = float3{x, 1.0f, z - 1.0f};
    t.p2 = float3{x, 0.0f, z + 1.0f};
    return t;
}

Triangle wall(int k, int lane) { return wall_at(kWallSpacing * float(k), lane); }

struct Box {
    float3 lo, hi;
};

Box bounds_of(const Triangle &t) {
    Box b;
    for (int c = 0; c < 3; c++) {
        b.lo[c] = std::min({t.p0[c], t.p1[c], t.p2[c]});
        b.hi[c] = std::max({t.p0[c], t.p1[c], t.p2[c]});
    }
    return b;
}

Box merge(const Box &a, const Box &b) {
    Box out;
    for (int c = 0; c < 3; c++) {
        out.lo[c] = std::min(a.lo[c], b.lo[c]);
        out.hi[c] = std::max(a.hi[c], b.hi[c]);
    }
    return out;
}

using NodeRow = _tree_layout2;
using Tile = _tree_layout1;

static_assert(sizeof(NodeRow) == 16 + 6 * 16, "a node row is Embree's AABBNode<4>");
static_assert(sizeof(Tile) == 3 * 3 * 16 + 16,
              "a tile is three vertices by component and an id, four wide");

struct Builder {
    std::vector<Tile> tiles;
    std::vector<NodeRow> rows;

    static void put(Tile &tile, unsigned lane, const Triangle &t, uint32_t id) {
        tile.p0.x[lane] = t.p0[0];
        tile.p0.y[lane] = t.p0[1];
        tile.p0.z[lane] = t.p0[2];
        tile.p1.x[lane] = t.p1[0];
        tile.p1.y[lane] = t.p1[1];
        tile.p1.z[lane] = t.p1[2];
        tile.p2.x[lane] = t.p2[0];
        tile.p2.y[lane] = t.p2[1];
        tile.p2.z[lane] = t.p2[2];
        tile.id[lane] = id;
    }

    uint32_t leaf(const std::vector<Triangle> &tris, Box &box, uint32_t first_id) {
        const uint32_t first = uint32_t(tiles.size());
        const uint32_t count = uint32_t((tris.size() + 3) / 4);
        tiles.resize(tiles.size() + count);
        box = bounds_of(tris.front());
        const float x_first = tris.front().p0[0];
        const float z_first = tris.front().p0[2] + 1.0f;
        const Triangle decoy = wall_at(x_first - 0.5f, int(z_first / kLaneSpacing));
        for (size_t s = 0; s < 4 * size_t(count); s++) {
            Tile &tile = tiles[first + s / 4];
            if (s < tris.size()) {
                put(tile, unsigned(s % 4), tris[s], first_id + uint32_t(s));
                box = merge(box, bounds_of(tris[s]));
            } else {
                put(tile, unsigned(s % 4), decoy, kNoId);
                box = merge(box, bounds_of(decoy));
            }
        }
        return (first << 4) | (8u + count);
    }

    static void fill(NodeRow &n, const uint32_t children[4], const Box boxes[4]) {
        for (int i = 0; i < 4; i++) {
            n.children[i] = children[i];
            n.lower_x[i] = boxes[i].lo[0];
            n.lower_y[i] = boxes[i].lo[1];
            n.lower_z[i] = boxes[i].lo[2];
            n.upper_x[i] = boxes[i].hi[0];
            n.upper_y[i] = boxes[i].hi[1];
            n.upper_z[i] = boxes[i].hi[2];
        }
    }

    uint32_t interior(const uint32_t children[4], const Box boxes[4], Box &box) {
        const uint32_t row = uint32_t(rows.size());
        rows.emplace_back();
        fill(rows.back(), children, boxes);
        box = boxes[0];
        for (int i = 1; i < 4; i++) {
            box = merge(box, boxes[i]);
        }
        return row << 4;
    }
};

_tree_layout0 build_tree(Builder &b) {
    b.rows.emplace_back();
    uint32_t root_children[4];
    Box root_boxes[4];
    uint32_t next_id = 0;
    for (int quarter = 0; quarter < 4; quarter++) {
        uint32_t leaves[4];
        Box leaf_boxes[4];
        int l = 0;
        for (int k = 2 * quarter; k < 2 * quarter + 2; k++) {
            for (int lanes = 0; lanes < 2; lanes++) {
                std::vector<Triangle> tris = {wall(k, 2 * lanes), wall(k, 2 * lanes + 1)};
                leaves[l] = b.leaf(tris, leaf_boxes[l], next_id);
                next_id += 2;
                l++;
            }
        }
        root_children[quarter] = b.interior(leaves, leaf_boxes, root_boxes[quarter]);
    }
    Builder::fill(b.rows[0], root_children, root_boxes);

    static bonsai_buffer tiles_buffer, rows_buffer;
    tiles_buffer = bonsai_buffer_wrap(b.tiles.data(), b.tiles.size() * sizeof(Tile));
    rows_buffer = bonsai_buffer_wrap(b.rows.data(), b.rows.size() * sizeof(NodeRow));
    _tree_layout0 tree;
    tree.pCount = uint32_t(4 * b.tiles.size());
    tree.group0_prims = &tiles_buffer;
    tree.nCount = uint32_t(b.rows.size());
    tree.group1_row = &rows_buffer;
    return tree;
}

bool same_record(const Record &a, const Record &b) {
    return a.d == b.d && a.p0[0] == b.p0[0] && a.p0[1] == b.p0[1] && a.p0[2] == b.p0[2];
}

// A ray with the wall it must hit, or none.
struct RayCase {
    const char *what;
    Ray ray;
    bool hits;
    float wall_x;
};

struct PointCase {
    const char *what;
    Point point;
    float wall_x;
};

int failures = 0;

void check(const char *what, bool right, const Record &batch) {
    std::cout << what << ": d=" << batch.d << " x=" << batch.p0[0]
              << (right ? " -- ok" : " -- WRONG") << '\n';
    if (!right) {
        failures++;
    }
}

} // namespace

int main() {
    Builder b;
    _tree_layout0 tree = build_tree(b);
    const float inf = std::numeric_limits<float>::infinity();

    const float3 forward = float3{1.0f, 0.0f, 0.0f};
    const float3 backward = float3{-1.0f, 0.0f, 0.0f};
    std::vector<RayCase> ray_cases;
    std::vector<PointCase> point_cases;
    for (int lane = 0; lane < kLanes; lane++) {
        const float z = kLaneSpacing * float(lane);
        ray_cases.push_back({"+x", Ray{float3{-1.0f, 0.0f, z}, forward}, true, 0.0f});
        ray_cases.push_back({"-x", Ray{float3{100.0f, 0.0f, z}, backward}, true,
                             kWallSpacing * float(kWalls - 1)});
        ray_cases.push_back({"+x from 35", Ray{float3{35.0f, 0.0f, z}, forward}, true, 40.0f});
        ray_cases.push_back({"+x short", Ray{float3{-1.0f, 0.0f, z}, forward, 0.5f}, false, 0.0f});
        for (int k = 0; k < kWalls; k += 3) {
            point_cases.push_back({"closest before wall",
                                   Point{float3{kWallSpacing * float(k) - 1.0f, 0.0f, z}},
                                   kWallSpacing * float(k)});
        }
        point_cases.push_back({"closest past wall 2",
                               Point{float3{kWallSpacing * 2.0f + 2.0f, 0.0f, z}}, 20.0f});
    }
    ray_cases.push_back({"+x between lanes", Ray{float3{-1.0f, 0.0f, 0.5f * kLaneSpacing}, forward},
                         false, 0.0f});
    ray_cases.push_back({"diagonal", Ray{float3{-1.0f, 0.0f, -1.0f}, float3{1.0f, 0.0f, 0.5f}},
                         true, 0.0f});
    ray_cases.push_back({"+z beside", Ray{float3{30.0f, 2.0f, -10.0f}, float3{0.0f, 0.0f, 1.0f}},
                         false, 0.0f});
    point_cases.push_back({"closest from far", Point{float3{-100.0f, 0.0f, 0.0f}}, 0.0f});

    // The batches: the records written into `hits`, pre-filled with a value
    // no query answers, so that a slot the batch left alone shows.
    std::vector<Ray> rays;
    for (const RayCase &c : ray_cases) {
        rays.push_back(c.ray);
    }
    std::vector<Point> points;
    for (const PointCase &c : point_cases) {
        points.push_back(c.point);
    }
    const Record untouched{-1.0f, float3{-1.0f, -1.0f, -1.0f}};
    std::vector<Record> ray_hits(rays.size(), untouched);
    std::vector<Record> point_hits(points.size(), untouched);
    trace_all(uint32_t(rays.size()), rays.data(), ray_hits.data(), tree);
    closest_all(uint32_t(points.size()), points.data(), point_hits.data(), tree);

    for (size_t i = 0; i < ray_cases.size(); i++) {
        const RayCase &c = ray_cases[i];
        _option0 single;
        trace(single, c.ray, tree);
        const Record &batch = ray_hits[i];
        bool right;
        if (c.hits) {
            right = single.set && same_record(single.value._field1, batch) &&
                    batch.p0[0] == c.wall_x && batch.d < inf;
        } else {
            right = !single.set && batch.d == inf && batch.p0[0] == 0.0f;
        }
        check(c.what, right, batch);
    }
    for (size_t i = 0; i < point_cases.size(); i++) {
        const PointCase &c = point_cases[i];
        __tuple_0 single;
        closest(single, c.point, tree);
        const Record &batch = point_hits[i];
        const bool right = same_record(single._field1, batch) && batch.p0[0] == c.wall_x &&
                           batch.d == std::fabs(c.point.vec[0] - c.wall_x);
        check(c.what, right, batch);
    }
    if (failures != 0) {
        std::cout << failures << " WRONG\n";
        return 1;
    }
    std::cout << "every batch record is its single query's\n";
    return 0;
}
