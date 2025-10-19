#pragma once

#include <cstdint>
#include "runtime/bonsai_cpp.h"

struct Point {
    float x;
    float y;
};
struct _tree_layout1 {
    float xl;
    float xh;
    float yl;
    float yh;
    uint8_t nPrims;
    uint24_t offset;
} __attribute__((packed));
struct _tree_layout0 {
    uint64_t pCount;
    Point* prims;
    uint64_t nCount;
    _tree_layout1* group0_index;
} __attribute__((packed));
struct _tree_layout2 {
} __attribute__((packed));
struct _tree_layout3 {
} __attribute__((packed));

set<std::tuple<Point, Point>> chebyshev(const float value, const set<Point>& input0, const set<Point>& input1);
set<std::tuple<Point, Point>> chebyshev_dual(const float value, const _tree_layout0& tree0, const _tree_layout0& tree1);
set<std::tuple<Point, set<Point>>> chebyshev_single(const float value, const set<Point>& input0, const _tree_layout0& tree1);
set<std::tuple<Point, Point>> cosine(const float value, const set<Point>& input0, const set<Point>& input1);
set<std::tuple<Point, Point>> cosine_dual(const float value, const _tree_layout0& tree0, const _tree_layout0& tree1);
set<std::tuple<Point, set<Point>>> cosine_single(const float value, const set<Point>& input0, const _tree_layout0& tree1);
set<std::tuple<Point, Point>> donut(const float rmin, const float rmax, const set<Point>& input0, const set<Point>& input1);
set<std::tuple<Point, Point>> donut_dual(const float rmin, const float rmax, const _tree_layout0& tree0, const _tree_layout0& tree1);
set<std::tuple<Point, set<Point>>> donut_single(const float rmin, const float rmax, const set<Point>& input0, const _tree_layout0& tree1);
set<std::tuple<Point, Point>> euclidean(const float radius, const set<Point>& input0, const set<Point>& input1);
set<std::tuple<Point, Point>> euclidean_dual(const float radius, const _tree_layout0& tree0, const _tree_layout0& tree1);
set<std::tuple<Point, set<Point>>> euclidean_single(const float radius, const set<Point>& input0, const _tree_layout0& tree1);
set<std::tuple<Point, Point>> manhattan(const float value, const set<Point>& input0, const set<Point>& input1);
set<std::tuple<Point, Point>> manhattan_dual(const float value, const _tree_layout0& tree0, const _tree_layout0& tree1);
set<std::tuple<Point, set<Point>>> manhattan_single(const float value, const set<Point>& input0, const _tree_layout0& tree1);
