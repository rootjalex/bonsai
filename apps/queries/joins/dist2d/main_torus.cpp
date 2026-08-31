#include "torus_gen.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

std::ostream &operator<<(std::ostream &os, const Point &point) {
    os << "Point {" << point.x << ", " << point.y << "}";
    return os;
}

bool operator<(const Point &a, const Point &b) {
    return (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
}

bool operator==(const Point &a, const Point &b) {
    return (a.x == b.x) && (a.y == b.y);
}

std::ostream &operator<<(std::ostream &os, const std::tuple<Point, Point> &p) {
    os << "(" << std::get<0>(p) << ", " << std::get<1>(p) << ")";
    return os;
}

set<Point> generate_random_circle(std::mt19937 &rng, size_t size,
                                  float radius) {
    set<Point> result;
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> radius_dist(0.0f,
                                                      1.0f); // for sqrt scaling

    while (result.size() < size) {
        float theta = angle_dist(rng);
        float r =
            radius * std::sqrt(radius_dist(rng)); // uniform sampling in circle
        float x = r * std::cos(theta);
        float y = r * std::sin(theta);
        result.push_back(Point{x, y});
    }
    return result;
}

void export_to_csv(const set<Point> &input_set, const std::string &filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename
                  << std::endl;
        return;
    }

    out << "x,y\n"; // CSV header
    input_set.for_each(
        [&](const Point &point) { out << point.x << ", " << point.y << "\n"; });

    out.close();
}

#define USE_APPROX_SAH

uint64_t build_range(_tree_layout0 &tree, uint64_t &next_node, uint64_t low,
                     uint64_t high, bool prev_split_on_x) {
    constexpr uint64_t MAX_LEAF_COUNT = 8;

    uint64_t count = high - low;
    uint64_t this_index = next_node++;
    assert(this_index < tree.nCount);

    // tree.group0_index[this_index].dCount = high - low;

    // Compute bounding box for current range
    float xl = tree.prims[low].x, xh = tree.prims[low].x;
    float yl = tree.prims[low].y, yh = tree.prims[low].y;
    for (uint64_t i = low + 1; i < high; ++i) {
        xl = std::min(xl, tree.prims[i].x);
        xh = std::max(xh, tree.prims[i].x);
        yl = std::min(yl, tree.prims[i].y);
        yh = std::max(yh, tree.prims[i].y);
    }
    tree.group0_index[this_index].xl = xl;
    tree.group0_index[this_index].xh = xh;
    tree.group0_index[this_index].yl = yl;
    tree.group0_index[this_index].yh = yh;

    if (count <= MAX_LEAF_COUNT) {
        // Leaf node
        tree.group0_index[this_index].nPrims = count;
        tree.group0_index[this_index].offset = low;
    } else {
        tree.group0_index[this_index].nPrims = 0;

        // Choose split axis: longest dimension
        bool split_on_x = (xh - xl) >= (yh - yl);

        // Sort on that axis
        if (split_on_x && !prev_split_on_x) {
            std::sort(tree.prims + low, tree.prims + high,
                      [](const Point &a, const Point &b) { return a.x < b.x; });
        } else if (!split_on_x && prev_split_on_x) {
            std::sort(tree.prims + low, tree.prims + high,
                      [](const Point &a, const Point &b) { return a.y < b.y; });
        }

        // Fast binned split: pick index that minimizes left/right interval
        // ratio
        const bool sx = split_on_x;
        const float first = sx ? tree.prims[low].x : tree.prims[low].y;
        const float last = sx ? tree.prims[high - 1].x : tree.prims[high - 1].y;

        float best_ratio = std::numeric_limits<float>::max();
        uint64_t best_mid = low + count / 2;

        if (sx) {
            for (uint64_t i = 1; i < count; ++i) {
                float left = tree.prims[low + i - 1].x - first;
                float right = last - tree.prims[low + i].x;
                float ratio = std::abs(left / (right + 1e-9f) - 1.0f);
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    best_mid = low + i;
                }
            }
        } else {
            for (uint64_t i = 1; i < count; ++i) {
                float left = tree.prims[low + i - 1].y - first;
                float right = last - tree.prims[low + i].y;
                float ratio = std::abs(left / (right + 1e-9f) - 1.0f);
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    best_mid = low + i;
                }
            }
        }

        uint64_t mid = best_mid;

        // Recursively build subtrees
        uint64_t left = build_range(tree, next_node, low, mid, split_on_x);
        uint64_t right = build_range(tree, next_node, mid, high, split_on_x);

        // Set split offset (offset from this node to right child)
        uint64_t offset = right - this_index;
        tree.group0_index[this_index].offset = offset;
    }
    return this_index;
}

_tree_layout0 build_tree(const set<Point> &input) {
    _tree_layout0 tree;
    tree.pCount = input.size();
    tree.prims = static_cast<Point *>(std::malloc(sizeof(Point) * tree.pCount));
    if (!tree.prims) {
        throw std::bad_alloc();
    }

    std::copy(input.data.begin(), input.data.end(), tree.prims);

    // Sort on query dimensions
    constexpr uint64_t MAX_LEAF_COUNT = 8;

    // Safe conservative tree estimate.
    tree.nCount = 2 * tree.pCount - 1;
    tree.group0_index = static_cast<_tree_layout1 *>(
        std::malloc(sizeof(_tree_layout1) * tree.nCount));
    if (!tree.group0_index) {
        std::free(tree.prims);
        throw std::bad_alloc();
    }

    uint64_t next_node = 0;

    // Sort on x initially
    std::sort(tree.prims, tree.prims + tree.pCount,
              [](const Point &a, const Point &b) { return a.x < b.x; });

    build_range(tree, next_node, /*low=*/0, /*high=*/tree.pCount, true);

    return tree;
}

#define PROFILE 0

auto benchmark_donut(const set<Point> &input0, const set<Point> &input1,
                     const _tree_layout0 &tree0, const _tree_layout0 &tree1,
                     bool &nested_timedout, bool &single_timedout,
                     bool &dual_timedout, const int k, const int m) {
    float value0 = 10;
    float value1 = 20;
#ifndef PROFILE
    std::cout << "donut query, bounds = " << value0 << ", " << value1
              << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_join < PROFILE == 1, set<std::tuple<Point, Point>>,
           set < std::tuple < Point,
           set < Point >>>> ("donut", input0, input1, tree0, tree1, k, m, donut,
                             donut_single, donut_dual, nested_timedout,
                             single_timedout, dual_timedout, value0, value1);
}

template <typename T>
void pretty_print_vector(const std::vector<T> &vec) {
    bool first = true;
    std::cout << "[";
    for (const auto &v : vec) {
        if (!first) {
            std::cout << ", ";
        }
        first = false;
        std::cout << v;
    }
    std::cout << "]";
}

// #define EXPORT 1

int main(int argc, char **argv) {
    const int k = 7; // total runs
    const int m = 1; // number of fastest and slowest to drop

    // Parse radius from command line if provided
    float radius = 5.0f;
    if (argc > 1) {
        try {
            radius = std::stof(argv[1]);
        } catch (const std::exception &e) {
            std::cerr << "Invalid radius provided. Using default 5.0\n";
            radius = 5.0f;
        }
    }

    // std::mt19937 rng(std::random_device{}());
    // For consistent results
    std::mt19937 rng(42);

    std::vector<size_t> test_sizes = {
        1ull << 8,  1ull << 9,  1ull << 10, 1ull << 11, 1ull << 12, 1ull << 13,
        1ull << 14, 1ull << 15, 1ull << 16, 1ull << 17, 1ull << 18, 1ull << 19,
        1ull << 20, 1ull << 21, 1ull << 22, 1ull << 23, 1ull << 24, 1ull << 25,
        1ull << 26, 1ull << 27, 1ull << 28, 1ull << 29, 1ull << 30,
        // 1ull << 31, 1ull << 32
    };

#ifdef PROFILE
    pretty_print_vector(test_sizes);
    std::cout << std::endl;

    bool nested_timedout = false, single_timedout = false,
         dual_timedout = false;

#endif
    for (size_t size : test_sizes) {
#ifndef PROFILE
        std::cout << "\n--- Test with input size: " << size << " ---"
                  << std::endl;
#endif
        // Generate input
        auto input_set0 = generate_random_circle(rng, size, radius);
        auto input_set1 = generate_random_circle(rng, size, radius);

        // Build tree
        auto t_build_start = std::chrono::high_resolution_clock::now();
        const auto input_tree0 = build_tree(input_set0);
        const auto input_tree1 = build_tree(input_set1);
        auto t_build_end = std::chrono::high_resolution_clock::now();
        int64_t build_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_build_end -
                                                                 t_build_start)
                .count();
#if PROFILE
        std::cout << "build_tree() time: " << build_time << " ns\n";
        // verify_IntervalTree(0, input_tree);
#endif

        {
            auto [nested, single, dual] = benchmark_donut(
                input_set0, input_set1, input_tree0, input_tree1,
                nested_timedout, single_timedout, dual_timedout, k, m);
            std::cout << "donut: (" << size << ", " << nested << ", "
                      << build_time << ", " << single << ", " << dual << ")"
                      << std::endl;
        }

        std::free(input_tree0.prims);
        std::free(input_tree0.group0_index);
        std::free(input_tree1.prims);
        std::free(input_tree1.group0_index);
    }
    return 0;
}
