#include "joins_gen.h"
#include <algorithm>
#include <cassert>
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

// -------- Choose one by uncommenting or defining via -D flag --------
// #define USE_UNIFORM
// #define USE_NORMAL
// #define USE_EXPONENTIAL
// #define USE_LOGNORMAL
// #define USE_CAUCHY
// #define USE_WEIBULL

set<Point> generate_random_set(std::mt19937 &rng, size_t size,
                               float min_val = -1000, float max_val = 1000) {
    set<Point> result;
#if defined(USE_NORMAL)
    // Centered at 0, stddev so most values fall in [min, max]
    std::normal_distribution<float> dist(0.0f, (max_val - min_val) / 4.0f);

#elif defined(USE_EXPONENTIAL)
    // Shifted exponential: λ controls spread; result shifted by min_val
    std::exponential_distribution<float> dist(1.0f / (max_val - min_val));

#elif defined(USE_LOGNORMAL)
    // log-normal parameters — mean and stddev of the underlying normal
    std::lognormal_distribution<float> dist(0.0f, (max_val - min_val) / 4.0f);

#elif defined(USE_CAUCHY)
    // Heavy-tailed: median=0, scale controls width
    std::cauchy_distribution<float> dist(0.0f, (max_val - min_val) / 10.0f);

#elif defined(USE_WEIBULL)
    // Shape > 0, scale > 0; used in survival analysis, reliability
    std::weibull_distribution<float> dist(2.0f, (max_val - min_val) / 2.0f);

#else
    std::uniform_real_distribution<float> dist(min_val, max_val);
#endif

    std::function<float()> get_value = [&]() -> float {
        float val = dist(rng);

#if defined(USE_EXPONENTIAL)
        x += min_val;
#elif defined(USE_LOGNORMAL) || defined(USE_CAUCHY) || defined(USE_WEIBULL)
        x = min_val + fmod(val, max_val - min_val); // wrap into range
#endif

        // Optional clamp for distributions that might go out of range
        if (val < min_val || val > max_val || !std::isfinite(val))
            return get_value();
        return val;
    };

    while (result.size() < size) {
        float x = get_value();
        float y = get_value();

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

    out << "x, y\n"; // CSV header
    input_set.for_each([&](const Point &point) { out << point.x << ", " << point.y << "\n"; });

    out.close();
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

    // TODO: add parameters that pass the xl/xh/yl/yh values.
    std::function<uint64_t(uint64_t, uint64_t, uint64_t)> handle_range =
        [&](uint64_t low, uint64_t high, uint64_t depth) -> uint64_t {
        // assert(depth < MAX_TREE_DEPTH);

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
            reinterpret_cast<_tree_layout3 *>(
                &tree.group0_index[this_index].split0on_nPrims)
                ->pOffset = low;
        } else {
            tree.group0_index[this_index].nPrims = 0;

            // Choose split axis: longest dimension
            bool split_on_x = (xh - xl) >= (yh - yl);

            // Sort on that axis
            if (split_on_x) {
                std::sort(
                    tree.prims + low, tree.prims + high,
                    [](const Point &a, const Point &b) { return a.x < b.x; });
            } else {
                std::sort(
                    tree.prims + low, tree.prims + high,
                    [](const Point &a, const Point &b) { return a.y < b.y; });
            }

            // Split in the middle (median)
            uint64_t mid = low + count / 2;

            // Recursively build subtrees
            uint64_t left = handle_range(low, mid, depth + 1);
            uint64_t right = handle_range(mid, high, depth + 1);

            // Set split offset (offset from this node to right child)
            uint64_t offset = right - this_index;
            reinterpret_cast<_tree_layout2 *>(
                &tree.group0_index[this_index].split0on_nPrims)
                ->offset = offset;
        }
        return this_index;
    };

    // TODO: pass bounding box info computed via sort...?
    handle_range(/*low=*/0, /*high=*/tree.pCount, /*depth=*/0);
    return tree;
}

#define PROFILE 0

auto benchmark_chebyshev(const set<Point> &input0, const set<Point> &input1, const _tree_layout0 &tree0, const _tree_layout0 &tree1,
                         const int k, const int m) {
    float value = 0.0001;
#ifndef PROFILE
    std::cout << "Chebyshev query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_join<PROFILE == 1, set <std::tuple<Point, Point>>, set<std::tuple<Point, set<Point>>>>
            ("chebyshev", input0, input1, tree0, tree1, k, m,
             chebyshev, chebyshev_single, chebyshev_dual, value);
}

auto benchmark_cosine(const set<Point> &input0, const set<Point> &input1, const _tree_layout0 &tree0, const _tree_layout0 &tree1,
                         const int k, const int m) {
    float value = 0.0001;
#ifndef PROFILE
    std::cout << "Cosine query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_join<PROFILE == 1, set <std::tuple<Point, Point>>, set<std::tuple<Point, set<Point>>>>
            ("cosine", input0, input1, tree0, tree1, k, m,
             cosine, cosine_single, cosine_dual, value);
}


auto benchmark_donut(const set<Point> &input0, const set<Point> &input1, const _tree_layout0 &tree0, const _tree_layout0 &tree1,
                         const int k, const int m) {
    float value0 = 0.0001;
    float value1 = 0.0002;
#ifndef PROFILE
    std::cout << "donut query, bounds = " << value0 << ", " << value1 << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_join<PROFILE == 1, set <std::tuple<Point, Point>>, set<std::tuple<Point, set<Point>>>>
            ("donut", input0, input1, tree0, tree1, k, m,
             donut, donut_single, donut_dual, value0, value1);
}


auto benchmark_euclidean(const set<Point> &input0, const set<Point> &input1, const _tree_layout0 &tree0, const _tree_layout0 &tree1,
                         const int k, const int m) {
    float value = 0.0001;
#ifndef PROFILE
    std::cout << "euclidean query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_join<PROFILE == 1, set <std::tuple<Point, Point>>, set<std::tuple<Point, set<Point>>>>
            ("euclidean", input0, input1, tree0, tree1, k, m,
             euclidean, euclidean_single, euclidean_dual, value);
}


auto benchmark_manhattan(const set<Point> &input0, const set<Point> &input1, const _tree_layout0 &tree0, const _tree_layout0 &tree1,
                         const int k, const int m) {
    float value = 0.0001;
#ifndef PROFILE
    std::cout << "manhattan query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    return benchmark_join<PROFILE == 1, set <std::tuple<Point, Point>>, set<std::tuple<Point, set<Point>>>>
            ("manhattan", input0, input1, tree0, tree1, k, m,
             manhattan, manhattan_single, manhattan_dual, value);
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

int main() {
    const int k = 3; // total runs
    const int m = 0;  // number of fastest and slowest to drop

    // std::mt19937 rng(std::random_device{}());
    // For consistent results
    std::mt19937 rng(42);

    std::vector<size_t> test_sizes = {
        1ull << 8,  1ull << 9,  1ull << 10, 1ull << 11, 1ull << 12,
        1ull << 13, 1ull << 14, 1ull << 15, 1ull << 16, 1ull << 17,
        1ull << 18,1ull << 19, 1ull << 20,
        // 1ull << 21, 1ull << 22,
        // 1ull << 23, 1ull << 24, 1ull << 25, 1ull << 26, 1ull << 27,
        // 1ull << 28, 1ull << 29, 1ull << 30, 1ull << 31, (1ull << 32) - 1
    };
#if defined(USE_NORMAL)
    std::cout << "normal distribution" << std::endl;
#elif defined(USE_EXPONENTIAL)
    std::cout << "exponential distribution" << std::endl;
#elif defined(USE_LOGNORMAL)
    std::cout << "lognormal distribution" << std::endl;
#elif defined(USE_CAUCHY)
    std::cout << "cauchy distribution" << std::endl;
#elif defined(USE_WEIBULL)
    std::cout << "weibull distribution" << std::endl;
#else
    std::cout << "uniform distribution" << std::endl;
#endif

#ifdef PROFILE
    pretty_print_vector(test_sizes);
    std::cout << std::endl;
    static constexpr int N_BENCHMARKS = 1;
    // std::vector<std::pair<std::string, std::vector<double>>> results(
    //     N_BENCHMARKS);
    // results[0].first = "abs(a.x - b.x) < 0.1";
    // results[0].second.reserve(test_sizes.size());

#endif
    for (size_t size : test_sizes) {
#ifndef PROFILE
        std::cout << "\n--- Test with input size: " << size << " ---"
                  << std::endl;
#endif
        // Generate input
        auto input_set0 = generate_random_set(rng, size);
        auto input_set1 = generate_random_set(rng, size);

#ifdef EXPORT
        export_to_csv(input_set0,
                      "/Users/ajroot/projects/learn-sql/data/input_a_" +
                          std::to_string(size) + ".csv");
        export_to_csv(input_set1,
                      "/Users/ajroot/projects/learn-sql/data/input_b_" +
                          std::to_string(size) + ".csv");
#endif

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
        // verify_IntervalTree(0, input_tree0);
        // verify_IntervalTree(0, input_tree1);

        {
        auto [nested, single, dual] = benchmark_chebyshev(input_set0, input_set1, input_tree0, input_tree1, k, m);

    
// #ifdef PROFILE
//         results[0].second.push_back((dual > 0) ? (double)nested / (double)dual : std::numeric_limits<double>::max());
// #endif
        std::cout << "chebyshev: (" << size << ", " << nested << ", " << build_time << ", " << single << ", " << dual << ")" << std::endl;
        }

        // {
        // auto [nested, single, dual] = benchmark_cosine(input_set0, input_set1, input_tree0, input_tree1, k, m);
        // std::cout << "cosine: (" << size << ", " << nested << ", " << build_time << ", " << single << ", " << dual << ")" << std::endl;
        // }

        {
        auto [nested, single, dual] = benchmark_donut(input_set0, input_set1, input_tree0, input_tree1, k, m);
        std::cout << "donut: (" << size << ", " << nested << ", " << build_time << ", " << single << ", " << dual << ")" << std::endl;
        }

        {
        auto [nested, single, dual] = benchmark_euclidean(input_set0, input_set1, input_tree0, input_tree1, k, m);
        std::cout << "euclidean: (" << size << ", " << nested << ", " << build_time << ", " << single << ", " << dual << ")" << std::endl;
        }

        {
        auto [nested, single, dual] = benchmark_manhattan(input_set0, input_set1, input_tree0, input_tree1, k, m);
        std::cout << "manhattan: (" << size << ", " << nested << ", " << build_time << ", " << single << ", " << dual << ")" << std::endl;
        }

        std::free(input_tree0.prims);
        std::free(input_tree0.group0_index);
        std::free(input_tree1.prims);
        std::free(input_tree1.group0_index);
    }
// #ifdef PROFILE
//     for (const auto &res : results) {
//         std::cout << "(\"" << res.first << "\", ";
//         pretty_print_vector(res.second);
//         std::cout << ")," << std::endl;
//     }
// #endif
    return 0;
}
