#include "queries_gen.h"
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

_tree_layout0 copy_tree(const _tree_layout4 &src) {
    _tree_layout0 dst;
    dst.pCount = src.pCount;
    dst.nCount = src.nCount;

    // Same primitives.
    dst.prims = src.prims;

    // Allocate destination nodes
    dst.group0_index = static_cast<_tree_layout1 *>(
        std::malloc(sizeof(_tree_layout1) * dst.nCount));
    if (!dst.group0_index) {
        std::free(dst.prims);
        throw std::bad_alloc();
    }

    // Fast copy of tree node data (dropping dCount)
    for (uint64_t i = 0; i < dst.nCount; ++i) {
        const _tree_layout5 &srcNode = src.group0_index[i];
        _tree_layout1 &dstNode = dst.group0_index[i];

        // Copy all fields *except* dCount
        dstNode.xl = srcNode.xl;
        dstNode.xh = srcNode.xh;
        dstNode.yl = srcNode.yl;
        dstNode.yh = srcNode.yh;
        dstNode.nPrims = srcNode.nPrims;
        dstNode.split0on_nPrims = srcNode.split0on_nPrims;
    }

    return dst;
}

#define USE_APPROX_SAH

_tree_layout4 build_tree(const set<Point> &input) {
    _tree_layout4 tree;
    tree.pCount = input.size();
    tree.prims = static_cast<Point *>(std::malloc(sizeof(Point) * tree.pCount));
    if (!tree.prims) {
        throw std::bad_alloc();
    }

    std::copy(input.data.begin(), input.data.end(), tree.prims);

    // TODO: always sort on larger dimension (x or y).
    // std::sort(tree.prims, tree.prims + tree.pCount);

    constexpr uint64_t MAX_LEAF_COUNT = 8;

    // Safe conservative tree estimate.
    tree.nCount = 2 * tree.pCount - 1;
    tree.group0_index = static_cast<_tree_layout5 *>(
        std::malloc(sizeof(_tree_layout5) * tree.nCount));
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

        tree.group0_index[this_index].dCount = high - low;

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

#ifdef USE_APPROX_SAH
            // Fast binned split: pick index that minimizes left/right interval
            // ratio
            uint64_t best_mid = low + count / 2;
            float best_ratio = std::numeric_limits<float>::max();
            for (uint64_t i = 1; i < count; ++i) {
                float left_size =
                    (split_on_x ? tree.prims[low + i - 1].x
                                : tree.prims[low + i - 1].y) -
                    (split_on_x ? tree.prims[low].x : tree.prims[low].y);
                float right_size = (split_on_x ? tree.prims[high - 1].x
                                               : tree.prims[high - 1].y) -
                                   (split_on_x ? tree.prims[low + i].x
                                               : tree.prims[low + i].y);

                float ratio = std::abs(left_size / (right_size + 1e-9) - 1.0f);
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    best_mid = low + i;
                }
            }

            uint64_t mid = best_mid;

            // Recursively build subtrees
            uint64_t left = handle_range(low, mid, depth + 1);
            uint64_t right = handle_range(mid, high, depth + 1);
#else
            // Split in the middle (median)
            uint64_t mid = low + count / 2;

            // Recursively build subtrees
            uint64_t left = handle_range(low, mid, depth + 1);
            uint64_t right = handle_range(mid, high, depth + 1);
#endif
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

void verify_IntervalTree(const uint64_t input_index,
                         const _tree_layout0 input) {
    if (input.group0_index[input_index].nPrims == 0u) {
        verify_IntervalTree(input_index + 1u, input);
        verify_IntervalTree(
            input_index +
                (uint64_t)(reinterpret<_tree_layout2>(
                               input.group0_index[input_index].split0on_nPrims)
                               .offset),
            input);
    } else {
        for (uint64_t _idx0 = 0u;
             _idx0 < (uint64_t)(input.group0_index[input_index].nPrims);
             _idx0 += 1u) {
            const Point prim =
                input.prims[(uint64_t)(reinterpret<_tree_layout3>(
                                           input.group0_index[input_index]
                                               .split0on_nPrims)
                                           .pOffset) +
                            _idx0];
            if (input.group0_index[input_index].xl > prim.x) {
                std::cout << "Tree verification failed at index: "
                          << input_index << " with prim = " << prim
                          << " and lb(x) = " << input.group0_index[input_index].xl
                          << std::endl;
                abort();
            }
            if (input.group0_index[input_index].yl > prim.y) {
                std::cout << "Tree verification failed at index: "
                          << input_index << " with prim = " << prim
                          << " and lb(y) = " << input.group0_index[input_index].yl
                          << std::endl;
                abort();
            }
            if (input.group0_index[input_index].xh < prim.x) {
                std::cout << "Tree verification failed at index: "
                          << input_index << " with prim = " << prim
                          << " and ub(x) = " << input.group0_index[input_index].xh
                          << std::endl;
                abort();
            }
            if (input.group0_index[input_index].yh < prim.y) {
                std::cout << "Tree verification failed at index: "
                          << input_index << " with prim = " << prim
                          << " and ub(y) = " << input.group0_index[input_index].yh
                          << std::endl;
                abort();
            }
        }
    }
}

#define PROFILE 0

double benchmark_absd_query(const set<Point> &input, const _tree_layout0 &tree,
                             const int k, const int m) {
    float value = 1;
#ifndef PROFILE
    std::cout << "Absd query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, set<Point>>(
        "abs(x - y) <= 10", input, tree, k, m, absd_query, absd_query_fast,
        value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_absd_count_query(const set<Point> &input,
                                  const _tree_layout0 &tree, const int k,
                                  const int m) {
    float value = 1;
#ifndef PROFILE
    std::cout << "Absd query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, uint64_t>(
        "COUNT(abs(x - y) <= 10)", input, tree, k, m, absd_count_query,
        absd_count_query_fast, value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_absd_count_aug_query(const set<Point> &input,
                                      const _tree_layout4 &tree, const int k,
                                      const int m) {
    float value = 1;
#ifndef PROFILE
    std::cout << "Absd query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, uint64_t>(
        "aug COUNT(abs(x - y) <= 10)", input, tree, k, m, absd_count_query,
        absd_count_query_fast_aug, value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_abss_query(const set<Point> &input, const _tree_layout0 &tree,
                            const int k, const int m) {
    float value = 1;
#ifndef PROFILE
    std::cout << "Abss query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, set<Point>>(
        "abs(x + y) <= 10", input, tree, k, m, abss_query, abss_query_fast,
        value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_abss_count_query(const set<Point> &input,
                                  const _tree_layout0 &tree, const int k,
                                  const int m) {
    float value = 1;
#ifndef PROFILE
    std::cout << "Abss query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, uint64_t>(
        "COUNT(abs(x + y) <= 10)", input, tree, k, m, abss_count_query,
        abss_count_query_fast, value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_abss_count_aug_query(const set<Point> &input,
                                      const _tree_layout4 &tree, const int k,
                                      const int m) {
    float value = 1;
#ifndef PROFILE
    std::cout << "Abss query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, uint64_t>(
        "aug COUNT(abs(x + y) <= 10)", input, tree, k, m, abss_count_query,
        abss_count_query_fast_aug, value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_circle_query(const set<Point> &input,
                              const _tree_layout0 &tree, const int k,
                              const int m) {
    float value = 10;
#ifndef PROFILE
    std::cout << "Circle query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, set<Point>>(
        "x^2 + y^2 <= 100", input, tree, k, m, circle_query, circle_query_fast,
        value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_circle_count_query(const set<Point> &input,
                                    const _tree_layout0 &tree, const int k,
                                    const int m) {
    float value = 10;
#ifndef PROFILE
    std::cout << "Circle query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, uint64_t>(
        "COUNT(x^2 + y^2 <= 100)", input, tree, k, m, circle_count_query,
        circle_count_query_fast, value);
    return static_cast<double>(linear) / indexed;
}

double benchmark_circle_count_aug_query(const set<Point> &input,
                                        const _tree_layout4 &tree, const int k,
                                        const int m) {
    float value = 10;
#ifndef PROFILE
    std::cout << "Circle query, value = " << value << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
#endif
    auto [linear, indexed] = benchmark_1d_queries<PROFILE == 1, uint64_t>(
        "aug COUNT(x^2 + y^2 <= 100)", input, tree, k, m, circle_count_query,
        circle_count_query_fast_aug, value);
    return static_cast<double>(linear) / indexed;
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
    const int k = 14; // total runs
    const int m = 2;  // number of fastest and slowest to drop

    // std::mt19937 rng(std::random_device{}());
    // For consistent results
    std::mt19937 rng(42);

    std::vector<size_t> test_sizes = {
        1ull << 8,  1ull << 9,  1ull << 10, 1ull << 11, 1ull << 12,
        1ull << 13, 1ull << 14, 1ull << 15, 1ull << 16, 1ull << 17,
        1ull << 18, 1ull << 19, 1ull << 20, 1ull << 21, 1ull << 22,
        1ull << 23, 1ull << 24, 1ull << 25, 1ull << 26, 1ull << 27,
        1ull << 28, // 1ull << 29, 1ull << 30, 1ull << 31, (1ull << 32) - 1
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
    static constexpr int N_BENCHMARKS = 9;
    std::vector<std::pair<std::string, std::vector<double>>> results(
        N_BENCHMARKS);
    results[0].first = "abs(x - y) <= 1";
    results[0].second.reserve(test_sizes.size());
    results[1].first = "abs(x + y) <= 1";
    results[1].second.reserve(test_sizes.size());
    results[2].first = "x^2 + y^2 <= 10";
    results[2].second.reserve(test_sizes.size());

    results[3].first = "COUNT(abs(x - y) <= 1)";
    results[3].second.reserve(test_sizes.size());
    results[4].first = "COUNT(abs(x + y) <= 1)";
    results[4].second.reserve(test_sizes.size());
    results[5].first = "COUNT(x^2 + y^2 <= 10)";
    results[5].second.reserve(test_sizes.size());

    results[6].first = "aug COUNT(abs(x - y) <= 1)";
    results[6].second.reserve(test_sizes.size());
    results[7].first = "aug COUNT(abs(x + y) <= 1)";
    results[7].second.reserve(test_sizes.size());
    results[8].first = "aug COUNT(x^2 + y^2 <= 10)";
    results[8].second.reserve(test_sizes.size());

#endif
    for (size_t size : test_sizes) {
        std::cout << size << std::endl;
#ifndef PROFILE
        std::cout << "\n--- Test with input size: " << size << " ---"
                  << std::endl;
#endif
        // Generate input
        auto input_set = generate_random_set(rng, size);

#ifdef EXPORT
        export_to_csv(input_set,
                      "/Users/ajroot/projects/learn-sql/data/input_" +
                          std::to_string(size) + ".csv");
#endif

        // Build tree
        auto t_build_start = std::chrono::high_resolution_clock::now();
        const auto input_tree_aug = build_tree(input_set);
        const auto input_tree = copy_tree(input_tree_aug);
        auto t_build_end = std::chrono::high_resolution_clock::now();
        int64_t build_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_build_end -
                                                                 t_build_start)
                .count();
#if PROFILE
        std::cout << "build_tree() time: " << build_time << " ns\n";
        // verify_IntervalTree(0, input_tree);
#endif

#ifdef PROFILE
        results[0].second.push_back(
#endif
            benchmark_absd_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[1].second.push_back(
#endif
            benchmark_abss_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[2].second.push_back(
#endif
            benchmark_circle_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[3].second.push_back(
#endif
            benchmark_absd_count_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[4].second.push_back(
#endif
            benchmark_abss_count_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[5].second.push_back(
#endif
            benchmark_circle_count_query(input_set, input_tree, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[6].second.push_back(
#endif
            benchmark_absd_count_aug_query(input_set, input_tree_aug, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[7].second.push_back(
#endif
            benchmark_abss_count_aug_query(input_set, input_tree_aug, k, m)
#ifdef PROFILE
        )
#endif
            ;

#ifdef PROFILE
        results[8].second.push_back(
#endif
            benchmark_circle_count_aug_query(input_set, input_tree_aug, k, m)
#ifdef PROFILE
        )
#endif
            ;

        std::free(input_tree.prims);
        std::free(input_tree.group0_index);
        std::free(input_tree_aug.group0_index);
    }
#ifdef PROFILE
    for (const auto &res : results) {
        std::cout << "(\"" << res.first << "\", ";
        pretty_print_vector(res.second);
        std::cout << ")," << std::endl;
    }
#endif
    return 0;
}
