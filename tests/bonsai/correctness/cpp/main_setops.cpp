// Drives the queries in setops.bonsai and checks each against a linear scan
// over the same data. The generated traversals prune, so agreeing with an
// unpruned scan on every query and every range is the property that matters.

#include "setops.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace {

// Build the packed interval tree that setops.bonsai's layout describes: the
// primitives sorted, and a balanced index over them where each node records
// the range of values beneath it.
Input build_tree(std::vector<float> values) {
    constexpr uint64_t MAX_LEAF_COUNT = 4;

    Input tree;
    tree.pCount = values.size();
    tree.prims = static_cast<float *>(std::malloc(sizeof(float) * tree.pCount));
    std::sort(values.begin(), values.end());
    std::copy(values.begin(), values.end(), tree.prims);

    tree.count = 2 * tree.pCount - 1;
    tree.nodes = static_cast<Nodes *>(std::malloc(sizeof(Nodes) * tree.count));

    uint64_t next_node = 0;
    std::function<uint64_t(uint64_t, uint64_t)> handle_range =
        [&](uint64_t low, uint64_t high) -> uint64_t {
        const uint64_t count = high - low;
        const uint64_t this_index = next_node++;
        assert(this_index < tree.count);

        tree.nodes[this_index].low = tree.prims[low];
        tree.nodes[this_index].high = tree.prims[high - 1];

        if (count <= MAX_LEAF_COUNT) {
            tree.nodes[this_index].nPrims = count;
            reinterpret_cast<Arm_Leaf *>(
                &tree.nodes[this_index].split0on_nPrims)
                ->pOffset = low;
        } else {
            tree.nodes[this_index].nPrims = 0;
            const uint64_t mid = low + count / 2;
            handle_range(low, mid);
            const uint64_t right = handle_range(mid, high);
            reinterpret_cast<Arm_Interior *>(
                &tree.nodes[this_index].split0on_nPrims)
                ->offset = right - this_index;
        }
        return this_index;
    };
    handle_range(0, tree.pCount);
    return tree;
}

int failures = 0;

template <typename T>
void check(const char *what, float lo, float hi, T got, T want) {
    if (got != want) {
        std::cout << "MISMATCH " << what << " over [" << lo << ", " << hi
                  << "]: got " << got << ", want " << want << "\n";
        failures++;
    }
}

} // namespace

int main() {
    std::vector<float> values;
    for (int i = 0; i < 64; i++) {
        // Spread unevenly so subtrees have different extents.
        values.push_back(static_cast<float>((i * 37) % 64));
    }
    const Input tree = build_tree(values);

    // Ranges chosen to cover an empty result, a result entirely inside one
    // subtree, a result spanning several, and one containing everything.
    const float bounds[][2] = {
        {100.f, 200.f}, {-5.f, -1.f}, {0.f, 0.f},  {10.f, 12.f},
        {0.f, 31.f},    {32.f, 63.f}, {7.f, 45.f}, {-1.f, 64.f},
    };

    for (const auto &b : bounds) {
        const float lo = b[0], hi = b[1];

        uint64_t want_count = 0;
        float want_sum = 0.f;
        float want_min = std::numeric_limits<float>::infinity();
        float want_max = -std::numeric_limits<float>::infinity();
        bool want_any = false, want_all = true;
        for (float v : values) {
            const bool in = lo <= v && v <= hi;
            want_any = want_any || in;
            want_all = want_all && in;
            if (!in) {
                continue;
            }
            want_count++;
            want_sum += v;
            want_min = std::min(want_min, v);
            want_max = std::max(want_max, v);
        }

        check("count", lo, hi, q_count(lo, hi, &tree), want_count);
        check("sum", lo, hi, q_sum(lo, hi, &tree), want_sum);
        check("any", lo, hi, q_any(lo, hi, &tree), want_any);
        check("all", lo, hi, q_all(lo, hi, &tree), want_all);

        const std::optional<float> got_min = q_min(lo, hi, &tree);
        const std::optional<float> got_max = q_max(lo, hi, &tree);
        check("minimum set", lo, hi, got_min.has_value(), want_count != 0);
        check("maximum set", lo, hi, got_max.has_value(), want_count != 0);
        if (want_count != 0) {
            check("minimum", lo, hi, *got_min, want_min);
            check("maximum", lo, hi, *got_max, want_max);
        }
    }

    std::cout << (failures == 0 ? "all queries match a linear scan\n"
                                : "FAILED\n");
    std::free(tree.prims);
    std::free(tree.nodes);
    return failures == 0 ? 0 : 1;
}
