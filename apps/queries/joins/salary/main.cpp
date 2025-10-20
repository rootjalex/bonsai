#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "joins_gen.h"

#include "apps/queries/joins/salary/joins_gen.h"


std::vector<Employee> load_csv(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file '" << filename << "'\n";
        abort();
    }

    std::vector<Employee> employees;
    std::string line;

    // skip header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: empty file or failed to read header\n";
        abort();
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string name, dept_str, salary_str, tax_str;

        if (!std::getline(iss, name, ',')) continue;
        if (!std::getline(iss, dept_str, ',')) continue;
        if (!std::getline(iss, salary_str, ',')) continue;
        if (!std::getline(iss, tax_str, ',')) continue;

        try {
            Employee e;
            e.salary = std::stof(salary_str);
            e.tax = std::stof(tax_str);
            employees.push_back(e);
        } catch (const std::exception& ex) {
            std::cout << "Warning: skipping malformed line: " << line << "\n";
        }
    }

    // Print summary
    std::cout << "Loaded " << employees.size() << " employees.\n";
    return employees;
}

#define USE_APPROX_SAH

_tree_layout0 build_tree(const set<Employee> &input) {
    _tree_layout0 tree;
    tree.pCount = input.size();
    tree.prims = static_cast<Employee *>(std::malloc(sizeof(Employee) * tree.pCount));
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

#ifdef USE_APPROX_SAH
    // TODO: set root_min_salary, root_max_salary, root_min_tax, root_max_tax
    if (tree.pCount == 0) {
        throw std::runtime_error("Empty input set");
    }

    float root_min_salary = input.data.begin()->salary;
    float root_max_salary = input.data.begin()->salary;
    float root_min_tax    = input.data.begin()->tax;
    float root_max_tax    = input.data.begin()->tax;

    for (const auto &e : input.data) {
        root_min_salary = std::min(root_min_salary, e.salary);
        root_max_salary = std::max(root_max_salary, e.salary);
        root_min_tax    = std::min(root_min_tax, e.tax);
        root_max_tax    = std::max(root_max_tax, e.tax);
    }
#endif

    std::function<uint64_t(uint64_t, uint64_t, uint64_t)> handle_range =
        [&](uint64_t low, uint64_t high, uint64_t depth) -> uint64_t {

        uint64_t count = high - low;
        uint64_t this_index = next_node++;
        assert(this_index < tree.nCount);

        tree.group0_index[this_index].c = high - low;

        // Compute bounding box for current range
        float sl = tree.prims[low].salary, sh = tree.prims[low].salary;
        float tl = tree.prims[low].tax, th = tree.prims[low].tax;
        for (uint64_t i = low + 1; i < high; ++i) {
            sl = std::min(sl, tree.prims[i].salary);
            sh = std::max(sh, tree.prims[i].salary);
            tl = std::min(tl, tree.prims[i].tax);
            th = std::max(th, tree.prims[i].tax);
        }
        tree.group0_index[this_index].sl = sl;
        tree.group0_index[this_index].sh = sh;
        tree.group0_index[this_index].tl = tl;
        tree.group0_index[this_index].th = th;

        if (count <= MAX_LEAF_COUNT) {
            // Leaf node
            tree.group0_index[this_index].nPrims = count;
            tree.group0_index[this_index].offset = low;
        } else {
            tree.group0_index[this_index].nPrims = 0;

            // Choose split axis: longest dimension
            bool split_on_x = (sh - sl) / (root_max_salary - root_min_salary) >= (th - tl) / (root_max_tax - root_min_tax);

            // Sort on that axis
            if (split_on_x) {
                std::sort(
                    tree.prims + low, tree.prims + high,
                    [](const Employee &a, const Employee &b) { return a.salary < b.salary; });
            } else {
                std::sort(
                    tree.prims + low, tree.prims + high,
                    [](const Employee &a, const Employee &b) { return a.tax < b.tax; });
            }

#ifdef USE_APPROX_SAH
            // Fast binned split: pick index that minimizes left/right interval
            // ratio
            uint64_t best_mid = low + count / 2;
            float best_ratio = std::numeric_limits<float>::max();
            for (uint64_t i = 1; i < count; ++i) {
                float left_size =
                    (split_on_x ? tree.prims[low + i - 1].salary
                                : tree.prims[low + i - 1].tax) -
                    (split_on_x ? tree.prims[low].salary : tree.prims[low].tax);
                float right_size = (split_on_x ? tree.prims[high - 1].salary
                                               : tree.prims[high - 1].tax) -
                                   (split_on_x ? tree.prims[low + i].salary
                                               : tree.prims[low + i].tax);

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
            tree.group0_index[this_index].offset = offset;
        }
        return this_index;
    };

    // TODO: pass bounding box info computed via sort...?
    handle_range(/*low=*/0, /*high=*/tree.pCount, /*depth=*/0);
    return tree;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file>\n";
        return 1;
    }

    const int k = 7; // total runs
    const int m = 1; // number of fastest and slowest to drop

    const std::string filename = argv[1];
    const set<Employee> employees = load_csv(filename);

    // TODO: measure tree build time.
    auto start = std::chrono::high_resolution_clock::now();
    const auto tree = build_tree(employees);
    auto end = std::chrono::high_resolution_clock::now();

    auto build_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    auto [nested, single, dual] = benchmark_join<false, uint64_t, uint64_t>("salary join", employees, employees, tree, tree, k, m, query_nested, query_single, query_dual);

    std::cout << "(\"bonsai_salary\", " << employees.size() << ", " << (double)nested / (1e9) << ", " << (double)build_time / (1e9) << ", " << (double)single / (1e9) << ", " << (double)dual / (1e9) << ")" << std::endl;

    return 0;
}
