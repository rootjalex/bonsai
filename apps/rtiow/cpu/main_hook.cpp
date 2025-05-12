// clang++ -std=c++20 -O3 apps/rtiow/main_hook.cpp apps/rtiow/main.o -o
// apps/rtiow/main_runner
// ./main_runner &> bonsai_image.ppm
#include <chrono>
#include <fstream> // for std::ofstream
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "main.h"
#include <cassert>

constexpr uint32_t LAMBERTIAN = 0;
constexpr uint32_t METAL = 1;
constexpr uint32_t DIALECTRIC = 2;

constexpr uint32_t MAX_TREE_DEPTH = 64;

constexpr double pi = 3.1415926535897932385;

inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

inline vec3_float min(const vec3_float &a, const vec3_float &b) {
    return vec3_float{std::fminf(a[0], b[0]), std::fminf(a[1], b[1]),
                      std::fminf(a[2], b[2])};
}

inline vec3_float max(const vec3_float &a, const vec3_float &b) {
    return vec3_float{std::fmaxf(a[0], b[0]), std::fmaxf(a[1], b[1]),
                      std::fmaxf(a[2], b[2])};
}

_spheres_layout0 build_tree_simple(std::vector<MaterialSphere> &spheres,
                                   size_t max_prims) {
    _spheres_layout0 tree;
    tree.pCount = spheres.size();
    tree.prims = spheres.data();
    // Just do a simple split, don't even sort for now.
    // First compute the number of nodes needed.
    // Store at most two spheres per leaf node.
    // Then build the tree.
    size_t leaf_count = (tree.pCount + (max_prims - 1)) / max_prims;
    size_t internal_count = leaf_count - 1;
    tree.count = leaf_count + internal_count;
    tree.spheres_index =
        (_spheres_layout1 *)malloc(sizeof(_spheres_layout1) * tree.count);

    uint32_t next_node = 0;

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        uint32_t this_index = next_node++;

        if (count <= 2) {
            // Leaf node
            tree.spheres_index[this_index].nPrims = count;
            *reinterpret_cast<uint16_t *>(
                &tree.spheres_index[this_index].spheres_spliton_nPrims) = low;
            if (count == 1) {
                tree.spheres_index[this_index].center = spheres[low].s.center;
                tree.spheres_index[this_index].radius = spheres[low].s.radius;
            } else if (count == 2) {
                Sphere merged;
                bounding_sphere(merged, spheres[low].s, spheres[low + 1].s);
                tree.spheres_index[this_index].center = merged.center;
                tree.spheres_index[this_index].radius = merged.radius;
            }
        } else {
            // Internal node
            tree.spheres_index[this_index].nPrims = 0;

            vec3_float min_bound = spheres[low].s.center;
            vec3_float max_bound = spheres[low].s.center;

            for (uint32_t i = low + 1; i < high; ++i) {
                min_bound = min(min_bound, spheres[i].s.center);
                max_bound = max(max_bound, spheres[i].s.center);
            }

            // Choose axis with greatest extent
            vec3_float extent = max_bound - min_bound;
            int axis = 0;
            if (extent[1] > extent[0])
                axis = 1;
            if (extent[2] > extent[axis])
                axis = 2;

            tree.spheres_index[this_index].axis = axis;

            // Partition at midpoint along chosen axis
            auto mid_iter = spheres.begin() + low + count / 2;
            std::nth_element(
                spheres.begin() + low, mid_iter, spheres.begin() + high,
                [axis](const MaterialSphere &a, const MaterialSphere &b) {
                    return a.s.center[axis] < b.s.center[axis];
                });

            uint32_t mid = low + count / 2;

            uint32_t left = handle_range(low, mid, depth + 1);
            uint32_t right = handle_range(mid, high, depth + 1);

            // Set split offset (offset from this node to right child)
            uint32_t offset = right - this_index;
            *reinterpret_cast<uint16_t *>(
                &tree.spheres_index[this_index].spheres_spliton_nPrims) =
                offset;

            // Compute bounding volume
            Sphere merged;
            bounding_sphere(merged,
                            {tree.spheres_index[left].center,
                             tree.spheres_index[left].radius},
                            {tree.spheres_index[right].center,
                             tree.spheres_index[right].radius});

            tree.spheres_index[this_index].center = merged.center;
            tree.spheres_index[this_index].radius = merged.radius;
        }

        return this_index;
    };

    handle_range(/*low=*/0, /*high=*/tree.pCount, /*depth=*/0);
    return tree;
}

int main(int argc, char *argv[]) {
    using clock = std::chrono::high_resolution_clock;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <outfile>\n";
        return 1;
    }

    auto t0 = clock::now();

    std::vector<MaterialSphere> spheres{
        // Ground
        {Sphere{{0, -1000, 0}, 1000}, LAMBERTIAN, {0.5, 0.5, 0.5}, 0.0},

        {Sphere{{0, 1, 0}, 1}, DIALECTRIC, {0, 0, 0}, 1.5},
        {Sphere{{-4, 1, 0}, 1}, LAMBERTIAN, {0.4, 0.2, 0.1}, 0.0},
        {Sphere{{4, 1, 0}, 1}, METAL, {0.7, 0.6, 0.5}, 0.0},
    };

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_float();
            vec3_float center = {static_cast<float>(a + 0.9 * random_float()),
                                 0.2,
                                 static_cast<float>(b + 0.9 * random_float())};

            vec3_float a = {4, 0.2, 0};

            vec3_float diff = (center - a);
            float len =
                sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

            if (len > 0.9) {
                if (choose_mat < 0.8) {
                    // diffuse
                    vec3_float r0 = {random_float(), random_float(),
                                     random_float()};
                    vec3_float r1 = {random_float(), random_float(),
                                     random_float()};
                    auto albedo = r0 * r1;
                    spheres.push_back(
                        {Sphere{center, 0.2}, LAMBERTIAN, albedo, 0.0});
                } else if (choose_mat < 0.95) {
                    // metal
                    vec3_float albedo = {random_float(0.5, 1),
                                         random_float(0.5, 1),
                                         random_float(0.5, 1)};
                    float fuzz = random_float(0, 0.5);
                    spheres.push_back(
                        {Sphere{center, 0.2}, METAL, albedo, fuzz});
                } else {
                    // glass
                    spheres.push_back(
                        {Sphere{center, 0.2}, DIALECTRIC, {0, 0, 0}, 1.5});
                }
            }
        }
    }

    _spheres_layout0 tree = build_tree_simple(spheres, 1);

    Camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.width = 1423; // makes height = 800
    cam.samples_per_pixel = 50;
    cam.max_depth = 10;

    cam.vfov = 20;
    cam.lookfrom = {13, 2, 3};
    cam.lookat = {0, 0, 0};
    cam.vup = {0, 1, 0};

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    int image_width = cam.width;
    float image_height = (int)(cam.width / cam.aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto t1 = clock::now();

    // Render
    int *im = (int *)image(cam, tree);

    auto t2 = clock::now();

    const char *output_filename = argv[1];
    std::ofstream out(output_filename);

    if (!out) {
        std::cerr << "Error: Cannot open file " << output_filename
                  << " for writing\n";
        free(im);
        return 1;
    }

    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int ir = im[(j * image_width + i) * 3 + 0];
            int ig = im[(j * image_width + i) * 3 + 1];
            int ib = im[(j * image_width + i) * 3 + 2];
            out << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    auto t3 = clock::now();

    auto setup_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto render_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto write_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << "Setup time: " << setup_ms << " ms\n";
    std::cout << "Render time: " << render_ms << " ms\n";
    std::cout << "Write-to-output time: " << write_ms << " ms\n";

    free(im);
    return 0;
}
