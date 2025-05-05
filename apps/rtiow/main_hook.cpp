// clang++ -std=c++20 -O3 apps/rtiow/main_hook.cpp apps/rtiow/main.o -o
// apps/rtiow/main_runner
// ./main_runner &> bonsai_image.ppm
#include <functional>
#include <iostream>
#include <vector>

#include "main.h"

constexpr uint32_t LAMBERTIAN = 0;
constexpr uint32_t METAL = 1;

_spheres_layout1 build_tree_simple(std::vector<MaterialSphere> &spheres,
                                   size_t max_prims) {
    _spheres_layout1 tree;
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
        (_spheres_layout0 *)malloc(sizeof(_spheres_layout0) * tree.count);

    uint32_t next_node = 0;

    std::function<uint32_t(uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high) -> uint32_t {
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

            uint32_t mid = low + count / 2;

            uint32_t left = handle_range(low, mid);
            uint32_t right = handle_range(mid, high);

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

    handle_range(0, tree.pCount);
    return tree;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <width>\n";
        return 1;
    }

    // Parse width and height from input strings
    int image_width = std::stoi(argv[1]);
    float aspect_ratio = 16.0 / 9.0;
    float image_height = (int)(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    // int image_height = std::stoi(argv[2]);

    std::vector<MaterialSphere> spheres{
        // Ground
        {Sphere{{0, -100.5, -1}, 100}, LAMBERTIAN, {0.8, 0.8, 0.0}},
        // Center
        {Sphere{{0.0, 0.0, -1.2}, 0.5}, LAMBERTIAN, {0.1, 0.2, 0.5}},
        // Left
        {Sphere{{-1.0, 0.0, -1.0}, 0.5}, METAL, {0.8, 0.8, 0.8}},
        // Right
        {Sphere{{1.0, 0.0, -1.0}, 0.5}, METAL, {0.8, 0.6, 0.2}},
    };

    _spheres_layout1 tree = build_tree_simple(spheres, 1);

    // Render
    int *im = (int *)image(image_width, tree);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int ir = im[(j * image_width + i) * 3 + 0];
            int ig = im[(j * image_width + i) * 3 + 1];
            int ib = im[(j * image_width + i) * 3 + 2];
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    free(im);
}
