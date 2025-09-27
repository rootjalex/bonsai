#include "helpers.h"
#include "rtiow.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace {

__host__ inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

__host__ inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

__host__ Camera setup_camera() {
    Camera camera;
    camera.aspect_ratio = 16.0 / 9.0;
    camera.width = 1200; // makes height = 675
    camera.samples_per_pixel = 50;
    camera.max_depth = 20;

    camera.vfov = 20;
    camera.lookfrom = {13, 2, 3};
    camera.lookat = {0, 0, 0};
    camera.vup = {0, 1, 0};

    camera.defocus_angle = 0.6;
    camera.focus_dist = 10.0;
    return camera;
}

__host__ std::vector<MaterialSphere> setup_spheres() {
    constexpr uint32_t LAMBERTIAN = 0;
    constexpr uint32_t METAL = 1;
    constexpr uint32_t DIALECTRIC = 2;

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
            float3 center = {static_cast<float>(a + 0.9 * random_float()), 0.2,
                             static_cast<float>(b + 0.9 * random_float())};

            float3 a = {4, 0.2, 0};

            float3 diff = (center - a);
            float len =
                sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

            if (len > 0.9) {
                if (choose_mat < 0.8) {
                    // diffuse
                    float3 r0 = {random_float(), random_float(),
                                 random_float()};
                    float3 r1 = {random_float(), random_float(),
                                 random_float()};
                    auto albedo = r0 * r1;
                    spheres.push_back(
                        {Sphere{center, 0.2}, LAMBERTIAN, albedo, 0.0});
                } else if (choose_mat < 0.95) {
                    // metal
                    float3 albedo = {random_float(0.5, 1), random_float(0.5, 1),
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
    return spheres;
}

__host__ int save_image(int64_t image_height, int64_t image_width,
                        const int *image, const std::string &output_filename) {
    std::ofstream out(output_filename);

    if (!out) {
        std::cerr << "Error: Cannot open output file: `" << output_filename
                  << "` for writing\n";
        return 1;
    }

    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int ir = image[(j * image_width + i) * 3 + 0];
            int ig = image[(j * image_width + i) * 3 + 1];
            int ib = image[(j * image_width + i) * 3 + 2];
            out << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    return 0;
}

__host__ Sphere get_bounding_sphere(const BVH *node) {
    if (std::holds_alternative<Interior>(*node)) {
        const Interior &interior = std::get<Interior>(*node);
        return {interior.center, interior.radius};
    }
    if (std::holds_alternative<Leaf>(*node)) {
        const Leaf &leaf = std::get<Leaf>(*node);
        return {leaf.center, leaf.radius};
    }
    assert(false && "unexpected");
}

__host__ void free_canonical_tree(BVH *node) {
    if (std::holds_alternative<Interior>(*node)) {
        Interior &interior = std::get<Interior>(*node);
        free_canonical_tree(interior.left);
        free_canonical_tree(interior.right);
        cudaFree(&interior);
        return;
    }

    if (std::holds_alternative<Leaf>(*node)) {
        Leaf &leaf = std::get<Leaf>(*node);
        cudaFree(leaf.data);
        cudaFree(&leaf);
        return;
    }

    assert(false && "unexpected");
}

// Builds the canonical tree using a median split.
__host__ BVH *build_canonical_tree(std::vector<MaterialSphere> &spheres) {
    constexpr uint32_t MAX_TREE_DEPTH = 64;
    std::function<BVH *(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth = 0) -> BVH * {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        if (count <= 2) {
            float3 center = spheres[low].s.center;
            float radius = spheres[low].s.radius;
            if (count == 2) {
                Sphere merged =
                    bounding_sphere(&spheres[low].s, &spheres[low + 1].s);
                center = merged.center;
                radius = merged.radius;
            }
            auto *data =
                (MaterialSphere *)(malloc(sizeof(MaterialSphere) * count));
            for (int i = 0; i < count; ++i) {
                data[i] = spheres[low + i];
            }
            return new BVH(Leaf{
                .center = center,
                .radius = radius,
                .nprims = static_cast<uint8_t>(count),
                .data = data,
            });
        }

        // Internal node
        float3 min_bound = spheres[low].s.center;
        float3 max_bound = spheres[low].s.center;

        for (uint32_t i = low + 1; i < high; ++i) {
            min_bound = min(min_bound, spheres[i].s.center);
            max_bound = max(max_bound, spheres[i].s.center);
        }

        // Choose axis with greatest extent
        float3 extent = max_bound - min_bound;
        int axis = 0;
        float ex = extent.x;
        float ey = extent.y;
        float ez = extent.z;

        if (ey > ex)
            axis = 1;
        if (ez > ((axis == 0) ? ex : ey))
            axis = 2;

        // Partition at midpoint along chosen axis
        auto mid_it = spheres.begin() + low + count / 2;
        std::nth_element(spheres.begin() + low, mid_it, spheres.begin() + high,
                         [&](const MaterialSphere &a, const MaterialSphere &b) {
                             if (axis == 0)
                                 return a.s.center.x < b.s.center.x;
                             if (axis == 1)
                                 return a.s.center.y < b.s.center.y;
                             return a.s.center.z < b.s.center.z;
                         });

        const uint32_t mid = low + count / 2;
        BVH *left = partition(low, mid, depth + 1);
        BVH *right = partition(mid, high, depth + 1);

        // Compute bounding volume
        Sphere a = get_bounding_sphere(left), b = get_bounding_sphere(right);
        Sphere merged = bounding_sphere(&a, &b);

        return new BVH(Interior{
            .center = merged.center,
            .radius = merged.radius,
            .left = left,
            .right = right,
        });
    };

    return partition(/*low=*/0, /*high=*/spheres.size(), /*depth=*/0);
}

__host__ BVH *copy_to_device(BVH *node) {
    if (node == nullptr) {
        return nullptr;
    }

    BVH *root;
    cudaMalloc(&root, sizeof(BVH));

    if (std::holds_alternative<Interior>(*node)) {
        const Interior &h_interior = std::get<Interior>(*node);
        Interior temp_interior = h_interior;
        temp_interior.left = copy_to_device(h_interior.left);
        temp_interior.right = copy_to_device(h_interior.right);
        BVH temp_bvh_variant = temp_interior;
        cudaMemcpy(root, &temp_bvh_variant, sizeof(BVH),
                   cudaMemcpyHostToDevice);
    } else if (std::holds_alternative<Leaf>(*node)) {
        const Leaf &h_leaf = std::get<Leaf>(*node);
        Leaf temp_leaf = h_leaf;
        if (h_leaf.nprims > 0 && h_leaf.data != nullptr) {
            size_t data_size = h_leaf.nprims * sizeof(MaterialSphere);
            cudaMalloc(&temp_leaf.data, data_size);
            cudaMemcpy(temp_leaf.data, h_leaf.data, data_size,
                       cudaMemcpyHostToDevice);
        } else {
            temp_leaf.data = nullptr;
        }
        BVH temp_bvh_variant = temp_leaf;
        cudaMemcpy(root, &temp_bvh_variant, sizeof(BVH),
                   cudaMemcpyHostToDevice);
    }
    free(node);
    return root;
}

} // namespace

// ---------------------------
// RTIOW cuda main hook
// ---------------------------

int main(int argc, char **argv) {
    using clock = std::chrono::high_resolution_clock;
    assert(argc == 2);
    std::string output_filename = argv[1];

    std::vector<MaterialSphere> spheres = setup_spheres();

    std::cout << "-- building canonical tree" << std::endl;
    auto ct_begin = clock::now();
    BVH *node = build_canonical_tree(spheres);
    node = copy_to_device(node);
    auto ct_end = clock::now();

    std::cout << "-- building specialized tree" << std::endl;
    auto st_begin = clock::now();
    Spheres tree = build_spheres(node);
    auto st_end = clock::now();

    free_canonical_tree(node);
    std::cout << "-- canonical tree free" << std::endl;

    Camera camera = setup_camera();
    std::cout << "-- camera set up" << std::endl;

    int image_width = camera.width;
    float image_height = (int)(camera.width / camera.aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto t1 = clock::now();

    std::cout << "-- rendering image" << std::endl;
    int *im = (int *)image(&camera, &tree);

    auto t2 = clock::now();

    if (save_image(image_height, image_width, im, output_filename)) {
        cudaFree(im);
        std::cout << "save image failed!\n";
        return 1;
    }

    auto t3 = clock::now();
    auto ct_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(ct_end - ct_begin)
            .count();
    auto st_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(st_end - st_begin)
            .count();
    auto render_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto write_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << "canonical tree   : " << ct_time << "ms\n";
    std::cout << "specialized tree : " << st_time << "ms\n";
    std::cout << "render time      : " << render_ms << " ms\n";
    std::cout << "write time       : " << write_ms << " ms\n";

    cudaFree(im);
    return 0;
}
