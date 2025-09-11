// clang++ -std=c++20 -O3 apps/rtiow/main_hook.cpp apps/rtiow/main.o -o
// apps/rtiow/main_runner
// ./main_runner &> bonsai_image.ppm
#include <chrono>
#include <fstream> // for std::ofstream
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "rtiow.h"
#include <cassert>

constexpr uint32_t LAMBERTIAN = 0;
constexpr uint32_t METAL = 1;
constexpr uint32_t DIALECTRIC = 2;

constexpr uint32_t MAX_TREE_DEPTH = 64;

constexpr double PI = 3.1415926535897932385;

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

struct Interior;
struct Leaf;

using BVH = std::variant<Interior *, Leaf *>;
using f32 = float;

struct Interior {
    vec3_float center;
    f32 radius;
    BVH left;
    BVH right;
};

struct Leaf {
    vec3_float center;
    f32 radius;
    uint16_t nprims;
    MaterialSphere *data;
};

Sphere get_bounding_sphere(const BVH &bvh) {
    if (std::holds_alternative<Interior *>(bvh)) {
        const Interior *interior = std::get<Interior *>(bvh);
        return {interior->center, interior->radius};
    }
    if (std::holds_alternative<Leaf *>(bvh)) {
        const Leaf *leaf = std::get<Leaf *>(bvh);
        return {leaf->center, leaf->radius};
    }
    assert(false && "unexpected");
}

void free_canonical_tree(BVH node) {
    if (std::holds_alternative<Interior *>(node)) {
        Interior *interior = std::get<Interior *>(node);
        free_canonical_tree(interior->left);
        free_canonical_tree(interior->right);
        free(interior);
        return;
    }

    if (std::holds_alternative<Leaf *>(node)) {
        Leaf *leaf = std::get<Leaf *>(node);
        free(leaf->data);
        free(leaf);
        return;
    }

    assert(false && "unexpected");
}

BVH build_canonical_tree(std::vector<MaterialSphere> &spheres) {
    std::function<BVH(uint32_t, uint32_t, uint32_t)> partition =
        [&](uint32_t low, uint32_t high, uint32_t depth = 0) -> BVH {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        if (count <= 2) {
            vec3_float center = spheres[low].s.center;
            f32 radius = spheres[low].s.radius;
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
            return new Leaf{
                .center = center,
                .radius = radius,
                .nprims = (uint16_t)count,
                .data = data,
            };
        }

        // Internal node
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

        // Partition at midpoint along chosen axis
        auto mid_it = spheres.begin() + low + count / 2;
        std::nth_element(spheres.begin() + low, mid_it, spheres.begin() + high,
                         [&](const MaterialSphere &a, const MaterialSphere &b) {
                             return a.s.center[axis] < b.s.center[axis];
                         });

        const uint32_t mid = low + count / 2;
        BVH left = partition(low, mid, depth + 1);
        BVH right = partition(mid, high, depth + 1);

        // Compute bounding volume
        Sphere a = get_bounding_sphere(left), b = get_bounding_sphere(right);
        Sphere merged = bounding_sphere(&a, &b);

        return new Interior{
            .center = merged.center,
            .radius = merged.radius,
            .left = left,
            .right = right,
        };
    };

    return partition(/*low=*/0, /*high=*/spheres.size(), /*depth=*/0);
}

int main(int argc, char *argv[]) {
    using clock = std::chrono::high_resolution_clock;
    std::string output_filename;
    if (argc == 2) {
        output_filename = argv[1];
    } else {
        output_filename = "rtiow-cpu-image.ppm";
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

    std::cerr << "building canonical tree" << std::endl;
    BVH node = build_canonical_tree(spheres);
    std::cerr << "building specialized tree" << std::endl;
    Spheres tree = build_spheres(node);

    free_canonical_tree(node);

    Camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.width = 1200; // makes height = 675
    cam.samples_per_pixel = 50;
    cam.max_depth = 20;

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
    int *im = (int *)image(&cam, tree);

    auto t2 = clock::now();

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
    // TODO(cgyurgyik): free the specialized tree.
    return 0;
}
