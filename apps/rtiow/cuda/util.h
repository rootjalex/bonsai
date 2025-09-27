#pragma once

#include "cuda/rtiow.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

__host__ float random_float();

#ifndef __CUDA_ARCH__
__host__ inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}
#endif // __CUDA_ARCH__

__host__ float random_float(float min, float max);
#ifndef __CUDA_ARCH__
__host__ inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}
#endif // __CUDA_ARCH__

__host__ Camera setup_camera();
#ifndef __CUDA_ARCH__
__host__ inlineCamera setup_camera() {
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
#endif // __CUDA_ARCH__

__host__ std::vector<MaterialSphere> setup_spheres();
#ifndef __CUDA_ARCH__
__host__ inline std::vector<MaterialSphere> setup_spheres() {
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
            vec3_float center = {static_cast<float>(a + 0.9 * random_float()),
                                 0.2,
                                 static_cast<float>(b + 0.9 * random_float())};

            vec3_float a = {4, 0.2, 0};

            vec3_float diff = (center - a);
            float len =
                sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

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
    return spheres;
}
#endif // __CUDA_ARCH__

__host__ int save_image(int64_t image_height, int64_t image_width,
                        const int *image, const std::string &output_filename);
#ifndef __CUDA_ARCH__
__host__ inline int save_image(int64_t image_height, int64_t image_width,
                               const int *image,
                               const std::string &output_filename) {
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
#endif // __CUDA_ARCH__