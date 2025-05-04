// clang++ -std=c++20 -O3 apps/rtiow/main_hook.cpp apps/rtiow/main.o -o
// apps/rtiow/main_runner
// ./main_runner &> bonsai_image.ppm
#include <iostream>

#include "main.h"

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

    _spheres_layout1 tree;
    tree.pCount = 2;
    tree.prims = (Sphere *)malloc(sizeof(Sphere) * tree.pCount);
    // world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    tree.prims[0].center[0] = 0;
    tree.prims[0].center[1] = 0;
    tree.prims[0].center[2] = -1;
    tree.prims[0].radius = 0.5;
    // world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));
    tree.prims[1].center[0] = 0;
    tree.prims[1].center[1] = -100.5;
    tree.prims[1].center[2] = -1;
    tree.prims[1].radius = 100;

    tree.count = 3;
    tree.spheres_index =
        (_spheres_layout0 *)malloc(tree.count * sizeof(_spheres_layout0));
    // Parent
    {
        tree.spheres_index[0].nPrims = 0;
        // second child offset offset = 0
        *reinterpret_cast<uint16_t *>(
            &tree.spheres_index[0].spheres_spliton_nPrims) = 2;

        Sphere root_bv;
        bounding_sphere(root_bv, tree.prims[0], tree.prims[1]);

        tree.spheres_index[0].center = root_bv.center;
        tree.spheres_index[0].radius = root_bv.radius;
    }

    // Left child
    {
        tree.spheres_index[1].nPrims = 1;
        *reinterpret_cast<uint16_t *>(
            &tree.spheres_index[1].spheres_spliton_nPrims) = 0;
        tree.spheres_index[1].center = tree.prims[0].center;
        tree.spheres_index[1].radius = tree.prims[0].radius;
    }

    // Right child
    {
        tree.spheres_index[2].nPrims = 1;
        *reinterpret_cast<uint16_t *>(
            &tree.spheres_index[2].spheres_spliton_nPrims) = 1;
        tree.spheres_index[2].center = tree.prims[1].center;
        tree.spheres_index[2].radius = tree.prims[1].radius;
    }

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
