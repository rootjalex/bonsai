#include "canonical_tree.h"
#include "helpers.h"
#include "rtiow.h"
#include "util.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// ---------------------------
// RTIOW cuda main hook
// ---------------------------

int main(int argc, char **argv) {
    using clock = std::chrono::high_resolution_clock;
    assert(argc == 2);
    std::string output_filename = argv[1];

    const std::vector<MaterialSphere> spheres = setup_spheres();

    std::cout << "-- building canonical tree" << std::endl;
    auto ct_begin = clock::now();
    BVH *node = build_canonical_tree(spheres);
    auto ct_end = clock::now();

    std::cout << "-- building specialized tree" << std::endl;
    auto st_begin = clock::now();
    Spheres tree;
    build_spheres(&tree, node);
    auto st_end = clock::now();

    free_canonical_tree(node);

    Camera camera = setup_camera();

    int image_width = camera.width;
    float image_height = (int)(camera.width / camera.aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto t1 = clock::now();

    std::cout << "-- rendering image" << std::endl;
    int *im = (int *)image(&camera, &tree);

    auto t2 = clock::now();

    if (save_image(image_height, image_width, im, output_filename)) {
        cudaFree(im);
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
