// clang++ -std=c++20 -O3 apps/rtiow/main_hook.cpp apps/rtiow/main.o -o
// apps/rtiow/main_runner
// ./main_runner &> bonsai_image.ppm
#include <iostream>

#include "main.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height>\n";
        return 1;
    }

    // Parse width and height from input strings
    int image_width = std::stoi(argv[1]);
    int image_height = std::stoi(argv[2]);

    // Render
    int *im = (int *)image(image_width, image_height);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int ir = im[(j * image_width + i) * 3 + 0];
            int ig = im[(j * image_width + i) * 3 + 1];
            int ib = im[(j * image_width + i) * 3 + 2];
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}
