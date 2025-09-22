#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<Ray> load_rays_binary(const std::string &filename,
                                  int64_t ray_count) {
    std::vector<Ray> rays;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename
                  << " for reading\n";
        return rays;
    }

    // Read number of rays
    size_t count;
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (ray_count > count) {
        std::cout << "the requested ray count: " << ray_count
                  << " is greater than the total ray count: " << count
                  << " You need to re-generate the rays.";
    }
    assert(ray_count <= count);

    rays.reserve(ray_count);

    // Read ray data
    for (size_t i = 0; i < ray_count; ++i) {
        Ray ray;
        file.read(reinterpret_cast<char *>(&ray.o[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.o[2]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&ray.d[2]), sizeof(float));
        rays.push_back(ray);
    }

    file.close();
    return rays;
}
