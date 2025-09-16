#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<Ray> load_rays_binary(const std::string &filename) {
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

    rays.reserve(count);

    // Read ray data
    for (size_t i = 0; i < count; ++i) {
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
