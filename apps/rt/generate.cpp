#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>

// Utilities for generating random rays used in ray-tracing queries.

struct vec3_float {
    float x, y, z;
    vec3_float(std::initializer_list<float> list) {
        auto it = list.begin();
        x = *it++;
        y = *it++;
        z = *it;
    }
    vec3_float() : x(0), y(0), z(0) {}
};

struct Ray {
    vec3_float o; // origin
    vec3_float d; // direction
};

inline float random_float(std::mt19937 &generator, float min = -1.0,
                          float max = 1.0) {
    std::uniform_real_distribution<float> urd(min, max);
    return urd(generator);
}

vec3_float random_unit_vector(std::mt19937 &generator) {
    float x = random_float(generator);
    float y = random_float(generator);
    float z = random_float(generator);

    // Normalize the vector
    float length = std::sqrt(x * x + y * y + z * z);
    if (length > 0) {
        x /= length;
        y /= length;
        z /= length;
    }

    return vec3_float({x, y, z});
}

std::vector<Ray> generate_random_rays(int count, float origin_range = 10.0f) {
    std::vector<Ray> rays;
    rays.reserve(count);

    std::random_device rd;
    std::mt19937 generator(rd());

    for (int i = 0; i < count; ++i) {
        Ray ray;

        ray.o = vec3_float({
            random_float(generator, -origin_range, origin_range),
            random_float(generator, -origin_range, origin_range),
            random_float(generator, -origin_range, origin_range),
        });

        ray.d = random_unit_vector(generator);

        rays.push_back(ray);
    }

    return rays;
}

bool save_rays_binary(const std::vector<Ray> &rays,
                      const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: could not open file " << filename
                  << " for writing\n";
        return false;
    }

    // Write number of rays
    size_t count = rays.size();
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));

    // Write ray data
    for (const auto &ray : rays) {
        file.write(reinterpret_cast<const char *>(&ray.o.x), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.o.y), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.o.z), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.d.x), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.d.y), sizeof(float));
        file.write(reinterpret_cast<const char *>(&ray.d.z), sizeof(float));
    }

    file.close();
    std::cout << "saved " << count << " rays to " << filename << std::endl;
    return true;
}

int main(int argc, char *argv[]) {
    assert(argc == 3);
    std::string filename = argv[1];
    int64_t count = std::stoi(argv[2]);
    std::vector<Ray> rays = generate_random_rays(count);
    save_rays_binary(rays, filename);
}