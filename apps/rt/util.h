#include <filesystem>
#include <random>
#include <string>
#include <vector>

// Loads the object file at filename, and fills the points and trinagles arrays.
// Format is assumed to be Wavefront OBJ.
template <typename S>
bool load_object_file(const std::string &object,
                      std::vector<vector<S, 3>> &points,
                      std::vector<IndexTriangle> &triangles) {
    std::filesystem::path current_path = std::filesystem::current_path();
    while (current_path.has_parent_path()) {
        if (std::filesystem::exists(current_path / "bonsai")) {
            break;
        }
        current_path = current_path.parent_path();
    }

    std::string path = "apps/rt/objects/" + object + ".obj";
    FILE *file = fopen(path.data(), "rb");
    if (file == nullptr) {
        std::cerr << "file path: " << path << " does not exist" << std::endl;
        return false;
    }

    bool has_normal = false;
    bool has_texture = false;
    char line_buffer[2000];
    while (fgets(line_buffer, 2000, file)) {
        char *first_token = strtok(line_buffer, "\r\n\t ");
        if (!first_token || first_token[0] == '#' || first_token[0] == 0)
            continue;

        switch (first_token[0]) {
        case 'v': {
            if (first_token[1] == 'n') {
                strtok(nullptr, "\t ");
                strtok(nullptr, "\t ");
                strtok(nullptr, "\t ");
                has_normal = true;
            } else if (first_token[1] == 't') {
                strtok(nullptr, "\t ");
                strtok(nullptr, "\t ");
                has_texture = true;
            } else {
                S x = (S)atof(strtok(nullptr, "\t "));
                S y = (S)atof(strtok(nullptr, "\t "));
                S z = (S)atof(strtok(nullptr, "\t "));
                points.push_back(vector<S, 3>({x, y, z}));
            }
        } break;
        case 'f': {
            IndexTriangle tri;
            char *data[30];
            int n = 0;
            while ((data[n] = strtok(nullptr, "\t \r\n")) != nullptr) {
                if (strlen(data[n]))
                    n++;
            }

            for (int t = 0; t < (n - 2); ++t) {
                if ((!has_texture) && (!has_normal)) {
                    tri[0] = atoi(data[0]) - 1;
                    tri[1] = atoi(data[1]) - 1;
                    tri[2] = atoi(data[2]) - 1;
                } else {
                    const char *v1;
                    for (int i = 0; i < 3; i++) {
                        // vertex ID
                        if (i == 0)
                            v1 = data[0];
                        else
                            v1 = data[t + i];

                        tri[i] = atoi(v1) - 1;
                    }
                }
                triangles.push_back(tri);
            }
        }
        }
    }
    return true;
}

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
