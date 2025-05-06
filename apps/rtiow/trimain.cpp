// clang++ -std=c++20 -O3 trimain.cpp ../../deps/tinyply/source/tinyply.cpp -o
// trimain.out
#include "../../deps/tinyply/source/tinyply.h"
#include <array>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <infile>\n";
        return 1;
    }

    std::string filename(argv[1]);

    std::ifstream ss(filename, std::ios::binary);
    if (!ss) {
        std::cerr << "Failed to open PLY file\n";
        return 1;
    }

    tinyply::PlyFile file;
    file.parse_header(ss);

    std::shared_ptr<tinyply::PlyData> vertices, faces;

    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
    faces = file.request_properties_from_element(
        "face", {"vertex_indices"}); // could also be "vertex_index"

    file.read(ss);

    // Load vertices
    std::vector<std::array<float, 3>> vertex_list;
    if (vertices && vertices->count > 0) {
        const float *verts = reinterpret_cast<float *>(vertices->buffer.get());
        for (size_t i = 0; i < vertices->count; i++) {
            vertex_list.push_back(
                {verts[i * 3 + 0], verts[i * 3 + 1], verts[i * 3 + 2]});
        }
    }

    // Load triangle faces
    std::vector<std::array<int, 3>> triangle_list;
    if (faces && faces->count > 0) {
        const uint8_t *ptr = faces->buffer.get();
        for (size_t i = 0; i < faces->count; i++) {
            uint8_t vertex_count = *ptr; // assume uint8_t count for list size
            ptr += 1;
            if (vertex_count == 3) {
                std::array<int, 3> tri;
                for (int j = 0; j < 3; ++j) {
                    tri[j] = *reinterpret_cast<const uint32_t *>(ptr);
                    ptr += sizeof(uint32_t);
                }
                triangle_list.push_back(tri);
            } else {
                // Skip non-triangle faces
                ptr += vertex_count * sizeof(uint32_t);
            }
        }
    }

    std::cout << "Loaded " << vertex_list.size() << " vertices and "
              << triangle_list.size() << " triangles\n";

    return 0;
}
