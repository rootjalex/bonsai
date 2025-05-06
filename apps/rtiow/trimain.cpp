// clang++ -std=c++20 -O3 trimain.cpp ../../deps/tinyply/source/tinyply.cpp -o
// trimain.out
// #include "../../deps/tinyply/source/tinyply.h"

#include "trimain.h"

#include <array>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <outfile>\n";
        return 1;
    }

    // Define one triangle
    MaterialTriangle tri;
    tri.t.p0 = {0.0f, 0.0f, 0.0f};
    tri.t.p1 = {1.0f, 0.0f, 0.0f};
    tri.t.p2 = {0.0f, 1.0f, 0.0f};
    tri.material = 0; // Lambertian
    tri.albedo = {0.5, 0.5, 0.5};
    tri.fuzz = 0.0;

    AABB aabb;
    bounding_box(aabb, tri.t);

    _triangles_layout1 tree;
    tree.pCount = 1;
    tree.prims = &tri;

    tree.count = 0;
    tree.triangles_index =
        (_triangles_layout0 *)malloc(sizeof(_triangles_layout0) * tree.count);
    tree.triangles_index[0].low = aabb.low;
    tree.triangles_index[0].high = aabb.high;
    tree.triangles_index[0].nPrims = 1;
    *reinterpret_cast<uint16_t *>(
        &tree.triangles_index[0].triangles_spliton_nPrims) = 0;

    Camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.width = 400;
    cam.samples_per_pixel = 10;
    cam.max_depth = 10;

    cam.vfov = 20;
    cam.lookfrom = {0.5, 0.5, 10};
    cam.lookat = {0.5, 0.5, 0};
    cam.vup = {0, 1, 0};

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    int image_width = cam.width;
    float image_height = (int)(cam.width / cam.aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Render
    int *im = (int *)image(cam, tree);

    const char *output_filename = argv[1];
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

    return 0;
}

/*
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
*/
