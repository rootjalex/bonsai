#include "bonsai_cpp.h"
#include "canonical_tree.h"
#include "rt.h"
#include "util.h"

#include <chrono>
#include <iostream>
#include <random>

namespace {

// Runs an collision detection test on the two OBJ files for Bonsai and FCL.
template <typename S>
void run_test(const std::string &object) {
    if constexpr (!(std::is_floating_point_v<S> && sizeof(S) == 4)) {
        std::cerr << "the bonsai kernel currently assumes f32 input";
        exit(-1);
    }

    using clock = std::chrono::high_resolution_clock;
    std::vector<vector<S, 3>> vertices;
    std::vector<IndexTriangle> triangles;
    if (!load_object_file(object, vertices, triangles)) {
        exit(-1);
    }

    auto ct_begin = clock::now();
    BVH *canonical_tree = build_canonical_tree<S>(vertices, triangles);
    auto ct_end = clock::now();

    Triangles tree = build_triangles(canonical_tree);

    auto st_begin = clock::now();
    free_canonical_tree(canonical_tree);
    auto st_end = clock::now();

    std::vector<Ray> rays = generate_random_rays(4096);
    std::vector<Triangle> hits;
    auto trace_begin = clock::now();
    for (const Ray &ray : rays) {
        if (std::optional<Triangle> tri = trace(&ray, &tree)) {
            hits.push_back(*tri);
        }
    }
    auto trace_end = clock::now();

    auto ct_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(ct_end - ct_begin)
            .count();
    auto st_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(st_end - st_begin)
            .count();
    auto trace_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          trace_end - trace_begin)
                          .count();
    std::cout << "hits             : " << hits.size() << "\n";
    std::cout << "canonical tree   : " << ct_time << "ms\n";
    std::cout << "specialized tree : " << st_time << "ms\n";
    std::cout << "trace time       : " << trace_time << " ms\n";
    std::cout << "--------------------------------\n";
}

} // namespace

int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::string object = argv[1];
    // Wavefront OBJ files are taken from FCL [1] from `prims` [2].
    // [1] https://github.com/flexible-collision-library/fcl
    // [2] https://github.com/nickdesaulniers/prims/tree/master/meshes
    run_test<float>(object);
    return 0;
}
