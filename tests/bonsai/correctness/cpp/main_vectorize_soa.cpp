#include "vectorize_soa.h"

#include <iostream>

// Used in vectorize_soa.bonsai
namespace {

void show(const std::array<float, 8> &v) {
    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0) {
            std::cout << ' ';
        }
        std::cout << v[i];
    }
    std::cout << '\n';
}

} // namespace

int main() {
    // The generated code reads this array one component at a time, at
    // whatever stride the backend says a float3 occupies. That has to be the
    // same stride the C++ side lays the array out at, or the two disagree
    // about where lane i's data is -- it is four floats, not three, because
    // a three-float vector is padded.
    static_assert(sizeof(float3) == 4 * sizeof(float),
                  "generated code and this driver must agree on the stride");

    // Distinct values everywhere, so reading the wrong stride or component
    // cannot coincidentally agree.
    std::array<float3, 8> pts;
    for (size_t i = 0; i < pts.size(); i++) {
        pts[i] = float3{float(i), float(10 * i), float(100 * i)};
    }

    std::array<float, 8> out{};

    // (i + 10i + 100i) * 2 = 222i
    scale_each(pts, 2, out);
    show(out);

    // Squared distance from (1, 1, 1).
    distance_from(pts, float3{1, 1, 1}, out);
    show(out);

    std::array<float3, 8> other;
    for (size_t i = 0; i < other.size(); i++) {
        // Deliberately smaller in some components and larger in others, so
        // the per-component select has to disagree within one lane.
        other[i] = float3{float(100 - i), float(5 * i), float(50)};
    }
    pick_vector(pts, other, out);
    show(out);
}
