#include "vectorize_visect.h"

#include <cmath>
#include <iostream>
#include <limits>

// Used in vectorize_visect.bonsai
namespace {

// The same slab test, written scalar, as the reference the gang is checked
// against. It follows the bonsai source operation for operation so that the
// two agree exactly rather than approximately.
float gamma_of(int n) {
    const float e = std::numeric_limits<float>::epsilon();
    return (float(n) * e) / (1 - float(n) * e);
}

bool intersects(const float3 &o, const float3 &d, const AABB &b) {
    const float3 inv_dir = float3{1, 1, 1} / d;

    float t_min[3];
    float t_max[3];
    for (int k = 0; k < 3; k++) {
        const bool neg = inv_dir[k] < 0;
        const float low = neg ? b.high[k] : b.low[k];
        const float high = neg ? b.low[k] : b.high[k];
        t_min[k] = (low - o[k]) * inv_dir[k];
        t_max[k] = (high - o[k]) * inv_dir[k] * (1 + 2 * gamma_of(3));
    }

    if (t_min[0] > t_max[1] || t_min[1] > t_max[0]) {
        return false;
    }
    float tmin = std::fmax(t_min[0], t_min[1]);
    float tmax = std::fmin(t_max[0], t_max[1]);
    if (tmin > t_max[2] || t_min[2] > tmax) {
        return false;
    }
    tmin = std::fmax(tmin, t_min[2]);
    tmax = std::fmin(tmax, t_max[2]);
    return tmin <= tmax;
}

} // namespace

int main() {
    AABB box;
    box.low = float3{-1, -1, -1};
    box.high = float3{1, 1, 1};

    // A mix of rays: through the box from several directions, aimed away,
    // aimed past a corner, and starting inside. Directions avoid zero
    // components, whose infinities would make the comparison order matter.
    // Half hit and half miss, so neither an inverted mask nor a lane read
    // from the wrong offset can agree with the reference by accident.
    std::array<float3, 8> origins = {
        float3{-5, 0, 0},    float3{-5, 3, 0},   float3{0, 0, -5},
        float3{0, 5, 0},     float3{-5, -5, -5}, float3{0, 0, 0},
        float3{-5, 0, 4},    float3{3, -4, 2},
    };
    std::array<float3, 8> directions = {
        float3{1, 0.01f, 0.01f},   float3{1, 0.01f, 0.01f},
        float3{0.01f, 0.01f, 1},   float3{0.01f, 1, 0.01f},
        float3{1, 1, 1},           float3{1, 1, 1},
        float3{1, 0.01f, 0.01f},   float3{-1, 1, -1},
    };

    std::array<bool, 8> got;
    got.fill(false);
    visect_box(box, origins, directions, got);

    bool ok = true;
    for (size_t i = 0; i < got.size(); i++) {
        const bool want = intersects(origins[i], directions[i], box);
        std::cout << (got[i] ? 1 : 0);
        if (got[i] != want) {
            ok = false;
        }
    }
    std::cout << '\n' << (ok ? "matches scalar" : "DIFFERS from scalar") << '\n';
}
