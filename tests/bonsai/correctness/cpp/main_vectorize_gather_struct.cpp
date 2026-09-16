#include "vectorize_gather_struct.h"

#include <iostream>

// Used in vectorize_gather_struct.bonsai.
//
// Sixteen vertices with every field distinct, eight lanes reading them in a
// scrambled order with repeats, and two lanes whose index is past the end --
// which the branch keeps from being read at all, masked or not. Each lane
// must match the scalar pick.
int main() {
    std::array<Vertex, 16> verts;
    for (int i = 0; i < 16; i++) {
        verts[i].p = float3{float(i), float(10 * i), float(-i)};
        verts[i].w = 0.5f * float(i + 1);
        verts[i].id = 1000 + i;
        verts[i].flag = (i % 3) == 0;
    }
    const std::array<int32_t, 8> idx = {5, 0, 15, 5, 99, 7, 2, 40};
    std::array<float, 8> out{};
    pick_all(verts, idx, out);
    bool same = true;
    for (size_t l = 0; l < idx.size(); l++) {
        same = same && pick_one(verts, idx[l]) == out[l];
        if (l != 0) {
            std::cout << ' ';
        }
        std::cout << out[l];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
}
