#include "vectorize_gather_union.h"

#include <iostream>

// Used in vectorize_gather_union.bonsai.
//
// Twelve variants, four of each kind in a scrambled order and with distinct
// payloads, eight lanes reading them in an order that mixes the kinds within
// one gang, one lane past the end. Built through the generated constructors,
// so the tag numbering stays the compiler's. Each lane must match the scalar
// pick.
int main() {
    std::array<Blob, 12> blobs;
    for (int i = 0; i < 12; i++) {
        switch (i % 3) {
        case 0:
            Blob_Whole(blobs[i], 1.5f * float(i + 1));
            break;
        case 1:
            Blob_Split(blobs[i], i, 10 * i);
            break;
        default:
            Blob_Wide(blobs[i], float3{float(i), 2.0f * float(i), -0.5f},
                      (i % 2) == 0);
            break;
        }
    }
    const std::array<int32_t, 8> idx = {0, 1, 2, 11, 7, 4, 50, 9};
    std::array<float, 8> out{};
    pick_all(blobs, idx, out);
    bool same = true;
    for (size_t l = 0; l < idx.size(); l++) {
        same = same && pick_one(blobs, idx[l]) == out[l];
        if (l != 0) {
            std::cout << ' ';
        }
        std::cout << out[l];
    }
    std::cout << '\n' << (same ? "same as scalar" : "DIFFERS from scalar")
              << '\n';
}
