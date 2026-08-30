#include "mut-dynamic-array.h"

#include <cstdlib>
#include <iostream>
#include <vector>

// Used in mut-dynamic-array.bonsai.
//
// The pointers are deliberately heap-allocated and their values checked after
// the call as well as their contents: the bug this guards against overwrote
// the caller's pointer variable with the first element, so a test that only
// looked at the data through a stale copy of the pointer could still pass.
int main() {
    const uint32_t n = 8;

    int32_t *ints = (int32_t *)malloc(sizeof(int32_t) * n);
    int32_t *ints_before = ints;
    fill(n, ints);

    std::vector<float> in(n);
    for (uint32_t i = 0; i < n; i++) {
        in[i] = float(i);
    }
    float *scaled = (float *)malloc(sizeof(float) * n);
    float *scaled_before = scaled;
    scale(n, 0.5f, in.data(), scaled);

    float3 *vecs = (float3 *)malloc(sizeof(float3) * n);
    float3 *vecs_before = vecs;
    splat(n, vecs);

    // The same three again, through parameters with no size on the type.
    int32_t *u_ints = (int32_t *)malloc(sizeof(int32_t) * n);
    int32_t *u_ints_before = u_ints;
    fill_unsized(n, u_ints);

    float *u_scaled = (float *)malloc(sizeof(float) * n);
    float *u_scaled_before = u_scaled;
    copy_unsized(n, in.data(), u_scaled);

    float3 *u_vecs = (float3 *)malloc(sizeof(float3) * n);
    float3 *u_vecs_before = u_vecs;
    splat_unsized(n, u_vecs);

    size_t wrong = 0;
    wrong += (ints != ints_before) + (scaled != scaled_before) +
             (vecs != vecs_before);
    wrong += (u_ints != u_ints_before) + (u_scaled != u_scaled_before) +
             (u_vecs != u_vecs_before);
    for (uint32_t i = 0; i < n; i++) {
        wrong += ints[i] != int32_t(i) * 3;
        wrong += scaled[i] != float(i) * 0.5f;
        wrong += vecs[i][0] != float(i);
        wrong += vecs[i][1] != float(i) + 1.0f;
        wrong += vecs[i][2] != float(i) + 2.0f;

        wrong += u_ints[i] != ints[i];
        wrong += u_scaled[i] != scaled[i];
        for (int c = 0; c < 3; c++) {
            wrong += u_vecs[i][c] != vecs[i][c];
        }
    }

    std::cout << "wrong: " << wrong << '\n';
    std::cout << "spot: " << ints[7] << ' ' << scaled[7] << ' ' << vecs[7][2]
              << '\n';

    free(ints);
    free(scaled);
    free(vecs);
    free(u_ints);
    free(u_scaled);
    free(u_vecs);
}
