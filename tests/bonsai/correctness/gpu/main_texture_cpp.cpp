#include "texture-cpp.h"

#include <cstdio>
#include <vector>

// Used in texture-cpp.bonsai.
int main() {
    // Two textures over the same two-level pyramid: level 0 is 2 by 2 --
    // red and green on the first row, blue and white on the second -- and
    // level 1 is their average. Texture 0 clamps at its edge, texture 1 has
    // a black border outside it.
    const uint32_t widths[2] = {2, 1};
    const uint32_t heights[2] = {2, 1};
    const float level0[16] = {1, 0, 0, 1, 0, 1, 0, 1,
                              0, 0, 1, 1, 1, 1, 1, 1};
    const float level1[4] = {0.5f, 0.5f, 0.5f, 1};
    const float *const levels[2] = {level0, level1};
    std::vector<uint64_t> handles = {
        bonsai_cuda_texture_create(2, widths, heights, levels, /*wrap=*/1,
                                   /*max_anisotropy=*/8),
        bonsai_cuda_texture_create(2, widths, heights, levels, /*wrap=*/2,
                                   /*max_anisotropy=*/8)};

    // The lookups: which texture, where, with what footprint.
    struct Lookup {
        uint32_t texture;
        float2 st, dstdx, dstdy;
    };
    const Lookup lookups[] = {
        // The red texel's centre, and the white one's.
        {0, {0.25f, 0.25f}, {0, 0}, {0, 0}},
        {0, {0.75f, 0.75f}, {0, 0}, {0, 0}},
        // Midway from red to green; the middle of all four.
        {0, {0.5f, 0.25f}, {0, 0}, {0, 0}},
        {0, {0.5f, 0.5f}, {0, 0}, {0, 0}},
        // Past the right edge: clamped to green, or the border.
        {0, {1.5f, 0.25f}, {0, 0}, {0, 0}},
        {1, {1.5f, 0.25f}, {0, 0}, {0, 0}},
        // A footprint three texels across, the same both ways: the pyramid's
        // top. (Wide along one axis only it would be filtered
        // anisotropically -- the object asks for anisotropy 8, as pbrt's GPU
        // textures do -- and the taps' average is the unit's to place, so
        // that is not pinned here.)
        {0, {0.25f, 0.25f}, {1.5f, 0}, {0, 1.5f}},
        // Green's centre on the second texture.
        {1, {0.75f, 0.25f}, {0, 0}, {0, 0}},
    };
    const int32_t n = int32_t(sizeof(lookups) / sizeof(lookups[0]));
    std::vector<uint32_t> texture(n);
    std::vector<float2> st(n), dstdx(n), dstdy(n);
    for (int32_t i = 0; i < n; i++) {
        texture[i] = lookups[i].texture;
        st[i] = lookups[i].st;
        dstdx[i] = lookups[i].dstdx;
        dstdy[i] = lookups[i].dstdy;
    }
    std::vector<float4> out(n, float4{-1, -1, -1, -1});

    bonsai_buffer texture_buffer =
        bonsai_buffer_wrap(texture.data(), n * sizeof(uint32_t));
    bonsai_buffer st_buffer = bonsai_buffer_wrap(st.data(), n * sizeof(float2));
    bonsai_buffer dstdx_buffer =
        bonsai_buffer_wrap(dstdx.data(), n * sizeof(float2));
    bonsai_buffer dstdy_buffer =
        bonsai_buffer_wrap(dstdy.data(), n * sizeof(float2));
    bonsai_buffer out_buffer = bonsai_buffer_wrap(out.data(), n * sizeof(float4));
    bonsai_buffer handles_buffer =
        bonsai_buffer_wrap(handles.data(), handles.size() * sizeof(uint64_t));
    // In run's parameter order: its own, then the extern.
    bonsai_buffer *buffers[] = {&texture_buffer, &st_buffer, &dstdx_buffer,
                                &dstdy_buffer, &out_buffer, &handles_buffer};
    static_assert(sizeof(buffers) / sizeof(buffers[0]) == sizeof(run_sides));

    bonsai_buffer_stage_all(buffers, run_sides, 6);
    bonsai_buffer_implicit_copies(0);
    run(n, &texture_buffer, &st_buffer, &dstdx_buffer, &dstdy_buffer,
        &out_buffer, &handles_buffer);
    bonsai_buffer_stage(&out_buffer, BONSAI_HOST);
    for (int32_t i = 0; i < n; i++) {
        printf("%d: %.3f %.3f %.3f %.3f\n", i, double(out[i].x), double(out[i].y),
               double(out[i].z), double(out[i].w));
    }

    for (bonsai_buffer *b : buffers) {
        bonsai_buffer_free(b);
    }
    for (uint64_t h : handles) {
        bonsai_cuda_texture_destroy(h);
    }
}
