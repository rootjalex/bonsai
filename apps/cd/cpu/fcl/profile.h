#pragma once
#include <cstdint>

// #ifdef AJR_PROFILING
inline uint32_t bonsai_aabb_counter = 0;
inline uint32_t bonsai_tri_counter = 0;
inline uint32_t bonsai_rec_counter = 0;

inline uint32_t fcl_aabb_counter = 0;
inline uint32_t fcl_tri_counter = 0;
inline uint32_t fcl_rec_counter = 0;

inline void ajr_profiler_reset() {
    bonsai_aabb_counter = 0;
    bonsai_tri_counter = 0;
    bonsai_rec_counter = 0;
    fcl_aabb_counter = 0;
    fcl_tri_counter = 0;
    fcl_rec_counter = 0;
}
// #endif