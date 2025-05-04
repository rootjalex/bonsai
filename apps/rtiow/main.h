#pragma once

#include <cstdint>

extern "C" {
struct Sphere {
    float center[3];
    float radius;
};
struct _spheres_layout0 {
    float center[3];
    float radius;
    uint16_t nPrims;
    uint8_t spheres_spliton_nPrims[2];
} __attribute__((packed));
struct _spheres_layout1 {
    uint32_t pCount;
    Sphere* prims;
    uint32_t count;
    _spheres_layout0* spheres_index;
} __attribute__((packed));

int32_t*** image(const int32_t width, const _spheres_layout1& spheres);
}
