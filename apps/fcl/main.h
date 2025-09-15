#pragma once

#include <cstdint>

extern "C" {
typedef float vec3_float __attribute__((vector_size(12)));
struct Triangle {
    vec3_float p0;
    vec3_float p1;
    vec3_float p2;
};
struct __tuple_0 {
    Triangle _field0;
    Triangle _field1;
};
typedef uint8_t vec64_uint8_t __attribute__((vector_size(64)));
struct __dyn_array0 {
    __tuple_0* buffer;
    int32_t size = 0;
    int32_t capacity;
    vec64_uint8_t mutex;
};
typedef uint8_t vec4_uint8_t __attribute__((vector_size(4)));
struct _tree_layout1 {
    vec3_float low;
    uint32_t nPrims;
    uint8_t axis;
    uint8_t pad0;
    vec3_float high;
    vec4_uint8_t split0on_nPrims;
} __attribute__((packed));
struct _tree_layout0 {
    uint32_t pCount;
    Triangle* prims;
    uint32_t count;
    _tree_layout1* group0_index;
} __attribute__((packed));

void collisions(__dyn_array0& _ret0, const _tree_layout0& triangles1, const _tree_layout0& triangles2);
}
