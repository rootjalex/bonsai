// Compilation steps:
// 1. Compile RTIOW with Bonsai, and include it here.
//    `./build/compiler -i <rtiow-path> -o main.cu -b cuda`
// 2. Compile this file:
//    `nvcc -O3 main.cu -o main`
// 3. Run it:
//    `./main`

// Pulled from [1], with our own addendums for Bonsai.
// [1] https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

#include "cuda_runtime.h"
#include "curand_kernel.h"

// For math.h
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <initializer_list>
#include <math.h>
#include <type_traits>

// For the main hook
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

typedef unsigned int uint;
typedef unsigned short ushort;

////////////////////////////////////////////////////////////////////////////////
// bool
////////////////////////////////////////////////////////////////////////////////
struct bool2 {
    bool x;
    bool y;
};
struct bool3 {
    bool x;
    bool y;
    bool z;
};
struct bool4 {
    bool x;
    bool y;
    bool z;
    bool w;
};

__forceinline__ __host__ __device__ bool2 make_bool2(bool x, bool y) {
    return bool2{.x = x, .y = y};
}

__forceinline__ __host__ __device__ bool3 make_bool3(bool x, bool y, bool z) {
    return bool3{.x = x, .y = y, .z = z};
}

__forceinline__ __host__ __device__ bool4 make_bool4(bool x, bool y, bool z,
                                                     bool w) {
    return bool4{.x = x, .y = y, .z = z, .w = w};
}

__forceinline__ __host__ __device__ bool2 make_bool2(bool s) {
    return bool2{.x = s, .y = s};
}

__forceinline__ __host__ __device__ bool3 make_bool3(bool s) {
    return bool3{.x = s, .y = s, .z = s};
}

__forceinline__ __host__ __device__ bool4 make_bool4(bool s) {
    return bool4{.x = s, .y = s, .z = s, .w = s};
}

__forceinline__ bool4 operator<(float4 a, float4 b) {
    return make_bool4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}
__forceinline__ bool4 operator<(int4 a, int4 b) {
    return make_bool4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}
__forceinline__ bool4 operator<(uint4 a, uint4 b) {
    return make_bool4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}

__forceinline__ bool2 operator<(float2 a, float2 b) {
    return make_bool2(a.x < b.x, a.y < b.y);
}
__forceinline__ bool2 operator<(int2 a, int2 b) {
    return make_bool2(a.x < b.x, a.y < b.y);
}
__forceinline__ bool2 operator<(uint2 a, uint2 b) {
    return make_bool2(a.x < b.x, a.y < b.y);
}
__forceinline__ bool3 operator<(float3 a, float3 b) {
    return make_bool3(a.x < b.x, a.y < b.y, a.z < b.z);
}
__forceinline__ bool3 operator<(int3 a, int3 b) {
    return make_bool3(a.x < b.x, a.y < b.y, a.z < b.z);
}
__forceinline__ bool3 operator<(uint3 a, uint3 b) {
    return make_bool3(a.x < b.x, a.y < b.y, a.z < b.z);
}

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 make_float2(float s) {
    return make_float2(s, s);
}
__forceinline__ __host__ __device__ float2 make_float2(float3 a) {
    return make_float2(a.x, a.y);
}
__forceinline__ __host__ __device__ float2 make_float2(int2 a) {
    return make_float2(float(a.x), float(a.y));
}
__forceinline__ __host__ __device__ float2 make_float2(uint2 a) {
    return make_float2(float(a.x), float(a.y));
}

__forceinline__ __host__ __device__ int2 make_int2(int s) {
    return make_int2(s, s);
}
__forceinline__ __host__ __device__ int2 make_int2(int3 a) {
    return make_int2(a.x, a.y);
}
__forceinline__ __host__ __device__ int2 make_int2(uint2 a) {
    return make_int2(int(a.x), int(a.y));
}
__forceinline__ __host__ __device__ int2 make_int2(float2 a) {
    return make_int2(int(a.x), int(a.y));
}

__forceinline__ __host__ __device__ uint2 make_uint2(uint s) {
    return make_uint2(s, s);
}
__forceinline__ __host__ __device__ uint2 make_uint2(uint3 a) {
    return make_uint2(a.x, a.y);
}
__forceinline__ __host__ __device__ uint2 make_uint2(int2 a) {
    return make_uint2(uint(a.x), uint(a.y));
}

__forceinline__ __host__ __device__ float3 make_float3(float s) {
    return make_float3(s, s, s);
}
__forceinline__ __host__ __device__ float3 make_float3(float2 a) {
    return make_float3(a.x, a.y, 0.0f);
}
__forceinline__ __host__ __device__ float3 make_float3(float2 a, float s) {
    return make_float3(a.x, a.y, s);
}
__forceinline__ __host__ __device__ float3 make_float3(float4 a) {
    return make_float3(a.x, a.y, a.z);
}
__forceinline__ __host__ __device__ float3 make_float3(int3 a) {
    return make_float3(float(a.x), float(a.y), float(a.z));
}
__forceinline__ __host__ __device__ float3 make_float3(uint3 a) {
    return make_float3(float(a.x), float(a.y), float(a.z));
}

__forceinline__ __host__ __device__ int3 make_int3(int s) {
    return make_int3(s, s, s);
}
__forceinline__ __host__ __device__ int3 make_int3(int2 a) {
    return make_int3(a.x, a.y, 0);
}
__forceinline__ __host__ __device__ int3 make_int3(int2 a, int s) {
    return make_int3(a.x, a.y, s);
}
__forceinline__ __host__ __device__ int3 make_int3(uint3 a) {
    return make_int3(int(a.x), int(a.y), int(a.z));
}
__forceinline__ __host__ __device__ int3 make_int3(float3 a) {
    return make_int3(int(a.x), int(a.y), int(a.z));
}

__forceinline__ __host__ __device__ uint3 make_uint3(uint s) {
    return make_uint3(s, s, s);
}
__forceinline__ __host__ __device__ uint3 make_uint3(uint2 a) {
    return make_uint3(a.x, a.y, 0);
}
__forceinline__ __host__ __device__ uint3 make_uint3(uint2 a, uint s) {
    return make_uint3(a.x, a.y, s);
}
__forceinline__ __host__ __device__ uint3 make_uint3(uint4 a) {
    return make_uint3(a.x, a.y, a.z);
}
__forceinline__ __host__ __device__ uint3 make_uint3(int3 a) {
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

__forceinline__ __host__ __device__ float4 make_float4(float s) {
    return make_float4(s, s, s, s);
}
__forceinline__ __host__ __device__ float4 make_float4(float3 a) {
    return make_float4(a.x, a.y, a.z, 0.0f);
}
__forceinline__ __host__ __device__ float4 make_float4(float3 a, float w) {
    return make_float4(a.x, a.y, a.z, w);
}
__forceinline__ __host__ __device__ float4 make_float4(int4 a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
__forceinline__ __host__ __device__ float4 make_float4(uint4 a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

__forceinline__ __host__ __device__ int4 make_int4(int s) {
    return make_int4(s, s, s, s);
}
__forceinline__ __host__ __device__ int4 make_int4(int3 a) {
    return make_int4(a.x, a.y, a.z, 0);
}
__forceinline__ __host__ __device__ int4 make_int4(int3 a, int w) {
    return make_int4(a.x, a.y, a.z, w);
}
__forceinline__ __host__ __device__ int4 make_int4(uint4 a) {
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
__forceinline__ __host__ __device__ int4 make_int4(float4 a) {
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}

__forceinline__ __host__ __device__ uint4 make_uint4(uint s) {
    return make_uint4(s, s, s, s);
}
__forceinline__ __host__ __device__ uint4 make_uint4(uint3 a) {
    return make_uint4(a.x, a.y, a.z, 0);
}
__forceinline__ __host__ __device__ uint4 make_uint4(uint3 a, uint w) {
    return make_uint4(a.x, a.y, a.z, w);
}
__forceinline__ __host__ __device__ uint4 make_uint4(int4 a) {
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator-(float2 &a) {
    return make_float2(-a.x, -a.y);
}
__forceinline__ __host__ __device__ int2 operator-(int2 &a) {
    return make_int2(-a.x, -a.y);
}
__forceinline__ __host__ __device__ float3 operator-(float3 &a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__forceinline__ __host__ __device__ int3 operator-(int3 &a) {
    return make_int3(-a.x, -a.y, -a.z);
}
__forceinline__ __host__ __device__ float4 operator-(float4 &a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
__forceinline__ __host__ __device__ int4 operator-(int4 &a) {
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}
__forceinline__ __host__ __device__ float2 operator+(float2 a, float b) {
    return make_float2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ float2 operator+(float b, float2 a) {
    return make_float2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ void operator+=(float2 &a, float b) {
    a.x += b;
    a.y += b;
}

__forceinline__ __host__ __device__ int2 operator+(int2 a, int2 b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ void operator+=(int2 &a, int2 b) {
    a.x += b.x;
    a.y += b.y;
}
__forceinline__ __host__ __device__ int2 operator+(int2 a, int b) {
    return make_int2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ int2 operator+(int b, int2 a) {
    return make_int2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ void operator+=(int2 &a, int b) {
    a.x += b;
    a.y += b;
}

__forceinline__ __host__ __device__ uint2 operator+(uint2 a, uint2 b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ void operator+=(uint2 &a, uint2 b) {
    a.x += b.x;
    a.y += b.y;
}
__forceinline__ __host__ __device__ uint2 operator+(uint2 a, uint b) {
    return make_uint2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ uint2 operator+(uint b, uint2 a) {
    return make_uint2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ void operator+=(uint2 &a, uint b) {
    a.x += b;
    a.y += b;
}

__forceinline__ __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ void operator+=(float3 &a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__forceinline__ __host__ __device__ float3 operator+(float3 a, float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ void operator+=(float3 &a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

__forceinline__ __host__ __device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ void operator+=(int3 &a, int3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__forceinline__ __host__ __device__ int3 operator+(int3 a, int b) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ void operator+=(int3 &a, int b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

__forceinline__ __host__ __device__ uint3 operator+(uint3 a, uint3 b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ void operator+=(uint3 &a, uint3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
__forceinline__ __host__ __device__ uint3 operator+(uint3 a, uint b) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ void operator+=(uint3 &a, uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
}

__forceinline__ __host__ __device__ int3 operator+(int b, int3 a) {
    return make_int3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ uint3 operator+(uint b, uint3 a) {
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ float3 operator+(float b, float3 a) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
__forceinline__ __host__ __device__ float4 operator+(float4 a, float b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(float4 &a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__forceinline__ __host__ __device__ int4 operator+(int4 a, int4 b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(int4 &a, int4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
__forceinline__ __host__ __device__ int4 operator+(int4 a, int b) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ int4 operator+(int b, int4 a) {
    return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(int4 &a, int b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

__forceinline__ __host__ __device__ uint4 operator+(uint4 a, uint4 b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ void operator+=(uint4 &a, uint4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
__forceinline__ __host__ __device__ uint4 operator+(uint4 a, uint b) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ uint4 operator+(uint b, uint4 a) {
    return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(uint4 &a, uint b) {
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ void operator-=(float2 &a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
__forceinline__ __host__ __device__ float2 operator-(float2 a, float b) {
    return make_float2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ float2 operator-(float b, float2 a) {
    return make_float2(b - a.x, b - a.y);
}
__forceinline__ __host__ __device__ void operator-=(float2 &a, float b) {
    a.x -= b;
    a.y -= b;
}

__forceinline__ __host__ __device__ int2 operator-(int2 a, int2 b) {
    return make_int2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ void operator-=(int2 &a, int2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
__forceinline__ __host__ __device__ int2 operator-(int2 a, int b) {
    return make_int2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ int2 operator-(int b, int2 a) {
    return make_int2(b - a.x, b - a.y);
}
__forceinline__ __host__ __device__ void operator-=(int2 &a, int b) {
    a.x -= b;
    a.y -= b;
}

__forceinline__ __host__ __device__ uint2 operator-(uint2 a, uint2 b) {
    return make_uint2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ void operator-=(uint2 &a, uint2 b) {
    a.x -= b.x;
    a.y -= b.y;
}
__forceinline__ __host__ __device__ uint2 operator-(uint2 a, uint b) {
    return make_uint2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ uint2 operator-(uint b, uint2 a) {
    return make_uint2(b - a.x, b - a.y);
}
__forceinline__ __host__ __device__ void operator-=(uint2 &a, uint b) {
    a.x -= b;
    a.y -= b;
}

__forceinline__ __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ void operator-=(float3 &a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
__forceinline__ __host__ __device__ float3 operator-(float3 a, float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ float3 operator-(float b, float3 a) {
    return make_float3(b - a.x, b - a.y, b - a.z);
}
__forceinline__ __host__ __device__ void operator-=(float3 &a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__forceinline__ __host__ __device__ int3 operator-(int3 a, int3 b) {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ void operator-=(int3 &a, int3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
__forceinline__ __host__ __device__ int3 operator-(int3 a, int b) {
    return make_int3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ int3 operator-(int b, int3 a) {
    return make_int3(b - a.x, b - a.y, b - a.z);
}
__forceinline__ __host__ __device__ void operator-=(int3 &a, int b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__forceinline__ __host__ __device__ uint3 operator-(uint3 a, uint3 b) {
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ void operator-=(uint3 &a, uint3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
__forceinline__ __host__ __device__ uint3 operator-(uint3 a, uint b) {
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ uint3 operator-(uint b, uint3 a) {
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
__forceinline__ __host__ __device__ void operator-=(uint3 &a, uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ void operator-=(float4 &a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ void operator-=(float4 &a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__forceinline__ __host__ __device__ int4 operator-(int4 a, int4 b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ void operator-=(int4 &a, int4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
__forceinline__ __host__ __device__ int4 operator-(int4 a, int b) {
    return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ int4 operator-(int b, int4 a) {
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
__forceinline__ __host__ __device__ void operator-=(int4 &a, int b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__forceinline__ __host__ __device__ uint4 operator-(uint4 a, uint4 b) {
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ void operator-=(uint4 &a, uint4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
__forceinline__ __host__ __device__ uint4 operator-(uint4 a, uint b) {
    return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ uint4 operator-(uint b, uint4 a) {
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
__forceinline__ __host__ __device__ void operator-=(uint4 &a, uint b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ void operator*=(float2 &a, float2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
__forceinline__ __host__ __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ float2 operator*(float b, float2 a) {
    return make_float2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ void operator*=(float2 &a, float b) {
    a.x *= b;
    a.y *= b;
}

__forceinline__ __host__ __device__ int2 operator*(int2 a, int2 b) {
    return make_int2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ void operator*=(int2 &a, int2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
__forceinline__ __host__ __device__ int2 operator*(int2 a, int b) {
    return make_int2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ int2 operator*(int b, int2 a) {
    return make_int2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ void operator*=(int2 &a, int b) {
    a.x *= b;
    a.y *= b;
}

__forceinline__ __host__ __device__ uint2 operator*(uint2 a, uint2 b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ void operator*=(uint2 &a, uint2 b) {
    a.x *= b.x;
    a.y *= b.y;
}
__forceinline__ __host__ __device__ uint2 operator*(uint2 a, uint b) {
    return make_uint2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ uint2 operator*(uint b, uint2 a) {
    return make_uint2(b * a.x, b * a.y);
}
__forceinline__ __host__ __device__ void operator*=(uint2 &a, uint b) {
    a.x *= b;
    a.y *= b;
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ void operator*=(float3 &a, float3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
__forceinline__ __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ float3 operator*(float b, float3 a) {
    return make_float3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(float3 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__forceinline__ __host__ __device__ int3 operator*(int3 a, int3 b) {
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ void operator*=(int3 &a, int3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
__forceinline__ __host__ __device__ int3 operator*(int3 a, int b) {
    return make_int3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ int3 operator*(int b, int3 a) {
    return make_int3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(int3 &a, int b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__forceinline__ __host__ __device__ uint3 operator*(uint3 a, uint3 b) {
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ void operator*=(uint3 &a, uint3 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
__forceinline__ __host__ __device__ uint3 operator*(uint3 a, uint b) {
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ uint3 operator*(uint b, uint3 a) {
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
__forceinline__ __host__ __device__ void operator*=(uint3 &a, uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ void operator*=(float4 &a, float4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__forceinline__ __host__ __device__ int4 operator*(int4 a, int4 b) {
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ void operator*=(int4 &a, int4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
__forceinline__ __host__ __device__ int4 operator*(int4 a, int b) {
    return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ int4 operator*(int b, int4 a) {
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(int4 &a, int b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

__forceinline__ __host__ __device__ uint4 operator*(uint4 a, uint4 b) {
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ void operator*=(uint4 &a, uint4 b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
__forceinline__ __host__ __device__ uint4 operator*(uint4 a, uint b) {
    return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ uint4 operator*(uint b, uint4 a) {
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
__forceinline__ __host__ __device__ void operator*=(uint4 &a, uint b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 operator/(float2 a, float2 b) {
    return make_float2(a.x / b.x, a.y / b.y);
}
__forceinline__ __host__ __device__ void operator/=(float2 &a, float2 b) {
    a.x /= b.x;
    a.y /= b.y;
}
__forceinline__ __host__ __device__ float2 operator/(float2 a, float b) {
    return make_float2(a.x / b, a.y / b);
}
__forceinline__ __host__ __device__ void operator/=(float2 &a, float b) {
    a.x /= b;
    a.y /= b;
}
__forceinline__ __host__ __device__ float2 operator/(float b, float2 a) {
    return make_float2(b / a.x, b / a.y);
}

__forceinline__ __host__ __device__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__forceinline__ __host__ __device__ void operator/=(float3 &a, float3 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
__forceinline__ __host__ __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
__forceinline__ __host__ __device__ void operator/=(float3 &a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
__forceinline__ __host__ __device__ float3 operator/(float b, float3 a) {
    return make_float3(b / a.x, b / a.y, b / a.z);
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__forceinline__ __host__ __device__ void operator/=(float4 &a, float4 b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
__forceinline__ __host__ __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__forceinline__ __host__ __device__ void operator/=(float4 &a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
__forceinline__ __host__ __device__ float4 operator/(float b, float4 a) {
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 min(float2 a, float2 b) {
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
__forceinline__ __host__ __device__ float3 min(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__forceinline__ __host__ __device__ float4 min(float4 a, float4 b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z),
                       fminf(a.w, b.w));
}

__forceinline__ __host__ __device__ int2 min(int2 a, int2 b) {
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}
__forceinline__ __host__ __device__ int3 min(int3 a, int3 b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
__forceinline__ __host__ __device__ int4 min(int4 a, int4 b) {
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z),
                     min(a.w, b.w));
}

__forceinline__ __host__ __device__ uint2 min(uint2 a, uint2 b) {
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}
__forceinline__ __host__ __device__ uint3 min(uint3 a, uint3 b) {
    return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
__forceinline__ __host__ __device__ uint4 min(uint4 a, uint4 b) {
    return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z),
                      min(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 max(float2 a, float2 b) {
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
__forceinline__ __host__ __device__ float3 max(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__forceinline__ __host__ __device__ float4 max(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z),
                       fmaxf(a.w, b.w));
}

__forceinline__ __host__ __device__ int2 max(int2 a, int2 b) {
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}
__forceinline__ __host__ __device__ int3 max(int3 a, int3 b) {
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
__forceinline__ __host__ __device__ int4 max(int4 a, int4 b) {
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z),
                     max(a.w, b.w));
}

__forceinline__ __host__ __device__ uint2 max(uint2 a, uint2 b) {
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}
__forceinline__ __host__ __device__ uint3 max(uint3 a, uint3 b) {
    return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
__forceinline__ __host__ __device__ uint4 max(uint4 a, uint4 b) {
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z),
                      max(a.w, b.w));
}

__forceinline__ __host__ __device__ uint max(uint2 a) { return max(a.x, a.y); }
__forceinline__ __host__ __device__ uint max(uint3 a) {
    return max(max(a.x, a.y), a.z);
}
__forceinline__ __host__ __device__ uint max(uint4 a) {
    return max(max(max(a.x, a.y), a.z), a.w);
}

__forceinline__ __host__ __device__ int max(int2 a) { return max(a.x, a.y); }
__forceinline__ __host__ __device__ int max(int3 a) {
    return max(max(a.x, a.y), a.z);
}
__forceinline__ __host__ __device__ int max(int4 a) {
    return max(max(max(a.x, a.y), a.z), a.w);
}

__forceinline__ __host__ __device__ float max(float2 a) {
    return fmaxf(a.x, a.y);
}
__forceinline__ __host__ __device__ float max(float3 a) {
    return fmaxf(fmaxf(a.x, a.y), a.z);
}
__forceinline__ __host__ __device__ float max(float4 a) {
    return fmaxf(fmaxf(fmaxf(a.x, a.y), a.z), a.w);
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ __host__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}
__forceinline__ __device__ __host__ float2 lerp(float2 a, float2 b, float t) {
    return a + t * (b - a);
}
__forceinline__ __device__ __host__ float3 lerp(float3 a, float3 b, float t) {
    return a + t * (b - a);
}
__forceinline__ __device__ __host__ float4 lerp(float4 a, float4 b, float t) {
    return a + t * (b - a);
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}
__forceinline__ __host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ __host__ __device__ float dot(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __host__ __device__ int dot(int2 a, int2 b) {
    return a.x * b.x + a.y * b.y;
}
__forceinline__ __host__ __device__ int dot(int3 a, int3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ __host__ __device__ int dot(int4 a, int4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__forceinline__ __host__ __device__ uint dot(uint2 a, uint2 b) {
    return a.x * b.x + a.y * b.y;
}
__forceinline__ __host__ __device__ uint dot(uint3 a, uint3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__forceinline__ __host__ __device__ uint dot(uint4 a, uint4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float length(float2 v) {
    return sqrtf(dot(v, v));
}
__forceinline__ __host__ __device__ float length(float3 v) {
    return sqrtf(dot(v, v));
}
__forceinline__ __host__ __device__ float length(float4 v) {
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 normalize(float2 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
__forceinline__ __host__ __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
__forceinline__ __host__ __device__ float4 normalize(float4 v) {
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 floorf(float2 v) {
    return make_float2(floorf(v.x), floorf(v.y));
}
__forceinline__ __host__ __device__ float3 floorf(float3 v) {
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
__forceinline__ __host__ __device__ float4 floorf(float4 v) {
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float fracf(float v) {
    return v - floorf(v);
}
__forceinline__ __host__ __device__ float2 fracf(float2 v) {
    return make_float2(fracf(v.x), fracf(v.y));
}
__forceinline__ __host__ __device__ float3 fracf(float3 v) {
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
__forceinline__ __host__ __device__ float4 fracf(float4 v) {
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 fmodf(float2 a, float2 b) {
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
__forceinline__ __host__ __device__ float3 fmodf(float3 a, float3 b) {
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
__forceinline__ __host__ __device__ float4 fmodf(float4 a, float4 b) {
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z),
                       fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float2 abs(float2 v) {
    return make_float2(fabs(v.x), fabs(v.y));
}
__forceinline__ __host__ __device__ float3 abs(float3 v) {
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
__forceinline__ __host__ __device__ float4 abs(float4 v) {
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

__forceinline__ __host__ __device__ int2 abs(int2 v) {
    return make_int2(abs(v.x), abs(v.y));
}
__forceinline__ __host__ __device__ int3 abs(int3 v) {
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
__forceinline__ __host__ __device__ int4 abs(int4 v) {
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

////////////////////////////////////////////////////////////////////////////////
// idxmax, idxmin
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ uint32_t idxmax(int2 v) {
    uint32_t idx = 0u;
    idx = (v.y > v.x) ? 1u : 0u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmax(int3 v) {
    uint32_t idx = 0u;
    if (v.y > v.x)
        idx = 1u;
    if (v.z > (idx == 0u ? v.x : (idx == 1u ? v.y : v.x)))
        idx = 2u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmax(int4 v) {
    uint32_t idx = 0u;
    if (v.y > v.x)
        idx = 1u;
    if (v.z > (idx == 0u ? v.x : v.y))
        idx = 2u;
    if (v.w > (idx == 0u ? v.x : (idx == 1u ? v.y : v.z)))
        idx = 3u;
    return idx;
}

__forceinline__ __host__ __device__ uint32_t idxmax(uint2 v) {
    uint32_t idx = 0u;
    idx = (v.y > v.x) ? 1u : 0u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmax(uint3 v) {
    uint32_t idx = 0u;
    if (v.y > v.x)
        idx = 1u;
    if (v.z > (idx == 0u ? v.x : (idx == 1u ? v.y : v.x)))
        idx = 2u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmax(uint4 v) {
    uint32_t idx = 0u;
    if (v.y > v.x)
        idx = 1u;
    if (v.z > (idx == 0u ? v.x : v.y))
        idx = 2u;
    if (v.w > (idx == 0u ? v.x : (idx == 1u ? v.y : v.z)))
        idx = 3u;
    return idx;
}

__forceinline__ __host__ __device__ uint32_t idxmax(float2 v) {
    uint32_t idx = 0u;
    idx = (v.y > v.x) ? 1u : 0u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmax(float3 v) {
    uint32_t idx = 0u;
    if (v.y > v.x)
        idx = 1u;
    if (v.z > (idx == 0u ? v.x : (idx == 1u ? v.y : v.x)))
        idx = 2u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmax(float4 v) {
    uint32_t idx = 0u;
    if (v.y > v.x)
        idx = 1u;
    if (v.z > (idx == 0u ? v.x : v.y))
        idx = 2u;
    if (v.w > (idx == 0u ? v.x : (idx == 1u ? v.y : v.z)))
        idx = 3u;
    return idx;
}

__forceinline__ __host__ __device__ uint32_t idxmin(int2 v) {
    uint32_t idx = 0u;
    idx = (v.y < v.x) ? 1u : 0u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmin(int3 v) {
    uint32_t idx = 0u;
    if (v.y < v.x)
        idx = 1u;
    if (v.z < (idx == 0u ? v.x : (idx == 1u ? v.y : v.x)))
        idx = 2u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmin(int4 v) {
    uint32_t idx = 0u;
    if (v.y < v.x)
        idx = 1u;
    if (v.z < (idx == 0u ? v.x : v.y))
        idx = 2u;
    if (v.w < (idx == 0u ? v.x : (idx == 1u ? v.y : v.z)))
        idx = 3u;
    return idx;
}

__forceinline__ __host__ __device__ uint32_t idxmin(uint2 v) {
    uint32_t idx = 0u;
    idx = (v.y < v.x) ? 1u : 0u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmin(uint3 v) {
    uint32_t idx = 0u;
    if (v.y < v.x)
        idx = 1u;
    if (v.z < (idx == 0u ? v.x : (idx == 1u ? v.y : v.x)))
        idx = 2u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmin(uint4 v) {
    uint32_t idx = 0u;
    if (v.y < v.x)
        idx = 1u;
    if (v.z < (idx == 0u ? v.x : v.y))
        idx = 2u;
    if (v.w < (idx == 0u ? v.x : (idx == 1u ? v.y : v.z)))
        idx = 3u;
    return idx;
}

__forceinline__ __host__ __device__ uint32_t idxmin(float2 v) {
    uint32_t idx = 0u;
    idx = (v.y < v.x) ? 1u : 0u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmin(float3 v) {
    uint32_t idx = 0u;
    if (v.y < v.x)
        idx = 1u;
    if (v.z < (idx == 0u ? v.x : (idx == 1u ? v.y : v.x)))
        idx = 2u;
    return idx;
}
__forceinline__ __host__ __device__ uint32_t idxmin(float4 v) {
    uint32_t idx = 0u;
    if (v.y < v.x)
        idx = 1u;
    if (v.z < (idx == 0u ? v.x : v.y))
        idx = 2u;
    if (v.w < (idx == 0u ? v.x : (idx == 1u ? v.y : v.z)))
        idx = 3u;
    return idx;
}

////////////////////////////////////////////////////////////////////////////////
// sum
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ int32_t sum(int2 v) { return v.x + v.y; }
__forceinline__ __host__ __device__ int32_t sum(int3 v) {
    return v.x + v.y + v.z;
}
__forceinline__ __host__ __device__ int32_t sum(int4 v) {
    return v.x + v.y + v.z + v.w;
}

__forceinline__ __host__ __device__ uint32_t sum(uint2 v) { return v.x + v.y; }
__forceinline__ __host__ __device__ uint32_t sum(uint3 v) {
    return v.x + v.y + v.z;
}
__forceinline__ __host__ __device__ uint32_t sum(uint4 v) {
    return v.x + v.y + v.z + v.w;
}

__forceinline__ __host__ __device__ float sum(float2 v) { return v.x + v.y; }
__forceinline__ __host__ __device__ float sum(float3 v) {
    return v.x + v.y + v.z;
}
__forceinline__ __host__ __device__ float sum(float4 v) {
    return v.x + v.y + v.z + v.w;
}

__forceinline__ __host__ __device__ int32_t mul(int2 v) { return v.x * v.y; }
__forceinline__ __host__ __device__ int32_t mul(int3 v) {
    return v.x * v.y * v.z;
}
__forceinline__ __host__ __device__ int32_t mul(int4 v) {
    return v.x * v.y * v.z * v.w;
}

__forceinline__ __host__ __device__ uint32_t mul(uint2 v) { return v.x * v.y; }
__forceinline__ __host__ __device__ uint32_t mul(uint3 v) {
    return v.x * v.y * v.z;
}
__forceinline__ __host__ __device__ uint32_t mul(uint4 v) {
    return v.x * v.y * v.z * v.w;
}

__forceinline__ __host__ __device__ float mul(float2 v) { return v.x * v.y; }
__forceinline__ __host__ __device__ float mul(float3 v) {
    return v.x * v.y * v.z;
}
__forceinline__ __host__ __device__ float mul(float4 v) {
    return v.x * v.y * v.z * v.w;
}

////////////////////////////////////////////////////////////////////////////////
// shuffle
////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ int2
shuffle(int2 v, std::initializer_list<uint32_t> indices) {
    int2 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}
__forceinline__ __host__ __device__ int3
shuffle(int3 v, std::initializer_list<uint32_t> indices) {
    int3 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    case 2:
        r.x = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    case 2:
        r.y = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.z = v.x;
        break;
    case 1:
        r.z = v.y;
        break;
    case 2:
        r.z = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}
__forceinline__ __host__ __device__ int4
shuffle(int4 v, std::initializer_list<uint32_t> indices) {
    int4 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    case 2:
        r.x = v.z;
        break;
    case 3:
        r.x = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    case 2:
        r.y = v.z;
        break;
    case 3:
        r.y = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.z = v.x;
        break;
    case 1:
        r.z = v.y;
        break;
    case 2:
        r.z = v.z;
        break;
    case 3:
        r.z = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.w = v.x;
        break;
    case 1:
        r.w = v.y;
        break;
    case 2:
        r.w = v.z;
        break;
    case 3:
        r.w = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}

__forceinline__ __host__ __device__ uint2
shuffle(uint2 v, std::initializer_list<uint32_t> indices) {
    uint2 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}
__forceinline__ __host__ __device__ uint3
shuffle(uint3 v, std::initializer_list<uint32_t> indices) {
    uint3 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    case 2:
        r.x = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    case 2:
        r.y = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.z = v.x;
        break;
    case 1:
        r.z = v.y;
        break;
    case 2:
        r.z = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}
__forceinline__ __host__ __device__ uint4
shuffle(uint4 v, std::initializer_list<uint32_t> indices) {
    uint4 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    case 2:
        r.x = v.z;
        break;
    case 3:
        r.x = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    case 2:
        r.y = v.z;
        break;
    case 3:
        r.y = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.z = v.x;
        break;
    case 1:
        r.z = v.y;
        break;
    case 2:
        r.z = v.z;
        break;
    case 3:
        r.z = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.w = v.x;
        break;
    case 1:
        r.w = v.y;
        break;
    case 2:
        r.w = v.z;
        break;
    case 3:
        r.w = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}

__forceinline__ __host__ __device__ float2
shuffle(float2 v, std::initializer_list<uint32_t> indices) {
    float2 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}
__forceinline__ __host__ __device__ float3
shuffle(float3 v, std::initializer_list<uint32_t> indices) {
    float3 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    case 2:
        r.x = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    case 2:
        r.y = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.z = v.x;
        break;
    case 1:
        r.z = v.y;
        break;
    case 2:
        r.z = v.z;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}
__forceinline__ __host__ __device__ float4
shuffle(float4 v, std::initializer_list<uint32_t> indices) {
    float4 r;
    auto it = indices.begin();
    switch (*it++) {
    case 0:
        r.x = v.x;
        break;
    case 1:
        r.x = v.y;
        break;
    case 2:
        r.x = v.z;
        break;
    case 3:
        r.x = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.y = v.x;
        break;
    case 1:
        r.y = v.y;
        break;
    case 2:
        r.y = v.z;
        break;
    case 3:
        r.y = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.z = v.x;
        break;
    case 1:
        r.z = v.y;
        break;
    case 2:
        r.z = v.z;
        break;
    case 3:
        r.z = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    switch (*it++) {
    case 0:
        r.w = v.x;
        break;
    case 1:
        r.w = v.y;
        break;
    case 2:
        r.w = v.z;
        break;
    case 3:
        r.w = v.w;
        break;
    default:
        __builtin_unreachable();
    }
    return r;
}

template <typename T>
__forceinline__ __host__ __device__ T argmin(T *current, T update) {
    if (current->_field0 < update._field0) {
        return *current;
    }
    return update;
}

template <typename T>
__forceinline__ __host__ __device__ T *argmax(T *current, T update) {
    if (current->_field0 > update._field0) {
        return current;
    }
    return &update;
}

// Jesus christ
template <typename O, typename I>
__forceinline__ __host__ __device__ O bonsai_reinterpret(I input) {
    static_assert(sizeof(O) == sizeof(I));
    static_assert(std::is_trivially_copyable<O>::value);
    static_assert(std::is_trivially_copyable<I>::value);
    I *i = &input;
    O *output;
    output = reinterpret_cast<O *>(i);
    return *output;
}

struct AABB {
    float3 low;
    float3 high;
};

struct Camera {
    float aspect_ratio;
    int32_t width;
    uint32_t samples_per_pixel = 100u;
    int32_t max_depth = 10;
    float vfov = static_cast<float>(90);
    float3 lookfrom = float3{static_cast<float>(0), static_cast<float>(0),
                             static_cast<float>(0)};
    float3 lookat = float3{static_cast<float>(0), static_cast<float>(0),
                           static_cast<float>(-1)};
    float3 vup = float3{static_cast<float>(0), static_cast<float>(1),
                        static_cast<float>(0)};
    float defocus_angle = static_cast<float>(0);
    float focus_dist = static_cast<float>(10);
};

struct FInterval {
    float low;
    float high;
};

struct Sphere {
    float3 center;
    float radius;
};

struct MaterialSphere {
    Sphere s;
    uint32_t material;
    float3 albedo;
    float fuzz;
};

struct Point {
    float3 vec;
};

struct Ray {
    float3 o;
    float3 d;
    float tmax = INFINITY;
};

struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};

struct TriangleIntersection {
    float b0;
    float b1;
    float b2;
    float t;
};

struct __tuple_0 {
    float _field0;
    MaterialSphere _field1;
};

struct __tuple_1 {
    Point _field0;
    Point _field1;
};

struct _option0 {
    MaterialSphere value;
    bool set = false;
};

struct _option1 {
    FInterval value;
    bool set = false;
};

struct _option2 {
    TriangleIntersection value;
    bool set = false;
};

struct _spheres_layout1 {
    float3 center;
    float radius;
    uint16_t nPrims;
    uchar2 spheres_spliton_nPrims;
} __attribute__((packed));

struct _spheres_layout0 {
    uint32_t pCount;
    MaterialSphere *prims; // of size pCount
    uint32_t count;
    _spheres_layout1 *spheres_index; // of size count
} __attribute__((packed));

struct _spheres_split_layout2 {
    uint16_t offset;
} __attribute__((packed));

struct _spheres_split_layout3 {
    uint16_t pOffset;
} __attribute__((packed));

struct hit_record {
    float3 p;
    float3 normal;
    float t;
    bool front_face;
};

struct scatter_record {
    float3 attenuation;
    Ray ray;
    bool hit;
};

Point ClosestPtPointAABB(Point *pt, AABB *a) {
    return Point{min(max((*pt).vec, (*a).low), (*a).high)};
}

float SqDistPointAABB(Point *pt, AABB *a) {
    float3 v = (*pt).vec;
    float3 sqLow = (((*a).low - v) * ((*a).low - v));
    float3 low = make_float3(
        ((v < (*a).low).x ? sqLow.x : make_float3(static_cast<float>(0)).x),
        ((v < (*a).low).y ? sqLow.y : make_float3(static_cast<float>(0)).y),
        ((v < (*a).low).z ? sqLow.z : make_float3(static_cast<float>(0)).z));
    float3 sqHigh = ((v - (*a).high) * (v - (*a).high));
    float3 high = make_float3(
        (((*a).high < v).x ? sqHigh.x : make_float3(static_cast<float>(0)).x),
        (((*a).high < v).y ? sqHigh.y : make_float3(static_cast<float>(0)).y),
        (((*a).high < v).z ? sqHigh.z : make_float3(static_cast<float>(0)).z));
    return sum((low + high));
}

float __prod_diff_f32(float a, float b, float c, float d) {
    float cd = (c * d);
    float diff = fma(a, b, -cd);
    float err = fma(-c, d, cd);
    return (diff + err);
}

float __sqlen_f32(float3 v) { return sum((v * v)); }

__device__ _option1 intersectsp_ray_sphere(Ray *r, Sphere *s) {
    float3 oc = ((*s).center - (*r).o);
    float a = sum(((*r).d * (*r).d));
    float h = dot((*r).d, oc);
    float c = (sum((oc * oc)) - ((*s).radius * (*s).radius));
    float disc = ((h * h) - (a * c));
    if (disc < static_cast<float>(0)) {
        return _option1{};
    }
    float sqrtd = sqrt(disc);
    float root0 = ((h - sqrtd) / a);
    float root1 = ((h + sqrtd) / a);
    FInterval interval = FInterval{min(root0, root1), max(root0, root1)};
    return _option1{interval, true};
}

__device__ float distmax_Ray_Sphere(Ray *r, Sphere *s) {
    _option1 interval = intersectsp_ray_sphere(r, s);
    if (interval.set) {
        FInterval extract = interval.value;
        return extract.high;
    }
    return INFINITY;
}

__device__ float distmin_Ray_Sphere(Ray *r, Sphere *s) {
    _option1 interval = intersectsp_ray_sphere(r, s);
    if (interval.set) {
        FInterval extract = interval.value;
        return extract.low;
    }
    return -INFINITY;
}

__device__ bool intersects_Ray_Sphere(Ray *ray, Sphere *s) {
    _option1 interval = intersectsp_ray_sphere(ray, s);
    if (interval.set) {
        FInterval extract = interval.value;
        return ((extract.low < (*ray).tmax) &
                (static_cast<float>(0) < extract.high));
    }
    return false;
}

__device__ void _recloop_func0(uint16_t spheres__index,
                               _spheres_layout0 *spheres, Ray *r,
                               __tuple_0 *_best0) {
    Sphere _lv2 = Sphere{(*spheres).spheres_index[spheres__index].center,
                         (*spheres).spheres_index[spheres__index].radius};
    if (intersects_Ray_Sphere(r, (&_lv2))) {
        Sphere _lv1 = Sphere{(*spheres).spheres_index[spheres__index].center,
                             (*spheres).spheres_index[spheres__index].radius};
        if (static_cast<float>(0.001) < distmax_Ray_Sphere(r, (&_lv1))) {
            Sphere _lv0 =
                Sphere{(*spheres).spheres_index[spheres__index].center,
                       (*spheres).spheres_index[spheres__index].radius};
            if (distmin_Ray_Sphere(r, (&_lv0)) < (*_best0)._field0) {
                if ((*spheres).spheres_index[spheres__index].nPrims == 0u) {
                    _recloop_func0(spheres__index + 1u, spheres, r, _best0);
                    _recloop_func0(
                        spheres__index +
                            bonsai_reinterpret<_spheres_split_layout2>(
                                (*spheres)
                                    .spheres_index[spheres__index]
                                    .spheres_spliton_nPrims)
                                .offset,
                        spheres, r, _best0);
                } else {
                    for (uint16_t _idx0 = 0u;
                         _idx0 <
                         (*spheres).spheres_index[spheres__index].nPrims;
                         _idx0 += 1u) {
                        if (intersects_Ray_Sphere(
                                r, (&(*spheres)
                                         .prims[bonsai_reinterpret<
                                                    _spheres_split_layout3>(
                                                    (*spheres)
                                                        .spheres_index
                                                            [spheres__index]
                                                        .spheres_spliton_nPrims)
                                                    .pOffset +
                                                _idx0]
                                         .s))) {
                            if (static_cast<float>(0.001) <
                                distmin_Ray_Sphere(
                                    r,
                                    (&(*spheres)
                                          .prims
                                              [bonsai_reinterpret<
                                                   _spheres_split_layout3>(
                                                   (*spheres)
                                                       .spheres_index
                                                           [spheres__index]
                                                       .spheres_spliton_nPrims)
                                                   .pOffset +
                                               _idx0]
                                          .s))) {
                                if (distmin_Ray_Sphere(
                                        r,
                                        (&(*spheres)
                                              .prims
                                                  [bonsai_reinterpret<
                                                       _spheres_split_layout3>(
                                                       (*spheres)
                                                           .spheres_index
                                                               [spheres__index]
                                                           .spheres_spliton_nPrims)
                                                       .pOffset +
                                                   _idx0]
                                              .s)) < (*_best0)._field0) {
                                    *_best0 = argmin(
                                        _best0,
                                        __tuple_0{
                                            distmin_Ray_Sphere(
                                                r,
                                                (&(*spheres)
                                                      .prims
                                                          [bonsai_reinterpret<
                                                               _spheres_split_layout3>(
                                                               (*spheres)
                                                                   .spheres_index
                                                                       [spheres__index]
                                                                   .spheres_spliton_nPrims)
                                                               .pOffset +
                                                           _idx0]
                                                      .s)),
                                            (*spheres).prims
                                                [bonsai_reinterpret<
                                                     _spheres_split_layout3>(
                                                     (*spheres)
                                                         .spheres_index
                                                             [spheres__index]
                                                         .spheres_spliton_nPrims)
                                                     .pOffset +
                                                 _idx0]});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

__device__ float3 random_in_unit_disk(curandState *local_state) {
    float r = sqrt(curand_uniform(local_state));
    float theta = ((static_cast<float>(2) * static_cast<float>(3.14159)) *
                   curand_uniform(local_state));
    return float3{r * cos(theta), r * sin(theta), static_cast<float>(0)};
}

__device__ float3 defocus_disk_sample(float3 center, float3 defocus_disk_u,
                                      float3 defocus_disk_v,
                                      curandState *local_state) {
    float3 p = random_in_unit_disk(local_state);
    return ((center + (make_float3(p.x) * defocus_disk_u)) +
            (make_float3(p.y) * defocus_disk_v));
}

__device__ Ray build_ray(int32_t i, int32_t j, Camera *cam,
                         curandState *local_state) {
    int32_t width = (*cam).width;
    int32_t _height = (int32_t)((float)width / (*cam).aspect_ratio);
    int32_t *height = &_height;
    *height = (((*height) < 1) ? 1 : (*height));
    float theta =
        (((*cam).vfov * static_cast<float>(3.14159)) / static_cast<float>(180));
    float h = tan(theta / static_cast<float>(2));
    float viewport_height = ((static_cast<float>(2) * h) * (*cam).focus_dist);
    float viewport_width =
        (viewport_height * ((float)width / (float)(*height)));
    float3 camera_center = (*cam).lookfrom;
    float3 w = (((*cam).lookfrom - (*cam).lookat) /
                make_float3(length((*cam).lookfrom - (*cam).lookat)));
    float3 u =
        (cross((*cam).vup, w) / make_float3(length(cross((*cam).vup, w))));
    float3 v = cross(w, u);
    float3 viewport_u = (make_float3(viewport_width) * u);
    float3 viewport_v = (make_float3(viewport_height) * -v);
    float3 pixel_delta_u = (viewport_u / make_float3((float)width));
    float3 pixel_delta_v = (viewport_v / make_float3((float)(*height)));
    float3 viewport_upper_left =
        (((camera_center - (make_float3((*cam).focus_dist) * w)) -
          (viewport_u / make_float3(static_cast<float>(2)))) -
         (viewport_v / make_float3(static_cast<float>(2))));
    float3 pixel00_loc =
        (viewport_upper_left + (make_float3(static_cast<float>(0.5)) *
                                (pixel_delta_u + pixel_delta_v)));
    float defocus_radius =
        ((*cam).focus_dist *
         tan((((*cam).defocus_angle / static_cast<float>(2)) *
              static_cast<float>(3.14159)) /
             static_cast<float>(180)));
    float3 defocus_disk_u = (u * make_float3(defocus_radius));
    float3 defocus_disk_v = (v * make_float3(defocus_radius));
    float3 offset =
        float3{curand_uniform(local_state) - static_cast<float>(0.5),
               curand_uniform(local_state) - static_cast<float>(0.5),
               static_cast<float>(0)};
    float3 pixel_sample =
        ((pixel00_loc + (make_float3(((float)i + offset.x)) * pixel_delta_u)) +
         (make_float3(((float)j + offset.y)) * pixel_delta_v));
    float3 _ray_origin = camera_center;
    float3 *ray_origin = &_ray_origin;
    if (static_cast<float>(0) < (*cam).defocus_angle) {
        *ray_origin = defocus_disk_sample(camera_center, defocus_disk_u,
                                          defocus_disk_v, local_state);
    }
    float3 ray_direction = (pixel_sample - (*ray_origin));
    return Ray{(*ray_origin), ray_direction, INFINITY};
}

__device__ _option0 _traverse_tree0(Ray *r, _spheres_layout0 *spheres) {
    __tuple_0 __best0 = __tuple_0{INFINITY, MaterialSphere{}};
    __tuple_0 *_best0 = &__best0;
    int32_t __queue_count0 = 1;
    int32_t *_queue_count0 = &__queue_count0;
    uint16_t _queue0[64];
    _queue0[0] = 0u;
    do {
        *_queue_count0 -= 1;
        uint16_t spheres__index = _queue0[(*_queue_count0)];
        Sphere _lv2 = Sphere{(*spheres).spheres_index[spheres__index].center,
                             (*spheres).spheres_index[spheres__index].radius};
        if (intersects_Ray_Sphere(r, (&_lv2))) {
            Sphere _lv1 =
                Sphere{(*spheres).spheres_index[spheres__index].center,
                       (*spheres).spheres_index[spheres__index].radius};
            if (static_cast<float>(0.001) < distmax_Ray_Sphere(r, (&_lv1))) {
                Sphere _lv0 =
                    Sphere{(*spheres).spheres_index[spheres__index].center,
                           (*spheres).spheres_index[spheres__index].radius};
                if (distmin_Ray_Sphere(r, (&_lv0)) < (*_best0)._field0) {
                    if ((*spheres).spheres_index[spheres__index].nPrims == 0u) {
                        _queue0[(*_queue_count0)] = (spheres__index + 1u);
                        *_queue_count0 += 1;
                        _queue0[(*_queue_count0)] =
                            (spheres__index +
                             bonsai_reinterpret<_spheres_split_layout2>(
                                 (*spheres)
                                     .spheres_index[spheres__index]
                                     .spheres_spliton_nPrims)
                                 .offset);
                        *_queue_count0 += 1;
                    } else {
                        for (uint16_t _idx0 = 0u;
                             _idx0 <
                             (*spheres).spheres_index[spheres__index].nPrims;
                             _idx0 += 1u) {
                            if (intersects_Ray_Sphere(
                                    r,
                                    (&(*spheres)
                                          .prims
                                              [bonsai_reinterpret<
                                                   _spheres_split_layout3>(
                                                   (*spheres)
                                                       .spheres_index
                                                           [spheres__index]
                                                       .spheres_spliton_nPrims)
                                                   .pOffset +
                                               _idx0]
                                          .s))) {
                                if (static_cast<float>(0.001) <
                                    distmin_Ray_Sphere(
                                        r,
                                        (&(*spheres)
                                              .prims
                                                  [bonsai_reinterpret<
                                                       _spheres_split_layout3>(
                                                       (*spheres)
                                                           .spheres_index
                                                               [spheres__index]
                                                           .spheres_spliton_nPrims)
                                                       .pOffset +
                                                   _idx0]
                                              .s))) {
                                    if (distmin_Ray_Sphere(
                                            r,
                                            (&(*spheres)
                                                  .prims
                                                      [bonsai_reinterpret<
                                                           _spheres_split_layout3>(
                                                           (*spheres)
                                                               .spheres_index
                                                                   [spheres__index]
                                                               .spheres_spliton_nPrims)
                                                           .pOffset +
                                                       _idx0]
                                                  .s)) < (*_best0)._field0) {
                                        *_best0 = argmin(
                                            _best0,
                                            __tuple_0{
                                                distmin_Ray_Sphere(
                                                    r,
                                                    (&(*spheres)
                                                          .prims
                                                              [bonsai_reinterpret<
                                                                   _spheres_split_layout3>(
                                                                   (*spheres)
                                                                       .spheres_index
                                                                           [spheres__index]
                                                                       .spheres_spliton_nPrims)
                                                                   .pOffset +
                                                               _idx0]
                                                          .s)),
                                                (*spheres).prims
                                                    [bonsai_reinterpret<
                                                         _spheres_split_layout3>(
                                                         (*spheres)
                                                             .spheres_index
                                                                 [spheres__index]
                                                             .spheres_spliton_nPrims)
                                                         .pOffset +
                                                     _idx0]});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } while ((*_queue_count0) != 0);
    return (((*_best0)._field0 != INFINITY) ? _option0{(*_best0)._field1, true}
                                            : _option0{});
}

__device__ hit_record get_hit_record(Ray *r, Sphere *s) {
    float t = distmin_Ray_Sphere(r, s);
    float3 p = ((*r).o + (make_float3(t) * (*r).d));
    float3 outward_normal = ((p - (*s).center) / make_float3((*s).radius));
    bool front_face = (dot((*r).d, outward_normal) < static_cast<float>(0));
    float3 normal = (front_face ? outward_normal : -outward_normal);
    hit_record record = hit_record{p, normal, t, front_face};
    return record;
}

__device__ float3 random_unit_vector(curandState *local_state) {
    float x1 = (static_cast<float>(-1) +
                ((static_cast<float>(1) - static_cast<float>(-1)) *
                 curand_uniform(local_state)));
    float x2 = (static_cast<float>(-1) +
                ((static_cast<float>(1) - static_cast<float>(-1)) *
                 curand_uniform(local_state)));
    float s = (((x1 * x1) + (x2 * x2)) + static_cast<float>(1e-08));
    float factor = sqrt(static_cast<float>(2) / s);
    float x = (factor * x1);
    float y = (factor * x2);
    float z = (static_cast<float>(1) - (static_cast<float>(2) * s));
    float len = sqrt(((x * x) + (y * y)) + (z * z));
    return float3{x / len, y / len, z / len};
}

__device__ float reflectance(float cos_theta, float refract_idx) {
    float r0 = ((static_cast<float>(1) - refract_idx) /
                (static_cast<float>(1) + refract_idx));
    float r1 = (r0 * r0);
    return (r1 +
            ((static_cast<float>(1) - r1) *
             pow(static_cast<float>(1) - cos_theta, static_cast<float>(5))));
}

__device__ float3 refract(float3 uv, float3 n, float etai_over_etat) {
    // TODO(cgyurgyik): the fuck is this?
    //
    // error: calling a constexpr __host__ function("min") from a __device__
    // function("scatter") is not allowed. The experimental flag
    // '--expt-relaxed-constexpr' can be used to allow this.
    float cos_theta = min(dot(-uv, n), static_cast<float>(1));
    float3 r_out_perp =
        (make_float3(etai_over_etat) * (uv + (make_float3(cos_theta) * n)));
    float3 r_out_parallel =
        (make_float3(-sqrt(
             abs(static_cast<float>(1) - sum((r_out_perp * r_out_perp))))) *
         n);
    return (r_out_perp + r_out_parallel);
}

__device__ scatter_record scatter(Ray *ray, MaterialSphere *ms,
                                  curandState *local_state) {
    hit_record hit = get_hit_record(ray, (&(*ms).s));
    if ((*ms).material == 0u) {
        float3 _scatter_dir = (hit.normal + random_unit_vector(local_state));
        float3 *scatter_dir = &_scatter_dir;
        if (((abs((*scatter_dir).x) < static_cast<float>(1e-08)) &
             (abs((*scatter_dir).y) < static_cast<float>(1e-08))) &
            (abs((*scatter_dir).z) < static_cast<float>(1e-08))) {
            *scatter_dir = hit.normal;
        }
        Ray l_scattered = Ray{hit.p, (*scatter_dir), INFINITY};
        return scatter_record{(*ms).albedo, l_scattered, true};
    } else {
        if ((*ms).material == 1u) {
            float3 ref = ((*ray).d - (make_float3((static_cast<float>(2) *
                                                   dot((*ray).d, hit.normal))) *
                                      hit.normal));
            float3 reflected =
                ((ref / make_float3(length(ref))) +
                 (make_float3((*ms).fuzz) * random_unit_vector(local_state)));
            Ray m_scattered = Ray{hit.p, reflected, INFINITY};
            return scatter_record{(*ms).albedo, m_scattered, true};
        } else {
            float ri = (hit.front_face ? (static_cast<float>(1) / (*ms).fuzz)
                                       : (*ms).fuzz);
            float3 unit_dir = ((*ray).d / make_float3(length((*ray).d)));
            float cos_theta =
                min(dot(-unit_dir, hit.normal), static_cast<float>(1));
            float sin_theta =
                sqrt(static_cast<float>(1) - (cos_theta * cos_theta));
            bool cannot_refract =
                ((static_cast<float>(1) < (ri * sin_theta)) |
                 (curand_uniform(local_state) < reflectance(cos_theta, ri)));
            float3 direction =
                (cannot_refract
                     ? (unit_dir - (make_float3((static_cast<float>(2) *
                                                 dot(unit_dir, hit.normal))) *
                                    hit.normal))
                     : refract(unit_dir, hit.normal, ri));
            Ray d_scattered = Ray{hit.p, direction, INFINITY};
            return scatter_record{make_float3(static_cast<float>(1),
                                              static_cast<float>(1),
                                              static_cast<float>(1)),
                                  d_scattered, true};
        }
    }
}

__device__ float3 sample(Ray *r, int32_t depth, float3 mult,
                         _spheres_layout0 *spheres, curandState *local_state) {
    Ray _S_r = (*r);
    Ray *S_r = &_S_r;
    int32_t _S_depth = depth;
    int32_t *S_depth = &_S_depth;
    float3 _S_mult = mult;
    float3 *S_mult = &_S_mult;
    _spheres_layout0 _S_spheres = (*spheres);
    _spheres_layout0 *S_spheres = &_S_spheres;
    do {
        if ((*S_depth) <= 0) {
            return make_float3(0, 0, 0);
        }
        _option0 isect = _traverse_tree0(S_r, S_spheres);
        if (isect.set) {
            scatter_record data = scatter(S_r, (&isect.value), local_state);
            if (data.hit) {
                *S_r = data.ray;
                *S_depth = ((*S_depth) - 1);
                *S_mult = ((*S_mult) * data.attenuation);
                *S_spheres = (*S_spheres);
                continue;
            } else {
                return make_float3(0, 0, 0);
            }
        }
        float3 unit_direction = ((*S_r).d / make_float3(length((*S_r).d)));
        float a = ((float)0.5 * (unit_direction.y + 1));
        return ((*S_mult) *
                ((make_float3((1 - a)) * make_float3(1, 1, 1)) +
                 (make_float3(a) * make_float3((float)0.5, (float)0.7, 1))));
    } while (true);
}

__device__ float3 _traverse_array1(int32_t i, int32_t j, Camera *c,
                                   _spheres_layout0 *spheres,
                                   curandState *local_state) {
    float3 __alloc1 = make_float3(static_cast<float>(0));
    float3 *_alloc1 = &__alloc1;
    for (uint32_t _i0 = 0u; _i0 < (*c).samples_per_pixel; _i0 += 1u) {
        Ray _lv0 = build_ray(i, j, c, local_state);
        *_alloc1 += sample((&_lv0), (*c).max_depth, make_float3(1.0), spheres,
                           local_state);
    }
    return (*_alloc1);
}

__device__ float3 pixel(int32_t i, int32_t j, Camera *c,
                        _spheres_layout0 *spheres, curandState *local_state) {
    return (_traverse_array1(i, j, c, spheres, local_state) /
            make_float3((float)(*c).samples_per_pixel));
}

__device__ float linear_to_gamma_f(float l) {
    if (static_cast<float>(0) < l) {
        return sqrt(l);
    }
    return static_cast<float>(0);
}

__device__ int3 to_rgb(float3 v) {
    float3 corrected = float3{linear_to_gamma_f(v.x), linear_to_gamma_f(v.y),
                              linear_to_gamma_f(v.z)};
    return make_int3((make_float3(static_cast<float>(256)) *
                      min(max(corrected, make_float3(static_cast<float>(0))),
                          make_float3(static_cast<float>(0.999))))
                         .x,
                     (make_float3(static_cast<float>(256)) *
                      min(max(corrected, make_float3(static_cast<float>(0))),
                          make_float3(static_cast<float>(0.999))))
                         .y,
                     (make_float3(static_cast<float>(256)) *
                      min(max(corrected, make_float3(static_cast<float>(0))),
                          make_float3(static_cast<float>(0.999))))
                         .z);
}

__global__ void traverse_kernel(int *output, int width, int height, Camera *c,
                                _spheres_layout0 *spheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    int idx = (y * width + x) * 3;
    curandState local_state;
    // Use pixel id as seed.
    curand_init(idx, 0, 0, &local_state);

    int3 rgb = to_rgb(pixel(x, y, c, spheres, &local_state));
    output[idx + 0] = rgb.x;
    output[idx + 1] = rgb.y;
    output[idx + 2] = rgb.z;
}

__host__ int *_parallel_traverse_array0(Camera *c, int32_t height,
                                        _spheres_layout0 *spheres) {
    int width = c->width;
    int total_pixels = width * height;
    int total_size = total_pixels * 3 * sizeof(int);

    // Device memory
    int *device_output;
    Camera *d_camera;
    _spheres_layout0 *d_spheres;

    cudaMalloc(&device_output, total_size);
    cudaMalloc(&d_camera, sizeof(Camera));

    cudaMalloc(&d_spheres, sizeof(_spheres_layout0));

    // Copy the primitives array
    MaterialSphere *d_prims;
    cudaMalloc(&d_prims, spheres->pCount * sizeof(MaterialSphere));
    cudaMemcpy(d_prims, spheres->prims,
               spheres->pCount * sizeof(MaterialSphere),
               cudaMemcpyHostToDevice);

    // Copy the spheres_index array
    _spheres_layout1 *d_spheres_index;
    cudaMalloc(&d_spheres_index, spheres->count * sizeof(_spheres_layout1));
    cudaMemcpy(d_spheres_index, spheres->spheres_index,
               spheres->count * sizeof(_spheres_layout1),
               cudaMemcpyHostToDevice);

    // Create a shallow copy of the struct, then patch the device pointers
    _spheres_layout0 h_spheres_copy = *spheres;
    h_spheres_copy.prims = d_prims;
    h_spheres_copy.spheres_index = d_spheres_index;

    // Now copy the patched struct
    cudaMemcpy(d_spheres, &h_spheres_copy, sizeof(_spheres_layout0),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_camera, c, sizeof(Camera), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15) / 16, (height + 15) / 16);

    // Kernel launch
    traverse_kernel<<<gridDim, blockDim>>>(device_output, width, height,
                                           d_camera, d_spheres);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Host memory for result
    int *host_output = (int *)malloc(total_size);
    cudaMemcpy(host_output, device_output, total_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(device_output);
    cudaFree(d_camera);
    cudaFree(d_spheres);
    cudaFree(d_prims);
    cudaFree(d_spheres_index);
    return host_output;
}

Sphere _unexported_bounding_sphere(Sphere *a, Sphere *b) {
    float3 d = ((*b).center - (*a).center);
    float dist_sq = sum((d * d));
    float dist = sqrt(dist_sq);
    if ((dist + (*b).radius) <= (*a).radius) {
        return (*a);
    } else {
        if ((dist + (*a).radius) <= (*b).radius) {
            return (*b);
        }
    }
    float new_radius =
        (static_cast<float>(0.5) * ((dist + (*a).radius) + (*b).radius));
    float3 direction =
        ((static_cast<float>(0) < dist)
             ? (d / make_float3(dist))
             : make_float3(static_cast<float>(1), static_cast<float>(0),
                           static_cast<float>(0)));
    float3 new_center =
        ((*a).center + (direction * make_float3((new_radius - (*a).radius))));
    return Sphere{new_center, new_radius};
}

float3 at(Ray *r, float t) { return ((*r).o + (make_float3(t) * (*r).d)); }

void bounding_sphere(Sphere *_ret0, Sphere *a, Sphere *b) {
    float3 d = ((*b).center - (*a).center);
    float dist_sq = sum((d * d));
    float dist = sqrt(dist_sq);
    if ((dist + (*b).radius) <= (*a).radius) {
        *_ret0 = (*a);
        return;
    } else {
        if ((dist + (*a).radius) <= (*b).radius) {
            *_ret0 = (*b);
            return;
        }
    }
    float new_radius =
        (static_cast<float>(0.5) * ((dist + (*a).radius) + (*b).radius));
    float3 direction =
        ((static_cast<float>(0) < dist)
             ? (d / make_float3(dist))
             : make_float3(static_cast<float>(1), static_cast<float>(0),
                           static_cast<float>(0)));
    float3 new_center =
        ((*a).center + (direction * make_float3((new_radius - (*a).radius))));
    *_ret0 = Sphere{new_center, new_radius};
    return;
}

float3 clamp(float3 x, float low, float high) {
    return min(max(x, make_float3(low)), make_float3(high));
}

__tuple_1 closestPointonTriangle(Point *pt, Triangle *tri) {
    float3 p = (*pt).vec;
    float3 a = (*tri).p0;
    float3 b = (*tri).p1;
    float3 c = (*tri).p2;
    float3 ab = (b - a);
    float3 ac = (c - a);
    float3 ap = (p - a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= static_cast<float>(0)) {
        if (d2 <= static_cast<float>(0)) {
            return __tuple_1{Point{a}, Point{float3{static_cast<float>(1),
                                                    static_cast<float>(0),
                                                    static_cast<float>(0)}}};
        }
    }
    float3 bp = (p - b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (static_cast<float>(0) <= d3) {
        if (d4 <= d3) {
            return __tuple_1{Point{b}, Point{float3{static_cast<float>(0),
                                                    static_cast<float>(1),
                                                    static_cast<float>(0)}}};
        }
    }
    float vc = ((d1 * d4) - (d3 * d2));
    if (vc <= static_cast<float>(0)) {
        if (static_cast<float>(0) <= d1) {
            if (d3 <= static_cast<float>(0)) {
                float v0 = (d1 / (d1 - d3));
                return __tuple_1{Point{a + (make_float3(v0) * ab)},
                                 Point{float3{static_cast<float>(1) - v0, v0,
                                              static_cast<float>(0)}}};
            }
        }
    }
    float3 cp = (p - c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (static_cast<float>(0) <= d6) {
        if (d5 <= d6) {
            return __tuple_1{Point{c}, Point{float3{static_cast<float>(0),
                                                    static_cast<float>(0),
                                                    static_cast<float>(1)}}};
        }
    }
    float vb = ((d5 * d2) - (d1 * d6));
    if (vb <= static_cast<float>(0)) {
        if (static_cast<float>(0) <= d2) {
            if (d6 <= static_cast<float>(0)) {
                float w0 = (d2 / (d2 - d6));
                return __tuple_1{Point{a + (make_float3(w0) * ac)},
                                 Point{float3{static_cast<float>(1) - w0,
                                              static_cast<float>(0), w0}}};
            }
        }
    }
    float va = ((d3 * d6) - (d5 * d4));
    if (va <= static_cast<float>(0)) {
        if (static_cast<float>(0) <= (d4 - d3)) {
            if (static_cast<float>(0) <= (d5 - d6)) {
                float w1 = ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
                return __tuple_1{Point{b + (make_float3(w1) * (c - b))},
                                 Point{float3{static_cast<float>(0),
                                              static_cast<float>(1) - w1, w1}}};
            }
        }
    }
    float denom = (static_cast<float>(1) / ((va + vb) + vc));
    float v = (vb * denom);
    float w = (vc * denom);
    float u = (va * denom);
    return __tuple_1{Point{(a + (ab * make_float3(v))) + (ac * make_float3(w))},
                     Point{float3{u, v, w}}};
}

float3 cross_(float3 v0, float3 v1) {
    return float3{__prod_diff_f32(v0.y, v1.z, v0.z, v1.y),
                  __prod_diff_f32(v0.z, v1.x, v0.x, v1.z),
                  __prod_diff_f32(v0.x, v1.y, v0.y, v1.x)};
}

float degrees_to_radians(float degrees) {
    return ((degrees * static_cast<float>(3.14159)) / static_cast<float>(180));
}

_option1 intersectsp_ray_aabb(Ray *r, AABB *b) {
    float3 invDir = (make_float3(static_cast<float>(1)) / (*r).d);
    bool3 dirIsNeg = (invDir < make_float3(static_cast<float>(0)));
    float3 low_parts = make_float3((dirIsNeg.x ? (*b).high.x : (*b).low.x),
                                   (dirIsNeg.y ? (*b).high.y : (*b).low.y),
                                   (dirIsNeg.z ? (*b).high.z : (*b).low.z));
    float3 high_parts = make_float3((dirIsNeg.x ? (*b).low.x : (*b).high.x),
                                    (dirIsNeg.y ? (*b).low.y : (*b).high.y),
                                    (dirIsNeg.z ? (*b).low.z : (*b).high.z));
    float3 _tMin = ((low_parts - (*r).o) * invDir);
    float3 *tMin = &_tMin;
    float3 _tMax = ((high_parts - (*r).o) * invDir);
    float3 *tMax = &_tMax;
    *tMax *= (static_cast<float>(1) +
              (static_cast<float>(2) *
               (((float)3 * static_cast<float>(5.96046e-08)) /
                (static_cast<float>(1) -
                 ((float)3 * static_cast<float>(5.96046e-08))))));
    if (((*tMax).y < (*tMin).x) || ((*tMax).x < (*tMin).y)) {
        return _option1{};
    }
    float _tmin = max((*tMin).x, (*tMin).y);
    float *tmin = &_tmin;
    float _tmax = min((*tMax).x, (*tMax).y);
    float *tmax = &_tmax;
    if (((*tMax).z < (*tmin)) || ((*tmax) < (*tMin).z)) {
        return _option1{};
    }
    *tmin = max((*tmin), (*tMin).z);
    *tmax = min((*tmax), (*tMax).z);
    return _option1{FInterval{(*tmin), (*tmax)}, true};
}

float distmax_Ray_AABB(Ray *r, AABB *b) {
    _option1 interval = intersectsp_ray_aabb(r, b);
    if (interval.set) {
        FInterval extract = interval.value;
        return extract.high;
    }
    return INFINITY;
}

_option2 intersectsp_ray_tri(Ray *ray, Triangle *tri) {
    if (sum((cross_((*tri).p2 - (*tri).p0, (*tri).p1 - (*tri).p0) *
             cross_((*tri).p2 - (*tri).p0, (*tri).p1 - (*tri).p0))) ==
        static_cast<float>(0)) {
        return _option2{};
    }
    float3 _p0t = ((*tri).p0 - (*ray).o);
    float3 *p0t = &_p0t;
    float3 _p1t = ((*tri).p1 - (*ray).o);
    float3 *p1t = &_p1t;
    float3 _p2t = ((*tri).p2 - (*ray).o);
    float3 *p2t = &_p2t;
    uint32_t kz = idxmax(abs((*ray).d));
    uint32_t kx = ((kz + 1u) % 3u);
    uint32_t ky = ((kx + 1u) % 3u);
    float3 d = shuffle((*ray).d, {kx, ky, kz});
    *p0t = shuffle((*p0t), {kx, ky, kz});
    *p1t = shuffle((*p1t), {kx, ky, kz});
    *p2t = shuffle((*p2t), {kx, ky, kz});
    float Sx = (-d.x / d.z);
    float Sy = (-d.y / d.z);
    float Sz = (static_cast<float>(1) / d.z);
    *p0t += (Sx * (*p0t).z);
    *p0t += (Sy * (*p0t).z);
    *p1t += (Sx * (*p1t).z);
    *p1t += (Sy * (*p1t).z);
    *p2t += (Sx * (*p2t).z);
    *p2t += (Sy * (*p2t).z);
    float e0 = __prod_diff_f32((*p1t).x, (*p2t).y, (*p1t).y, (*p2t).x);
    float e1 = __prod_diff_f32((*p2t).x, (*p0t).y, (*p2t).y, (*p0t).x);
    float e2 = __prod_diff_f32((*p0t).x, (*p1t).y, (*p0t).y, (*p1t).x);
    if (((e0 < static_cast<float>(0)) || (e1 < static_cast<float>(0))) ||
        (e2 < static_cast<float>(0))) {
        if (((static_cast<float>(0) < e0) || (static_cast<float>(0) < e1)) ||
            (static_cast<float>(0) < e2)) {
            return _option2{};
        }
    }
    float det = ((e0 + e1) + e2);
    if (det == static_cast<float>(0)) {
        return _option2{};
    }
    *p0t *= Sz;
    *p1t *= Sz;
    *p2t *= Sz;
    float tScaled = (((e0 * (*p0t).z) + (e1 * (*p1t).z)) + (e2 * (*p2t).z));
    if ((det < static_cast<float>(0)) && ((static_cast<float>(0) <= tScaled) ||
                                          (tScaled < ((*ray).tmax * det)))) {
        return _option2{};
    } else {
        if ((static_cast<float>(0) < det) &&
            ((tScaled <= static_cast<float>(0)) ||
             (((*ray).tmax * det) < tScaled))) {
            return _option2{};
        }
    }
    float invDet = (static_cast<float>(1) / det);
    float b0 = (e0 * invDet);
    float b1 = (e1 * invDet);
    float b2 = (e2 * invDet);
    float t = (tScaled * invDet);
    float maxZt = max(abs(float3{(*p0t).z, (*p1t).z, (*p2t).z}));
    float deltaZ = ((((float)3 * static_cast<float>(5.96046e-08)) /
                     (static_cast<float>(1) -
                      ((float)3 * static_cast<float>(5.96046e-08)))) *
                    maxZt);
    float maxXt = max(abs(float3{(*p0t).x, (*p1t).x, (*p2t).x}));
    float maxYt = max(abs(float3{(*p0t).y, (*p1t).y, (*p2t).y}));
    float deltaX = ((((float)5 * static_cast<float>(5.96046e-08)) /
                     (static_cast<float>(1) -
                      ((float)5 * static_cast<float>(5.96046e-08)))) *
                    (maxXt + maxZt));
    float deltaY = ((((float)5 * static_cast<float>(5.96046e-08)) /
                     (static_cast<float>(1) -
                      ((float)5 * static_cast<float>(5.96046e-08)))) *
                    (maxYt + maxZt));
    float deltaE = (static_cast<float>(2) *
                    (((((((float)2 * static_cast<float>(5.96046e-08)) /
                         (static_cast<float>(1) -
                          ((float)2 * static_cast<float>(5.96046e-08)))) *
                        maxXt) *
                       maxYt) +
                      (deltaY * maxXt)) +
                     (deltaX * maxYt)));
    float maxE = max(abs(float3{e0, e1, e2}));
    float deltaT = ((static_cast<float>(3) *
                     (((((((float)3 * static_cast<float>(5.96046e-08)) /
                          (static_cast<float>(1) -
                           ((float)3 * static_cast<float>(5.96046e-08)))) *
                         maxE) *
                        maxZt) +
                       (deltaE * maxZt)) +
                      (deltaZ * maxE))) *
                    abs(invDet));
    if (t <= deltaT) {
        return _option2{};
    }
    return _option2{TriangleIntersection{b0, b1, b2, t}, true};
}

float distmax_Ray_Triangle(Ray *ray, Triangle *tri) {
    _option2 isect = intersectsp_ray_tri(ray, tri);
    if (isect.set) {
        TriangleIntersection isect_ = isect.value;
        return isect_.t;
    } else {
        return -INFINITY;
    }
}

float distmin_Point_AABB(Point *pt, AABB *a) {
    return sqrt(SqDistPointAABB(pt, a));
}

float distmin_Point_Triangle(Point *p, Triangle *tri) {
    __tuple_1 pts = closestPointonTriangle(p, tri);
    return length((*p).vec - pts._field0.vec);
}

float distmin_Ray_AABB(Ray *r, AABB *b) {
    _option1 interval = intersectsp_ray_aabb(r, b);
    if (interval.set) {
        FInterval extract = interval.value;
        return extract.low;
    }
    return -INFINITY;
}

__device__ float distmin_Ray_MaterialSphere(Ray *r, MaterialSphere *ms) {
    return distmin_Ray_Sphere(r, (&(*ms).s));
}

float distmin_Ray_Triangle(Ray *ray, Triangle *tri) {
    _option2 isect = intersectsp_ray_tri(ray, tri);
    if (isect.set) {
        TriangleIntersection isect_ = isect.value;
        return isect_.t;
    } else {
        return INFINITY;
    }
}

float gamma(int32_t n) {
    return (
        ((float)n * static_cast<float>(5.96046e-08)) /
        (static_cast<float>(1) - ((float)n * static_cast<float>(5.96046e-08))));
}

__host__ int *image(Camera *c, _spheres_layout0 *spheres) {
    int32_t _height = (int32_t)((float)(*c).width / (*c).aspect_ratio);
    int32_t *height = &_height;
    *height = (((*height) < 1) ? 1 : (*height));
    return _parallel_traverse_array0(c, (*height), spheres);
}

bool intersects_Ray_AABB(Ray *r, AABB *b) {
    _option1 interval = intersectsp_ray_aabb(r, b);
    if (interval.set) {
        FInterval extract = interval.value;
        return ((extract.low < (*r).tmax) &
                (static_cast<float>(0) < extract.high));
    }
    return false;
}

float len_squared(float3 v) { return sum((v * v)); }

__device__ float3 linear_to_gamma_v(float3 l) {
    return float3{linear_to_gamma_f(l.x), linear_to_gamma_f(l.y),
                  linear_to_gamma_f(l.z)};
}

bool near_zero(float3 v) {
    return (((abs(v.x) < static_cast<float>(1e-08)) &
             (abs(v.y) < static_cast<float>(1e-08))) &
            (abs(v.z) < static_cast<float>(1e-08)));
}

float random_float(float low, float high) {
    return (low + ((high - low) * static_cast<float>(rand()) /
                   static_cast<float>(RAND_MAX)));
}

float3 random_vec3f() {
    return float3{static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                  static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                  static_cast<float>(rand()) / static_cast<float>(RAND_MAX)};
}

float3 random_vec3f_in(float low, float high) {
    return float3{low + ((high - low) * static_cast<float>(rand()) /
                         static_cast<float>(RAND_MAX)),
                  low + ((high - low) * static_cast<float>(rand()) /
                         static_cast<float>(RAND_MAX)),
                  low + ((high - low) * static_cast<float>(rand()) /
                         static_cast<float>(RAND_MAX))};
}

float3 reflect(float3 v, float3 n) {
    return (v - (make_float3((static_cast<float>(2) * dot(v, n))) * n));
}

float3 sample_square() {
    return float3{static_cast<float>(rand()) / static_cast<float>(RAND_MAX) -
                      static_cast<float>(0.5),
                  static_cast<float>(rand()) / static_cast<float>(RAND_MAX) -
                      static_cast<float>(0.5),
                  static_cast<float>(0)};
}

float3 unit_vector(float3 v) { return (v / make_float3(length(v))); }

constexpr uint32_t LAMBERTIAN = 0;
constexpr uint32_t METAL = 1;
constexpr uint32_t DIALECTRIC = 2;

constexpr uint32_t MAX_TREE_DEPTH = 64;

inline float random_scalar_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float get_axis(float3 v, int index) {
    if (index == 0) {
        return v.x;
    }
    if (index == 1) {
        return v.y;
    }
    if (index == 2) {
        return v.z;
    }
    __builtin_unreachable();
}

_spheres_layout0 build_tree_simple(std::vector<MaterialSphere> &spheres,
                                   size_t max_prims) {
    _spheres_layout0 tree;
    tree.pCount = spheres.size();
    tree.prims = spheres.data();
    // Just do a simple split, don't even sort for now.
    // First compute the number of nodes needed.
    // Store at most two spheres per leaf node.
    // Then build the tree.
    size_t leaf_count = (tree.pCount + (max_prims - 1)) / max_prims;
    size_t internal_count = leaf_count - 1;
    tree.count = leaf_count + internal_count;
    tree.spheres_index =
        (_spheres_layout1 *)malloc(sizeof(_spheres_layout1) * tree.count);

    uint32_t next_node = 0;

    std::function<uint32_t(uint32_t, uint32_t, uint32_t)> handle_range =
        [&](uint32_t low, uint32_t high, uint32_t depth) -> uint32_t {
        assert(depth < MAX_TREE_DEPTH);
        uint32_t count = high - low;
        uint32_t this_index = next_node++;

        if (count <= 2) {
            // Leaf node
            tree.spheres_index[this_index].nPrims = count;
            *reinterpret_cast<uint16_t *>(
                &tree.spheres_index[this_index].spheres_spliton_nPrims) = low;
            if (count == 1) {
                tree.spheres_index[this_index].center = spheres[low].s.center;
                tree.spheres_index[this_index].radius = spheres[low].s.radius;
            } else if (count == 2) {
                Sphere merged;
                bounding_sphere(&merged, &spheres[low].s, &spheres[low + 1].s);
                tree.spheres_index[this_index].center = merged.center;
                tree.spheres_index[this_index].radius = merged.radius;
            }
        } else {
            // Internal node
            tree.spheres_index[this_index].nPrims = 0;

            float3 min_bound = spheres[low].s.center;
            float3 max_bound = spheres[low].s.center;

            for (uint32_t i = low + 1; i < high; ++i) {
                min_bound = min(min_bound, spheres[i].s.center);
                max_bound = max(max_bound, spheres[i].s.center);
            }

            // Choose axis with greatest extent
            float3 extent = max_bound - min_bound;
            int axis = 0;
            if (extent.y > extent.x)
                axis = 1;
            if (extent.z > extent.y)
                axis = 2;

            // Partition at midpoint along chosen axis
            auto mid_iter = spheres.begin() + low + count / 2;
            std::nth_element(
                spheres.begin() + low, mid_iter, spheres.begin() + high,
                [axis](const MaterialSphere &a, const MaterialSphere &b) {
                    return get_axis(a.s.center, axis) <
                           get_axis(b.s.center, axis);
                });

            uint32_t mid = low + count / 2;

            uint32_t left = handle_range(low, mid, depth + 1);
            uint32_t right = handle_range(mid, high, depth + 1);

            // Set split offset (offset from this node to right child)
            uint32_t offset = right - this_index;
            *reinterpret_cast<uint16_t *>(
                &tree.spheres_index[this_index].spheres_spliton_nPrims) =
                offset;

            // Compute bounding volume
            Sphere merged;
            Sphere s1 = Sphere{tree.spheres_index[left].center,
                               tree.spheres_index[left].radius};
            Sphere s2 = Sphere{tree.spheres_index[right].center,
                               tree.spheres_index[right].radius};
            bounding_sphere(&merged, &s1, &s2);

            tree.spheres_index[this_index].center = merged.center;
            tree.spheres_index[this_index].radius = merged.radius;
        }

        return this_index;
    };

    handle_range(/*low=*/0, /*high=*/tree.pCount, /*depth=*/0);
    return tree;
}

int main(int argc, char **argv) {
    using clock = std::chrono::high_resolution_clock;

    auto t0 = clock::now();

    std::vector<MaterialSphere> spheres{
        // Ground
        {Sphere{{0, -1000, 0}, 1000}, LAMBERTIAN, {0.5, 0.5, 0.5}, 0.0},

        {Sphere{{0, 1, 0}, 1}, DIALECTRIC, {0, 0, 0}, 1.5},
        {Sphere{{-4, 1, 0}, 1}, LAMBERTIAN, {0.4, 0.2, 0.1}, 0.0},
        {Sphere{{4, 1, 0}, 1}, METAL, {0.7, 0.6, 0.5}, 0.0},
    };
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_scalar_float();
            float3 center = {
                static_cast<float>(a + 0.9 * random_scalar_float()), 0.2,
                static_cast<float>(b + 0.9 * random_scalar_float())};

            float3 a = {4, 0.2, 0};

            float3 diff = (center - a);
            float len = length(diff);

            if (len > 0.9) {
                if (choose_mat < 0.8) {
                    // diffuse
                    float3 r0 = {random_scalar_float(), random_scalar_float(),
                                 random_scalar_float()};
                    float3 r1 = {random_scalar_float(), random_scalar_float(),
                                 random_scalar_float()};
                    auto albedo = r0 * r1;
                    spheres.push_back(
                        {Sphere{center, 0.2}, LAMBERTIAN, albedo, 0.0});
                } else if (choose_mat < 0.95) {
                    // metal
                    float3 albedo = {random_float(0.5, 1), random_float(0.5, 1),
                                     random_float(0.5, 1)};
                    float fuzz = random_float(0, 0.5);
                    spheres.push_back(
                        {Sphere{center, 0.2}, METAL, albedo, fuzz});
                } else {
                    // glass
                    spheres.push_back(
                        {Sphere{center, 0.2}, DIALECTRIC, {0, 0, 0}, 1.5});
                }
            }
        }
    }

    _spheres_layout0 tree = build_tree_simple(spheres, 1);
    Camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.width = 1423; // makes height = 800
    cam.samples_per_pixel = 50;
    cam.max_depth = 10;

    cam.vfov = 20;
    cam.lookfrom = {13, 2, 3};
    cam.lookat = {0, 0, 0};
    cam.vup = {0, 1, 0};

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    int image_width = cam.width;
    float image_height = (int)(cam.width / cam.aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    auto t1 = clock::now();
    // Render
    int *im = image(&cam, &tree);
    auto t2 = clock::now();
    std::string output_filename = "rtiow-image-cuda.ppm";
    std::ofstream out(output_filename);
    if (!out) {
        std::cerr << "Error: Cannot open file " << output_filename
                  << " for writing\n";
        cudaFree(im);
        return 1;
    }

    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int ir = im[(j * image_width + i) * 3 + 0];
            int ig = im[(j * image_width + i) * 3 + 1];
            int ib = im[(j * image_width + i) * 3 + 2];
            out << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    out.flush();
    auto t3 = clock::now();

    auto setup_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto render_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto write_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cerr << "Setup time: " << setup_ms << " ms\n";
    std::cerr << "Render time: " << render_ms << " ms\n";
    std::cerr << "Write-to-output time: " << write_ms << " ms\n";

    free(im);
    return 0;
}
