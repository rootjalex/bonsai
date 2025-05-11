#pragma once

// Pulled from [1], with our own addendums for Bonsai.
// [1] https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <math.h>
#include <type_traits>

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

__forceinline__ __host__ __device__ template <typename T>
T argmin(T *current, T update) {
    if (current->_field0 < update._field0) {
        return *current;
    }
    return update;
}

__forceinline__ __host__ __device__ template <typename T>
T *argmax(T *current, T update) {
    if (current->_field0 > update._field0) {
        return current;
    }
    return &update;
}

// Mimics curand_uniform by producing an output in (0, 1].
// https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE_1gf1ba3a7a4a53b2bee1d7c1c7b837c00d
template <typename T>
__forceinline__ __host__ T random() {
    T v = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX);
    // Scale to (0, 1].
    return (static_cast<T>(1.0) - std::numeric_limits<T>::epsilon()) * v +
           std::numeric_limits<T>::epsilon();
}

// Jesus christ
__forceinline__ __host__ __device__ template <typename O, typename I>
O bonsai_reinterpret(I input) {
    static_assert(sizeof(O) == sizeof(I));
    static_assert(std::is_trivially_copyable<O>::value);
    static_assert(std::is_trivially_copyable<I>::value);
    I *i = &input;
    O *output;
    output = reinterpret_cast<O *>(i);
    return *output;
}
