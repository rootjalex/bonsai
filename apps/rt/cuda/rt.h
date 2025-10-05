#include "helpers.h"

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/universal_vector.h>

#include <stdint.h>
#include <variant>

struct AABBNode;
struct OBBNode;
struct Leaf;
using BVH = std::variant<AABBNode, OBBNode, Leaf>;

struct AABB {
    float3 low;
    float3 high;
};

struct AABBNode {
    cuda::std::array<BVH *, 8> aabb_children;
    cuda::std::array<float3, 8> aabb_low;
    cuda::std::array<float3, 8> aabb_high;
};

struct FInterval {
    float low;
    float high;
};

struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};

struct Leaf {
    uint8_t nprims;
    Triangle *data;
};

struct OBB {
    float3 obb_low;
    float3 obb_high;
    cuda::std::array<float4, 3> orientation;
};

struct OBBNode {
    cuda::std::array<BVH *, 8> obb_children;
    cuda::std::array<float4, 3> orientation;
    cuda::std::array<float3, 8> obb_low;
    cuda::std::array<float3, 8> obb_high;
};

struct Point {
    float3 vec;
};

struct Ray {
    float3 o;
    float3 d;
    float tmax = INFINITY;
};

struct Sphere {
    float3 center;
    float radius;
};

struct TriangleIntersection {
    float b0;
    float b1;
    float b2;
    float t;
};

struct Aabbs {
    cuda::std::array<float3, 8> aabb_low;
    cuda::std::array<float3, 8> aabb_high;
    cuda::std::array<uint64_t, 8> aabb_children;
} __attribute__((packed));

struct Obbs {
    float3 mlo;
    float3 mex;
    cuda::std::array<uchar3, 8> qlo;
    cuda::std::array<uchar3, 8> qhi;
    cuda::std::array<char4, 3> qorientation;
    cuda::std::array<uint64_t, 8> obb_children;
} __attribute__((packed));

struct Triangles {
    uint64_t primitive_count;
    Triangle *primitives;
    uint64_t aabb_count;
    Aabbs *aabbs;
    uint64_t obb_count;
    Obbs *obbs;
} __attribute__((packed));

struct _ctx0 {
    int64_t n;
    cuda::std::optional<Triangle> *_alloc0;
    Ray *rays;
    Triangles *triangles;
};

__device__ float __prod_diff_f32(float a, float b, float c, float d) {
    float cd = (c * d);
    float diff = fmaf(a, b, -cd);
    float err = fmaf(-c, d, cd);
    return (diff + err);
}

__device__ cuda::std::array<float4, 3>
dequantize_orientation(cuda::std::array<char4, 3> x) {
    char4 _t0 = x[0];
    char4 _t4 = x[1];
    char4 _t8 = x[2];
    return cuda::std::array<float4, 3>{
        float4{(float)_t0.x / (float)127, (float)_t0.y / (float)127,
               (float)_t0.z / (float)127, (float)_t0.w / (float)127},
        float4{(float)_t4.x / (float)127, (float)_t4.y / (float)127,
               (float)_t4.z / (float)127, (float)_t4.w / (float)127},
        float4{(float)_t8.x / (float)127, (float)_t8.y / (float)127,
               (float)_t8.z / (float)127, (float)_t8.w / (float)127}};
}

__device__ float gamma(int32_t n) {
    float _t1 = ((float)n * (float)5.96046e-08);
    return (_t1 / (1 - _t1));
}

__device__ cuda::std::optional<FInterval> intersectsp_ray_aabb(Ray *r,
                                                               AABB *b) {
    float3 _t1 = (make_float3(1) / (*r).d);
    bool3 dirIsNeg = (_t1 < make_float3(0));
    float3 _t2 = (*b).high;
    float3 _t3 = (*b).low;
    float3 low_parts =
        make_float3((dirIsNeg.x ? _t2.x : _t3.x), (dirIsNeg.y ? _t2.y : _t3.y),
                    (dirIsNeg.z ? _t2.z : _t3.z));
    float3 high_parts =
        make_float3((dirIsNeg.x ? _t3.x : _t2.x), (dirIsNeg.y ? _t3.y : _t2.y),
                    (dirIsNeg.z ? _t3.z : _t2.z));
    float3 _t6 = (*r).o;
    float3 tMin = ((low_parts - _t6) * _t1);
    float3 tMax = ((high_parts - _t6) * _t1);
    tMax *= (1 + ((float)2 * gamma(3)));
    if ((tMax.y < tMin.x) || (tMax.x < tMin.y)) {
        return cuda::std::nullopt;
    }
    float tmin = fmaxf(tMin.x, tMin.y);
    float tmax = fminf(tMax.x, tMax.y);
    if ((tMax.z < tmin) || (tmax < tMin.z)) {
        return cuda::std::nullopt;
    }
    tmin = fmaxf(tmin, tMin.z);
    tmax = fminf(tmax, tMax.z);
    return FInterval{tmin, tmax};
}

__device__ float distmin_Ray_AABB(Ray *r, AABB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return extract.low;
    }
    return -INFINITY;
}

__device__ cuda::std::optional<FInterval> intersectsp_ray_obb(Ray *r, OBB *b) {
    cuda::std::array<float4, 3> _t0 = (*b).orientation;
    float4 _t1 = _t0[0];
    float _t2 = _t1.x;
    float3 _t3 = (*r).o;
    float _t4 = _t3.x;
    float _t8 = _t1.y;
    float _t10 = _t3.y;
    float _t15 = _t1.z;
    float _t17 = _t3.z;
    float _t23 = ((((_t2 * _t4) + (_t8 * _t10)) + (_t15 * _t17)) + _t1.w);
    float4 _t25 = _t0[1];
    float _t26 = _t25.x;
    float _t32 = _t25.y;
    float _t39 = _t25.z;
    float _t47 = ((((_t26 * _t4) + (_t32 * _t10)) + (_t39 * _t17)) + _t25.w);
    float4 _t49 = _t0[2];
    float _t50 = _t49.x;
    float _t56 = _t49.y;
    float _t63 = _t49.z;
    float _t71 = ((((_t50 * _t4) + (_t56 * _t10)) + (_t63 * _t17)) + _t49.w);
    float3 _t75 = (*r).d;
    float _t76 = _t75.x;
    float _t82 = _t75.y;
    float _t89 = _t75.z;
    float inv_dx = (1 / (((_t2 * _t76) + (_t8 * _t82)) + (_t15 * _t89)));
    float inv_dy = (1 / (((_t26 * _t76) + (_t32 * _t82)) + (_t39 * _t89)));
    float inv_dz = (1 / (((_t50 * _t76) + (_t56 * _t82)) + (_t63 * _t89)));
    float3 _t132 = (*b).obb_low;
    float _t135 = ((_t132.x - _t23) * inv_dx);
    float3 _t136 = (*b).obb_high;
    float _t139 = ((_t136.x - _t23) * inv_dx);
    float tmin_x = fminf(_t135, _t139);
    float tmax_x = fmaxf(_t135, _t139);
    float _t143 = ((_t132.y - _t47) * inv_dy);
    float _t147 = ((_t136.y - _t47) * inv_dy);
    float tmin_y = fminf(_t143, _t147);
    float tmax_y = fmaxf(_t143, _t147);
    float _t151 = ((_t132.z - _t71) * inv_dz);
    float _t155 = ((_t136.z - _t71) * inv_dz);
    float tmin_z = fminf(_t151, _t155);
    float tmax_z = fmaxf(_t151, _t155);
    float _t157 = fmaxf(fmaxf(tmin_x, tmin_y), tmin_z);
    float _t159 = fminf(fminf(tmax_x, tmax_y), tmax_z);
    if (_t159 < _t157) {
        return cuda::std::nullopt;
    }
    if (_t159 < 0) {
        return cuda::std::nullopt;
    }
    return FInterval{fmaxf(_t157, 0), _t159};
}

__device__ float distmin_Ray_OBB(Ray *r, OBB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_obb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return extract.low;
    }
    return -INFINITY;
}

__device__ float3 cross_(float3 v0, float3 v1) {
    float _t0 = v0.y;
    float _t1 = v1.z;
    float _t2 = v0.z;
    float _t3 = v1.y;
    float _t5 = v1.x;
    float _t6 = v0.x;
    return float3{__prod_diff_f32(_t0, _t1, _t2, _t3),
                  __prod_diff_f32(_t2, _t5, _t6, _t1),
                  __prod_diff_f32(_t6, _t3, _t0, _t5)};
}

__device__ cuda::std::optional<TriangleIntersection>
intersectsp_ray_tri(Ray *ray, Triangle *tri) {
    float3 _t0 = (*tri).p2;
    float3 _t1 = (*tri).p0;
    float3 _t2 = (*tri).p1;
    if (sum((cross_(_t0 - _t1, _t2 - _t1) * cross_(_t0 - _t1, _t2 - _t1))) ==
        0) {
        return cuda::std::nullopt;
    }
    float3 _t5 = (*ray).o;
    float3 p0t = (_t1 - _t5);
    float3 p1t = (_t2 - _t5);
    float3 p2t = (_t0 - _t5);
    float3 _t10 = (*ray).d;
    uint32_t kz = idxmax(abs(_t10));
    uint32_t kx = ((kz + 1u) % 3u);
    uint32_t ky = ((kx + 1u) % 3u);
    float3 d = shuffle(_t10, {kx, ky, kz});
    p0t = shuffle(p0t, {kx, ky, kz});
    p1t = shuffle(p1t, {kx, ky, kz});
    p2t = shuffle(p2t, {kx, ky, kz});
    float _t13 = d.z;
    float _t14 = (-d.x / _t13);
    float _t17 = (-d.y / _t13);
    float Sz = (1 / _t13);
    p0t.x += (_t14 * p0t.z);
    p0t.y += (_t17 * p0t.z);
    p1t.x += (_t14 * p1t.z);
    p1t.y += (_t17 * p1t.z);
    p2t.x += (_t14 * p2t.z);
    p2t.y += (_t17 * p2t.z);
    float _t20 = __prod_diff_f32(p1t.x, p2t.y, p1t.y, p2t.x);
    float _t21 = __prod_diff_f32(p2t.x, p0t.y, p2t.y, p0t.x);
    float _t22 = __prod_diff_f32(p0t.x, p1t.y, p0t.y, p1t.x);
    if (((_t20 < 0) || (_t21 < 0)) || (_t22 < 0)) {
        if (((0 < _t20) || (0 < _t21)) || (0 < _t22)) {
            return cuda::std::nullopt;
        }
    }
    float _t24 = ((_t20 + _t21) + _t22);
    if (_t24 == 0) {
        return cuda::std::nullopt;
    }
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    float tScaled = (((_t20 * p0t.z) + (_t21 * p1t.z)) + (_t22 * p2t.z));
    if ((_t24 < 0) && ((0 <= tScaled) || (tScaled < ((*ray).tmax * _t24)))) {
        return cuda::std::nullopt;
    } else {
        if ((0 < _t24) &&
            ((tScaled <= 0) || (((*ray).tmax * _t24) < tScaled))) {
            return cuda::std::nullopt;
        }
    }
    float invDet = (1 / _t24);
    float b0 = (_t20 * invDet);
    float b1 = (_t21 * invDet);
    float b2 = (_t22 * invDet);
    float t = (tScaled * invDet);
    float maxZt = max(abs(float3{p0t.z, p1t.z, p2t.z}));
    float _t25 = gamma(3);
    float deltaZ = (_t25 * maxZt);
    float maxXt = max(abs(float3{p0t.x, p1t.x, p2t.x}));
    float maxYt = max(abs(float3{p0t.y, p1t.y, p2t.y}));
    float _t26 = gamma(5);
    float deltaX = (_t26 * (maxXt + maxZt));
    float deltaY = (_t26 * (maxYt + maxZt));
    float deltaE =
        ((float)2 * ((((gamma(2) * maxXt) * maxYt) + (deltaY * maxXt)) +
                     (deltaX * maxYt)));
    float maxE = max(abs(float3{_t20, _t21, _t22}));
    float deltaT = (((float)3 * ((((_t25 * maxE) * maxZt) + (deltaE * maxZt)) +
                                 (deltaZ * maxE))) *
                    fabsf(invDet));
    if (t <= deltaT) {
        return cuda::std::nullopt;
    }
    return TriangleIntersection{b0, b1, b2, t};
}

__device__ float distmin_Ray_Triangle(Ray *ray, Triangle *tri) {
    cuda::std::optional<TriangleIntersection> isect =
        intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        TriangleIntersection isect_ = *isect;
        return isect_.t;
    } else {
        return INFINITY;
    }
}

__device__ bool intersects_Ray_AABB(Ray *r, AABB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return ((extract.low < (*r).tmax) & (0 < extract.high));
    }
    return false;
}

__device__ bool intersects_Ray_OBB(Ray *r, OBB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_obb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return ((extract.low < (*r).tmax) & (0 < extract.high));
    }
    return false;
}

__device__ cuda::std::optional<Triangle> _traverse_tree0(Ray *ray,
                                                         Triangles *triangles) {
    cuda::std::tuple<float, Triangle> _best0 =
        cuda::std::tuple<float, Triangle>{INFINITY, Triangle{}};
    int32_t _queue_count0 = 1;
    uint64_t _queue0[64];
    _queue0[0] = 2u;
    do {
        _queue_count0 -= 1;
        uint64_t index = _queue0[_queue_count0];
        if (index == 18446744073709551615u) {
            if (_queue_count0 <= 0) {
                break;
            } else {
                continue;
            }
        }
        uint64_t _t814 = slice<0, 2>(index);
        if (_t814 == 2u) {
            Aabbs _t18 = (*triangles).aabbs[slice<4, 63>(index)];
            cuda::std::array<float3, 8> _t19 = _t18.aabb_low;
            cuda::std::array<float3, 8> _t24 = _t18.aabb_high;
            AABB _t26 = AABB{_t19[0], _t24[0]};
            if (intersects_Ray_AABB(ray, (&_t26))) {
                if (distmin_Ray_AABB(ray, (&_t26)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[0u];
                    _queue_count0 += 1;
                }
            }
            AABB _t53 = AABB{_t19[1], _t24[1]};
            if (intersects_Ray_AABB(ray, (&_t53))) {
                if (distmin_Ray_AABB(ray, (&_t53)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[1u];
                    _queue_count0 += 1;
                }
            }
            AABB _t80 = AABB{_t19[2], _t24[2]};
            if (intersects_Ray_AABB(ray, (&_t80))) {
                if (distmin_Ray_AABB(ray, (&_t80)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[2u];
                    _queue_count0 += 1;
                }
            }
            AABB _t107 = AABB{_t19[3], _t24[3]};
            if (intersects_Ray_AABB(ray, (&_t107))) {
                if (distmin_Ray_AABB(ray, (&_t107)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[3u];
                    _queue_count0 += 1;
                }
            }
            AABB _t134 = AABB{_t19[4], _t24[4]};
            if (intersects_Ray_AABB(ray, (&_t134))) {
                if (distmin_Ray_AABB(ray, (&_t134)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[4u];
                    _queue_count0 += 1;
                }
            }
            AABB _t161 = AABB{_t19[5], _t24[5]};
            if (intersects_Ray_AABB(ray, (&_t161))) {
                if (distmin_Ray_AABB(ray, (&_t161)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[5u];
                    _queue_count0 += 1;
                }
            }
            AABB _t188 = AABB{_t19[6], _t24[6]};
            if (intersects_Ray_AABB(ray, (&_t188))) {
                if (distmin_Ray_AABB(ray, (&_t188)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[6u];
                    _queue_count0 += 1;
                }
            }
            AABB _t215 = AABB{_t19[7], _t24[7]};
            if (intersects_Ray_AABB(ray, (&_t215))) {
                if (distmin_Ray_AABB(ray, (&_t215)) <
                    cuda::std::get<0>(_best0)) {
                    _queue0[_queue_count0] = _t18.aabb_children[7u];
                    _queue_count0 += 1;
                }
            }
        } else {
            if (_t814 == 3u) {
                Obbs _t257 = (*triangles).obbs[slice<4, 63>(index)];
                float3 _t258 = _t257.mlo;
                float3 _t262 = _t257.mex;
                cuda::std::array<float3, 8> _t267 = cuda::std::array<float3, 8>{
                    _t258 + ((make_float3(_t257.qlo[0].x, _t257.qlo[0].y,
                                          _t257.qlo[0].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[1].x, _t257.qlo[1].y,
                                          _t257.qlo[1].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[2].x, _t257.qlo[2].y,
                                          _t257.qlo[2].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[3].x, _t257.qlo[3].y,
                                          _t257.qlo[3].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[4].x, _t257.qlo[4].y,
                                          _t257.qlo[4].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[5].x, _t257.qlo[5].y,
                                          _t257.qlo[5].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[6].x, _t257.qlo[6].y,
                                          _t257.qlo[6].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qlo[7].x, _t257.qlo[7].y,
                                          _t257.qlo[7].z) /
                              make_float3((float)255)) *
                             _t262)};
                cuda::std::array<float3, 8> _t281 = cuda::std::array<float3, 8>{
                    _t258 + ((make_float3(_t257.qhi[0].x, _t257.qhi[0].y,
                                          _t257.qhi[0].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[1].x, _t257.qhi[1].y,
                                          _t257.qhi[1].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[2].x, _t257.qhi[2].y,
                                          _t257.qhi[2].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[3].x, _t257.qhi[3].y,
                                          _t257.qhi[3].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[4].x, _t257.qhi[4].y,
                                          _t257.qhi[4].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[5].x, _t257.qhi[5].y,
                                          _t257.qhi[5].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[6].x, _t257.qhi[6].y,
                                          _t257.qhi[6].z) /
                              make_float3((float)255)) *
                             _t262),
                    _t258 + ((make_float3(_t257.qhi[7].x, _t257.qhi[7].y,
                                          _t257.qhi[7].z) /
                              make_float3((float)255)) *
                             _t262)};
                cuda::std::array<float4, 3> _t287 =
                    dequantize_orientation(_t257.qorientation);
                OBB _t288 = OBB{_t267[0], _t281[0], _t287};
                if (intersects_Ray_OBB(ray, (&_t288))) {
                    if (distmin_Ray_OBB(ray, (&_t288)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[0u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t361 = OBB{_t267[1], _t281[1], _t287};
                if (intersects_Ray_OBB(ray, (&_t361))) {
                    if (distmin_Ray_OBB(ray, (&_t361)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[1u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t434 = OBB{_t267[2], _t281[2], _t287};
                if (intersects_Ray_OBB(ray, (&_t434))) {
                    if (distmin_Ray_OBB(ray, (&_t434)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[2u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t507 = OBB{_t267[3], _t281[3], _t287};
                if (intersects_Ray_OBB(ray, (&_t507))) {
                    if (distmin_Ray_OBB(ray, (&_t507)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[3u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t580 = OBB{_t267[4], _t281[4], _t287};
                if (intersects_Ray_OBB(ray, (&_t580))) {
                    if (distmin_Ray_OBB(ray, (&_t580)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[4u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t653 = OBB{_t267[5], _t281[5], _t287};
                if (intersects_Ray_OBB(ray, (&_t653))) {
                    if (distmin_Ray_OBB(ray, (&_t653)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[5u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t726 = OBB{_t267[6], _t281[6], _t287};
                if (intersects_Ray_OBB(ray, (&_t726))) {
                    if (distmin_Ray_OBB(ray, (&_t726)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[6u];
                        _queue_count0 += 1;
                    }
                }
                OBB _t799 = OBB{_t267[7], _t281[7], _t287};
                if (intersects_Ray_OBB(ray, (&_t799))) {
                    if (distmin_Ray_OBB(ray, (&_t799)) <
                        cuda::std::get<0>(_best0)) {
                        _queue0[_queue_count0] = _t257.obb_children[7u];
                        _queue_count0 += 1;
                    }
                }
            } else {
                uint64_t _t811 = slice<4, 63>(index);
                for (uint64_t _idx0 = _t811;
                     _idx0 <
                     (_t811 + (uint64_t)(uint8_t)(slice<1, 8>(index) + 1u));
                     _idx0 += 1u) {
                    Triangle _t810 = (*triangles).primitives[_idx0];
                    if (intersectsp_ray_tri(ray, (&_t810)).has_value()) {
                        float _t807 = distmin_Ray_Triangle(ray, (&_t810));
                        if (_t807 < cuda::std::get<0>(_best0)) {
                            _best0 = argmin(_best0,
                                            cuda::std::tuple<float, Triangle>{
                                                _t807, _t810});
                        }
                    }
                }
            }
        }
    } while (_queue_count0 != 0);
    return ((cuda::std::get<0>(_best0) != INFINITY)
                ? cuda::std::optional<Triangle>{cuda::std::get<1>(_best0)}
                : cuda::std::nullopt);
}

__global__ void _parfunc0(_ctx0 ctx0) {
    int64_t tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int64_t _i0 = tid;
    if (ctx0.n <= tid) {
        return;
    }
    Ray _lv0 = ctx0.rays[_i0];
    ctx0._alloc0[_i0] = _traverse_tree0((&_lv0), ctx0.triangles);
    return;
}

__host__ cuda::std::optional<Triangle> *_traverse_array0(int64_t n, Ray *rays,
                                                         Triangles *triangles) {
    cuda::std::optional<Triangle> *_alloc0;
    cudaError_t err = cudaMalloc((void **)&_alloc0,
                                 n * sizeof(cuda::std::optional<Triangle>));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed:\n");
        fprintf(stderr, "  Error name: %s\n", cudaGetErrorName(err));
        fprintf(stderr, "  Error string: %s\n", cudaGetErrorString(err));
        fprintf(stderr, "  Requested size: %zu bytes\n",
                n * sizeof(cuda::std::optional<Triangle>));
        assert(false);
    }
    Ray *d_rays;
    cudaMallocAndCopyToDevice((void **)&d_rays, rays, n * sizeof(Ray));
    Triangle *__primitives;
    cudaMallocAndCopyToDevice((void **)&__primitives, (*triangles).primitives,
                              (*triangles).primitive_count * sizeof(Triangle));
    Aabbs *__aabbs;
    cudaMallocAndCopyToDevice((void **)&__aabbs, (*triangles).aabbs,
                              (*triangles).aabb_count * sizeof(Aabbs));
    Obbs *__obbs;
    cudaMallocAndCopyToDevice((void **)&__obbs, (*triangles).obbs,
                              (*triangles).obb_count * sizeof(Obbs));
    Triangles h_triangles = *triangles;
    h_triangles.primitives = __primitives;
    h_triangles.aabbs = __aabbs;
    h_triangles.obbs = __obbs;
    Triangles *d_triangles;
    cudaMallocAndCopyToDevice((void **)&d_triangles, &h_triangles,
                              sizeof(Triangles));
    _ctx0 ctx = _ctx0{n, _alloc0, d_rays, d_triangles};
    _parfunc0<<<((n + 255) / 256), 256>>>(ctx);
    cudaDeviceSynchronize();
    cuda::std::optional<Triangle> *h__alloc0;
    mallocAndCopyFromDevice((void **)&h__alloc0, _alloc0,
                            n * sizeof(cuda::std::optional<Triangle>));
    cudaFree(__primitives);
    cudaFree(__aabbs);
    cudaFree(__obbs);
    cudaFree(_alloc0);
    cudaFree(d_rays);
    cudaFree(d_triangles);
    _alloc0 = h__alloc0;
    return _alloc0;
}

__host__ float3 compute_merged_extent(cuda::std::array<float3, 8> lo,
                                      cuda::std::array<float3, 8> hi) {
    float3 mlo = min(
        lo[0],
        min(lo[1],
            min(lo[2], min(lo[3], min(lo[4], min(lo[5], min(lo[6], lo[7])))))));
    float3 mhi = max(
        hi[0],
        max(hi[1],
            max(hi[2], max(hi[3], max(hi[4], max(hi[5], max(hi[6], hi[7])))))));
    return (mhi - mlo);
}

__host__ uchar3 to_u8_ceil(float3 f) {
    float3 f1 = ceilf(f);
    float3 f2 = max(make_float3(0), min(f1, make_float3((float)255)));
    return make_uchar3(f2.x, f2.y, f2.z);
}

__host__ cuda::std::array<uchar3, 8>
quantize_bounds_high(cuda::std::array<float3, 8> low,
                     cuda::std::array<float3, 8> high) {
    float3 mlo = min(
        low[0],
        min(low[1],
            min(low[2],
                min(low[3], min(low[4], min(low[5], min(low[6], low[7])))))));
    float3 mex = compute_merged_extent(low, high);
    return cuda::std::array<uchar3, 8>{
        to_u8_ceil(((high[0] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[1] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[2] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[3] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[4] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[5] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[6] - mlo) / mex) * make_float3((float)255)),
        to_u8_ceil(((high[7] - mlo) / mex) * make_float3((float)255))};
}

__host__ uchar3 to_u8_floor(float3 f) {
    float3 f1 = floorf(f);
    float3 f2 = max(make_float3(0), min(f1, make_float3((float)255)));
    return make_uchar3(f2.x, f2.y, f2.z);
}

__host__ cuda::std::array<uchar3, 8>
quantize_bounds_low(cuda::std::array<float3, 8> low,
                    cuda::std::array<float3, 8> high) {
    float3 mlo = min(
        low[0],
        min(low[1],
            min(low[2],
                min(low[3], min(low[4], min(low[5], min(low[6], low[7])))))));
    float3 mex = compute_merged_extent(low, high);
    return cuda::std::array<uchar3, 8>{
        to_u8_floor(((low[0] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[1] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[2] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[3] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[4] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[5] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[6] - mlo) / mex) * make_float3((float)255)),
        to_u8_floor(((low[7] - mlo) / mex) * make_float3((float)255))};
}

__host__ int8_t to_i8(float f) {
    float f1 = roundf(f);
    float f2 = fmaxf((float)-128, fminf(f1, (float)127));
    return (int8_t)f2;
}

__host__ cuda::std::array<char4, 3>
quantize_orientation(cuda::std::array<float4, 3> x) {
    float4 _t0 = x[0];
    float4 _t4 = x[1];
    float4 _t8 = x[2];
    return cuda::std::array<char4, 3>{
        char4{to_i8(_t0.x * (float)127), to_i8(_t0.y * (float)127),
              to_i8(_t0.z * (float)127), to_i8(_t0.w * (float)127)},
        char4{to_i8(_t4.x * (float)127), to_i8(_t4.y * (float)127),
              to_i8(_t4.z * (float)127), to_i8(_t4.w * (float)127)},
        char4{to_i8(_t8.x * (float)127), to_i8(_t8.y * (float)127),
              to_i8(_t8.z * (float)127), to_i8(_t8.w * (float)127)}};
}

__host__ uint64_t rec_build_triangles(BVH *node_, Triangles *ST,
                                      size_t *aabbs_index, size_t *obbs_index,
                                      size_t *primitives_index) {
    if (!node_) {
        return 18446744073709551615u;
    }
    if (std::holds_alternative<AABBNode>(*node_)) {
        const AABBNode &node = std::get<AABBNode>(*node_);
        size_t this_index = (*aabbs_index);
        (*aabbs_index) += 1;
        (*ST).aabbs[this_index].aabb_low = node.aabb_low;
        (*ST).aabbs[this_index].aabb_high = node.aabb_high;
        uint64_t aabb_children_index[8];
        for (int32_t __r = 0; __r < 8; __r += 1) {
            aabb_children_index[__r] =
                rec_build_triangles(node.aabb_children[__r], ST, aabbs_index,
                                    obbs_index, primitives_index);
            (*ST).aabbs[this_index].aabb_children[__r] =
                aabb_children_index[__r];
        }
        return ((this_index << (uint64_t)4) | (uint64_t)2);
    } else if (std::holds_alternative<OBBNode>(*node_)) {
        const OBBNode &node = std::get<OBBNode>(*node_);
        size_t this_index = (*obbs_index);
        (*obbs_index) += 1;
        (*ST).obbs[this_index].mlo =
            min(node.obb_low[0],
                min(node.obb_low[1],
                    min(node.obb_low[2],
                        min(node.obb_low[3],
                            min(node.obb_low[4],
                                min(node.obb_low[5],
                                    min(node.obb_low[6], node.obb_low[7])))))));
        (*ST).obbs[this_index].mex =
            compute_merged_extent(node.obb_low, node.obb_high);
        (*ST).obbs[this_index].qlo =
            quantize_bounds_low(node.obb_low, node.obb_high);
        (*ST).obbs[this_index].qhi =
            quantize_bounds_high(node.obb_low, node.obb_high);
        (*ST).obbs[this_index].qorientation =
            quantize_orientation(node.orientation);
        uint64_t obb_children_index[8];
        for (int32_t __r = 0; __r < 8; __r += 1) {
            obb_children_index[__r] =
                rec_build_triangles(node.obb_children[__r], ST, aabbs_index,
                                    obbs_index, primitives_index);
            (*ST).obbs[this_index].obb_children[__r] = obb_children_index[__r];
        }
        return ((this_index << (uint64_t)4) | (uint64_t)3);
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        uint64_t poffset = (*primitives_index);
        for (uint8_t __p = 0u; __p < node.nprims; __p += 1u) {
            (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
        }
        (*primitives_index) += node.nprims;
        return ((poffset << (uint64_t)4) |
                ((uint64_t)node.nprims << (uint64_t)2));
    }
}

__host__ void rec_count_triangles(BVH *node_, Triangles *ST) {
    if (!node_) {
        return;
    }
    if (std::holds_alternative<AABBNode>(*node_)) {
        const AABBNode &node = std::get<AABBNode>(*node_);
        for (int32_t __r = 0; __r < 8; __r += 1) {
            rec_count_triangles(node.aabb_children[__r], ST);
        }
        (*ST).aabb_count += 1u;
    } else if (std::holds_alternative<OBBNode>(*node_)) {
        const OBBNode &node = std::get<OBBNode>(*node_);
        for (int32_t __r = 0; __r < 8; __r += 1) {
            rec_count_triangles(node.obb_children[__r], ST);
        }
        (*ST).obb_count += 1u;
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        (*ST).primitive_count += node.nprims;
    }
}

__host__ Triangles build_triangles(BVH *CT) {
    Triangles ST;
    size_t primitives_index = 0;
    size_t aabbs_index = 0;
    size_t obbs_index = 0;
    ST.primitive_count = 0u;
    ST.aabb_count = 0u;
    ST.obb_count = 0u;
    rec_count_triangles(CT, (&ST));
    printf("p:%lu,a:%lu,o:%lu", ST.primitive_count, ST.aabb_count,
           ST.obb_count);
    Triangle *primitives = reinterpret_cast<Triangle *>(
        malloc(sizeof(Triangle) * ST.primitive_count));
    ST.primitives = primitives;
    Aabbs *aabbs =
        reinterpret_cast<Aabbs *>(malloc(sizeof(Aabbs) * ST.aabb_count));
    ST.aabbs = aabbs;
    Obbs *obbs = reinterpret_cast<Obbs *>(malloc(sizeof(Obbs) * ST.obb_count));
    ST.obbs = obbs;
    rec_build_triangles(CT, (&ST), (&aabbs_index), (&obbs_index),
                        (&primitives_index));
    return ST;
}

__host__ cuda::std::optional<Triangle> *chrt(int64_t n, Ray *rays,
                                             Triangles *triangles) {
    return _traverse_array0(n, rays, triangles);
}
