#include "helpers.h"

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/universal_vector.h>

#include <stdint.h>
#include <variant>

struct Interior;
struct Leaf;
using BVH = std::variant<Interior, Leaf>;

struct AABB {
    float3 low;
    float3 high;
};

struct FInterval {
    float low;
    float high;
};

struct Interior {
    cuda::std::array<BVH *, 8> children;
    cuda::std::array<float3, 8> lo;
    cuda::std::array<float3, 8> hi;
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

struct alignas(32) Interiors {
    cuda::std::array<float3, 8> lo;
    cuda::std::array<float3, 8> hi;
    cuda::std::array<uint64_t, 8> children;
} __attribute__((packed));

struct Triangles {
    uint64_t primitive_count;
    Triangle *primitives;
    uint64_t interior_count;
    Interiors *interiors;
} __attribute__((packed));

__host__ float __prod_diff_f32(float a, float b, float c, float d) {
    float cd = (c * d);
    float diff = fmaf(a, b, -cd);
    float err = fmaf(-c, d, cd);
    return (diff + err);
}

__host__ float gamma(int32_t n) {
    float _t1 = ((float)n * (float)5.96046e-08);
    return (_t1 / (1 - _t1));
}

__host__ cuda::std::optional<FInterval> intersectsp_ray_aabb(Ray *r, AABB *b) {
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

__host__ float distmin_Ray_AABB(Ray *r, AABB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return extract.low;
    }
    return -INFINITY;
}

__host__ float3 cross_(float3 v0, float3 v1) {
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

__host__ cuda::std::optional<TriangleIntersection>
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

__host__ float distmin_Ray_Triangle(Ray *ray, Triangle *tri) {
    cuda::std::optional<TriangleIntersection> isect =
        intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        TriangleIntersection isect_ = *isect;
        return isect_.t;
    } else {
        return INFINITY;
    }
}

__host__ bool intersects_Ray_AABB(Ray *r, AABB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return ((extract.low < (*r).tmax) & (0 < extract.high));
    }
    return false;
}

__host__ void _recloop_func0(uint64_t I, Ray *ray, Triangles *triangles,
                             cuda::std::tuple<float, Triangle> *_best0) {
    if (I == 18446744073709551615u) {
        return;
    }
    if (slice<0, 2>(I) == 1u) {
        Interiors _t17 = (*triangles).interiors[slice<7, 63>(I)];
        cuda::std::array<float3, 8> _t18 = _t17.lo;
        cuda::std::array<float3, 8> _t23 = _t17.hi;
        AABB _t25 = AABB{_t18[0], _t23[0]};
        if (intersects_Ray_AABB(ray, (&_t25))) {
            if (distmin_Ray_AABB(ray, (&_t25)) < cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[0u], ray, triangles, _best0);
            }
        }
        AABB _t51 = AABB{_t18[1], _t23[1]};
        if (intersects_Ray_AABB(ray, (&_t51))) {
            if (distmin_Ray_AABB(ray, (&_t51)) < cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[1u], ray, triangles, _best0);
            }
        }
        AABB _t77 = AABB{_t18[2], _t23[2]};
        if (intersects_Ray_AABB(ray, (&_t77))) {
            if (distmin_Ray_AABB(ray, (&_t77)) < cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[2u], ray, triangles, _best0);
            }
        }
        AABB _t103 = AABB{_t18[3], _t23[3]};
        if (intersects_Ray_AABB(ray, (&_t103))) {
            if (distmin_Ray_AABB(ray, (&_t103)) <
                cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[3u], ray, triangles, _best0);
            }
        }
        AABB _t129 = AABB{_t18[4], _t23[4]};
        if (intersects_Ray_AABB(ray, (&_t129))) {
            if (distmin_Ray_AABB(ray, (&_t129)) <
                cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[4u], ray, triangles, _best0);
            }
        }
        AABB _t155 = AABB{_t18[5], _t23[5]};
        if (intersects_Ray_AABB(ray, (&_t155))) {
            if (distmin_Ray_AABB(ray, (&_t155)) <
                cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[5u], ray, triangles, _best0);
            }
        }
        AABB _t181 = AABB{_t18[6], _t23[6]};
        if (intersects_Ray_AABB(ray, (&_t181))) {
            if (distmin_Ray_AABB(ray, (&_t181)) <
                cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[6u], ray, triangles, _best0);
            }
        }
        AABB _t207 = AABB{_t18[7], _t23[7]};
        if (intersects_Ray_AABB(ray, (&_t207))) {
            if (distmin_Ray_AABB(ray, (&_t207)) <
                cuda::std::get<0>((*_best0))) {
                _recloop_func0(_t17.children[7u], ray, triangles, _best0);
            }
        }
    } else {
        uint64_t _t218 = slice<7, 63>(I);
        for (uint64_t _idx0 = _t218;
             _idx0 < (_t218 + (uint64_t)(uint8_t)(slice<3, 6>(I) + 1u));
             _idx0 += 1u) {
            Triangle _t217 = (*triangles).primitives[_idx0];
            if (intersectsp_ray_tri(ray, (&_t217)).has_value()) {
                float _t215 = distmin_Ray_Triangle(ray, (&_t217));
                if (_t215 < cuda::std::get<0>((*_best0))) {
                    (*_best0) =
                        argmin(_best0,
                               cuda::std::tuple<float, Triangle>{_t215, _t217});
                }
            }
        }
    }
    return;
}

__host__ cuda::std::optional<Triangle> _traverse_tree0(Ray *ray,
                                                       Triangles *triangles) {
    cuda::std::tuple<float, Triangle> _best0 =
        cuda::std::tuple<float, Triangle>{INFINITY, Triangle{}};
    _recloop_func0(1u, ray, triangles, (&_best0));
    return ((cuda::std::get<0>(_best0) != INFINITY)
                ? cuda::std::optional<Triangle>{cuda::std::get<1>(_best0)}
                : cuda::std::nullopt);
}

__host__ cuda::std::optional<Triangle> *_traverse_array0(int64_t n, Ray *rays,
                                                         Triangles *triangles) {
    cuda::std::optional<Triangle> *_alloc0;
    (void)cudaMalloc((void **)&_alloc0,
                     n * sizeof(cuda::std::optional<Triangle>));
    for (int64_t _i0 = 0; _i0 < n; _i0 += 1) {
        Ray _lv0 = rays[_i0];
        _alloc0[_i0] = _traverse_tree0((&_lv0), triangles);
    }
    return _alloc0;
}

__host__ uint64_t rec_build_triangles(BVH *node_, Triangles *ST,
                                      size_t *interiors_index,
                                      size_t *primitives_index) {
    if (!node_) {
        return 18446744073709551615u;
    }
    if (std::holds_alternative<Interior>(*node_)) {
        const Interior &node = std::get<Interior>(*node_);
        size_t this_index = (*interiors_index);
        (*interiors_index) += 1;
        (*ST).interiors[this_index].lo = node.lo;
        (*ST).interiors[this_index].hi = node.hi;
        uint64_t children_index[8];
        for (int32_t __r = 0; __r < 8; __r += 1) {
            children_index[__r] = rec_build_triangles(
                node.children[__r], ST, interiors_index, primitives_index);
            (*ST).interiors[this_index].children[__r] = children_index[__r];
        }
        return ((this_index << (uint64_t)7) | (uint64_t)1);
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        uint64_t poffset = (*primitives_index);
        for (uint8_t __p = 0u; __p < node.nprims; __p += 1u) {
            (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
        }
        (*primitives_index) += node.nprims;
        return ((poffset << (uint64_t)7) |
                (((uint64_t)node.nprims - (uint64_t)1) << (uint64_t)3));
    }
}

__host__ void rec_count_triangles(BVH *node_, Triangles *ST) {
    if (!node_) {
        return;
    }
    if (std::holds_alternative<Interior>(*node_)) {
        const Interior &node = std::get<Interior>(*node_);
        for (int32_t __r = 0; __r < 8; __r += 1) {
            rec_count_triangles(node.children[__r], ST);
        }
        (*ST).interior_count += 1u;
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        (*ST).primitive_count += node.nprims;
    }
}

__host__ Triangles build_triangles(BVH *CT) {
    printf("building triangles!\n");
    Triangles ST;
    size_t primitives_index = 0;
    size_t interiors_index = 0;
    ST.primitive_count = 0u;
    ST.interior_count = 0u;
    rec_count_triangles(CT, (&ST));
    Triangle *primitives;
    cudaError_t err =
        cudaMalloc((void **)&primitives, ST.primitive_count * sizeof(Triangle));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        std::cerr << "Tried to allocate: " << ST.primitive_count
                  << " triangles\n";
        exit(1);
    }

    ST.primitives = primitives;
    Interiors *interiors;
    err =
        cudaMalloc((void **)&interiors, ST.interior_count * sizeof(Interiors));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        std::cerr << "Tried to allocate: " << ST.interior_count
                  << " interiors\n";
        exit(1);
    }
    ST.interiors = interiors;
    rec_build_triangles(CT, (&ST), (&interiors_index), (&primitives_index));
    return ST;
}

__host__ cuda::std::optional<Triangle> *chrt(int64_t n, Ray *rays,
                                             Triangles *triangles) {
    return _traverse_array0(n, rays, triangles);
}

__host__ cuda::std::optional<Triangle> trace(Ray *ray, Triangles *triangles) {
    return _traverse_tree0(ray, triangles);
}
