#include "apps/queries/rt/rt.h"
struct AABB {
    float3 low;
    float3 high;
};
struct FInterval {
    float low;
    float high;
};
struct TriangleIntersection {
    float b0;
    float b1;
    float b2;
    float t;
};

std::optional<FInterval> intersectsp_ray_aabb(const Ray r, const AABB b) {
    const float3 invDir = float3{1.00000000f} / r.d;
    const float3 t0 = (b.low - r.o) * invDir;
    const float3 t1 = (b.high - r.o) * invDir;
    const float3 tNear = min(t0, t1);
    const float3 tFar = max(t0, t1);
    const float _t0 = max(0.00000000f, reduce_max(tNear));
    const float tFarMin = reduce_min(tFar);
    if (tFarMin < _t0) {
        return std::nullopt;
    }
    return (std::optional<FInterval>)(FInterval{_t0, tFarMin});
}
float distmin_Ray_AABB(const Ray r, const AABB b) {
    const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        const FInterval extract = *interval;
        return extract.low;
    }
    return -std::numeric_limits<float>::infinity();
}
std::optional<float> intersectsp_ray_tri(const Ray r, const Triangle tri) {
    const float epsilon = 0.00000006f;
    const float3 pa = tri.p0;
    const float3 pb = tri.p1;
    const float3 pc = tri.p2;
    const float3 v1 = pb - pa;
    const float3 v2 = pc - pa;
    const float3 _t0 = r.d;
    const float3 _t1 = cross(_t0, v2);
    const float det = dot(v1, _t1);
    if (abs(det) <= epsilon) {
        return std::nullopt;
    }
    const float invDet = 1.00000000f / det;
    const float3 _t3 = r.o - pa;
    const float _t5 = dot(_t3, _t1) * invDet;
    if ((_t5 < 0.00000000f) || (1.00000000f < _t5)) {
        return std::nullopt;
    }
    const float3 q = cross(_t3, v1);
    const float _t8 = dot(_t0, q) * invDet;
    if ((_t8 < 0.00000000f) || (1.00000000f < (_t5 + _t8))) {
        return std::nullopt;
    }
    const float _t10 = dot(v2, q) * invDet;
    if (0.00000000f <= _t10) {
        if (_t10 <= r.tmax) {
            return std::make_optional<float>(_t10);
        }
    }
    return std::nullopt;
}
float distmin_Ray_Triangle(const Ray ray, const Triangle tri) {
    const std::optional<float> isect = intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        return *isect;
    } else {
        return std::numeric_limits<float>::infinity();
    }
}
bool intersects_Ray_AABB(const Ray r, const AABB b) {
#ifdef AJR_PROFILE
    distmin_aabb_counter++;
#endif
    const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        const FInterval extract = *interval;
        return (extract.low < r.tmax) & (0.00000000f < extract.high);
    }
    return false;
}
bool intersects_Ray_Triangle(const Ray ray, const Triangle tri) {
#ifdef AJR_PROFILE
    distmin_triangle_counter++;
#endif
    return intersectsp_ray_tri(ray, tri).has_value();
}
std::optional<Triangle> _traverse_tree0(const Ray ray,
                                        const _tree_layout0 &triangles) {
    std::tuple<float, Triangle> _best0 =
        std::make_tuple(std::numeric_limits<float>::infinity(), Triangle{});
    int32_t _queue_count0 = 1;
    uint32_t _queue0[128];
    _queue0[0] = 0u;
    do {
        _queue_count0 -= 1;
        const uint32_t triangles_index = _queue0[_queue_count0];
        const _tree_layout1 _t45 = triangles.group0_index[triangles_index];
        const AABB _t50 = AABB{_t45.low, _t45.high};
        if (intersects_Ray_AABB(ray, _t50)) {
            if (distmin_Ray_AABB(ray, _t50) < std::get<0>(_best0)) {
                const uint32_t _t35 = _t45.nPrims;
                if (_t35 == 0u) {
                    _queue0[_queue_count0] = triangles_index + 1u;
                    _queue0[_queue_count0 + 1] = triangles_index + _t45.offset;
                    _queue_count0 += 2;
                } else {
                    for (uint32_t _idx0 = 0u; _idx0 < _t35; _idx0 += 1u) {
                        const Triangle _t29 =
                            triangles.prims[_t45.offset + _idx0];
                        if (intersects_Ray_Triangle(ray, _t29)) {
                            const float _t22 = distmin_Ray_Triangle(ray, _t29);
                            if (_t22 < std::get<0>(_best0)) {
                                _best0 = std::make_tuple(_t22, _t29);
                            }
                        }
                    }
                }
            }
        }
    } while (_queue_count0 != 0);
    return ((std::get<0>(_best0) != std::numeric_limits<float>::infinity())
                ? std::make_optional<Triangle>(std::get<1>(_best0))
                : std::nullopt);
}
float prod_diff(const float a, const float b, const float c, const float d) {
    const float cd = c * d;
    const float diff = fma(a, b, -cd);
    const float err = fma(-c, d, cd);
    return diff + err;
}
float3 cross_(const float3 v0, const float3 v1) {
    const float _t0 = v0[1u];
    const float _t1 = v1[2u];
    const float _t2 = v0[2u];
    const float _t3 = v1[1u];
    const float _t5 = v1[0u];
    const float _t6 = v0[0u];
    return vector<float, 3>({prod_diff(_t0, _t1, _t2, _t3),
                             prod_diff(_t2, _t5, _t6, _t1),
                             prod_diff(_t6, _t3, _t0, _t5)});
}
float distmax_Ray_AABB(const Ray r, const AABB b) {
    const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        const FInterval extract = *interval;
        return extract.high;
    }
    return std::numeric_limits<float>::infinity();
}
float distmax_Ray_Triangle(const Ray ray, const Triangle tri) {
    const std::optional<float> isect = intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        return *isect;
    } else {
        return -std::numeric_limits<float>::infinity();
    }
}
float gamma(const int32_t n) {
    const float _t1 = (float)(n) * 0.00000006f;
    return _t1 / (1.00000000f - _t1);
}
float orient3d(const float3 a, const float3 b, const float3 c, const float3 d) {
    const float3 ad = a - d;
    const float3 bd = b - d;
    const float3 cd = c - d;
    return dot(ad, cross(bd, cd));
}
bool point_in_triangle(const float3 p, const Triangle t, const float3 q) {
    return ((orient3d(p, q, t.p0, t.p1) <= 0.00000000f) &&
            (orient3d(p, q, t.p1, t.p2) <= 0.00000000f)) &&
           (orient3d(p, q, t.p2, t.p0) <= 0.00000000f);
}
float sqlen(const float3 v) { return reduce_add(v * v); }
std::optional<Triangle> trace(const Ray ray, const _tree_layout0 &triangles) {
    return _traverse_tree0(ray, triangles);
}
