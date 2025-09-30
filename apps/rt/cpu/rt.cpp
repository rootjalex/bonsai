#include "apps/rt/cpu/rt.h"
Point ClosestPtPointAABB(const Point *__restrict__ pt,
                         const AABB *__restrict__ a) {
    return Point{.vec = min(max((*pt).vec, (*a).low), (*a).high)};
}
float SqDistPointAABB(const Point *__restrict__ pt,
                      const AABB *__restrict__ a) {
    const vec3_float v = (*pt).vec;
    const vec3_float sqLow = (((*a).low - v) * ((*a).low - v));
    const vec3_float low = select((v < (*a).low), sqLow, vec3_float{0.0f});
    const vec3_float sqHigh = ((v - (*a).high) * (v - (*a).high));
    const vec3_float high = select(((*a).high < v), sqHigh, vec3_float{0.0f});
    return reduce_add((low + high));
}
float __prod_diff_f32(const float a, const float b, const float c,
                      const float d) {
    const float cd = (c * d);
    const float diff = fmaf(a, b, (-cd));
    const float err = fmaf((-c), d, cd);
    return (diff + err);
}
float __sqlen_f32(const vec3_float v) { return reduce_add((v * v)); }
float gamma(const int32_t n) {
    const float _t1 = ((float)(n) * 0.00000006f);
    return (_t1 / (1.0f - _t1));
}
std::optional<FInterval> intersectsp_ray_aabb(const Ray *__restrict__ r,
                                              const AABB *__restrict__ b) {
    const vec3_float _t1 = (vec3_float{1.0f} / (*r).d);
    const vec3_bool dirIsNeg = (_t1 < vec3_float{0.0f});
    const vec3_float _t2 = (*b).high;
    const vec3_float _t3 = (*b).low;
    const vec3_float low_parts = select(dirIsNeg, _t2, _t3);
    const vec3_float high_parts = select(dirIsNeg, _t3, _t2);
    const vec3_float _t6 = (*r).o;
    vec3_float tMin = ((low_parts - _t6) * _t1);
    vec3_float tMax = ((high_parts - _t6) * _t1);
    tMax *= (1.0f + (2.0f * gamma(3)));
    if (((tMax[1u] < tMin[0u]) || (tMax[0u] < tMin[1u]))) {
        return std::nullopt;
    }
    float tmin = max(tMin[0u], tMin[1u]);
    float tmax = min(tMax[0u], tMax[1u]);
    if (((tMax[2u] < tmin) || (tmax < tMin[2u]))) {
        return std::nullopt;
    }
    tmin = max(tmin, tMin[2u]);
    tmax = min(tmax, tMax[2u]);
    return (std::optional<FInterval>)(FInterval{.low = tmin, .high = tmax});
}
float distmin_Ray_AABB(const Ray *__restrict__ r, const AABB *__restrict__ b) {
    const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        const FInterval &extract = *interval;
        return extract.low;
    }
    return (-std::numeric_limits<float>::infinity());
}
vec3_float cross_(const vec3_float v0, const vec3_float v1) {
    const float _t0 = v0[1u];
    const float _t1 = v1[2u];
    const float _t2 = v0[2u];
    const float _t3 = v1[1u];
    const float _t5 = v1[0u];
    const float _t6 = v0[0u];
    return vec3_float{__prod_diff_f32(_t0, _t1, _t2, _t3),
                      __prod_diff_f32(_t2, _t5, _t6, _t1),
                      __prod_diff_f32(_t6, _t3, _t0, _t5)};
}
std::optional<TriangleIntersection>
intersectsp_ray_tri(const Ray *__restrict__ ray,
                    const Triangle *__restrict__ tri) {
    const vec3_float _t0 = (*tri).p2;
    const vec3_float _t1 = (*tri).p0;
    const vec3_float _t2 = (*tri).p1;
    if (reduce_add((cross_((_t0 - _t1), (_t2 - _t1)) *
                    cross_((_t0 - _t1), (_t2 - _t1)))) == 0.0f) {
        return std::nullopt;
    }
    const vec3_float _t5 = (*ray).o;
    vec3_float p0t = (_t1 - _t5);
    vec3_float p1t = (_t2 - _t5);
    vec3_float p2t = (_t0 - _t5);
    const vec3_float _t10 = (*ray).d;
    const uint32_t kz = reduce_idxmax(abs(_t10));
    const uint32_t kx = ((kz + 1u) % 3u);
    const uint32_t ky = ((kx + 1u) % 3u);
    const vec3_float d = shuffle(_t10, {kx, ky, kz});
    p0t = shuffle(p0t, {kx, ky, kz});
    p1t = shuffle(p1t, {kx, ky, kz});
    p2t = shuffle(p2t, {kx, ky, kz});
    const float _t13 = d[2u];
    const float _t14 = ((-d[0u]) / _t13);
    const float _t17 = ((-d[1u]) / _t13);
    const float Sz = (1.0f / _t13);
    p0t[0u] += (_t14 * p0t[2u]);
    p0t[1u] += (_t17 * p0t[2u]);
    p1t[0u] += (_t14 * p1t[2u]);
    p1t[1u] += (_t17 * p1t[2u]);
    p2t[0u] += (_t14 * p2t[2u]);
    p2t[1u] += (_t17 * p2t[2u]);
    const float _t20 = __prod_diff_f32(p1t[0u], p2t[1u], p1t[1u], p2t[0u]);
    const float _t21 = __prod_diff_f32(p2t[0u], p0t[1u], p2t[1u], p0t[0u]);
    const float _t22 = __prod_diff_f32(p0t[0u], p1t[1u], p0t[1u], p1t[0u]);
    if ((((_t20 < 0.0f) || (_t21 < 0.0f)) || (_t22 < 0.0f))) {
        if ((((0.0f < _t20) || (0.0f < _t21)) || (0.0f < _t22))) {
            return std::nullopt;
        }
    }
    const float _t24 = ((_t20 + _t21) + _t22);
    if (_t24 == 0.0f) {
        return std::nullopt;
    }
    p0t[2u] *= Sz;
    p1t[2u] *= Sz;
    p2t[2u] *= Sz;
    const float tScaled =
        (((_t20 * p0t[2u]) + (_t21 * p1t[2u])) + (_t22 * p2t[2u]));
    if (((_t24 < 0.0f) &&
         ((0.0f <= tScaled) || (tScaled < ((*ray).tmax * _t24))))) {
        return std::nullopt;
    } else if (((0.0f < _t24) &&
                ((tScaled <= 0.0f) || (((*ray).tmax * _t24) < tScaled)))) {
        return std::nullopt;
    }
    const float invDet = (1.0f / _t24);
    const float b0 = (_t20 * invDet);
    const float b1 = (_t21 * invDet);
    const float b2 = (_t22 * invDet);
    const float t = (tScaled * invDet);
    const float maxZt = reduce_max(abs(vec3_float{p0t[2u], p1t[2u], p2t[2u]}));
    const float _t25 = gamma(3);
    const float deltaZ = (_t25 * maxZt);
    const float maxXt = reduce_max(abs(vec3_float{p0t[0u], p1t[0u], p2t[0u]}));
    const float maxYt = reduce_max(abs(vec3_float{p0t[1u], p1t[1u], p2t[1u]}));
    const float _t26 = gamma(5);
    const float deltaX = (_t26 * (maxXt + maxZt));
    const float deltaY = (_t26 * (maxYt + maxZt));
    const float deltaE =
        (2.0f * ((((gamma(2) * maxXt) * maxYt) + (deltaY * maxXt)) +
                 (deltaX * maxYt)));
    const float maxE = reduce_max(abs(vec3_float{_t20, _t21, _t22}));
    const float deltaT =
        ((3.0f *
          ((((_t25 * maxE) * maxZt) + (deltaE * maxZt)) + (deltaZ * maxE))) *
         abs(invDet));
    if ((t <= deltaT)) {
        return std::nullopt;
    }
    return (std::optional<TriangleIntersection>)(TriangleIntersection{
        .b0 = b0, .b1 = b1, .b2 = b2, .t = t});
}
float distmin_Ray_Triangle(const Ray *__restrict__ ray,
                           const Triangle *__restrict__ tri) {
    const std::optional<TriangleIntersection> isect =
        intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        const TriangleIntersection &isect_ = *isect;
        return isect_.t;
    } else {
        return std::numeric_limits<float>::infinity();
    }
}
bool intersects_Ray_AABB(const Ray *__restrict__ r,
                         const AABB *__restrict__ b) {
    const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        const FInterval &extract = *interval;
        return ((extract.low < (*r).tmax) & (0.0f < extract.high));
    }
    return false;
}
void _recloop_func0(const uint64_t I, const Ray *__restrict__ ray,
                    const Triangles *__restrict__ triangles,
                    std::tuple<float, Triangle> *__restrict__ _best0) {
    if (I == 18446744073709551615u) {
        return;
    }
    if (slice<0, 2>(I) == 1u) {
        const Interiors &_t17 = (*triangles).interiors[slice<7, 63>(I)];
        const vec8_vec3_float _t18 = _t17.lo;
        const vec8_vec3_float _t23 = _t17.hi;
        const AABB _t25 = AABB{.low = _t18[0], .high = _t23[0]};
        if (intersects_Ray_AABB(ray, (&_t25))) {
            if ((distmin_Ray_AABB(ray, (&_t25)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[0u], ray, triangles, _best0);
            }
        }
        const AABB _t51 = AABB{.low = _t18[1], .high = _t23[1]};
        if (intersects_Ray_AABB(ray, (&_t51))) {
            if ((distmin_Ray_AABB(ray, (&_t51)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[1u], ray, triangles, _best0);
            }
        }
        const AABB _t77 = AABB{.low = _t18[2], .high = _t23[2]};
        if (intersects_Ray_AABB(ray, (&_t77))) {
            if ((distmin_Ray_AABB(ray, (&_t77)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[2u], ray, triangles, _best0);
            }
        }
        const AABB _t103 = AABB{.low = _t18[3], .high = _t23[3]};
        if (intersects_Ray_AABB(ray, (&_t103))) {
            if ((distmin_Ray_AABB(ray, (&_t103)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[3u], ray, triangles, _best0);
            }
        }
        const AABB _t129 = AABB{.low = _t18[4], .high = _t23[4]};
        if (intersects_Ray_AABB(ray, (&_t129))) {
            if ((distmin_Ray_AABB(ray, (&_t129)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[4u], ray, triangles, _best0);
            }
        }
        const AABB _t155 = AABB{.low = _t18[5], .high = _t23[5]};
        if (intersects_Ray_AABB(ray, (&_t155))) {
            if ((distmin_Ray_AABB(ray, (&_t155)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[5u], ray, triangles, _best0);
            }
        }
        const AABB _t181 = AABB{.low = _t18[6], .high = _t23[6]};
        if (intersects_Ray_AABB(ray, (&_t181))) {
            if ((distmin_Ray_AABB(ray, (&_t181)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[6u], ray, triangles, _best0);
            }
        }
        const AABB _t207 = AABB{.low = _t18[7], .high = _t23[7]};
        if (intersects_Ray_AABB(ray, (&_t207))) {
            if ((distmin_Ray_AABB(ray, (&_t207)) < std::get<0>((*_best0)))) {
                _recloop_func0(_t17.children[7u], ray, triangles, _best0);
            }
        }
    } else {
        const uint64_t _t218 = slice<7, 63>(I);
        for (uint64_t _idx0 = _t218;
             _idx0 < (_t218 + (uint64_t)((uint8_t)((slice<3, 6>(I) + 1u))));
             ++_idx0) {
            const Triangle &_t217 = (*triangles).primitives[_idx0];
            if (intersectsp_ray_tri(ray, (&_t217)).has_value()) {
                const float _t215 = distmin_Ray_Triangle(ray, (&_t217));
                if ((_t215 < std::get<0>((*_best0)))) {
                    (*_best0) = argmin<float, Triangle>(
                        _best0, std::tuple<float, Triangle>{_t215, _t217});
                }
            }
        }
    }
    return;
}
std::optional<Triangle>
_traverse_tree0(const Ray *__restrict__ ray,
                const Triangles *__restrict__ triangles) {
    std::tuple<float, Triangle> _best0 = std::tuple<float, Triangle>{
        std::numeric_limits<float>::infinity(), Triangle{}};
    _recloop_func0(1u, ray, triangles, (&_best0));
    return ((std::get<0>(_best0) != std::numeric_limits<float>::infinity())
                ? std::optional<Triangle>{std::get<1>(_best0)}
                : std::nullopt);
}
std::optional<Triangle> *
_traverse_array0(const int64_t n, const Ray *rays,
                 const Triangles *__restrict__ triangles) {
    std::optional<Triangle> *_alloc0 =
        reinterpret_cast<std::optional<Triangle> *>(
            malloc(sizeof(std::optional<Triangle>) * n));
    for (int64_t _i0 = 0; _i0 < n; ++_i0) {
        const Ray &_lv0 = rays[_i0];
        _alloc0[_i0] = _traverse_tree0((&_lv0), triangles);
    }
    return _alloc0;
}
bool axis(const vec3_float A, const vec3_float extents, const vec3_float v0,
          const vec3_float v1, const vec3_float v2) {
    const float R = dot(extents, abs(A));
    const vec3_float _t3 = vec3_float{dot(v0, A), dot(v1, A), dot(v2, A)};
    return reduce_and(((_t3 <= vec3_float{R}) & (vec3_float{(-R)} <= _t3)));
}
uint64_t rec_build_triangles(const BVH *__restrict__ node,
                             Triangles *__restrict__ ST,
                             size_t *__restrict__ interiors_index,
                             size_t *__restrict__ primitives_index) {
    if ((!node)) {
        return 18446744073709551615u;
    }
    return std::visit(
        overloaded{[&](const Interior &node) {
                       const size_t this_index = (*interiors_index);
                       (*interiors_index) += 1u;
                       (*ST).interiors[this_index].lo = node.lo;
                       (*ST).interiors[this_index].hi = node.hi;
                       std::array<uint64_t, 8> children_index;
                       for (int32_t __r = 0; __r < 8; ++__r) {
                           children_index[__r] = rec_build_triangles(
                               node.children[__r], ST, interiors_index,
                               primitives_index);
                           (*ST).interiors[this_index].children[__r] =
                               children_index[__r];
                       }
                       return ((this_index << (uint64_t)(7)) | (uint64_t)(1));
                   },
                   [&](const Leaf &node) {
                       const uint64_t poffset = (*primitives_index);
                       for (uint8_t __p = 0u; __p < node.nprims; ++__p) {
                           (*ST).primitives[(__p + (*primitives_index))] =
                               node.data[__p];
                       }
                       (*primitives_index) += node.nprims;
                       return ((poffset << (uint64_t)(7)) |
                               (((uint64_t)(node.nprims) - (uint64_t)(1))
                                << (uint64_t)(3)));
                   }},
        *node);
}
void rec_count_triangles(const BVH *__restrict__ node,
                         Triangles *__restrict__ ST,
                         size_t *__restrict__ size_interiors) {
    if ((!node)) {
        return;
    }
    return std::visit(
        overloaded{
            [&](const Interior &node) {
                for (int32_t __r = 0; __r < 8; ++__r) {
                    rec_count_triangles(node.children[__r], ST, size_interiors);
                }
                (*size_interiors) += 1u;
            },
            [&](const Leaf &node) { (*ST).primitive_count += node.nprims; }},
        *node);
}
Triangles build_triangles(const BVH *__restrict__ CT) {
    Triangles ST;
    size_t primitives_index = 0;
    size_t size_interiors = 0;
    size_t interiors_index = 0u;
    ST.primitive_count = 0u;
    rec_count_triangles(CT, (&ST), (&size_interiors));
    Triangle *primitives = reinterpret_cast<Triangle *>(
        malloc(sizeof(Triangle) * ST.primitive_count));
    ST.primitives = primitives;
    Interiors *interiors = reinterpret_cast<Interiors *>(std::aligned_alloc(
        32, (((sizeof(Interiors) * size_interiors) + 31) / 32) * 32));
    ST.interiors = interiors;
    rec_build_triangles(CT, (&ST), (&interiors_index), (&primitives_index));
    return ST;
}
std::optional<Triangle> *chrt(const int64_t n, const Ray *rays,
                              const Triangles *__restrict__ triangles) {
    return _traverse_array0(n, rays, triangles);
}
vec3_float clamp(const vec3_float x, const float low, const float high) {
    return min(max(x, vec3_float{low}), vec3_float{high});
}
std::tuple<Point, Point>
closestPointonTriangle(const Point *__restrict__ pt,
                       const Triangle *__restrict__ tri) {
    const vec3_float p = (*pt).vec;
    const vec3_float a = (*tri).p0;
    const vec3_float b = (*tri).p1;
    const vec3_float c = (*tri).p2;
    const vec3_float ab = (b - a);
    const vec3_float ac = (c - a);
    const vec3_float ap = (p - a);
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if ((d1 <= 0.0f)) {
        if ((d2 <= 0.0f)) {
            return std::tuple<Point, Point>{
                Point{.vec = a}, Point{.vec = vec3_float{1.0f, 0.0f, 0.0f}}};
        }
    }
    const vec3_float bp = (p - b);
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if ((0.0f <= d3)) {
        if ((d4 <= d3)) {
            return std::tuple<Point, Point>{
                Point{.vec = b}, Point{.vec = vec3_float{0.0f, 1.0f, 0.0f}}};
        }
    }
    const float _t2 = ((d1 * d4) - (d3 * d2));
    if ((_t2 <= 0.0f)) {
        if ((0.0f <= d1)) {
            if ((d3 <= 0.0f)) {
                const float _t4 = (d1 / (d1 - d3));
                return std::tuple<Point, Point>{
                    Point{.vec = (a + (vec3_float{_t4} * ab))},
                    Point{.vec = vec3_float{(1.0f - _t4), _t4, 0.0f}}};
            }
        }
    }
    const vec3_float cp = (p - c);
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if ((0.0f <= d6)) {
        if ((d5 <= d6)) {
            return std::tuple<Point, Point>{
                Point{.vec = c}, Point{.vec = vec3_float{0.0f, 0.0f, 1.0f}}};
        }
    }
    const float _t7 = ((d5 * d2) - (d1 * d6));
    if ((_t7 <= 0.0f)) {
        if ((0.0f <= d2)) {
            if ((d6 <= 0.0f)) {
                const float _t9 = (d2 / (d2 - d6));
                return std::tuple<Point, Point>{
                    Point{.vec = (a + (vec3_float{_t9} * ac))},
                    Point{.vec = vec3_float{(1.0f - _t9), 0.0f, _t9}}};
            }
        }
    }
    const float _t12 = ((d3 * d6) - (d5 * d4));
    if ((_t12 <= 0.0f)) {
        const float _t19 = (d4 - d3);
        if ((0.0f <= _t19)) {
            const float _t18 = (d5 - d6);
            if ((0.0f <= _t18)) {
                const float _t17 = (_t19 / (_t19 + _t18));
                return std::tuple<Point, Point>{
                    Point{.vec = (b + (vec3_float{_t17} * (c - b)))},
                    Point{.vec = vec3_float{0.0f, (1.0f - _t17), _t17}}};
            }
        }
    }
    const float _t22 = (1.0f / ((_t12 + _t7) + _t2));
    const float v = (_t7 * _t22);
    const float w = (_t2 * _t22);
    const float u = (_t12 * _t22);
    return std::tuple<Point, Point>{
        Point{.vec = ((a + (ab * vec3_float{v})) + (ac * vec3_float{w}))},
        Point{.vec = vec3_float{u, v, w}}};
}
float degrees_to_radians(const float degrees) {
    return ((degrees * 3.14159274f) / 180.0f);
}
float distmax_Ray_AABB(const Ray *__restrict__ r, const AABB *__restrict__ b) {
    const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        const FInterval &extract = *interval;
        return extract.high;
    }
    return std::numeric_limits<float>::infinity();
}
std::optional<FInterval> intersectsp_ray_sphere(const Ray *__restrict__ r,
                                                const Sphere *__restrict__ s) {
    const vec3_float _t2 = ((*s).center - (*r).o);
    const vec3_float _t3 = (*r).d;
    const float a = reduce_add((_t3 * _t3));
    const float _t7 = dot(_t3, _t2);
    const float _t9 = (*s).radius;
    const float _t15 =
        ((_t7 * _t7) - (a * (reduce_add((_t2 * _t2)) - (_t9 * _t9))));
    if ((_t15 < 0.0f)) {
        return std::nullopt;
    }
    const float sqrtd = sqrtf(_t15);
    const float _t17 = ((_t7 - sqrtd) / a);
    const float _t19 = ((_t7 + sqrtd) / a);
    const FInterval interval =
        FInterval{.low = min(_t17, _t19), .high = max(_t17, _t19)};
    return (std::optional<FInterval>)(interval);
}
float distmax_Ray_Sphere(const Ray *__restrict__ r,
                         const Sphere *__restrict__ s) {
    const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
    if (interval.has_value()) {
        const FInterval &extract = *interval;
        return extract.high;
    }
    return std::numeric_limits<float>::infinity();
}
float distmax_Ray_Triangle(const Ray *__restrict__ ray,
                           const Triangle *__restrict__ tri) {
    const std::optional<TriangleIntersection> isect =
        intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        const TriangleIntersection &isect_ = *isect;
        return isect_.t;
    } else {
        return (-std::numeric_limits<float>::infinity());
    }
}
float distmin_Point_AABB(const Point *__restrict__ pt,
                         const AABB *__restrict__ a) {
    return sqrtf(SqDistPointAABB(pt, a));
}
float distmin_Point_Triangle(const Point *__restrict__ p,
                             const Triangle *__restrict__ tri) {
    const std::tuple<Point, Point> pts = closestPointonTriangle(p, tri);
    return norm(((*p).vec - std::get<0>(pts).vec));
}
float distmin_Ray_Sphere(const Ray *__restrict__ r,
                         const Sphere *__restrict__ s) {
    const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
    if (interval.has_value()) {
        const FInterval &extract = *interval;
        return extract.low;
    }
    return (-std::numeric_limits<float>::infinity());
}
bool intersects_AABB_AABB(const AABB *__restrict__ a,
                          const AABB *__restrict__ b) {
    const vec3_float low = max((*a).low, (*b).low);
    const vec3_float high = min((*a).high, (*b).high);
    return reduce_and((low <= high));
}
bool intersects_AABB_Sphere(const AABB *__restrict__ a,
                            const Sphere *__restrict__ s) {
    const vec3_float _t1 = (*s).center;
    const vec3_float _t6 = (_t1 - max((*a).low, min(_t1, (*a).high)));
    const float _t7 = (*s).radius;
    return (dot(_t6, _t6) <= (_t7 * _t7));
}
bool intersects_AABB_Triangle(const AABB *__restrict__ a,
                              const Triangle *__restrict__ b) {
    const vec3_float _t0 = (*a).low;
    const vec3_float _t1 = (*a).high;
    const vec3_float _t3 = ((_t0 + _t1) * vec3_float{0.50000000f});
    const vec3_float _t7 = ((_t1 - _t0) * vec3_float{0.50000000f});
    const vec3_float _t9 = ((*b).p0 - _t3);
    const vec3_float _t11 = ((*b).p1 - _t3);
    const vec3_float _t13 = ((*b).p2 - _t3);
    const vec3_float f0 = (_t11 - _t9);
    const vec3_float f1 = (_t13 - _t11);
    const vec3_float f2 = (_t9 - _t13);
    const vec3_float tri_min = min(min(_t9, _t11), _t13);
    const vec3_float tri_max = max(max(_t9, _t11), _t13);
    if ((!reduce_and(((tri_min <= _t7) & ((-_t7) <= tri_max))))) {
        return false;
    }
    const vec3_float A0 = cross(f0, vec3_float{1.0f, 0.0f, 0.0f});
    const vec3_float A1 = cross(f0, vec3_float{0.0f, 1.0f, 0.0f});
    const vec3_float A2 = cross(f0, vec3_float{0.0f, 0.0f, 1.0f});
    const vec3_float A3 = cross(f1, vec3_float{1.0f, 0.0f, 0.0f});
    const vec3_float A4 = cross(f1, vec3_float{0.0f, 1.0f, 0.0f});
    const vec3_float A5 = cross(f1, vec3_float{0.0f, 0.0f, 1.0f});
    const vec3_float A6 = cross(f2, vec3_float{1.0f, 0.0f, 0.0f});
    const vec3_float A7 = cross(f2, vec3_float{0.0f, 1.0f, 0.0f});
    const vec3_float A8 = cross(f2, vec3_float{0.0f, 0.0f, 1.0f});
    if ((!axis(A0, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A1, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A2, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A3, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A4, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A5, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A6, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A7, _t7, _t9, _t11, _t13))) {
        return false;
    }
    if ((!axis(A8, _t7, _t9, _t11, _t13))) {
        return false;
    }
    const vec3_float N = cross(f0, f1);
    const float Rn = dot(_t7, abs(N));
    const float dist = dot(_t9, N);
    if (((Rn < dist) || (dist < (-Rn)))) {
        return false;
    }
    return true;
}
bool intersects_Ray_Sphere(const Ray *__restrict__ ray,
                           const Sphere *__restrict__ s) {
    const std::optional<FInterval> interval = intersectsp_ray_sphere(ray, s);
    if (interval.has_value()) {
        const FInterval &extract = *interval;
        return ((extract.low < (*ray).tmax) & (0.0f < extract.high));
    }
    return false;
}
bool intersects_Ray_Triangle(const Ray *__restrict__ ray,
                             const Triangle *__restrict__ tri) {
    return intersectsp_ray_tri(ray, tri).has_value();
}
bool tri_tri_axis(const vec3_float A, const vec3_float a0, const vec3_float a1,
                  const vec3_float a2, const vec3_float b0, const vec3_float b1,
                  const vec3_float b2) {
    if (reduce_and(A == vec3_float{0.0f})) {
        return true;
    }
    const float PA0 = dot(a0, A);
    const float PA1 = dot(a1, A);
    const float PA2 = dot(a2, A);
    const float amin = min(min(PA0, PA1), PA2);
    const float amax = max(max(PA0, PA1), PA2);
    const float PB0 = dot(b0, A);
    const float PB1 = dot(b1, A);
    const float PB2 = dot(b2, A);
    const float bmin = min(min(PB0, PB1), PB2);
    const float bmax = max(max(PB0, PB1), PB2);
    return (!((amax < bmin) | (bmax < amin)));
}
bool intersects_Triangle_Triangle(const Triangle *__restrict__ a,
                                  const Triangle *__restrict__ b) {
    const vec3_float _t0 = (*a).p1;
    const vec3_float _t1 = (*a).p0;
    const vec3_float _t2 = (_t0 - _t1);
    const vec3_float _t3 = (*a).p2;
    const vec3_float _t5 = (_t3 - _t0);
    const vec3_float _t8 = (_t1 - _t3);
    const vec3_float _t9 = (*b).p1;
    const vec3_float _t10 = (*b).p0;
    const vec3_float _t11 = (_t9 - _t10);
    const vec3_float _t12 = (*b).p2;
    const vec3_float _t14 = (_t12 - _t9);
    const vec3_float _t17 = (_t10 - _t12);
    const vec3_float N0 = cross(_t2, _t5);
    const vec3_float N1 = cross(_t11, _t14);
    if ((!tri_tri_axis(N0, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(N1, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    const vec3_float E00 = cross(_t2, _t11);
    const vec3_float E01 = cross(_t2, _t14);
    const vec3_float E02 = cross(_t2, _t17);
    const vec3_float E10 = cross(_t5, _t11);
    const vec3_float E11 = cross(_t5, _t14);
    const vec3_float E12 = cross(_t5, _t17);
    const vec3_float E20 = cross(_t8, _t11);
    const vec3_float E21 = cross(_t8, _t14);
    const vec3_float E22 = cross(_t8, _t17);
    if ((!tri_tri_axis(E00, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E01, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E02, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E10, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E11, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E12, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E20, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E21, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    if ((!tri_tri_axis(E22, _t1, _t0, _t3, _t10, _t9, _t12))) {
        return false;
    }
    return true;
}
float len_squared(const vec3_float v) { return reduce_add((v * v)); }
float linear_to_gamma_f(const float l) {
    if ((0.0f < l)) {
        return sqrtf(l);
    }
    return 0.0f;
}
vec3_float linear_to_gamma_v(const vec3_float l) {
    return vec3_float{linear_to_gamma_f(l[0u]), linear_to_gamma_f(l[1u]),
                      linear_to_gamma_f(l[2u])};
}
bool near_zero(const vec3_float v) {
    return (((abs(v[0]) < 0.00000001f) & (abs(v[1]) < 0.00000001f)) &
            (abs(v[2]) < 0.00000001f));
}
vec3_float reflect(const vec3_float v, const vec3_float n) {
    return (v - (vec3_float{(2.0f * dot(v, n))} * n));
}
float reflectance(const float cos_theta, const float refract_idx) {
    const float _t2 = ((1.0f - refract_idx) / (1.0f + refract_idx));
    const float r1 = (_t2 * _t2);
    return (r1 + ((1.0f - r1) * powf((1.0f - cos_theta), 5.0f)));
}
vec3_float refract(const vec3_float uv, const vec3_float n,
                   const float etai_over_etat) {
    const float cos_theta = min(dot((-uv), n), 1.0f);
    const vec3_float _t2 =
        (vec3_float{etai_over_etat} * (uv + (vec3_float{cos_theta} * n)));
    const vec3_float r_out_parallel =
        (vec3_float{(-sqrtf(abs((1.0f - reduce_add((_t2 * _t2))))))} * n);
    return (_t2 + r_out_parallel);
}
std::optional<Triangle> trace(const Ray *__restrict__ ray,
                              const Triangles *__restrict__ triangles) {
    return _traverse_tree0(ray, triangles);
}
vec3_float unit_vector(const vec3_float v) { return (v / vec3_float{norm(v)}); }
