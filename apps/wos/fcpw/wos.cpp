#include "apps/wos/fcpw/wos.h"
Point ClosestPtPointAABB(const Point *__restrict__ pt,
                         const AABB *__restrict__ a) {
    return Point{.vec = min(max((*pt).vec, (*a).low), (*a).high)};
}
float SqDistPointAABB(const Point *__restrict__ pt,
                      const AABB *__restrict__ a) {
    const float3 v = (*pt).vec;
    const float3 sqLow = (((*a).low - v) * ((*a).low - v));
    const float3 low = select((v < (*a).low), sqLow, float3{0.0f});
    const float3 sqHigh = ((v - (*a).high) * (v - (*a).high));
    const float3 high = select(((*a).high < v), sqHigh, float3{0.0f});
    return reduce_add((low + high));
}
float __prod_diff_f32(const float a, const float b, const float c,
                      const float d) {
    const float cd = (c * d);
    const float diff = fmaf(a, b, (-cd));
    const float err = fmaf((-c), d, cd);
    return (diff + err);
}
std::tuple<Point, Point>
closestPointonTriangle(const Point *__restrict__ pt,
                       const Triangle *__restrict__ tri) {
    const float3 p = (*pt).vec;
    const float3 a = (*tri).p0;
    const float3 b = (*tri).p1;
    const float3 c = (*tri).p2;
    const float3 ab = (b - a);
    const float3 ac = (c - a);
    const float3 ap = (p - a);
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if ((d1 <= 0.0f)) {
        if ((d2 <= 0.0f)) {
            return std::tuple<Point, Point>{
                Point{.vec = a}, Point{.vec = float3{1.0f, 0.0f, 0.0f}}};
        }
    }
    const float3 bp = (p - b);
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if ((0.0f <= d3)) {
        if ((d4 <= d3)) {
            return std::tuple<Point, Point>{
                Point{.vec = b}, Point{.vec = float3{0.0f, 1.0f, 0.0f}}};
        }
    }
    const float _t209 = ((d1 * d4) - (d3 * d2));
    if ((_t209 <= 0.0f)) {
        if ((0.0f <= d1)) {
            if ((d3 <= 0.0f)) {
                const float _t211 = (d1 / (d1 - d3));
                return std::tuple<Point, Point>{
                    Point{.vec = (a + (float3{_t211} * ab))},
                    Point{.vec = float3{(1.0f - _t211), _t211, 0.0f}}};
            }
        }
    }
    const float3 cp = (p - c);
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if ((0.0f <= d6)) {
        if ((d5 <= d6)) {
            return std::tuple<Point, Point>{
                Point{.vec = c}, Point{.vec = float3{0.0f, 0.0f, 1.0f}}};
        }
    }
    const float _t214 = ((d5 * d2) - (d1 * d6));
    if ((_t214 <= 0.0f)) {
        if ((0.0f <= d2)) {
            if ((d6 <= 0.0f)) {
                const float _t216 = (d2 / (d2 - d6));
                return std::tuple<Point, Point>{
                    Point{.vec = (a + (float3{_t216} * ac))},
                    Point{.vec = float3{(1.0f - _t216), 0.0f, _t216}}};
            }
        }
    }
    const float _t219 = ((d3 * d6) - (d5 * d4));
    if ((_t219 <= 0.0f)) {
        const float _t62 = (d4 - d3);
        if ((0.0f <= _t62)) {
            const float _t61 = (d5 - d6);
            if ((0.0f <= _t61)) {
                const float _t221 = (_t62 / (_t62 + _t61));
                return std::tuple<Point, Point>{
                    Point{.vec = (b + (float3{_t221} * (c - b)))},
                    Point{.vec = float3{0.0f, (1.0f - _t221), _t221}}};
            }
        }
    }
    const float _t224 = (1.0f / ((_t219 + _t214) + _t209));
    const float v = (_t214 * _t224);
    const float w = (_t209 * _t224);
    const float u = (_t219 * _t224);
    return std::tuple<Point, Point>{
        Point{.vec = ((a + (ab * float3{v})) + (ac * float3{w}))},
        Point{.vec = float3{u, v, w}}};
}
float distmax_Point_AABB(const Point *__restrict__ pt,
                         const AABB *__restrict__ a) {
    const float3 _t79 = (*pt).vec;
    const float3 _t229 = min(((*a).low - _t79), (_t79 - (*a).high));
    return dot(_t229, _t229);
}
Triangle _traverse_tree0(const Point *__restrict__ p,
                         const Triangles *__restrict__ triangles) {
    std::tuple<float, Triangle> _best0 = std::tuple<float, Triangle>{
        std::numeric_limits<float>::infinity(), Triangle{}};
    int32_t _queue_count0 = 1;
    std::array<uint32_t, 64> _queue0;
    _queue0[0] = 0u;
    do {
        _queue_count0 -= 1;
        const uint32_t index = _queue0[_queue_count0];
        if (index == 4294967295u) {
            if ((_queue_count0 <= 0)) {
                break;
            } else {
                continue;
            }
        }
        const Nodes &_t195 = (*triangles).nodes[index];
        const AABB _t198 = AABB{.low = _t195.low, .high = _t195.high};
        if ((SqDistPointAABB(p, (&_t198)) < std::get<0>(_best0))) {
            const uint16_t _t35 = _t195.nprims;
            if (_t35 == 0u) {
                _best0 = argmin<float, Triangle>(
                    _best0, std::tuple<float, Triangle>{
                                distmax_Point_AABB(p, (&_t198)), Triangle{}});
                const uint32_t L = index + 1u;
                const uint32_t R =
                    index +
                    reinterpret<Arm_Interior>(_t195.split0on_nprims).offset;

                const Nodes &Ln = (*triangles).nodes[L];
                const float3 Llow = Ln.low;
                const float3 Lhigh = Ln.high;
                const AABB lAABB = AABB{.low = Llow, .high = Lhigh};
                float left_dist = SqDistPointAABB(p, &lAABB);
                const Nodes &Rn = (*triangles).nodes[R];
                const AABB rAABB = AABB{.low = Rn.low, .high = Rn.high};
                float right_dist = SqDistPointAABB(p, &rAABB);
                if (left_dist <= right_dist) {
                    _queue0[_queue_count0] = R;
                    _queue0[(_queue_count0 + 1)] = L;
                } else {
                    _queue0[_queue_count0] = L;
                    _queue0[(_queue_count0 + 1)] = R;
                }
                _queue_count0 += 2;
            } else {
                const uint32_t _t24 =
                    reinterpret<Arm_Leaf>(_t195.split0on_nprims).poffset;
                for (uint32_t _idx0 = _t24; _idx0 < (_t24 + (uint32_t)(_t35));
                     ++_idx0) {
                    const Triangle &_t202 = (*triangles).primitives[_idx0];
                    float _t14 = 0.0f;
                    const std::tuple<Point, Point> __inline1 =
                        closestPointonTriangle(p, (&_t202));
                    const float3 __inline0 =
                        ((*p).vec - std::get<0>(__inline1).vec);
                    _t14 = dot(__inline0, __inline0);
                    bool _t0 = (_t14 < std::get<0>(_best0));
                    if (_t0) {
                        float _t1 = _t14;
                        _best0 = std::tuple<float, Triangle>{_t1, _t202};
                    }
                }
            }
        }
    } while ((_queue_count0 != 0));
    return std::get<1>(_best0);
}
uint32_t rec_build_triangles(const BVH *__restrict__ node,
                             Triangles *__restrict__ ST,
                             size_t *__restrict__ nodes_index,
                             size_t *__restrict__ primitives_index) {
    if ((!node)) {
        return 4294967295u;
    }
    return std::visit(
        overloaded{[&](const Interior &node) {
                       const size_t this_index = (*nodes_index);
                       (*nodes_index) += 1u;
                       (*ST).nodes[this_index].low = node.low;
                       (*ST).nodes[this_index].high = node.high;
                       (*ST).nodes[this_index].nprims = 0;
                       const uint32_t left_index = rec_build_triangles(
                           node.left, ST, nodes_index, primitives_index);
                       const uint32_t right_index = rec_build_triangles(
                           node.right, ST, nodes_index, primitives_index);
                       reinterpret_cast<Arm_Interior *>(
                           &(*ST).nodes[this_index].split0on_nprims)
                           ->offset = (right_index - this_index);
                       return this_index;
                   },
                   [&](const Leaf &node) {
                       const size_t this_index = (*nodes_index);
                       (*nodes_index) += 1u;
                       (*ST).nodes[this_index].low = node.low;
                       (*ST).nodes[this_index].high = node.high;
                       (*ST).nodes[this_index].nprims = node.nprims;
                       reinterpret_cast<Arm_Leaf *>(
                           &(*ST).nodes[this_index].split0on_nprims)
                           ->poffset = (*primitives_index);
                       for (uint16_t __p = 0u; __p < node.nprims; ++__p) {
                           (*ST).primitives[(__p + (*primitives_index))] =
                               node.data[__p];
                       }
                       (*primitives_index) += node.nprims;
                       return this_index;
                   }},
        *node);
}
void rec_count_triangles(const BVH *__restrict__ node,
                         Triangles *__restrict__ ST) {
    if ((!node)) {
        return;
    }
    return std::visit(overloaded{[&](const Interior &node) {
                                     rec_count_triangles(node.left, ST);
                                     rec_count_triangles(node.right, ST);
                                     (*ST).node_count += 1u;
                                 },
                                 [&](const Leaf &node) {
                                     (*ST).primitive_count += node.nprims;
                                     (*ST).node_count += 1u;
                                 }},
                      *node);
}
Triangles build_triangles(const BVH *__restrict__ CT) {
    Triangles ST;
    size_t primitives_index = 0;
    size_t nodes_index = 0;
    ST.primitive_count = 0u;
    ST.node_count = 0u;
    rec_count_triangles(CT, (&ST));
    Triangle *primitives = reinterpret_cast<Triangle *>(
        malloc(sizeof(Triangle) * ST.primitive_count));
    ST.primitives = primitives;
    Nodes *nodes = reinterpret_cast<Nodes *>(std::aligned_alloc(
        32, (((sizeof(Nodes) * ST.node_count) + 31) / 32) * 32));
    ST.nodes = nodes;
    rec_build_triangles(CT, (&ST), (&nodes_index), (&primitives_index));
    return ST;
}
float3 clamp(const float3 x, const float low, const float high) {
    return min(max(x, float3{low}), float3{high});
}
Triangle closest_point(const Point *__restrict__ p,
                       const Triangles *__restrict__ triangles) {
    return _traverse_tree0(p, triangles);
}
float3 cross_(const float3 v0, const float3 v1) {
    const float _t66 = v0[1u];
    const float _t67 = v1[2u];
    const float _t68 = v0[2u];
    const float _t69 = v1[1u];
    const float _t71 = v1[0u];
    const float _t72 = v0[0u];
    return float3{__prod_diff_f32(_t66, _t67, _t68, _t69),
                  __prod_diff_f32(_t68, _t71, _t72, _t67),
                  __prod_diff_f32(_t72, _t69, _t66, _t71)};
}
float degrees_to_radians(const float degrees) {
    return ((degrees * 3.14159274f) / 180.0f);
}
float distmin_Point_AABB(const Point *__restrict__ pt,
                         const AABB *__restrict__ a) {
    return SqDistPointAABB(pt, a);
}
float distmin_Point_Triangle(const Point *__restrict__ p,
                             const Triangle *__restrict__ tri) {
    const std::tuple<Point, Point> pts = closestPointonTriangle(p, tri);
    const float3 _t233 = ((*p).vec - std::get<0>(pts).vec);
    return dot(_t233, _t233);
}
float gamma(const int32_t n) {
    const float _t235 = ((float)(n) * 0.00000006f);
    return (_t235 / (1.0f - _t235));
}
float len_squared(const float3 v) { return reduce_add((v * v)); }
float linear_to_gamma_f(const float l) {
    if ((0.0f < l)) {
        return sqrtf(l);
    }
    return 0.0f;
}
float3 linear_to_gamma_v(const float3 l) {
    return float3{linear_to_gamma_f(l[0u]), linear_to_gamma_f(l[1u]),
                  linear_to_gamma_f(l[2u])};
}
bool near_zero(const float3 v) {
    return (((abs(v[0]) < 0.00000001f) & (abs(v[1]) < 0.00000001f)) &
            (abs(v[2]) < 0.00000001f));
}
float3 reflect(const float3 v, const float3 n) {
    return (v - (float3{(2.0f * dot(v, n))} * n));
}
float reflectance(const float cos_theta, const float refract_idx) {
    const float _t238 = ((1.0f - refract_idx) / (1.0f + refract_idx));
    const float r1 = (_t238 * _t238);
    return (r1 + ((1.0f - r1) * powf((1.0f - cos_theta), 5.0f)));
}
float3 refract(const float3 uv, const float3 n, const float etai_over_etat) {
    const float cos_theta = min(dot((-uv), n), 1.0f);
    const float3 _t241 =
        (float3{etai_over_etat} * (uv + (float3{cos_theta} * n)));
    const float3 r_out_parallel =
        (float3{(-sqrtf(abs((1.0f - reduce_add((_t241 * _t241))))))} * n);
    return (_t241 + r_out_parallel);
}
float3 unit_vector(const float3 v) { return (v / float3{norm(v)}); }
