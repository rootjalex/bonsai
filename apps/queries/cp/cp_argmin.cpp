#include "apps/queries/cp/cp_gen.h"
struct AABB {
    float3 low;
    float3 high;
};

float SqDistPointAABB(const Point &pt, const AABB &a) {
    const float3 v = pt.vec;
    const float3 sqLow = (a.low - v) * (a.low - v);
    const float3 low = select(v < a.low, sqLow, float3{0.00000000f});
    const float3 sqHigh = (v - a.high) * (v - a.high);
    const float3 high = select(a.high < v, sqHigh, float3{0.00000000f});
    return reduce_add(low + high);
}
float distmax_Point_AABB(const Point &pt, const AABB &a) {
    const float3 _t1 = pt.vec;
    const float3 d = min(a.low - _t1, _t1 - a.high);
    return dot(d, d);
}
float distmin_Point_AABB(const Point &pt, const AABB &a) {
    const float3 _t1 = pt.vec;
    const float3 _t7 = max(max(a.low - _t1, _t1 - a.high), float3{0.00000000f});
    return dot(_t7, _t7);
}
std::tuple<Point, Point> closestPointonTriangle(const Point &pt,
                                                const Triangle &tri) {
    const float3 p = pt.vec;
    const float3 a = tri.p0;
    const float3 b = tri.p1;
    const float3 c = tri.p2;
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.00000000f) {
        if (d2 <= 0.00000000f) {
            return std::tuple<Point, Point>{
                Point{a}, Point{float3{1.00000000f, 0.00000000f, 0.00000000f}}};
        }
    }
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (0.00000000f <= d3) {
        if (d4 <= d3) {
            return std::tuple<Point, Point>{
                Point{b}, Point{float3{0.00000000f, 1.00000000f, 0.00000000f}}};
        }
    }
    const float _t2 = (d1 * d4) - (d3 * d2);
    if (_t2 <= 0.00000000f) {
        if (0.00000000f <= d1) {
            if (d3 <= 0.00000000f) {
                const float _t4 = d1 / (d1 - d3);
                return std::tuple<Point, Point>{
                    Point{a + (float3{_t4} * ab)},
                    Point{vector<float, 3>(
                        {1.00000000f - _t4, _t4, 0.00000000f})}};
            }
        }
    }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (0.00000000f <= d6) {
        if (d5 <= d6) {
            return std::tuple<Point, Point>{
                Point{c}, Point{float3{0.00000000f, 0.00000000f, 1.00000000f}}};
        }
    }
    const float _t7 = (d5 * d2) - (d1 * d6);
    if (_t7 <= 0.00000000f) {
        if (0.00000000f <= d2) {
            if (d6 <= 0.00000000f) {
                const float _t9 = d2 / (d2 - d6);
                return std::tuple<Point, Point>{
                    Point{a + (float3{_t9} * ac)},
                    Point{vector<float, 3>(
                        {1.00000000f - _t9, 0.00000000f, _t9})}};
            }
        }
    }
    const float _t12 = (d3 * d6) - (d5 * d4);
    if (_t12 <= 0.00000000f) {
        const float _t19 = d4 - d3;
        if (0.00000000f <= _t19) {
            const float _t18 = d5 - d6;
            if (0.00000000f <= _t18) {
                const float _t17 = _t19 / (_t19 + _t18);
                return std::tuple<Point, Point>{
                    Point{b + (float3{_t17} * (c - b))},
                    Point{vector<float, 3>(
                        {0.00000000f, 1.00000000f - _t17, _t17})}};
            }
        }
    }
    const float _t22 = 1.00000000f / ((_t12 + _t7) + _t2);
    const float v = _t7 * _t22;
    const float w = _t2 * _t22;
    const float u = _t12 * _t22;
    return std::tuple<Point, Point>{
        Point{(a + (ab * float3{v})) + (ac * float3{w})},
        Point{vector<float, 3>({u, v, w})}};
}
float distmin_Point_Triangle(const Point &p, const Triangle &tri) {
    const std::tuple<Point, Point> pts = closestPointonTriangle(p, tri);
    const float3 _t3 = p.vec - std::get<0>(pts).vec;
    return dot(_t3, _t3);
}
Triangle _traverse_tree0(const Point &p, const _tree_layout0 &tris) {
    std::tuple<float, Triangle> _best0 = std::tuple<float, Triangle>{
        std::numeric_limits<float>::infinity(), Triangle{}};
    int32_t _queue_count0 = 1;
    uint32_t _queue0[64];
    _queue0[0] = 0u;
    do {
        _queue_count0 -= 1;
        const uint32_t &tris_index = _queue0[_queue_count0];
        const _tree_layout1 *_t85 = tris.group0_index;
        const _tree_layout1 &_t86 = _t85[tris_index];
        const AABB _t91 = AABB{_t86.low, _t86.high};
        if (distmin_Point_AABB(p, _t91) < std::get<0>(_best0)) {
            const uint32_t _t84 = _t86.nPrims;
            if (_t84 == 0u) {
                _best0 = argmin<float, Triangle>(
                    _best0, std::tuple<float, Triangle>{
                                distmax_Point_AABB(p, _t91), Triangle{}});
                const uint32_t _t11 = tris_index + _t86.offset;
                const uint32_t _t18 = tris_index + 1u;
                const bool _sort_cmp0 =
                    distmin_Point_AABB(p,
                                       AABB{_t85[_t11].low, _t85[_t11].high}) <
                    distmin_Point_AABB(p,
                                       AABB{_t85[_t18].low, _t85[_t18].high});
                const uint32_t _sort_tmp0 = (_sort_cmp0 ? _t18 : _t11);
                const uint32_t _sort_tmp1 = (_sort_cmp0 ? _t11 : _t18);
                _queue0[_queue_count0] = _sort_tmp0;
                _queue0[_queue_count0 + 1] = _sort_tmp1;
                _queue_count0 += 2;
            } else {
                for (uint32_t _idx0 = 0u; _idx0 < _t84; _idx0 += 1u) {
                    const Triangle &_t77 = tris.prims[_t86.offset + _idx0];
                    const float _t78 = distmin_Point_Triangle(p, _t77);
                    if (_t78 < std::get<0>(_best0)) {
                        _best0 = std::tuple<float, Triangle>{_t78, _t77};
                    }
                }
            }
        }
    } while (_queue_count0 != 0);
    return std::get<1>(_best0);
}
Triangle closest(const Point &p, const _tree_layout0 &tris) {
    return _traverse_tree0(p, tris);
}
