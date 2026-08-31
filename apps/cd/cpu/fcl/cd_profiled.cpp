#include "apps/cd/cpu/fcl/cd.h"
#include "profile.h"

struct AABB {
    float3 low;
    float3 high;
};

bool intersects_AABB_AABB(const AABB a, const AABB b) {
    bonsai_aabb_counter++;
    const float3 low = max(a.low, b.low);
    const float3 high = min(a.high, b.high);
    return reduce_and(low <= high);
}
int32_t project6(const float3 ax, const float3 p1, const float3 p2,
                 const float3 p3, const float3 q1, const float3 q2,
                 const float3 q3) {
    const float P1 = dot(ax, p1);
    const float P2 = dot(ax, p2);
    const float P3 = dot(ax, p3);
    const float Q1 = dot(ax, q1);
    const float Q2 = dot(ax, q2);
    const float Q3 = dot(ax, q3);
    const float mn1 = min(min(P1, P2), P3);
    const float mx2 = max(max(Q1, Q2), Q3);
    if (mx2 < mn1) {
        return 0;
    }
    const float mx1 = max(max(P1, P2), P3);
    const float mn2 = min(min(Q1, Q2), Q3);
    if (mx1 < mn2) {
        return 0;
    }
    return 1;
}
bool fcl_triangle_intersect(const float3 P1, const float3 P2, const float3 P3,
                            const float3 Q1, const float3 Q2, const float3 Q3) {
    const float3 p2 = P2 - P1;
    const float3 p3 = P3 - P1;
    const float3 q1 = Q1 - P1;
    const float3 q2 = Q2 - P1;
    const float3 q3 = Q3 - P1;
    const float3 e2 = p3 - p2;
    const float3 n1 = cross(p2, e2);
    if (project6(n1, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 f1 = q2 - q1;
    const float3 f2 = q3 - q2;
    const float3 m1 = cross(f1, f2);
    if (project6(m1, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef11 = cross(p2, f1);
    if (project6(ef11, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef12 = cross(p2, f2);
    if (project6(ef12, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 f3 = q1 - q3;
    const float3 ef13 = cross(p2, f3);
    if (project6(ef13, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef21 = cross(e2, f1);
    if (project6(ef21, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef22 = cross(e2, f2);
    if (project6(ef22, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef23 = cross(e2, f3);
    if (project6(ef23, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 e3 = -p3;
    const float3 ef31 = cross(e3, f1);
    if (project6(ef31, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef32 = cross(e3, f2);
    if (project6(ef32, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 ef33 = cross(e3, f3);
    if (project6(ef33, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 g1 = cross(p2, n1);
    if (project6(g1, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 g2 = cross(e2, n1);
    if (project6(g2, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 g3 = cross(e3, n1);
    if (project6(g3, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 h1 = cross(f1, m1);
    if (project6(h1, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 h2 = cross(f2, m1);
    if (project6(h2, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    const float3 h3 = cross(f3, m1);
    if (project6(h3, float3{0.00000000f}, p2, p3, q1, q2, q3) == 0) {
        return false;
    }
    return true;
}
bool intersects_Triangle_Triangle(const Triangle a, const Triangle b) {
    bonsai_tri_counter++;
    return fcl_triangle_intersect(a.p0, a.p1, a.p2, b.p0, b.p1, b.p2);
}
void _recloop_func0(const uint32_t triangles1_index,
                    const uint32_t triangles2_index,
                    const _tree_layout0 &triangles1,
                    const _tree_layout4 &triangles2,
                    set<std::tuple<Triangle, Triangle>> &_dyn_alloc) {
    bonsai_rec_counter++;
    const _tree_layout1 _t72 = triangles1.group0_index[triangles1_index];
    const _tree_layout5 _t76 = triangles2.group0_index[triangles2_index];
    if (intersects_AABB_AABB(AABB{_t72.low, _t72.high},
                             AABB{_t76.low, _t76.high})) {
        const uint32_t _t70 = _t72.nPrims;
        if (_t70 == 0u) {
            if (_t76.nPrims == 0u) {
                const uint32_t _t0 = triangles1_index + 1u;
                const uint32_t _t1 = triangles2_index + 1u;
                _recloop_func0(_t0, _t1, triangles1, triangles2, _dyn_alloc);
                const uint32_t _t6 = triangles2_index + _t76.offset;
                _recloop_func0(_t0, _t6, triangles1, triangles2, _dyn_alloc);
                const uint32_t _t10 = triangles1_index + _t72.offset;
                _recloop_func0(_t10, _t1, triangles1, triangles2, _dyn_alloc);
                _recloop_func0(_t10, _t6, triangles1, triangles2, _dyn_alloc);
            } else {
                _recloop_func0(triangles1_index + 1u, triangles2_index,
                               triangles1, triangles2, _dyn_alloc);
                _recloop_func0(triangles1_index + _t72.offset, triangles2_index,
                               triangles1, triangles2, _dyn_alloc);
            }
        } else {
            const uint32_t _t66 = _t76.nPrims;
            if (_t66 == 0u) {
                _recloop_func0(triangles1_index, triangles2_index + 1u,
                               triangles1, triangles2, _dyn_alloc);
                _recloop_func0(triangles1_index, triangles2_index + _t76.offset,
                               triangles1, triangles2, _dyn_alloc);
            } else {
                for (uint32_t _idx0 = 0u; _idx0 < _t70; _idx0 += 1u) {
                    for (uint32_t _idx1 = 0u; _idx1 < _t66; _idx1 += 1u) {
                        const Triangle _t39 =
                            triangles1.prims[_t72.offset + _idx0];
                        const Triangle _t45 =
                            triangles2.prims[_t76.offset + _idx1];
                        if (intersects_Triangle_Triangle(_t39, _t45)) {
                            _dyn_alloc.push_back(std::make_tuple(_t39, _t45));
                        }
                    }
                }
            }
        }
    }
    return;
}
set<std::tuple<Triangle, Triangle>>
_traverse_tree0(const _tree_layout0 &triangles1,
                const _tree_layout4 &triangles2) {
    set<std::tuple<Triangle, Triangle>> _dyn_alloc;
    _recloop_func0(0u, 0u, triangles1, triangles2, _dyn_alloc);
    return _dyn_alloc;
}
bool axis(const float3 A, const float3 extents, const float3 v0,
          const float3 v1, const float3 v2) {
    const float R = dot(extents, abs(A));
    const float P0 = dot(v0, A);
    const float P1 = dot(v1, A);
    const float P2 = dot(v2, A);
    const float Pmin = min(min(P0, P1), P2);
    const float Pmax = max(max(P0, P1), P2);
    return (-R <= Pmax) & (Pmin <= R);
}
set<std::tuple<Triangle, Triangle>>
collisions(const _tree_layout0 &triangles1, const _tree_layout4 &triangles2) {
    return _traverse_tree0(triangles1, triangles2);
}
