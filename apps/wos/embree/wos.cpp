#include "apps/wos/embree/wos.h"
Point ClosestPtPointAABB(const Point* __restrict__ pt, const AABB* __restrict__ a) {
  return Point{.vec=min(max((*pt).vec, (*a).low), (*a).high)};
}
float SqDistPointAABB(const Point* __restrict__ pt, const AABB* __restrict__ a) {
  const float3 v = (*pt).vec;
  const float3 sqLow = (((*a).low - v) * ((*a).low - v));
  const float3 low = select((v < (*a).low), sqLow, float3{0.0f});
  const float3 sqHigh = ((v - (*a).high) * (v - (*a).high));
  const float3 high = select(((*a).high < v), sqHigh, float3{0.0f});
  return reduce_add((low + high));
}
float __prod_diff_f32(const float a, const float b, const float c, const float d) {
  const float cd = (c * d);
  const float diff = fmaf(a, b, (-cd));
  const float err = fmaf((-c), d, cd);
  return (diff + err);
}
std::tuple<Point, Point> closestPointonTriangle(const Point* __restrict__ pt, const Triangle* __restrict__ tri) {
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
      return std::tuple<Point, Point>{Point{.vec=a}, Point{.vec=float3{1.0f, 0.0f, 0.0f}}};
    }
  }
  const float3 bp = (p - b);
  const float d3 = dot(ab, bp);
  const float d4 = dot(ac, bp);
  if ((0.0f <= d3)) {
    if ((d4 <= d3)) {
      return std::tuple<Point, Point>{Point{.vec=b}, Point{.vec=float3{0.0f, 1.0f, 0.0f}}};
    }
  }
  const float _t468 = ((d1 * d4) - (d3 * d2));
  if ((_t468 <= 0.0f)) {
    if ((0.0f <= d1)) {
      if ((d3 <= 0.0f)) {
        const float _t470 = (d1 / (d1 - d3));
        return std::tuple<Point, Point>{Point{.vec=(a + (float3{_t470} * ab))}, Point{.vec=float3{(1.0f - _t470), _t470, 0.0f}}};
      }
    }
  }
  const float3 cp = (p - c);
  const float d5 = dot(ab, cp);
  const float d6 = dot(ac, cp);
  if ((0.0f <= d6)) {
    if ((d5 <= d6)) {
      return std::tuple<Point, Point>{Point{.vec=c}, Point{.vec=float3{0.0f, 0.0f, 1.0f}}};
    }
  }
  const float _t473 = ((d5 * d2) - (d1 * d6));
  if ((_t473 <= 0.0f)) {
    if ((0.0f <= d2)) {
      if ((d6 <= 0.0f)) {
        const float _t475 = (d2 / (d2 - d6));
        return std::tuple<Point, Point>{Point{.vec=(a + (float3{_t475} * ac))}, Point{.vec=float3{(1.0f - _t475), 0.0f, _t475}}};
      }
    }
  }
  const float _t478 = ((d3 * d6) - (d5 * d4));
  if ((_t478 <= 0.0f)) {
    const float _t279 = (d4 - d3);
    if ((0.0f <= _t279)) {
      const float _t278 = (d5 - d6);
      if ((0.0f <= _t278)) {
        const float _t480 = (_t279 / (_t279 + _t278));
        return std::tuple<Point, Point>{Point{.vec=(b + (float3{_t480} * (c - b)))}, Point{.vec=float3{0.0f, (1.0f - _t480), _t480}}};
      }
    }
  }
  const float _t483 = (1.0f / ((_t478 + _t473) + _t468));
  const float v = (_t473 * _t483);
  const float w = (_t468 * _t483);
  const float u = (_t478 * _t483);
  return std::tuple<Point, Point>{Point{.vec=((a + (ab * float3{v})) + (ac * float3{w}))}, Point{.vec=float3{u, v, w}}};
}
float3x8 dequantize_bounds_hi(const float3 mlo, const float3 mex, const Qbox38 bound) {
  const float rcp = (1.0f / 255.0f);
  return float3x8{(mlo + (((float3)(bound[0].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[1].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[2].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[3].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[4].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[5].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[6].hi) * float3{rcp}) * mex)), (mlo + (((float3)(bound[7].hi) * float3{rcp}) * mex))};
}
float3x8 dequantize_bounds_lo(const float3 mlo, const float3 mex, const Qbox38 bound) {
  const float rcp = (1.0f / 255.0f);
  return float3x8{(mlo + (((float3)(bound[0].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[1].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[2].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[3].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[4].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[5].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[6].lo) * float3{rcp}) * mex)), (mlo + (((float3)(bound[7].lo) * float3{rcp}) * mex))};
}
Triangle _traverse_tree0(const Point* __restrict__ p, const Triangles* __restrict__ triangles) {
  std::tuple<float, Triangle> _best0 = std::tuple<float, Triangle>{std::numeric_limits<float>::infinity(), Triangle{}};
  int32_t _queue_count0 = 1;
  std::array<uint32_t, 64> _queue0;
  _queue0[0] = 1u;
  do {
    _queue_count0 -= 1;
    const uint32_t I = _queue0[_queue_count0];
    if (I == 4294967295u) {
      if ((_queue_count0 <= 0)) {
        break;
      } else {
        continue;
      }
    }
    if (slice<0, 1>(slice<0, 1>(I)) == 1u) {
      const Interiors& _t446 = (*triangles).interiors[slice<2, 31>(I)];
      const float3 _t8 = _t446.mlo;
      const float3 _t12 = _t446.mex;
      const Qbox38 _t16 = _t446.child_bounds;
      const float3x8 _t17 = dequantize_bounds_lo(_t8, _t12, _t16);
      const float3x8 _t30 = dequantize_bounds_hi(_t8, _t12, _t16);
      const AABB _lv0 = AABB{.low=_t17[0], .high=_t30[0]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv0))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[0u];
        _queue_count0 += 1;
      }
      const AABB _lv1 = AABB{.low=_t17[1], .high=_t30[1]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv1))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[1u];
        _queue_count0 += 1;
      }
      const AABB _lv2 = AABB{.low=_t17[2], .high=_t30[2]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv2))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[2u];
        _queue_count0 += 1;
      }
      const AABB _lv3 = AABB{.low=_t17[3], .high=_t30[3]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv3))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[3u];
        _queue_count0 += 1;
      }
      const AABB _lv4 = AABB{.low=_t17[4], .high=_t30[4]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv4))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[4u];
        _queue_count0 += 1;
      }
      const AABB _lv5 = AABB{.low=_t17[5], .high=_t30[5]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv5))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[5u];
        _queue_count0 += 1;
      }
      const AABB _lv6 = AABB{.low=_t17[6], .high=_t30[6]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv6))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[6u];
        _queue_count0 += 1;
      }
      const AABB _lv7 = AABB{.low=_t17[7], .high=_t30[7]};
      if ((sqrtf(SqDistPointAABB(p, (&_lv7))) < std::get<0>(_best0))) {
        _queue0[_queue_count0] = _t446.children[7u];
        _queue_count0 += 1;
      }
    } else {
      const uint32_t _t258 = slice<7, 31>(I);
      for (uint32_t _idx0 = _t258; _idx0 < (_t258 + (uint32_t)((uint8_t)((slice<2, 6>(I) + 1u)))); ++_idx0) {
        const Triangle& _t464 = (*triangles).primitives[_idx0];
        float _t251 = 0.0f;
        const std::tuple<Point, Point> __inline0 = closestPointonTriangle(p, (&_t464));
        _t251 = norm(((*p).vec - std::get<0>(__inline0).vec));
        bool _t0 = (_t251 < std::get<0>(_best0));
        if (_t0) {
          _best0 = argmin<float, Triangle>(_best0, std::tuple<float, Triangle>{_t251, _t464});
        }
      }
    }
} while ((_queue_count0 != 0));
  return std::get<1>(_best0);
}
float3 compute_merged_extent(const float3x8 lo, const float3x8 hi) {
  const float3 mlo = min(lo[0], min(lo[1], min(lo[2], min(lo[3], min(lo[4], min(lo[5], min(lo[6], lo[7])))))));
  const float3 mhi = max(hi[0], max(hi[1], max(hi[2], max(hi[3], max(hi[4], max(hi[5], max(hi[6], hi[7])))))));
  return (mhi - mlo);
}
uint8_t3 to_u8_ceil(const float3 f) {
  const float3 f1 = ceil(f);
  const float3 f2 = max(float3{0.0f}, min(f1, float3{255.0f}));
  return (uint8_t3)(f2);
}
uint8_t3 to_u8_floor(const float3 f) {
  const float3 f1 = floor(f);
  const float3 f2 = max(float3{0.0f}, min(f1, float3{255.0f}));
  return (uint8_t3)(f2);
}
Qbox38 quantize_bounds(const float3x8 low, const float3x8 high) {
  const float3 _t299 = low[0];
  const float3 _t300 = low[1];
  const float3 _t301 = low[2];
  const float3 _t302 = low[3];
  const float3 _t303 = low[4];
  const float3 _t304 = low[5];
  const float3 _t305 = low[6];
  const float3 _t306 = low[7];
  const float3 _t492 = min(_t299, min(_t300, min(_t301, min(_t302, min(_t303, min(_t304, min(_t305, _t306)))))));
  const float3 mex = compute_merged_extent(low, high);
  const float3 _t494 = ((float3{1.0f} / mex) * float3{255.0f});
  return Qbox38{Qbox3{.lo=to_u8_floor(((_t299 - _t492) * _t494)), .hi=to_u8_ceil(((high[0] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t300 - _t492) * _t494)), .hi=to_u8_ceil(((high[1] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t301 - _t492) * _t494)), .hi=to_u8_ceil(((high[2] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t302 - _t492) * _t494)), .hi=to_u8_ceil(((high[3] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t303 - _t492) * _t494)), .hi=to_u8_ceil(((high[4] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t304 - _t492) * _t494)), .hi=to_u8_ceil(((high[5] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t305 - _t492) * _t494)), .hi=to_u8_ceil(((high[6] - _t492) * _t494))}, Qbox3{.lo=to_u8_floor(((_t306 - _t492) * _t494)), .hi=to_u8_ceil(((high[7] - _t492) * _t494))}};
}
uint32_t rec_build_triangles(const BVH* __restrict__ node, Triangles* __restrict__ ST, size_t* __restrict__ interiors_index, size_t* __restrict__ primitives_index) {
  if ((!node)) {
    return 4294967295u;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*interiors_index);
      (*interiors_index) += 1u;
      (*ST).interiors[this_index].mlo = min(node.lo[0], min(node.lo[1], min(node.lo[2], min(node.lo[3], min(node.lo[4], min(node.lo[5], min(node.lo[6], node.lo[7])))))));
      (*ST).interiors[this_index].mex = compute_merged_extent(node.lo, node.hi);
      (*ST).interiors[this_index].child_bounds = quantize_bounds(node.lo, node.hi);
      std::array<uint32_t, 8> children_index;
      for (int32_t __r = 0; __r < 8; ++__r) {
        children_index[__r] = rec_build_triangles(node.children[__r], ST, interiors_index, primitives_index);
        (*ST).interiors[this_index].children[__r] = children_index[__r];
      }
      return (uint32_t)(((this_index << (uint32_t)(2)) | (uint32_t)(1)));
    },
    [&](const Leaf& node) {
      const uint32_t poffset = (*primitives_index);
      for (uint8_t __p = 0u; __p < node.nprims; ++__p) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return (uint32_t)((((poffset << (uint32_t)(7)) | (((uint32_t)(node.nprims) - (uint32_t)(1)) << (uint32_t)(2))) | (uint32_t)(0)));
    }
  }, *node);
}
void rec_count_triangles(const BVH* __restrict__ node, Triangles* __restrict__ ST) {
  if ((!node)) {
    return;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      for (int32_t __r = 0; __r < 8; ++__r) {
        rec_count_triangles(node.children[__r], ST);
      }
      (*ST).interior_count += 1u;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
    }
  }, *node);
}
Triangles build_triangles(const BVH* __restrict__ CT) {
  Triangles ST;
  size_t primitives_index = 0;
  size_t interiors_index = 0;
  ST.primitive_count = 0u;
  ST.interior_count = 0u;
  rec_count_triangles(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Interiors* interiors = reinterpret_cast<Interiors*>(malloc(sizeof(Interiors) * ST.interior_count));
  ST.interiors = interiors;
  rec_build_triangles(CT, (&ST), (&interiors_index), (&primitives_index));
  return ST;
}
float3 clamp(const float3 x, const float low, const float high) {
  return min(max(x, float3{low}), float3{high});
}
Triangle closest_point(const Point* __restrict__ p, const Triangles* __restrict__ triangles) {
  return _traverse_tree0(p, triangles);
}
float3 compute_merged_low(const float3x8 low) {
  return min(low[0], min(low[1], min(low[2], min(low[3], min(low[4], min(low[5], min(low[6], low[7])))))));
}
float3 cross_(const float3 v0, const float3 v1) {
  const float _t283 = v0[1u];
  const float _t284 = v1[2u];
  const float _t285 = v0[2u];
  const float _t286 = v1[1u];
  const float _t288 = v1[0u];
  const float _t289 = v0[0u];
  return float3{__prod_diff_f32(_t283, _t284, _t285, _t286), __prod_diff_f32(_t285, _t288, _t289, _t284), __prod_diff_f32(_t289, _t286, _t283, _t288)};
}
float degrees_to_radians(const float degrees) {
  return ((degrees * 3.14159274f) / 180.0f);
}
float distmin_Point_AABB(const Point* __restrict__ pt, const AABB* __restrict__ a) {
  return sqrtf(SqDistPointAABB(pt, a));
}
float distmin_Point_Triangle(const Point* __restrict__ p, const Triangle* __restrict__ tri) {
  const std::tuple<Point, Point> pts = closestPointonTriangle(p, tri);
  return norm(((*p).vec - std::get<0>(pts).vec));
}
float gamma(const int32_t n) {
  const float _t485 = ((float)(n) * 0.00000006f);
  return (_t485 / (1.0f - _t485));
}
float len_squared(const float3 v) {
  return reduce_add((v * v));
}
float linear_to_gamma_f(const float l) {
  if ((0.0f < l)) {
    return sqrtf(l);
  }
  return 0.0f;
}
float3 linear_to_gamma_v(const float3 l) {
  return float3{linear_to_gamma_f(l[0u]), linear_to_gamma_f(l[1u]), linear_to_gamma_f(l[2u])};
}
bool near_zero(const float3 v) {
  return (((abs(v[0]) < 0.00000001f) & (abs(v[1]) < 0.00000001f)) & (abs(v[2]) < 0.00000001f));
}
float3 reflect(const float3 v, const float3 n) {
  return (v - (float3{(2.0f * dot(v, n))} * n));
}
float reflectance(const float cos_theta, const float refract_idx) {
  const float _t497 = ((1.0f - refract_idx) / (1.0f + refract_idx));
  const float r1 = (_t497 * _t497);
  return (r1 + ((1.0f - r1) * powf((1.0f - cos_theta), 5.0f)));
}
float3 refract(const float3 uv, const float3 n, const float etai_over_etat) {
  const float cos_theta = min(dot((-uv), n), 1.0f);
  const float3 _t500 = (float3{etai_over_etat} * (uv + (float3{cos_theta} * n)));
  const float3 r_out_parallel = (float3{(-sqrtf(abs((1.0f - reduce_add((_t500 * _t500))))))} * n);
  return (_t500 + r_out_parallel);
}
float3 unit_vector(const float3 v) {
  return (v / float3{norm(v)});
}
