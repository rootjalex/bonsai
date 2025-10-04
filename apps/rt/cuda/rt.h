#include "helpers.h"

#include <cuda/std/array>
#include <thrust/universal_vector.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>

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
  cuda::std::array<BVH*, 8> aabb_children;
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
  Triangle* data;
};

struct OBB {
  float3 obb_low;
  float3 obb_high;
  cuda::std::array<float4, 3> orientation;
};

struct OBBNode {
  cuda::std::array<BVH*, 8> obb_children;
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
  cuda::std::array<float4, 3> orientation;
  cuda::std::array<float3, 8> obb_low;
  cuda::std::array<float3, 8> obb_high;
  cuda::std::array<uint64_t, 8> obb_children;
} __attribute__((packed));

struct Triangles {
  uint64_t primitive_count;
  Triangle* primitives;
  uint64_t aabb_count;
  Aabbs* aabbs;
  uint64_t obb_count;
  Obbs* obbs;
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

__host__ cuda::std::optional<FInterval> intersectsp_ray_aabb(Ray* r, AABB* b) {
  float3 _t1 = (make_float3(1) / (*r).d);
  bool3 dirIsNeg = (_t1 < make_float3(0));
  float3 _t2 = (*b).high;
  float3 _t3 = (*b).low;
  float3 low_parts = make_float3((dirIsNeg.x ? _t2.x : _t3.x),(dirIsNeg.y ? _t2.y : _t3.y),(dirIsNeg.z ? _t2.z : _t3.z));
  float3 high_parts = make_float3((dirIsNeg.x ? _t3.x : _t2.x),(dirIsNeg.y ? _t3.y : _t2.y),(dirIsNeg.z ? _t3.z : _t2.z));
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

__host__ float distmin_Ray_AABB(Ray* r, AABB* b) {
  cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    FInterval extract = *interval;
    return extract.low;
  }
  return -INFINITY;
}

__host__ cuda::std::optional<FInterval> intersectsp_ray_obb(Ray* r, OBB* b) {
  cuda::std::array<float4, 3> _t0 = (*b).orientation;
  float4 _t1 = _t0.x;
  float _t2 = _t1.x;
  float3 _t3 = (*r).o;
  float _t4 = _t3.x;
  float _t8 = _t1.y;
  float _t10 = _t3.y;
  float _t15 = _t1.z;
  float _t17 = _t3.z;
  float _t23 = ((((_t2 * _t4) + (_t8 * _t10)) + (_t15 * _t17)) + _t1.w);
  float4 _t25 = _t0.y;
  float _t26 = _t25.x;
  float _t32 = _t25.y;
  float _t39 = _t25.z;
  float _t47 = ((((_t26 * _t4) + (_t32 * _t10)) + (_t39 * _t17)) + _t25.w);
  float4 _t49 = _t0.z;
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

__host__ float distmin_Ray_OBB(Ray* r, OBB* b) {
  cuda::std::optional<FInterval> interval = intersectsp_ray_obb(r, b);
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
  return float3{__prod_diff_f32(_t0, _t1, _t2, _t3), __prod_diff_f32(_t2, _t5, _t6, _t1), __prod_diff_f32(_t6, _t3, _t0, _t5)};
}

__host__ cuda::std::optional<TriangleIntersection> intersectsp_ray_tri(Ray* ray, Triangle* tri) {
  float3 _t0 = (*tri).p2;
  float3 _t1 = (*tri).p0;
  float3 _t2 = (*tri).p1;
  if (sum((cross_(_t0 - _t1, _t2 - _t1) * cross_(_t0 - _t1, _t2 - _t1))) == 0) {
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
    if ((0 < _t24) && ((tScaled <= 0) || (((*ray).tmax * _t24) < tScaled))) {
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
  float deltaE = ((float)2 * ((((gamma(2) * maxXt) * maxYt) + (deltaY * maxXt)) + (deltaX * maxYt)));
  float maxE = max(abs(float3{_t20, _t21, _t22}));
  float deltaT = (((float)3 * ((((_t25 * maxE) * maxZt) + (deltaE * maxZt)) + (deltaZ * maxE))) * fabsf(invDet));
  if (t <= deltaT) {
    return cuda::std::nullopt;
  }
  return TriangleIntersection{b0, b1, b2, t};
}

__host__ float distmin_Ray_Triangle(Ray* ray, Triangle* tri) {
  cuda::std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    TriangleIntersection isect_ = *isect;
    return isect_.t;
  } else {
    return INFINITY;
  }
}

__host__ bool intersects_Ray_AABB(Ray* r, AABB* b) {
  cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    FInterval extract = *interval;
    return ((extract.low < (*r).tmax) & (0 < extract.high));
  }
  return false;
}

__host__ bool intersects_Ray_OBB(Ray* r, OBB* b) {
  cuda::std::optional<FInterval> interval = intersectsp_ray_obb(r, b);
  if (interval.has_value()) {
    FInterval extract = *interval;
    return ((extract.low < (*r).tmax) & (0 < extract.high));
  }
  return false;
}

__host__ void _recloop_func0(uint64_t index, Ray* ray, Triangles* triangles, cuda::std::tuple<float, Triangle>* _best0) {
  if (index == 18446744073709551615u) {
    return;
  }
  uint64_t _t493 = slice<0, 2>(index);
  if (_t493 == 2u) {
    Aabbs _t17 = (*triangles).aabbs[slice<4, 63>(index)];
    cuda::std::array<float3, 8> _t18 = _t17.aabb_low;
    cuda::std::array<float3, 8> _t23 = _t17.aabb_high;
    AABB _t25 = AABB{_t18[0], _t23[0]};
    if (intersects_Ray_AABB(ray, (&_t25))) {
      if (distmin_Ray_AABB(ray, (&_t25)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[0u], ray, triangles, _best0);
      }
    }
    AABB _t51 = AABB{_t18[1], _t23[1]};
    if (intersects_Ray_AABB(ray, (&_t51))) {
      if (distmin_Ray_AABB(ray, (&_t51)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[1u], ray, triangles, _best0);
      }
    }
    AABB _t77 = AABB{_t18[2], _t23[2]};
    if (intersects_Ray_AABB(ray, (&_t77))) {
      if (distmin_Ray_AABB(ray, (&_t77)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[2u], ray, triangles, _best0);
      }
    }
    AABB _t103 = AABB{_t18[3], _t23[3]};
    if (intersects_Ray_AABB(ray, (&_t103))) {
      if (distmin_Ray_AABB(ray, (&_t103)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[3u], ray, triangles, _best0);
      }
    }
    AABB _t129 = AABB{_t18[4], _t23[4]};
    if (intersects_Ray_AABB(ray, (&_t129))) {
      if (distmin_Ray_AABB(ray, (&_t129)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[4u], ray, triangles, _best0);
      }
    }
    AABB _t155 = AABB{_t18[5], _t23[5]};
    if (intersects_Ray_AABB(ray, (&_t155))) {
      if (distmin_Ray_AABB(ray, (&_t155)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[5u], ray, triangles, _best0);
      }
    }
    AABB _t181 = AABB{_t18[6], _t23[6]};
    if (intersects_Ray_AABB(ray, (&_t181))) {
      if (distmin_Ray_AABB(ray, (&_t181)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[6u], ray, triangles, _best0);
      }
    }
    AABB _t207 = AABB{_t18[7], _t23[7]};
    if (intersects_Ray_AABB(ray, (&_t207))) {
      if (distmin_Ray_AABB(ray, (&_t207)) < cuda::std::get<0>((*_best0))) {
        _recloop_func0(_t17.aabb_children[7u], ray, triangles, _best0);
      }
    }
  } else {
    if (_t493 == 3u) {
      Obbs _t229 = (*triangles).obbs[slice<4, 63>(index)];
      cuda::std::array<float3, 8> _t230 = _t229.obb_low;
      cuda::std::array<float3, 8> _t235 = _t229.obb_high;
      cuda::std::array<float4, 3> _t240 = _t229.orientation;
      OBB _t241 = OBB{_t230[0], _t235[0], _t240};
      if (intersects_Ray_OBB(ray, (&_t241))) {
        if (distmin_Ray_OBB(ray, (&_t241)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[0u], ray, triangles, _best0);
        }
      }
      OBB _t275 = OBB{_t230[1], _t235[1], _t240};
      if (intersects_Ray_OBB(ray, (&_t275))) {
        if (distmin_Ray_OBB(ray, (&_t275)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[1u], ray, triangles, _best0);
        }
      }
      OBB _t309 = OBB{_t230[2], _t235[2], _t240};
      if (intersects_Ray_OBB(ray, (&_t309))) {
        if (distmin_Ray_OBB(ray, (&_t309)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[2u], ray, triangles, _best0);
        }
      }
      OBB _t343 = OBB{_t230[3], _t235[3], _t240};
      if (intersects_Ray_OBB(ray, (&_t343))) {
        if (distmin_Ray_OBB(ray, (&_t343)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[3u], ray, triangles, _best0);
        }
      }
      OBB _t377 = OBB{_t230[4], _t235[4], _t240};
      if (intersects_Ray_OBB(ray, (&_t377))) {
        if (distmin_Ray_OBB(ray, (&_t377)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[4u], ray, triangles, _best0);
        }
      }
      OBB _t411 = OBB{_t230[5], _t235[5], _t240};
      if (intersects_Ray_OBB(ray, (&_t411))) {
        if (distmin_Ray_OBB(ray, (&_t411)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[5u], ray, triangles, _best0);
        }
      }
      OBB _t445 = OBB{_t230[6], _t235[6], _t240};
      if (intersects_Ray_OBB(ray, (&_t445))) {
        if (distmin_Ray_OBB(ray, (&_t445)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[6u], ray, triangles, _best0);
        }
      }
      OBB _t479 = OBB{_t230[7], _t235[7], _t240};
      if (intersects_Ray_OBB(ray, (&_t479))) {
        if (distmin_Ray_OBB(ray, (&_t479)) < cuda::std::get<0>((*_best0))) {
          _recloop_func0(_t229.obb_children[7u], ray, triangles, _best0);
        }
      }
    } else {
      uint64_t _t490 = slice<4, 63>(index);
      for (uint64_t _idx0 = _t490; _idx0 < (_t490 + (uint64_t)(uint8_t)(slice<1, 8>(index) + 1u)); _idx0 += 1u) {
        Triangle _t489 = (*triangles).primitives[_idx0];
        if (intersectsp_ray_tri(ray, (&_t489)).has_value()) {
          float _t487 = distmin_Ray_Triangle(ray, (&_t489));
          if (_t487 < cuda::std::get<0>((*_best0))) {
            (*_best0) = argmin(_best0, cuda::std::tuple<float, Triangle>{_t487, _t489});
          }
        }
      }
    }
  }
  return;
}

__host__ cuda::std::optional<Triangle> _traverse_tree0(Ray* ray, Triangles* triangles) {
  cuda::std::tuple<float, Triangle> _best0 = cuda::std::tuple<float, Triangle>{INFINITY, Triangle{}};
  _recloop_func0(2, ray, triangles, (&_best0));
  return ((cuda::std::get<0>(_best0) != INFINITY) ? cuda::std::optional<Triangle>{cuda::std::get<1>(_best0)} : cuda::std::nullopt);
}

__host__ cuda::std::optional<Triangle>* _traverse_array0(int64_t n, Ray* rays, Triangles* triangles) {
  cuda::std::optional<Triangle>* _alloc0;
  (void)cudaMalloc((void**)&_alloc0, n * sizeof(cuda::std::optional<Triangle>));
  for (int64_t _i0 = 0; _i0 < n; _i0 += 1) {
    Ray _lv0 = rays[_i0];
    _alloc0[_i0] = _traverse_tree0((&_lv0), triangles);
  }
  return _alloc0;
}

__host__ uint64_t rec_build_triangles(BVH* node_, Triangles* ST, size_t* aabbs_index, size_t* obbs_index, size_t* primitives_index) {
  if (!node_) {
    return 18446744073709551615u;
  }
    if (std::holds_alternative<AABBNode>(*node_)) {
      const AABBNode& node = std::get<AABBNode>(*node_);
      size_t this_index = (*aabbs_index);
      (*aabbs_index) += 1;
      (*ST).aabbs[this_index].aabb_low = node.aabb_low;
      (*ST).aabbs[this_index].aabb_high = node.aabb_high;
      uint64_t aabb_children_index[8];
      for (int32_t __r = 0; __r < 8; __r += 1) {
        aabb_children_index[__r] = rec_build_triangles(node.aabb_children[__r], ST, aabbs_index, obbs_index, primitives_index);
        (*ST).aabbs[this_index].aabb_children[__r] = aabb_children_index[__r];
      }
      return (uint64_t)((this_index << 4u) | 2u);
    }
    else if (std::holds_alternative<OBBNode>(*node_)) {
      const OBBNode& node = std::get<OBBNode>(*node_);
      size_t this_index = (*obbs_index);
      (*obbs_index) += 1;
      (*ST).obbs[this_index].orientation = node.orientation;
      (*ST).obbs[this_index].obb_low = node.obb_low;
      (*ST).obbs[this_index].obb_high = node.obb_high;
      uint64_t obb_children_index[8];
      for (int32_t __r = 0; __r < 8; __r += 1) {
        obb_children_index[__r] = rec_build_triangles(node.obb_children[__r], ST, aabbs_index, obbs_index, primitives_index);
        (*ST).obbs[this_index].obb_children[__r] = obb_children_index[__r];
      }
      return (uint64_t)((this_index << 4u) | 3u);
    }
    else if (std::holds_alternative<Leaf>(*node_)) {
      const Leaf& node = std::get<Leaf>(*node_);
      uint64_t poffset = (*primitives_index);
      for (uint8_t __p = 0u; __p < node.nprims; __p += 1u) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return (uint64_t)((poffset << 4u) | (uint64_t)(node.nprims << 2u));
    }
}

__host__ void rec_count_triangles(BVH* node_, Triangles* ST) {
  if (!node_) {
    return;
  }
    if (std::holds_alternative<AABBNode>(*node_)) {
      const AABBNode& node = std::get<AABBNode>(*node_);
      for (int32_t __r = 0; __r < 8; __r += 1) {
        rec_count_triangles(node.aabb_children[__r], ST);
      }
      (*ST).aabb_count += 1u;
    }
    else if (std::holds_alternative<OBBNode>(*node_)) {
      const OBBNode& node = std::get<OBBNode>(*node_);
      for (int32_t __r = 0; __r < 8; __r += 1) {
        rec_count_triangles(node.obb_children[__r], ST);
      }
      (*ST).obb_count += 1u;
    }
    else if (std::holds_alternative<Leaf>(*node_)) {
      const Leaf& node = std::get<Leaf>(*node_);
      (*ST).primitive_count += node.nprims;
    }
}

__host__ Triangles build_triangles(BVH* CT) {
  Triangles ST;
  size_t primitives_index = 0;
  size_t aabbs_index = 0;
  size_t obbs_index = 0;
  ST.primitive_count = 0u;
  ST.aabb_count = 0u;
  ST.obb_count = 0u;
  rec_count_triangles(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Aabbs* aabbs = reinterpret_cast<Aabbs*>(malloc(sizeof(Aabbs) * ST.aabb_count));
  ST.aabbs = aabbs;
  Obbs* obbs = reinterpret_cast<Obbs*>(malloc(sizeof(Obbs) * ST.obb_count));
  ST.obbs = obbs;
  rec_build_triangles(CT, (&ST), (&aabbs_index), (&obbs_index), (&primitives_index));
  return ST;
}

__host__ cuda::std::optional<Triangle>* chrt(int64_t n, Ray* rays, Triangles* triangles) {
  return _traverse_array0(n, rays, triangles);
}
