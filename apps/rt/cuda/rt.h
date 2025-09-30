#include "helpers.h"

#include <cuda/std/array>
#include <thrust/universal_vector.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>

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
  cuda::std::array<BVH*, 8> children;
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
  Triangle* data;
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

struct Qbox3 {
  uchar3 lo;
  uchar3 hi;
};

struct alignas(32) Interiors {
  float3 mlo;
  float3 mex;
  cuda::std::array<Qbox3, 8> child_bounds;
  cuda::std::array<uint64_t, 8> children;
} __attribute__((packed));

struct Triangles {
  uint64_t primitive_count;
  Triangle* primitives;
  uint64_t interior_count;
  Interiors* interiors;
} __attribute__((packed));

struct _ctx0 {
  int64_t n;
  cuda::std::optional<Triangle>* _alloc0;
  Ray* rays;
  Triangles* triangles;
};

__host__ float __prod_diff_f32(float a, float b, float c, float d) {
  float cd = (c * d);
  float diff = fmaf(a, b, -cd);
  float err = fmaf(-c, d, cd);
  return (diff + err);
}

__host__ cuda::std::array<float3, 8> dequantize_bounds_hi(float3 mlo, float3 mex, cuda::std::array<Qbox3, 8> bound) {
  float rcp = (1 / (float)255);
  return cuda::std::array<float3, 8>{mlo + ((make_float3(bound[0].hi.x, bound[0].hi.y, bound[0].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[1].hi.x, bound[1].hi.y, bound[1].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[2].hi.x, bound[2].hi.y, bound[2].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[3].hi.x, bound[3].hi.y, bound[3].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[4].hi.x, bound[4].hi.y, bound[4].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[5].hi.x, bound[5].hi.y, bound[5].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[6].hi.x, bound[6].hi.y, bound[6].hi.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[7].hi.x, bound[7].hi.y, bound[7].hi.z) * make_float3(rcp)) * mex)};
}

__host__ cuda::std::array<float3, 8> dequantize_bounds_lo(float3 mlo, float3 mex, cuda::std::array<Qbox3, 8> bound) {
  float rcp = (1 / (float)255);
  return cuda::std::array<float3, 8>{mlo + ((make_float3(bound[0].lo.x, bound[0].lo.y, bound[0].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[1].lo.x, bound[1].lo.y, bound[1].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[2].lo.x, bound[2].lo.y, bound[2].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[3].lo.x, bound[3].lo.y, bound[3].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[4].lo.x, bound[4].lo.y, bound[4].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[5].lo.x, bound[5].lo.y, bound[5].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[6].lo.x, bound[6].lo.y, bound[6].lo.z) * make_float3(rcp)) * mex), mlo + ((make_float3(bound[7].lo.x, bound[7].lo.y, bound[7].lo.z) * make_float3(rcp)) * mex)};
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

__device__ __host__ cuda::std::optional<Triangle> _traverse_tree0(Ray* ray, Triangles* triangles) {
  cuda::std::tuple<float, Triangle> _best0 = cuda::std::tuple<float, Triangle>{INFINITY, Triangle{}};
  int32_t _queue_count0 = 1;
  uint64_t _queue0[64];
  _queue0[0] = 0u;
  do {
    _queue_count0 -= 1;
    uint64_t I = _queue0[_queue_count0];
    if (I == 18446744073709551615u) {
      continue;
    }
    if (slice<0, 2>(I) == 1u) {
      Interiors _t36 = (*triangles).interiors[slice<7, 63>(I)];
      float3 _t37 = _t36.mlo;
      float3 _t41 = _t36.mex;
      cuda::std::array<Qbox3, 8> _t45 = _t36.child_bounds;
      cuda::std::array<float3, 8> _t46 = dequantize_bounds_lo(_t37, _t41, _t45);
      cuda::std::array<float3, 8> _t60 = dequantize_bounds_hi(_t37, _t41, _t45);
      AABB _t62 = AABB{_t46[0], _t60[0]};
      if (intersects_Ray_AABB(ray, (&_t62))) {
        if (distmin_Ray_AABB(ray, (&_t62)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[0u];
          _queue_count0 += 1;
        }
      }
      AABB _t125 = AABB{_t46[1], _t60[1]};
      if (intersects_Ray_AABB(ray, (&_t125))) {
        if (distmin_Ray_AABB(ray, (&_t125)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[1u];
          _queue_count0 += 1;
        }
      }
      AABB _t188 = AABB{_t46[2], _t60[2]};
      if (intersects_Ray_AABB(ray, (&_t188))) {
        if (distmin_Ray_AABB(ray, (&_t188)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[2u];
          _queue_count0 += 1;
        }
      }
      AABB _t251 = AABB{_t46[3], _t60[3]};
      if (intersects_Ray_AABB(ray, (&_t251))) {
        if (distmin_Ray_AABB(ray, (&_t251)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[3u];
          _queue_count0 += 1;
        }
      }
      AABB _t314 = AABB{_t46[4], _t60[4]};
      if (intersects_Ray_AABB(ray, (&_t314))) {
        if (distmin_Ray_AABB(ray, (&_t314)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[4u];
          _queue_count0 += 1;
        }
      }
      AABB _t377 = AABB{_t46[5], _t60[5]};
      if (intersects_Ray_AABB(ray, (&_t377))) {
        if (distmin_Ray_AABB(ray, (&_t377)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[5u];
          _queue_count0 += 1;
        }
      }
      AABB _t440 = AABB{_t46[6], _t60[6]};
      if (intersects_Ray_AABB(ray, (&_t440))) {
        if (distmin_Ray_AABB(ray, (&_t440)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[6u];
          _queue_count0 += 1;
        }
      }
      AABB _t503 = AABB{_t46[7], _t60[7]};
      if (intersects_Ray_AABB(ray, (&_t503))) {
        if (distmin_Ray_AABB(ray, (&_t503)) < cuda::std::get<0>(_best0)) {
          _queue0[_queue_count0] = _t36.children[7u];
          _queue_count0 += 1;
        }
      }
    } else {
      uint64_t _t515 = slice<7, 63>(I);
      for (uint64_t _idx0 = _t515; _idx0 < (_t515 + (uint64_t)(uint8_t)(slice<3, 6>(I) + 1u)); _idx0 += 1u) {
        Triangle _t514 = (*triangles).primitives[_idx0];
        if (intersectsp_ray_tri(ray, (&_t514)).has_value()) {
          float _t511 = distmin_Ray_Triangle(ray, (&_t514));
          if (_t511 < cuda::std::get<0>(_best0)) {
            _best0 = argmin(_best0, cuda::std::tuple<float, Triangle>{_t511, _t514});
          }
        }
      }
    }
  } while (_queue_count0 != 0);
  return ((cuda::std::get<0>(_best0) != INFINITY) ? cuda::std::optional<Triangle>{cuda::std::get<1>(_best0)} : cuda::std::nullopt);
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

__host__ cuda::std::optional<Triangle>* _traverse_array0(int64_t n, Ray* rays, Triangles* triangles) {
  cuda::std::optional<Triangle>* _alloc0;
  (void)cudaMalloc((void**)&_alloc0, n * sizeof(cuda::std::optional<Triangle>));
  Ray* d_rays;
  cudaMallocAndCopyToDevice((void**)&d_rays, rays, n * sizeof(Ray));
  Triangle* __primitives;
  cudaMallocAndCopyToDevice((void**)&__primitives, (*triangles).primitives, (*triangles).primitive_count * sizeof(Triangle));
  Interiors* __interiors;
  cudaMallocAndCopyToDevice((void**)&__interiors, (*triangles).interiors, (*triangles).interior_count * sizeof(Interiors));
  Triangles h_triangles = *triangles;
  h_triangles.primitives = __primitives;
  h_triangles.interiors = __interiors;
  Triangles* d_triangles;
  cudaMallocAndCopyToDevice((void**)&d_triangles, &h_triangles, sizeof(Triangles));
  _ctx0 ctx = _ctx0{n, _alloc0, d_rays, d_triangles};
  _parfunc0<<<((n + 511) / 512), 512>>>(ctx);
  cudaDeviceSynchronize();
  cuda::std::optional<Triangle>* h__alloc0;
  mallocAndCopyFromDevice((void**)&h__alloc0, _alloc0, n * sizeof(cuda::std::optional<Triangle>));
  cudaFree(__primitives);
  cudaFree(__interiors);
  cudaFree(_alloc0);
  cudaFree(d_rays);
  cudaFree(d_triangles);
  _alloc0 = h__alloc0;
  return _alloc0;
}

__host__ float3 compute_merged_extent(cuda::std::array<float3, 8> lo, cuda::std::array<float3, 8> hi) {
  float3 mlo = min(lo[0], min(lo[1], min(lo[2], min(lo[3], min(lo[4], min(lo[5], min(lo[6], lo[7])))))));
  float3 mhi = max(hi[0], max(hi[1], max(hi[2], max(hi[3], max(hi[4], max(hi[5], max(hi[6], hi[7])))))));
  return (mhi - mlo);
}

__host__ uchar3 to_u8_ceil(float3 f) {
  float3 f1 = ceilf(f);
  float3 f2 = max(make_float3(0), min(f1, make_float3((float)255)));
  return make_uchar3(f2.x, f2.y, f2.z);
}

__host__ uchar3 to_u8_floor(float3 f) {
  float3 f1 = floorf(f);
  float3 f2 = max(make_float3(0), min(f1, make_float3((float)255)));
  return make_uchar3(f2.x, f2.y, f2.z);
}

__host__ cuda::std::array<Qbox3, 8> quantize_bounds(cuda::std::array<float3, 8> low, cuda::std::array<float3, 8> high) {
  float3 mlo = min(low[0], min(low[1], min(low[2], min(low[3], min(low[4], min(low[5], min(low[6], low[7])))))));
  float3 mex = compute_merged_extent(low, high);
  float3 _t1 = ((make_float3(1) / mex) * make_float3((float)255));
  return cuda::std::array<Qbox3, 8>{Qbox3{to_u8_floor((low[0] - mlo) * _t1), to_u8_ceil((high[0] - mlo) * _t1)}, Qbox3{to_u8_floor((low[1] - mlo) * _t1), to_u8_ceil((high[1] - mlo) * _t1)}, Qbox3{to_u8_floor((low[2] - mlo) * _t1), to_u8_ceil((high[2] - mlo) * _t1)}, Qbox3{to_u8_floor((low[3] - mlo) * _t1), to_u8_ceil((high[3] - mlo) * _t1)}, Qbox3{to_u8_floor((low[4] - mlo) * _t1), to_u8_ceil((high[4] - mlo) * _t1)}, Qbox3{to_u8_floor((low[5] - mlo) * _t1), to_u8_ceil((high[5] - mlo) * _t1)}, Qbox3{to_u8_floor((low[6] - mlo) * _t1), to_u8_ceil((high[6] - mlo) * _t1)}, Qbox3{to_u8_floor((low[7] - mlo) * _t1), to_u8_ceil((high[7] - mlo) * _t1)}};
}

__host__ uint64_t rec_build_triangles(BVH* node_, Triangles* ST, size_t* interiors_index, size_t* primitives_index) {
  if (!node_) {
    return 18446744073709551615u;
  }
    if (std::holds_alternative<Interior>(*node_)) {
      const Interior& node = std::get<Interior>(*node_);
      size_t this_index = (*interiors_index);
      (*interiors_index) += 1;
      (*ST).interiors[this_index].mlo = min(node.lo[0], min(node.lo[1], min(node.lo[2], min(node.lo[3], min(node.lo[4], min(node.lo[5], min(node.lo[6], node.lo[7])))))));
      (*ST).interiors[this_index].mex = compute_merged_extent(node.lo, node.hi);
      (*ST).interiors[this_index].child_bounds = quantize_bounds(node.lo, node.hi);
      uint64_t children_index[8];
      for (int32_t __r = 0; __r < 8; __r += 1) {
        children_index[__r] = rec_build_triangles(node.children[__r], ST, interiors_index, primitives_index);
        (*ST).interiors[this_index].children[__r] = children_index[__r];
      }
      return ((this_index << (uint64_t)7) | (uint64_t)1);
    }
    else if (std::holds_alternative<Leaf>(*node_)) {
      const Leaf& node = std::get<Leaf>(*node_);
      uint64_t poffset = (*primitives_index);
      for (uint8_t __p = 0u; __p < node.nprims; __p += 1u) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return ((poffset << (uint64_t)7) | (((uint64_t)node.nprims - (uint64_t)1) << (uint64_t)3));
    }
}

__host__ void rec_count_triangles(BVH* node_, Triangles* ST) {
  if (!node_) {
    return;
  }
    if (std::holds_alternative<Interior>(*node_)) {
      const Interior& node = std::get<Interior>(*node_);
      for (int32_t __r = 0; __r < 8; __r += 1) {
        rec_count_triangles(node.children[__r], ST);
      }
      (*ST).interior_count += 1u;
    }
    else if (std::holds_alternative<Leaf>(*node_)) {
      const Leaf& node = std::get<Leaf>(*node_);
      (*ST).primitive_count += node.nprims;
    }
}

__host__ Triangles build_triangles(BVH* CT) {
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

__host__ cuda::std::optional<Triangle>* chrt(int64_t n, Ray* rays, Triangles* triangles) {
  return _traverse_array0(n, rays, triangles);
}

__host__ cuda::std::optional<Triangle> trace(Ray* ray, Triangles* triangles) {
  return _traverse_tree0(ray, triangles);
}
