#include "helpers.h"

#include <cuda/std/array>
#include <thrust/universal_vector.h>
#include <cuda/std/optional>
#include <cuda/std/tuple>

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
  float3 low;
  float3 high;
  BVH* left;
  BVH* right;
};

struct Triangle {
  float3 p0;
  float3 p1;
  float3 p2;
};

struct Leaf {
  float3 low;
  float3 high;
  uint16_t nprims;
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

struct Arm_Interior {
  uint32_t offset;
} __attribute__((packed));

struct Arm_Leaf {
  uint32_t poffset;
} __attribute__((packed));

struct Nodes {
  int32_t q_min;
  int32_t q_max;
  uint8_t nprims;
  uchar4 split0on_nprims;
} __attribute__((packed));

struct Triangles {
  float3 wlow;
  float3 whigh;
  float3 bins;
  float3 bins_inv;
  uint32_t primitive_count;
  Triangle* primitives;
  uint32_t node_count;
  Nodes* nodes;
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
  uint32_t _queue0[64];
  _queue0[0] = 0u;
  do {
    _queue_count0 -= 1;
    uint32_t index = _queue0[_queue_count0];
    Nodes _t48 = (*triangles).nodes[index];
    float3 _t50 = (*triangles).bins;
    AABB _t60 = AABB{__fadd_rd((*triangles).wlow, float3{__fmul_rd((float)(_t48.q_min >> 20) & 1023, _t50.x), __fmul_rd((float)(_t48.q_min >> 10) & 1023, _t50.y), __fmul_rd((float)(_t48.q_min >> 0) & 1023, _t50.z)}), __fsub_ru((*triangles).whigh, float3{__fmul_rd((float)(_t48.q_max >> 20) & 1023, _t50.x), __fmul_rd((float)(_t48.q_max >> 10) & 1023, _t50.y), __fmul_rd((float)(_t48.q_max >> 0) & 1023, _t50.z)})};
    if (intersects_Ray_AABB(ray, (&_t60))) {
      if (distmin_Ray_AABB(ray, (&_t60)) < cuda::std::get<0>(_best0)) {
        uint8_t _t29 = _t48.nprims;
        if (_t29 == 0u) {
          _queue0[_queue_count0] = (index + 1u);
          _queue0[(_queue_count0 + 1)] = (index + bonsai_reinterpret<Arm_Interior>(_t48.split0on_nprims).offset);
          _queue_count0 += 2;
        } else {
          uint32_t _t18 = bonsai_reinterpret<Arm_Leaf>(_t48.split0on_nprims).poffset;
          for (uint32_t _idx0 = _t18; _idx0 < (_t18 + (uint32_t)_t29); _idx0 += 1u) {
            Triangle _t13 = (*triangles).primitives[_idx0];
            if (intersectsp_ray_tri(ray, (&_t13)).has_value()) {
              float _t10 = distmin_Ray_Triangle(ray, (&_t13));
              if (_t10 < cuda::std::get<0>(_best0)) {
                _best0 = argmin(_best0, cuda::std::tuple<float, Triangle>{_t10, _t13});
              }
            }
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
  Nodes* __nodes;
  cudaMallocAndCopyToDevice((void**)&__nodes, (*triangles).nodes, (*triangles).node_count * sizeof(Nodes));
  Triangles h_triangles = *triangles;
  h_triangles.primitives = __primitives;
  h_triangles.nodes = __nodes;
  Triangles* d_triangles;
  cudaMallocAndCopyToDevice((void**)&d_triangles, &h_triangles, sizeof(Triangles));
  _ctx0 ctx = _ctx0{n, _alloc0, d_rays, d_triangles};
  _parfunc0<<<((n + 511) / 512), 512>>>(ctx);
  cudaDeviceSynchronize();
  cuda::std::optional<Triangle>* h__alloc0;
  mallocAndCopyFromDevice((void**)&h__alloc0, _alloc0, n * sizeof(cuda::std::optional<Triangle>));
  cudaFree(__primitives);
  cudaFree(__nodes);
  cudaFree(_alloc0);
  cudaFree(d_rays);
  cudaFree(d_triangles);
  _alloc0 = h__alloc0;
  return _alloc0;
}

__host__ float3 build_bins_inverse(float3 low, float3 high) {
  float3 L1 = float3{__fsub_ru(high.x, low.x), __fsub_ru(high.y, low.y), __fsub_ru(high.z, low.z)};
  float3 L2 = make_float3(((L1 <= make_float3(0)).x ? make_float3(1).x : L1.x),((L1 <= make_float3(0)).y ? make_float3(1).y : L1.y),((L1 <= make_float3(0)).z ? make_float3(1).z : L1.z));
  return float3{__fdiv_rd((float)1023, L2.x), __fdiv_rd((float)1023, L2.y), __fdiv_rd((float)1023, L2.z)};
}

__host__ float3 build_bins(float3 low, float3 high) {
  float3 bins_inverse = build_bins_inverse(low, high);
  return float3{__frcp_rd(bins_inverse.x), __frcp_rd(bins_inverse.y), __frcp_rd(bins_inverse.z)};
}

__host__ int32_t quantize(float3 current, float3 world, float3 bin_inverse) {
  uint32_t x = (uint32_t)floorf(__fmul_rd(__fsub_rd(current.x, world.x), bin_inverse.x));
  uint32_t y = (uint32_t)floorf(__fmul_rd(__fsub_rd(current.y, world.y), bin_inverse.y));
  uint32_t z = (uint32_t)floorf(__fmul_rd(__fsub_rd(current.z, world.z), bin_inverse.z));
  return (int32_t)(((x << 20u) | (y << 10u)) | z);
}

__host__ uint32_t rec_build_triangles(BVH* node_, Triangles* ST, size_t* nodes_index, size_t* primitives_index) {
    if (std::holds_alternative<Interior>(*node_)) {
      const Interior& node = std::get<Interior>(*node_);
      size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      if (this_index == 0) {
        (*ST).wlow = node.low;
        (*ST).whigh = node.high;
        (*ST).bins_inv = build_bins_inverse(node.low, node.high);
        (*ST).bins = build_bins(node.low, node.high);
      }
      (*ST).nodes[this_index].q_min = quantize(node.low, (*ST).wlow, (*ST).bins_inv);
      (*ST).nodes[this_index].q_max = quantize(node.high, (*ST).whigh, (*ST).bins_inv);
      (*ST).nodes[this_index].nprims = 0;
      uint32_t left_index = rec_build_triangles(node.left, ST, nodes_index, primitives_index);
      uint32_t right_index = rec_build_triangles(node.right, ST, nodes_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    }
    else if (std::holds_alternative<Leaf>(*node_)) {
      const Leaf& node = std::get<Leaf>(*node_);
      size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].q_min = quantize(node.low, (*ST).wlow, (*ST).bins_inv);
      (*ST).nodes[this_index].q_max = quantize(node.high, (*ST).whigh, (*ST).bins_inv);
      (*ST).nodes[this_index].nprims = node.nprims;
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0u; __p < node.nprims; __p += 1u) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
}

__host__ void rec_count_triangles(BVH* node_, Triangles* ST) {
    if (std::holds_alternative<Interior>(*node_)) {
      const Interior& node = std::get<Interior>(*node_);
      rec_count_triangles(node.left, ST);
      rec_count_triangles(node.right, ST);
      (*ST).node_count += 1u;
    }
    else if (std::holds_alternative<Leaf>(*node_)) {
      const Leaf& node = std::get<Leaf>(*node_);
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1u;
    }
}

__host__ Triangles build_triangles(BVH* CT) {
  Triangles ST;
  size_t primitives_index = 0;
  size_t nodes_index = 0;
  ST.primitive_count = 0u;
  ST.node_count = 0u;
  rec_count_triangles(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Nodes* nodes = reinterpret_cast<Nodes*>(malloc(sizeof(Nodes) * ST.node_count));
  ST.nodes = nodes;
  rec_build_triangles(CT, (&ST), (&nodes_index), (&primitives_index));
  return ST;
}

__host__ cuda::std::optional<Triangle>* chrt(int64_t n, Ray* rays, Triangles* triangles) {
  return _traverse_array0(n, rays, triangles);
}

__host__ cuda::std::optional<Triangle> trace(Ray* ray, Triangles* triangles) {
  return _traverse_tree0(ray, triangles);
}
