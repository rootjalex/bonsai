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

struct Arm_Interior {
  uint32_t offset;
} __attribute__((packed));

struct Arm_Leaf {
  uint32_t poffset;
} __attribute__((packed));

struct alignas(16) Nodes {
  float3 low;
  float3 high;
  uint16_t nprims;
  uint8_t pad0;
  uchar4 split0on_nprims;
} __attribute__((packed));

struct Triangles {
  uint32_t primitive_count;
  Triangle* primitives;
  uint32_t node_count;
  Nodes* nodes;
} __attribute__((packed));

struct _ctx0 {
  int32_t n;
  Triangle* _alloc0;
  Point* pts;
  Triangles* triangles;
};

__device__ float SqDistPointAABB(Point* pt, AABB* a) {
  float3 v = (*pt).vec;
  float3 sqLow = (((*a).low - v) * ((*a).low - v));
  float3 low = make_float3(((v < (*a).low).x ? sqLow.x : make_float3(0).x),((v < (*a).low).y ? sqLow.y : make_float3(0).y),((v < (*a).low).z ? sqLow.z : make_float3(0).z));
  float3 sqHigh = ((v - (*a).high) * (v - (*a).high));
  float3 high = make_float3((((*a).high < v).x ? sqHigh.x : make_float3(0).x),(((*a).high < v).y ? sqHigh.y : make_float3(0).y),(((*a).high < v).z ? sqHigh.z : make_float3(0).z));
  return sum((low + high));
}

__device__ cuda::std::tuple<Point, Point> closestPointonTriangle(Point* pt, Triangle* tri) {
  float3 p = (*pt).vec;
  float3 a = (*tri).p0;
  float3 b = (*tri).p1;
  float3 c = (*tri).p2;
  float3 ab = (b - a);
  float3 ac = (c - a);
  float3 ap = (p - a);
  float d1 = dot(ab, ap);
  float d2 = dot(ac, ap);
  if (d1 <= 0) {
    if (d2 <= 0) {
      return cuda::std::tuple<Point, Point>{Point{a}, Point{float3{1, 0, 0}}};
    }
  }
  float3 bp = (p - b);
  float d3 = dot(ab, bp);
  float d4 = dot(ac, bp);
  if (0 <= d3) {
    if (d4 <= d3) {
      return cuda::std::tuple<Point, Point>{Point{b}, Point{float3{0, 1, 0}}};
    }
  }
  float _t165 = ((d1 * d4) - (d3 * d2));
  if (_t165 <= 0) {
    if (0 <= d1) {
      if (d3 <= 0) {
        float _t167 = (d1 / (d1 - d3));
        return cuda::std::tuple<Point, Point>{Point{a + (make_float3(_t167) * ab)}, Point{float3{1 - _t167, _t167, 0}}};
      }
    }
  }
  float3 cp = (p - c);
  float d5 = dot(ab, cp);
  float d6 = dot(ac, cp);
  if (0 <= d6) {
    if (d5 <= d6) {
      return cuda::std::tuple<Point, Point>{Point{c}, Point{float3{0, 0, 1}}};
    }
  }
  float _t170 = ((d5 * d2) - (d1 * d6));
  if (_t170 <= 0) {
    if (0 <= d2) {
      if (d6 <= 0) {
        float _t172 = (d2 / (d2 - d6));
        return cuda::std::tuple<Point, Point>{Point{a + (make_float3(_t172) * ac)}, Point{float3{1 - _t172, 0, _t172}}};
      }
    }
  }
  float _t175 = ((d3 * d6) - (d5 * d4));
  if (_t175 <= 0) {
    float _t64 = (d4 - d3);
    if (0 <= _t64) {
      float _t63 = (d5 - d6);
      if (0 <= _t63) {
        float _t177 = (_t64 / (_t64 + _t63));
        return cuda::std::tuple<Point, Point>{Point{b + (make_float3(_t177) * (c - b))}, Point{float3{0, 1 - _t177, _t177}}};
      }
    }
  }
  float _t180 = (1 / ((_t175 + _t170) + _t165));
  float v = (_t170 * _t180);
  float w = (_t165 * _t180);
  float u = (_t175 * _t180);
  return cuda::std::tuple<Point, Point>{Point{(a + (ab * make_float3(v))) + (ac * make_float3(w))}, Point{float3{u, v, w}}};
}

__device__ float distmax_Point_AABB(Point* pt, AABB* a) {
  float3 _t69 = (*pt).vec;
  float3 _t185 = min((*a).low - _t69, _t69 - (*a).high);
  return dot(_t185, _t185);
}

__device__ Triangle _traverse_tree0(Point* p, Triangles* triangles) {
  cuda::std::tuple<float, Triangle> _best0 = cuda::std::tuple<float, Triangle>{INFINITY, Triangle{}};
  int32_t _queue_count0 = 1;
  uint32_t _queue0[64];
  _queue0[0] = 0u;
  do {
    _queue_count0 -= 1;
    uint32_t index = _queue0[_queue_count0];
    if (index == 4294967295u) {
      if (_queue_count0 <= 0) {
        break;
      } else {
        continue;
      }
    }
    Nodes _t151 = (*triangles).nodes[index];
    AABB _t154 = AABB{_t151.low, _t151.high};
    if (SqDistPointAABB(p, (&_t154)) < cuda::std::get<0>(_best0)) {
      uint16_t _t37 = _t151.nprims;
      if (_t37 == 0u) {
        _best0 = argmin(_best0, cuda::std::tuple<float, Triangle>{distmax_Point_AABB(p, (&_t154)), Triangle{}});
        _queue0[_queue_count0] = (index + 1u);
        _queue0[(_queue_count0 + 1)] = (index + bonsai_reinterpret<Arm_Interior>(_t151.split0on_nprims).offset);
        _queue_count0 += 2;
      } else {
        uint32_t _t26 = bonsai_reinterpret<Arm_Leaf>(_t151.split0on_nprims).poffset;
        for (uint32_t _idx0 = _t26; _idx0 < (_t26 + (uint32_t)_t37); _idx0 += 1u) {
          Triangle _t158 = (*triangles).primitives[_idx0];
          float _t16 = 0;
          cuda::std::tuple<Point, Point> __inline1 = closestPointonTriangle(p, (&_t158));
          float3 __inline0 = ((*p).vec - cuda::std::get<0>(__inline1).vec);
          _t16 = dot(__inline0, __inline0);
          bool _t2 = (_t16 < cuda::std::get<0>(_best0));
          if (_t2) {
            float _t3 = _t16;
            _best0 = cuda::std::tuple<float, Triangle>{_t3, _t158};
          }
        }
      }
    }
  } while (_queue_count0 != 0);
  return cuda::std::get<1>(_best0);
}

__global__ void _parfunc0(_ctx0 ctx0) {
  int32_t tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int32_t _i0 = tid;
  if (ctx0.n <= tid) {
    return;
  }
  Point _lv0 = ctx0.pts[_i0];
  ctx0._alloc0[_i0] = _traverse_tree0((&_lv0), ctx0.triangles);
  return;
}

__host__ Triangle* _traverse_array0(int32_t n, Point* pts, Triangles* triangles) {
  Triangle* _alloc0;
  (void)cudaMalloc((void**)&_alloc0, n * sizeof(Triangle));
  Point* d_pts;
  cudaMallocAndCopyToDevice((void**)&d_pts, pts, n * sizeof(Point));
  Triangle* __primitives;
  cudaMallocAndCopyToDevice((void**)&__primitives, (*triangles).primitives, (*triangles).primitive_count * sizeof(Triangle));
  Nodes* __nodes;
  cudaMallocAndCopyToDevice((void**)&__nodes, (*triangles).nodes, (*triangles).node_count * sizeof(Nodes));
  Triangles h_triangles = *triangles;
  h_triangles.primitives = __primitives;
  h_triangles.nodes = __nodes;
  Triangles* d_triangles;
  cudaMallocAndCopyToDevice((void**)&d_triangles, &h_triangles, sizeof(Triangles));
  _ctx0 ctx = _ctx0{n, _alloc0, d_pts, d_triangles};
  _parfunc0<<<((n + 255) / 256), 256>>>(ctx);
  cudaDeviceSynchronize();
  Triangle* h__alloc0;
  mallocAndCopyFromDevice((void**)&h__alloc0, _alloc0, n * sizeof(Triangle));
  cudaFree(__primitives);
  cudaFree(__nodes);
  cudaFree(_alloc0);
  cudaFree(d_pts);
  cudaFree(d_triangles);
  _alloc0 = h__alloc0;
  return _alloc0;
}

__host__ uint32_t rec_build_triangles(BVH* node_, Triangles* ST, size_t* nodes_index, size_t* primitives_index) {
  if (!node_) {
    return 4294967295u;
  }
    if (std::holds_alternative<Interior>(*node_)) {
      const Interior& node = std::get<Interior>(*node_);
      size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].low = node.low;
      (*ST).nodes[this_index].high = node.high;
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
      (*ST).nodes[this_index].low = node.low;
      (*ST).nodes[this_index].high = node.high;
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
  if (!node_) {
    return;
  }
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

__host__ Triangle* closest_points(int32_t n, Point* pts, Triangles* triangles) {
  return _traverse_array0(n, pts, triangles);
}
