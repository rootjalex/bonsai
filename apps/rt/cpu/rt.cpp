#include "apps/rt/cpu/rt.h"
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
float __sqlen_f32(const float3 v) {
  return reduce_add((v * v));
}
float gamma(const int32_t n) {
  const float _t1 = ((float)(n) * 0.00000006f);
  return (_t1 / (1.0f - _t1));
}
std::optional<FInterval> intersectsp_ray_aabb(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const float3 _t1 = (float3{1.0f} / (*r).d);
  const bool3 dirIsNeg = (_t1 < float3{0.0f});
  const float3 _t2 = (*b).high;
  const float3 _t3 = (*b).low;
  const float3 low_parts = select(dirIsNeg, _t2, _t3);
  const float3 high_parts = select(dirIsNeg, _t3, _t2);
  const float3 _t6 = (*r).o;
  float3 tMin = ((low_parts - _t6) * _t1);
  float3 tMax = ((high_parts - _t6) * _t1);
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
  return (std::optional<FInterval>)(FInterval{.low=tmin, .high=tmax});
}
float distmin_Ray_AABB(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval& extract = *interval;
    return extract.low;
  }
  return (-std::numeric_limits<float>::infinity());
}
float3 cross_(const float3 v0, const float3 v1) {
  const float _t0 = v0[1u];
  const float _t1 = v1[2u];
  const float _t2 = v0[2u];
  const float _t3 = v1[1u];
  const float _t5 = v1[0u];
  const float _t6 = v0[0u];
  return float3{__prod_diff_f32(_t0, _t1, _t2, _t3), __prod_diff_f32(_t2, _t5, _t6, _t1), __prod_diff_f32(_t6, _t3, _t0, _t5)};
}
std::optional<TriangleIntersection> intersectsp_ray_tri(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  const float3 _t0 = (*tri).p2;
  const float3 _t1 = (*tri).p0;
  const float3 _t2 = (*tri).p1;
  if (reduce_add((cross_((_t0 - _t1), (_t2 - _t1)) * cross_((_t0 - _t1), (_t2 - _t1)))) == 0.0f) {
    return std::nullopt;
  }
  const float3 _t5 = (*ray).o;
  float3 p0t = (_t1 - _t5);
  float3 p1t = (_t2 - _t5);
  float3 p2t = (_t0 - _t5);
  const float3 _t10 = (*ray).d;
  const uint32_t kz = reduce_idxmax(abs(_t10));
  const uint32_t kx = ((kz + 1u) % 3u);
  const uint32_t ky = ((kx + 1u) % 3u);
  const float3 d = shuffle(_t10, {kx, ky, kz});
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
  const float tScaled = (((_t20 * p0t[2u]) + (_t21 * p1t[2u])) + (_t22 * p2t[2u]));
  if (((_t24 < 0.0f) && ((0.0f <= tScaled) || (tScaled < ((*ray).tmax * _t24))))) {
    return std::nullopt;
  } else if (((0.0f < _t24) && ((tScaled <= 0.0f) || (((*ray).tmax * _t24) < tScaled)))) {
    return std::nullopt;
  }
  const float invDet = (1.0f / _t24);
  const float b0 = (_t20 * invDet);
  const float b1 = (_t21 * invDet);
  const float b2 = (_t22 * invDet);
  const float t = (tScaled * invDet);
  const float maxZt = reduce_max(abs(float3{p0t[2u], p1t[2u], p2t[2u]}));
  const float _t25 = gamma(3);
  const float deltaZ = (_t25 * maxZt);
  const float maxXt = reduce_max(abs(float3{p0t[0u], p1t[0u], p2t[0u]}));
  const float maxYt = reduce_max(abs(float3{p0t[1u], p1t[1u], p2t[1u]}));
  const float _t26 = gamma(5);
  const float deltaX = (_t26 * (maxXt + maxZt));
  const float deltaY = (_t26 * (maxYt + maxZt));
  const float deltaE = (2.0f * ((((gamma(2) * maxXt) * maxYt) + (deltaY * maxXt)) + (deltaX * maxYt)));
  const float maxE = reduce_max(abs(float3{_t20, _t21, _t22}));
  const float deltaT = ((3.0f * ((((_t25 * maxE) * maxZt) + (deltaE * maxZt)) + (deltaZ * maxE))) * abs(invDet));
  if ((t <= deltaT)) {
    return std::nullopt;
  }
  return (std::optional<TriangleIntersection>)(TriangleIntersection{.b0=b0, .b1=b1, .b2=b2, .t=t});
}
float distmin_Ray_Triangle(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  const std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    const TriangleIntersection& isect_ = *isect;
    return isect_.t;
  } else {
    return std::numeric_limits<float>::infinity();
  }
}
bool intersects_Ray_AABB(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval& extract = *interval;
    return ((extract.low < (*r).tmax) & (0.0f < extract.high));
  }
  return false;
}
std::optional<Triangle> _traverse_tree0(const Ray* __restrict__ ray, const Triangles* __restrict__ triangles) {
  std::tuple<float, Triangle> _best0 = std::tuple<float, Triangle>{std::numeric_limits<float>::infinity(), Triangle{}};
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
    const Aabbs& _t39 = (*triangles).aabbs[index];
    const AABB _t44 = AABB{.low=_t39.low, .high=_t39.high};
    if (intersects_Ray_AABB(ray, (&_t44))) {
      if ((distmin_Ray_AABB(ray, (&_t44)) < std::get<0>(_best0))) {
        const Nodes& _t28 = (*triangles).nodes[index];
        const uint16_t _t29 = _t28.nprims;
        if (_t29 == 0u) {
          _queue0[_queue_count0] = (index + 1u);
          _queue0[(_queue_count0 + 1)] = (index + reinterpret<Arm_Interior>(_t28.split0on_nprims).offset);
          _queue_count0 += 2;
        } else {
          const uint32_t _t18 = reinterpret<Arm_Leaf>(_t28.split0on_nprims).poffset;
          for (uint32_t _idx0 = _t18; _idx0 < (_t18 + (uint32_t)(_t29)); ++_idx0) {
            const Triangle& _t13 = (*triangles).primitives[_idx0];
            if (intersectsp_ray_tri(ray, (&_t13)).has_value()) {
              const float _t10 = distmin_Ray_Triangle(ray, (&_t13));
              if ((_t10 < std::get<0>(_best0))) {
                _best0 = argmin<float, Triangle>(_best0, std::tuple<float, Triangle>{_t10, _t13});
              }
            }
          }
        }
      }
    }
} while ((_queue_count0 != 0));
  return ((std::get<0>(_best0) != std::numeric_limits<float>::infinity()) ? std::optional<Triangle>{std::get<1>(_best0)} : std::nullopt);
}
bool axis(const float3 A, const float3 extents, const float3 v0, const float3 v1, const float3 v2) {
  const float R = dot(extents, abs(A));
  const float3 _t3 = float3{dot(v0, A), dot(v1, A), dot(v2, A)};
  return reduce_and(((_t3 <= float3{R}) & (float3{(-R)} <= _t3)));
}
uint32_t rec_build_triangles(const BVH* __restrict__ node, Triangles* __restrict__ ST, size_t* __restrict__ nodes_index, size_t* __restrict__ primitives_index) {
  if ((!node)) {
    return 4294967295u;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1u;
      (*ST).aabbs[this_index].low = node.low;
      (*ST).aabbs[this_index].high = node.high;
      (*ST).nodes[this_index].nprims = 0;
      (*ST).nodes[this_index].axis = argmax((node.high - node.low));
      const uint32_t left_index = rec_build_triangles(node.left, ST, nodes_index, primitives_index);
      const uint32_t right_index = rec_build_triangles(node.right, ST, nodes_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1u;
      (*ST).aabbs[this_index].low = node.low;
      (*ST).aabbs[this_index].high = node.high;
      (*ST).nodes[this_index].nprims = node.nprims;
      (*ST).nodes[this_index].axis = argmax((node.high - node.low));
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0u; __p < node.nprims; ++__p) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_triangles(const BVH* __restrict__ node, Triangles* __restrict__ ST) {
  if ((!node)) {
    return;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_triangles(node.left, ST);
      rec_count_triangles(node.right, ST);
      (*ST).node_count += 1u;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1u;
    }
  }, *node);
}
Triangles build_triangles(const BVH* __restrict__ CT) {
  Triangles ST;
  size_t primitives_index = 0;
  size_t nodes_index = 0;
  ST.primitive_count = 0u;
  ST.node_count = 0u;
  ST.node_count = 0u;
  rec_count_triangles(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  std::cout << ";; primitives: " << sizeof(Triangle) <<  "," << ST.primitive_count<< "\n";
  ST.primitives = primitives;
  Aabbs* aabbs = reinterpret_cast<Aabbs*>(std::aligned_alloc(16, (((sizeof(Aabbs) * ST.node_count) + 15) / 16) * 16));
  std::cout << ";; aabbs: " << sizeof(Aabbs) <<  "," << ST.node_count<< "\n";
  ST.aabbs = aabbs;
  Nodes* nodes = reinterpret_cast<Nodes*>(std::aligned_alloc(16, (((sizeof(Nodes) * ST.node_count) + 15) / 16) * 16));
  std::cout << ";; nodes: " << sizeof(Nodes) <<  "," << ST.node_count<< "\n";
  ST.nodes = nodes;
  rec_build_triangles(CT, (&ST), (&nodes_index), (&primitives_index));
  return ST;
}
float3 clamp(const float3 x, const float low, const float high) {
  return min(max(x, float3{low}), float3{high});
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
  const float _t2 = ((d1 * d4) - (d3 * d2));
  if ((_t2 <= 0.0f)) {
    if ((0.0f <= d1)) {
      if ((d3 <= 0.0f)) {
        const float _t4 = (d1 / (d1 - d3));
        return std::tuple<Point, Point>{Point{.vec=(a + (float3{_t4} * ab))}, Point{.vec=float3{(1.0f - _t4), _t4, 0.0f}}};
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
  const float _t7 = ((d5 * d2) - (d1 * d6));
  if ((_t7 <= 0.0f)) {
    if ((0.0f <= d2)) {
      if ((d6 <= 0.0f)) {
        const float _t9 = (d2 / (d2 - d6));
        return std::tuple<Point, Point>{Point{.vec=(a + (float3{_t9} * ac))}, Point{.vec=float3{(1.0f - _t9), 0.0f, _t9}}};
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
        return std::tuple<Point, Point>{Point{.vec=(b + (float3{_t17} * (c - b)))}, Point{.vec=float3{0.0f, (1.0f - _t17), _t17}}};
      }
    }
  }
  const float _t22 = (1.0f / ((_t12 + _t7) + _t2));
  const float v = (_t7 * _t22);
  const float w = (_t2 * _t22);
  const float u = (_t12 * _t22);
  return std::tuple<Point, Point>{Point{.vec=((a + (ab * float3{v})) + (ac * float3{w}))}, Point{.vec=float3{u, v, w}}};
}
float degrees_to_radians(const float degrees) {
  return ((degrees * 3.14159274f) / 180.0f);
}
float distmax_Ray_AABB(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval& extract = *interval;
    return extract.high;
  }
  return std::numeric_limits<float>::infinity();
}
std::optional<FInterval> intersectsp_ray_sphere(const Ray* __restrict__ r, const Sphere* __restrict__ s) {
  const float3 _t2 = ((*s).center - (*r).o);
  const float3 _t3 = (*r).d;
  const float a = reduce_add((_t3 * _t3));
  const float _t7 = dot(_t3, _t2);
  const float _t9 = (*s).radius;
  const float _t15 = ((_t7 * _t7) - (a * (reduce_add((_t2 * _t2)) - (_t9 * _t9))));
  if ((_t15 < 0.0f)) {
    return std::nullopt;
  }
  const float sqrtd = sqrtf(_t15);
  const float _t17 = ((_t7 - sqrtd) / a);
  const float _t19 = ((_t7 + sqrtd) / a);
  const FInterval interval = FInterval{.low=min(_t17, _t19), .high=max(_t17, _t19)};
  return (std::optional<FInterval>)(interval);
}
float distmax_Ray_Sphere(const Ray* __restrict__ r, const Sphere* __restrict__ s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
  if (interval.has_value()) {
    const FInterval& extract = *interval;
    return extract.high;
  }
  return std::numeric_limits<float>::infinity();
}
float distmax_Ray_Triangle(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  const std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    const TriangleIntersection& isect_ = *isect;
    return isect_.t;
  } else {
    return (-std::numeric_limits<float>::infinity());
  }
}
float distmin_Point_AABB(const Point* __restrict__ pt, const AABB* __restrict__ a) {
  return sqrtf(SqDistPointAABB(pt, a));
}
float distmin_Point_Triangle(const Point* __restrict__ p, const Triangle* __restrict__ tri) {
  const std::tuple<Point, Point> pts = closestPointonTriangle(p, tri);
  return norm(((*p).vec - std::get<0>(pts).vec));
}
float distmin_Ray_Sphere(const Ray* __restrict__ r, const Sphere* __restrict__ s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
  if (interval.has_value()) {
    const FInterval& extract = *interval;
    return extract.low;
  }
  return (-std::numeric_limits<float>::infinity());
}
bool intersects_AABB_AABB(const AABB* __restrict__ a, const AABB* __restrict__ b) {
  const float3 low = max((*a).low, (*b).low);
  const float3 high = min((*a).high, (*b).high);
  return reduce_and((low <= high));
}
bool intersects_AABB_Sphere(const AABB* __restrict__ a, const Sphere* __restrict__ s) {
  const float3 _t1 = (*s).center;
  const float3 _t6 = (_t1 - max((*a).low, min(_t1, (*a).high)));
  const float _t7 = (*s).radius;
  return (dot(_t6, _t6) <= (_t7 * _t7));
}
bool intersects_AABB_Triangle(const AABB* __restrict__ a, const Triangle* __restrict__ b) {
  const float3 _t0 = (*a).low;
  const float3 _t1 = (*a).high;
  const float3 _t3 = ((_t0 + _t1) * float3{0.50000000f});
  const float3 _t7 = ((_t1 - _t0) * float3{0.50000000f});
  const float3 _t9 = ((*b).p0 - _t3);
  const float3 _t11 = ((*b).p1 - _t3);
  const float3 _t13 = ((*b).p2 - _t3);
  const float3 f0 = (_t11 - _t9);
  const float3 f1 = (_t13 - _t11);
  const float3 f2 = (_t9 - _t13);
  const float3 tri_min = min(min(_t9, _t11), _t13);
  const float3 tri_max = max(max(_t9, _t11), _t13);
  if ((!reduce_and(((tri_min <= _t7) & ((-_t7) <= tri_max))))) {
    return false;
  }
  const float3 A0 = cross(f0, float3{1.0f, 0.0f, 0.0f});
  const float3 A1 = cross(f0, float3{0.0f, 1.0f, 0.0f});
  const float3 A2 = cross(f0, float3{0.0f, 0.0f, 1.0f});
  const float3 A3 = cross(f1, float3{1.0f, 0.0f, 0.0f});
  const float3 A4 = cross(f1, float3{0.0f, 1.0f, 0.0f});
  const float3 A5 = cross(f1, float3{0.0f, 0.0f, 1.0f});
  const float3 A6 = cross(f2, float3{1.0f, 0.0f, 0.0f});
  const float3 A7 = cross(f2, float3{0.0f, 1.0f, 0.0f});
  const float3 A8 = cross(f2, float3{0.0f, 0.0f, 1.0f});
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
  const float3 N = cross(f0, f1);
  const float Rn = dot(_t7, abs(N));
  const float dist = dot(_t9, N);
  if (((Rn < dist) || (dist < (-Rn)))) {
    return false;
  }
  return true;
}
bool intersects_Ray_Sphere(const Ray* __restrict__ ray, const Sphere* __restrict__ s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(ray, s);
  if (interval.has_value()) {
    const FInterval& extract = *interval;
    return ((extract.low < (*ray).tmax) & (0.0f < extract.high));
  }
  return false;
}
bool intersects_Ray_Triangle(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  return intersectsp_ray_tri(ray, tri).has_value();
}
bool tri_tri_axis(const float3 A, const float3 a0, const float3 a1, const float3 a2, const float3 b0, const float3 b1, const float3 b2) {
  if (reduce_and(A == float3{0.0f})) {
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
bool intersects_Triangle_Triangle(const Triangle* __restrict__ a, const Triangle* __restrict__ b) {
  const float3 _t0 = (*a).p1;
  const float3 _t1 = (*a).p0;
  const float3 _t2 = (_t0 - _t1);
  const float3 _t3 = (*a).p2;
  const float3 _t5 = (_t3 - _t0);
  const float3 _t8 = (_t1 - _t3);
  const float3 _t9 = (*b).p1;
  const float3 _t10 = (*b).p0;
  const float3 _t11 = (_t9 - _t10);
  const float3 _t12 = (*b).p2;
  const float3 _t14 = (_t12 - _t9);
  const float3 _t17 = (_t10 - _t12);
  const float3 N0 = cross(_t2, _t5);
  const float3 N1 = cross(_t11, _t14);
  if ((!tri_tri_axis(N0, _t1, _t0, _t3, _t10, _t9, _t12))) {
    return false;
  }
  if ((!tri_tri_axis(N1, _t1, _t0, _t3, _t10, _t9, _t12))) {
    return false;
  }
  const float3 E00 = cross(_t2, _t11);
  const float3 E01 = cross(_t2, _t14);
  const float3 E02 = cross(_t2, _t17);
  const float3 E10 = cross(_t5, _t11);
  const float3 E11 = cross(_t5, _t14);
  const float3 E12 = cross(_t5, _t17);
  const float3 E20 = cross(_t8, _t11);
  const float3 E21 = cross(_t8, _t14);
  const float3 E22 = cross(_t8, _t17);
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
  const float _t2 = ((1.0f - refract_idx) / (1.0f + refract_idx));
  const float r1 = (_t2 * _t2);
  return (r1 + ((1.0f - r1) * powf((1.0f - cos_theta), 5.0f)));
}
float3 refract(const float3 uv, const float3 n, const float etai_over_etat) {
  const float cos_theta = min(dot((-uv), n), 1.0f);
  const float3 _t2 = (float3{etai_over_etat} * (uv + (float3{cos_theta} * n)));
  const float3 r_out_parallel = (float3{(-sqrtf(abs((1.0f - reduce_add((_t2 * _t2))))))} * n);
  return (_t2 + r_out_parallel);
}
std::optional<Triangle> trace(const Ray* __restrict__ ray, const Triangles* __restrict__ triangles) {
  return _traverse_tree0(ray, triangles);
}
float3 unit_vector(const float3 v) {
  return (v / float3{norm(v)});
}
