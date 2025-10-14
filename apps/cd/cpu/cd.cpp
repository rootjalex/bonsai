#include "apps/cd/cpu/cd.h"
int32_t project6(const float3 ax, const float3 p1, const float3 p2, const float3 p3, const float3 q1, const float3 q2, const float3 q3) {
  const float P1 = dot(ax, p1);
  const float P2 = dot(ax, p2);
  const float P3 = dot(ax, p3);
  const float Q1 = dot(ax, q1);
  const float Q2 = dot(ax, q2);
  const float Q3 = dot(ax, q3);
  const float mn1 = min(min(P1, P2), P3);
  const float mx2 = max(max(Q1, Q2), Q3);
  if ((mx2 < mn1)) {
    return 0;
  }
  const float mx1 = max(max(P1, P2), P3);
  const float mn2 = min(min(Q1, Q2), Q3);
  if ((mx1 < mn2)) {
    return 0;
  }
  return 1;
}
bool intersect_Triangle(const float3 P1, const float3 P2, const float3 P3, const float3 Q1, const float3 Q2, const float3 Q3) {
  const float3 p2 = (P2 - P1);
  const float3 p3 = (P3 - P1);
  const float3 q1 = (Q1 - P1);
  const float3 q2 = (Q2 - P1);
  const float3 q3 = (Q3 - P1);
  const float3 e2 = (p3 - p2);
  const float3 n1 = cross(p2, e2);
  if (project6(n1, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 f1 = (q2 - q1);
  const float3 f2 = (q3 - q2);
  const float3 m1 = cross(f1, f2);
  if (project6(m1, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef11 = cross(p2, f1);
  if (project6(ef11, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef12 = cross(p2, f2);
  if (project6(ef12, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 f3 = (q1 - q3);
  const float3 ef13 = cross(p2, f3);
  if (project6(ef13, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef21 = cross(e2, f1);
  if (project6(ef21, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef22 = cross(e2, f2);
  if (project6(ef22, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef23 = cross(e2, f3);
  if (project6(ef23, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 e3 = (-p3);
  const float3 ef31 = cross(e3, f1);
  if (project6(ef31, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef32 = cross(e3, f2);
  if (project6(ef32, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 ef33 = cross(e3, f3);
  if (project6(ef33, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 g1 = cross(p2, n1);
  if (project6(g1, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 g2 = cross(e2, n1);
  if (project6(g2, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 g3 = cross(e3, n1);
  if (project6(g3, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 h1 = cross(f1, m1);
  if (project6(h1, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 h2 = cross(f2, m1);
  if (project6(h2, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  const float3 h3 = cross(f3, m1);
  if (project6(h3, float3{0.0f}, p2, p3, q1, q2, q3) == 0) {
    return false;
  }
  return true;
}
bool intersects_AABB_AABB(const AABB* __restrict__ a, const AABB* __restrict__ b) {
  const float3 low = max((*a).low, (*b).low);
  const float3 high = min((*a).high, (*b).high);
  return reduce_and((low <= high));
}
void _recloop_func0(const uint32_t index1, const uint32_t index2, const Triangles1* __restrict__ triangles1, const Triangles2* __restrict__ triangles2, std::vector<std::tuple<Triangle, Triangle>>& _dyn_alloc) {
  if (index1 == 4294967295u) {
    return;
  }
  if (index2 == 4294967295u) {
    return;
  }
  const Nodes1& _t88 = (*triangles1).nodes1[index1];
  const Nodes2& _t92 = (*triangles2).nodes2[index2];
  const AABB _lv0 = AABB{.low=_t88.low, .high=_t88.high};
  const AABB _lv1 = AABB{.low=_t92.low, .high=_t92.high};
  if (intersects_AABB_AABB((&_lv0), (&_lv1))) {
    const uint16_t _t86 = _t88.nprims;
    if (_t86 == 0u) {
      if (_t92.nprims == 0u) {
        const uint32_t _t0 = (index1 + 1u);
        const uint32_t _t1 = (index2 + 1u);
        _recloop_func0(_t0, _t1, triangles1, triangles2, _dyn_alloc);
        const uint32_t _t8 = (index2 + reinterpret<Arm_Interior>(_t92.split0on_nprims).offset);
        _recloop_func0(_t0, _t8, triangles1, triangles2, _dyn_alloc);
        const uint32_t _t14 = (index1 + reinterpret<Arm_Interior>(_t88.split0on_nprims).offset);
        _recloop_func0(_t14, _t1, triangles1, triangles2, _dyn_alloc);
        _recloop_func0(_t14, _t8, triangles1, triangles2, _dyn_alloc);
      } else {
        _recloop_func0((index1 + 1u), index2, triangles1, triangles2, _dyn_alloc);
        _recloop_func0((index1 + reinterpret<Arm_Interior>(_t88.split0on_nprims).offset), index2, triangles1, triangles2, _dyn_alloc);
      }
    } else {
      const uint16_t _t82 = _t92.nprims;
      if (_t82 == 0u) {
        _recloop_func0(index1, (index2 + 1u), triangles1, triangles2, _dyn_alloc);
        _recloop_func0(index1, (index2 + reinterpret<Arm_Interior>(_t92.split0on_nprims).offset), triangles1, triangles2, _dyn_alloc);
      } else {
        const uint32_t _t71 = reinterpret<Arm_Leaf>(_t88.split0on_nprims).poffset;
        for (uint32_t _idx0 = _t71; _idx0 < (_t71 + (uint32_t)(_t86)); ++_idx0) {
          const uint32_t _t58 = reinterpret<Arm_Leaf>(_t92.split0on_nprims).poffset;
          for (uint32_t _idx1 = _t58; _idx1 < (_t58 + (uint32_t)(_t82)); ++_idx1) {
            const Triangle& _t47 = (*triangles1).primitives[_idx0];
            const Triangle& _t49 = (*triangles2).primitives[_idx1];
            if (intersect_Triangle(_t47.p0, _t47.p1, _t47.p2, _t49.p0, _t49.p1, _t49.p2)) {
              _dyn_alloc.push_back(std::tuple<Triangle, Triangle>{_t47, _t49});
            }
          }
        }
      }
    }
  }
  return;
}
std::vector<std::tuple<Triangle, Triangle>> _traverse_tree0(const Triangles1* __restrict__ triangles1, const Triangles2* __restrict__ triangles2) {
  std::vector<std::tuple<Triangle, Triangle>> _dyn_alloc;
  _dyn_alloc.reserve(16);
  _recloop_func0(0u, 0u, triangles1, triangles2, _dyn_alloc);
  return _dyn_alloc;
}
bool axis(const float3 A, const float3 extents, const float3 v0, const float3 v1, const float3 v2) {
  const float R = dot(extents, abs(A));
  const float P0 = dot(v0, A);
  const float P1 = dot(v1, A);
  const float P2 = dot(v2, A);
  const float Pmin = min(min(P0, P1), P2);
  const float Pmax = max(max(P0, P1), P2);
  return (((-R) <= Pmax) & (Pmin <= R));
}
uint32_t rec_build_triangles1(const BVH* __restrict__ node, Triangles1* __restrict__ ST, size_t* __restrict__ nodes1_index, size_t* __restrict__ primitives_index) {
  if ((!node)) {
    return 4294967295u;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes1_index);
      (*nodes1_index) += 1u;
      (*ST).nodes1[this_index].low = node.low;
      (*ST).nodes1[this_index].high = node.high;
      (*ST).nodes1[this_index].nprims = 0;
      (*ST).nodes1[this_index].axis = argmax((node.high - node.low));
      const uint32_t left_index = rec_build_triangles1(node.left, ST, nodes1_index, primitives_index);
      const uint32_t right_index = rec_build_triangles1(node.right, ST, nodes1_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes1[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes1_index);
      (*nodes1_index) += 1u;
      (*ST).nodes1[this_index].low = node.low;
      (*ST).nodes1[this_index].high = node.high;
      (*ST).nodes1[this_index].nprims = node.nprims;
      (*ST).nodes1[this_index].axis = argmax((node.high - node.low));
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes1[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0u; __p < node.nprims; ++__p) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_triangles1(const BVH* __restrict__ node, Triangles1* __restrict__ ST) {
  if ((!node)) {
    return;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_triangles1(node.left, ST);
      rec_count_triangles1(node.right, ST);
      (*ST).node_count += 1u;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1u;
    }
  }, *node);
}
Triangles1 build_triangles1(const BVH* __restrict__ CT) {
  Triangles1 ST;
  size_t primitives_index = 0;
  size_t nodes1_index = 0;
  ST.primitive_count = 0u;
  ST.node_count = 0u;
  rec_count_triangles1(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Nodes1* nodes1 = reinterpret_cast<Nodes1*>(std::aligned_alloc(32, (((sizeof(Nodes1) * ST.node_count) + 31) / 32) * 32));
  ST.nodes1 = nodes1;
  rec_build_triangles1(CT, (&ST), (&nodes1_index), (&primitives_index));
  return ST;
}
uint32_t rec_build_triangles2(const BVH* __restrict__ node, Triangles2* __restrict__ ST, size_t* __restrict__ nodes2_index, size_t* __restrict__ primitives_index) {
  if ((!node)) {
    return 4294967295u;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes2_index);
      (*nodes2_index) += 1u;
      (*ST).nodes2[this_index].low = node.low;
      (*ST).nodes2[this_index].high = node.high;
      (*ST).nodes2[this_index].nprims = 0;
      (*ST).nodes2[this_index].axis = argmax((node.high - node.low));
      const uint32_t left_index = rec_build_triangles2(node.left, ST, nodes2_index, primitives_index);
      const uint32_t right_index = rec_build_triangles2(node.right, ST, nodes2_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes2[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes2_index);
      (*nodes2_index) += 1u;
      (*ST).nodes2[this_index].low = node.low;
      (*ST).nodes2[this_index].high = node.high;
      (*ST).nodes2[this_index].nprims = node.nprims;
      (*ST).nodes2[this_index].axis = argmax((node.high - node.low));
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes2[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0u; __p < node.nprims; ++__p) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_triangles2(const BVH* __restrict__ node, Triangles2* __restrict__ ST) {
  if ((!node)) {
    return;
  }
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_triangles2(node.left, ST);
      rec_count_triangles2(node.right, ST);
      (*ST).node_count += 1u;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1u;
    }
  }, *node);
}
Triangles2 build_triangles2(const BVH* __restrict__ CT) {
  Triangles2 ST;
  size_t primitives_index = 0;
  size_t nodes2_index = 0;
  ST.primitive_count = 0u;
  ST.node_count = 0u;
  rec_count_triangles2(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Nodes2* nodes2 = reinterpret_cast<Nodes2*>(std::aligned_alloc(32, (((sizeof(Nodes2) * ST.node_count) + 31) / 32) * 32));
  ST.nodes2 = nodes2;
  rec_build_triangles2(CT, (&ST), (&nodes2_index), (&primitives_index));
  return ST;
}
std::vector<std::tuple<Triangle, Triangle>> collisions(const Triangles1* __restrict__ triangles1, const Triangles2* __restrict__ triangles2) {
  return _traverse_tree0(triangles1, triangles2);
}
bool intersects_Triangle_Triangle(const Triangle* __restrict__ a, const Triangle* __restrict__ b) {
  return intersect_Triangle((*a).p0, (*a).p1, (*a).p2, (*b).p0, (*b).p1, (*b).p2);
}
