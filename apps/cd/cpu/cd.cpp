#include "apps/cd/cpu/cd.h"
bool intersects_AABB_AABB(const AABB* a, const AABB* b) {
  const vec3_float low = max((*a).low, (*b).low);
  const vec3_float high = min((*a).high, (*b).high);
  return reduce_and((low <= high));
}
bool project6(const vec3_float ax, const vec3_float p1, const vec3_float p2, const vec3_float p3, const vec3_float q1, const vec3_float q2, const vec3_float q3) {
  const float P1 = dot(ax, p1);
  const float P2 = dot(ax, p2);
  const float P3 = dot(ax, p3);
  const float Q1 = dot(ax, q1);
  const float Q2 = dot(ax, q2);
  const float Q3 = dot(ax, q3);
  const float mn1 = min(min(P1, P2), P3);
  const float mx2 = max(max(Q1, Q2), Q3);
  if ((mx2 < mn1)) {
    return false;
  }
  const float mx1 = max(max(P1, P2), P3);
  const float mn2 = min(min(Q1, Q2), Q3);
  if ((mx1 < mn2)) {
    return false;
  }
  return true;
}
bool intersects_Triangle_Triangle(const Triangle* a, const Triangle* b) {
  const vec3_float p1 = (*a).p0;
  const vec3_float p2 = (*a).p1;
  const vec3_float p3 = (*a).p2;
  const vec3_float q1 = (*b).p0;
  const vec3_float q2 = (*b).p1;
  const vec3_float q3 = (*b).p2;
  const vec3_float e1 = (p2 - p1);
  const vec3_float e2 = (p3 - p2);
  const vec3_float e3 = (p1 - p3);
  const vec3_float f1 = (q2 - q1);
  const vec3_float f2 = (q3 - q2);
  const vec3_float f3 = (q1 - q3);
  const vec3_float n1 = cross(e1, e2);
  if ((!project6(n1, p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  const vec3_float m1 = cross(f1, f2);
  if ((!project6(m1, p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e1, f1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e1, f2), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e1, f3), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e2, f1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e2, f2), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e2, f3), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e3, f1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e3, f2), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e3, f3), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e1, n1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e2, n1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(e3, n1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(f1, m1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(f2, m1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  if ((!project6(cross(f3, m1), p1, p2, p3, q1, q2, q3))) {
    return false;
  }
  return true;
}
void _recloop_func0(const uint32_t index1, const uint32_t index2, const Triangles1* triangles1, const Triangles2* triangles2, std::vector<std::tuple<Triangle, Triangle>>& _dyn_alloc) {
  const AABB _lv4 = AABB{.low=(*triangles1).nodes[index2].low, .high=(*triangles1).nodes[index2].high};
  const AABB _lv5 = AABB{.low=(*triangles2).nodes[index2].low, .high=(*triangles2).nodes[index2].high};
  if (intersects_AABB_AABB((&_lv4), (&_lv5))) {
    if ((*triangles1).nodes[index2].nprims == 0) {
      if ((*triangles2).nodes[index2].nprims == 0) {
        _recloop_func0((index1 + 1), (index2 + 1), triangles1, triangles2, _dyn_alloc);
        _recloop_func0((index1 + 1), (index2 + reinterpret<Arm_Interior>((*triangles2).nodes[index2].split0on_nprims).offset), triangles1, triangles2, _dyn_alloc);
        _recloop_func0((index1 + reinterpret<Arm_Interior>((*triangles1).nodes[index2].split0on_nprims).offset), (index2 + 1), triangles1, triangles2, _dyn_alloc);
        _recloop_func0((index1 + reinterpret<Arm_Interior>((*triangles1).nodes[index2].split0on_nprims).offset), (index2 + reinterpret<Arm_Interior>((*triangles2).nodes[index2].split0on_nprims).offset), triangles1, triangles2, _dyn_alloc);
      } else {
        _recloop_func0((index1 + 1), index2, triangles1, triangles2, _dyn_alloc);
        _recloop_func0((index1 + reinterpret<Arm_Interior>((*triangles1).nodes[index2].split0on_nprims).offset), index2, triangles1, triangles2, _dyn_alloc);
      }
    } else if ((*triangles2).nodes[index2].nprims == 0) {
      _recloop_func0(index1, (index2 + 1), triangles1, triangles2, _dyn_alloc);
      _recloop_func0(index1, (index2 + reinterpret<Arm_Interior>((*triangles2).nodes[index2].split0on_nprims).offset), triangles1, triangles2, _dyn_alloc);
    } else {
      for (uint32_t _idx0 = reinterpret<Arm_Leaf>((*triangles1).nodes[index2].split0on_nprims).poffset; _idx0 < (reinterpret<Arm_Leaf>((*triangles1).nodes[index2].split0on_nprims).poffset + (uint32_t)((*triangles1).nodes[index2].nprims)); _idx0 += 1) {
        for (uint32_t _idx1 = reinterpret<Arm_Leaf>((*triangles2).nodes[index2].split0on_nprims).poffset; _idx1 < (reinterpret<Arm_Leaf>((*triangles2).nodes[index2].split0on_nprims).poffset + (uint32_t)((*triangles2).nodes[index2].nprims)); _idx1 += 1) {
          const Triangle _lv0 = (*triangles1).primitives[_idx0];
          const Triangle _lv1 = (*triangles2).primitives[_idx1];
          const Triangle _lv2 = (*triangles1).primitives[_idx0];
          const Triangle _lv3 = (*triangles2).primitives[_idx1];
          if (intersects_Triangle_Triangle((&_lv2), (&_lv3))) {
            _dyn_alloc.push_back(std::tuple<Triangle, Triangle>{_lv0, _lv1});
          }
        }
      }
    }
  }
  return;
}
std::vector<std::tuple<Triangle, Triangle>> _traverse_tree0(const Triangles1* triangles1, const Triangles2* triangles2) {
  std::vector<std::tuple<Triangle, Triangle>> _dyn_alloc;
  _dyn_alloc.reserve(16);
  _recloop_func0(0, 0, triangles1, triangles2, _dyn_alloc);
  return _dyn_alloc;
}
bool axis(const vec3_float A, const vec3_float extents, const vec3_float v0, const vec3_float v1, const vec3_float v2) {
  const float R = dot(extents, abs(A));
  const float P0 = dot(v0, A);
  const float P1 = dot(v1, A);
  const float P2 = dot(v2, A);
  const float Pmin = min(min(P0, P1), P2);
  const float Pmax = max(max(P0, P1), P2);
  return (((-R) <= Pmax) & (Pmin <= R));
}
uint32_t rec_build_triangles1(const BVH* node, Triangles1* ST, size_t* nodes_index, size_t* primitives_index) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].low = node.low;
      (*ST).nodes[this_index].high = node.high;
      (*ST).nodes[this_index].nprims = 0;
      (*ST).nodes[this_index].axis = argmax((node.high - node.low));
      const uint32_t left_index = rec_build_triangles1(node.left, ST, nodes_index, primitives_index);
      const uint32_t right_index = rec_build_triangles1(node.right, ST, nodes_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].low = node.low;
      (*ST).nodes[this_index].high = node.high;
      (*ST).nodes[this_index].nprims = node.nprims;
      (*ST).nodes[this_index].axis = argmax((node.high - node.low));
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0; __p < node.nprims; __p += 1) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_triangles1(const BVH* node, Triangles1* ST) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_triangles1(node.left, ST);
      rec_count_triangles1(node.right, ST);
      (*ST).node_count += 1;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1;
    }
  }, *node);
}
Triangles1 build_triangles1(const BVH* CT) {
  Triangles1 ST;
  size_t primitives_index = 0;
  size_t nodes_index = 0;
  rec_count_triangles1(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Nodes* nodes = reinterpret_cast<Nodes*>(malloc(sizeof(Nodes) * ST.node_count));
  ST.nodes = nodes;
  rec_build_triangles1(CT, (&ST), (&nodes_index), (&primitives_index));
  return ST;
}
uint32_t rec_build_triangles2(const BVH* node, Triangles2* ST, size_t* nodes_index, size_t* primitives_index) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].low = node.low;
      (*ST).nodes[this_index].high = node.high;
      (*ST).nodes[this_index].nprims = 0;
      (*ST).nodes[this_index].axis = argmax((node.high - node.low));
      const uint32_t left_index = rec_build_triangles2(node.left, ST, nodes_index, primitives_index);
      const uint32_t right_index = rec_build_triangles2(node.right, ST, nodes_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].low = node.low;
      (*ST).nodes[this_index].high = node.high;
      (*ST).nodes[this_index].nprims = node.nprims;
      (*ST).nodes[this_index].axis = argmax((node.high - node.low));
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0; __p < node.nprims; __p += 1) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_triangles2(const BVH* node, Triangles2* ST) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_triangles2(node.left, ST);
      rec_count_triangles2(node.right, ST);
      (*ST).node_count += 1;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1;
    }
  }, *node);
}
Triangles2 build_triangles2(const BVH* CT) {
  Triangles2 ST;
  size_t primitives_index = 0;
  size_t nodes_index = 0;
  rec_count_triangles2(CT, (&ST));
  Triangle* primitives = reinterpret_cast<Triangle*>(malloc(sizeof(Triangle) * ST.primitive_count));
  ST.primitives = primitives;
  Nodes* nodes = reinterpret_cast<Nodes*>(malloc(sizeof(Nodes) * ST.node_count));
  ST.nodes = nodes;
  rec_build_triangles2(CT, (&ST), (&nodes_index), (&primitives_index));
  return ST;
}
std::vector<std::tuple<Triangle, Triangle>> collisions(const Triangles1* triangles1, const Triangles2* triangles2) {
  return _traverse_tree0(triangles1, triangles2);
}
