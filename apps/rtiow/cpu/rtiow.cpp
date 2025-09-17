#include "apps/rtiow/cpu/rtiow.h"
Point ClosestPtPointAABB(const Point* __restrict__ pt, const AABB* __restrict__ a) {
  return Point{.vec=min(max((*pt).vec, (*a).low), (*a).high)};
}
float SqDistPointAABB(const Point* __restrict__ pt, const AABB* __restrict__ a) {
  const vec3_float v = (*pt).vec;
  const vec3_float sqLow = (((*a).low - v) * ((*a).low - v));
  const vec3_float low = select((v < (*a).low), sqLow, vec3_float{0.0f});
  const vec3_float sqHigh = ((v - (*a).high) * (v - (*a).high));
  const vec3_float high = select(((*a).high < v), sqHigh, vec3_float{0.0f});
  return reduce_add((low + high));
}
float __prod_diff_f32(const float a, const float b, const float c, const float d) {
  const float cd = (c * d);
  const float diff = fmaf(a, b, (-cd));
  const float err = fmaf((-c), d, cd);
  return (diff + err);
}
float __sqlen_f32(const vec3_float v) {
  return reduce_add((v * v));
}
vec3_float random_in_unit_disk() {
  const float _t0 = random_float<float>();
  const float _t1 = sqrtf(_t0);
  const float _t4 = ((2.0f * 3.14159274f) * _t0);
  return vec3_float{(_t1 * cosf(_t4)), (_t1 * sinf(_t4)), 0.0f};
}
vec3_float defocus_disk_sample(const vec3_float center, const vec3_float defocus_disk_u, const vec3_float defocus_disk_v) {
  const vec3_float p = random_in_unit_disk();
  return ((center + (vec3_float{p[0]} * defocus_disk_u)) + (vec3_float{p[1]} * defocus_disk_v));
}
vec3_float sample_square() {
  const float _t0 = random_float<float>();
  const float _t1 = (random_float<float>() - 0.50000000f);
  return vec3_float{_t1, _t1, 0.0f};
}
Ray build_ray(const int32_t i, const int32_t j, const Camera* __restrict__ cam) {
  const int32_t width = (*cam).width;
  int32_t height = (int32_t)(((float)(width) / (*cam).aspect_ratio));
  height = ((height < 1) ? 1 : height);
  const float theta = (((*cam).vfov * 3.14159274f) / 180.0f);
  const float h = tanf((theta / 2.0f));
  const float viewport_height = ((2.0f * h) * (*cam).focus_dist);
  const float viewport_width = (viewport_height * ((float)(width) / (float)(height)));
  const vec3_float camera_center = (*cam).lookfrom;
  const vec3_float w = (((*cam).lookfrom - (*cam).lookat) / vec3_float{norm(((*cam).lookfrom - (*cam).lookat))});
  const vec3_float u = (cross((*cam).vup, w) / vec3_float{norm(cross((*cam).vup, w))});
  const vec3_float v = cross(w, u);
  const vec3_float viewport_u = (vec3_float{viewport_width} * u);
  const vec3_float viewport_v = (vec3_float{viewport_height} * (-v));
  const vec3_float pixel_delta_u = (viewport_u / vec3_float{(float)(width)});
  const vec3_float pixel_delta_v = (viewport_v / vec3_float{(float)(height)});
  const vec3_float viewport_upper_left = (((camera_center - (vec3_float{(*cam).focus_dist} * w)) - (viewport_u / vec3_float{2.0f})) - (viewport_v / vec3_float{2.0f}));
  const vec3_float pixel00_loc = (viewport_upper_left + (vec3_float{0.50000000f} * (pixel_delta_u + pixel_delta_v)));
  const float defocus_radius = ((*cam).focus_dist * tanf(((((*cam).defocus_angle / 2.0f) * 3.14159274f) / 180.0f)));
  const vec3_float defocus_disk_u = (u * vec3_float{defocus_radius});
  const vec3_float defocus_disk_v = (v * vec3_float{defocus_radius});
  const vec3_float offset = sample_square();
  const vec3_float pixel_sample = ((pixel00_loc + (vec3_float{((float)(i) + offset[0])} * pixel_delta_u)) + (vec3_float{((float)(j) + offset[1])} * pixel_delta_v));
  vec3_float ray_origin = camera_center;
  if ((0.0f < (*cam).defocus_angle)) {
    ray_origin = defocus_disk_sample(camera_center, defocus_disk_u, defocus_disk_v);
  }
  const vec3_float ray_direction = (pixel_sample - ray_origin);
  return Ray{.o=ray_origin, .d=ray_direction, .tmax=std::numeric_limits<float>::infinity()};
}
std::optional<FInterval> intersectsp_ray_sphere(const Ray* __restrict__ r, const Sphere* __restrict__ s) {
  const vec3_float _t2 = ((*s).center - (*r).o);
  const vec3_float _t3 = (*r).d;
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
    const FInterval extract = *interval;
    return extract.high;
  }
  return std::numeric_limits<float>::infinity();
}
float distmin_Ray_Sphere(const Ray* __restrict__ r, const Sphere* __restrict__ s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.low;
  }
  return (-std::numeric_limits<float>::infinity());
}
bool intersects_Ray_Sphere(const Ray* __restrict__ ray, const Sphere* __restrict__ s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(ray, s);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return ((extract.low < (*ray).tmax) & (0.0f < extract.high));
  }
  return false;
}
std::optional<MaterialSphere> _traverse_tree0(const Ray* __restrict__ r, const Spheres* __restrict__ spheres) {
  std::tuple<float, MaterialSphere> _best0 = std::tuple<float, MaterialSphere>{std::numeric_limits<float>::infinity(), MaterialSphere{}};
  int32_t _queue_count0 = 1;
  uint32_t* _queue0 = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * 64));
  _queue0[0] = 0;
  do {
    _queue_count0 -= 1;
    const uint32_t index = _queue0[_queue_count0];
    const Nodes _t49 = (*spheres).nodes[index];
    const Sphere _t54 = Sphere{.center=_t49.center, .radius=_t49.radius};
    if (intersects_Ray_Sphere(r, (&_t54))) {
      if ((0.00100000f < distmax_Ray_Sphere(r, (&_t54)))) {
        if ((distmin_Ray_Sphere(r, (&_t54)) < std::get<0>(_best0))) {
          const uint8_t _t32 = _t49.nprims;
          if (_t32 == 0) {
            _queue0[_queue_count0] = (index + 1);
            _queue0[(_queue_count0 + 1)] = (index + (uint32_t)(reinterpret<Arm_Interior>(_t49.split0on_nprims).offset));
            _queue_count0 += 2;
          } else {
            const uint16_t _t21 = reinterpret<Arm_Leaf>(_t49.split0on_nprims).poffset;
            for (uint16_t _idx0 = _t21; _idx0 < (_t21 + (uint16_t)(_t32)); _idx0 += 1) {
              const MaterialSphere _t16 = (*spheres).primitives[_idx0];
              if (intersects_Ray_Sphere(r, (&_t16.s))) {
                const float _t14 = distmin_Ray_Sphere(r, (&_t16.s));
                if ((0.00100000f < _t14)) {
                  if ((_t14 < std::get<0>(_best0))) {
                    _best0 = argmin<float, MaterialSphere>(_best0, std::tuple<float, MaterialSphere>{_t14, _t16});
                  }
                }
              }
            }
          }
        }
      }
    }
} while ((_queue_count0 != 0));
  return ((std::get<0>(_best0) != std::numeric_limits<float>::infinity()) ? std::optional<MaterialSphere>{std::get<1>(_best0)} : std::nullopt);
}
Hit_record get_hit_record(const Ray* __restrict__ r, const Sphere* __restrict__ s) {
  const float t = distmin_Ray_Sphere(r, s);
  const vec3_float p = ((*r).o + (vec3_float{t} * (*r).d));
  const vec3_float outward_normal = ((p - (*s).center) / vec3_float{(*s).radius});
  const bool front_face = (dot((*r).d, outward_normal) < 0.0f);
  const vec3_float normal = (front_face ? outward_normal : (-outward_normal));
  const Hit_record record = Hit_record{.p=p, .normal=normal, .t=t, .front_face=front_face};
  return record;
}
vec3_float random_unit_vector() {
  const float x1 = (-1.0f + ((1.0f - -1.0f) * random_float<float>()));
  const float x2 = (-1.0f + ((1.0f - -1.0f) * random_float<float>()));
  const float _t3 = (((x1 * x1) + (x2 * x2)) + 0.00000001f);
  const float _t5 = sqrtf((2.0f / _t3));
  const float x = (_t5 * x1);
  const float y = (_t5 * x2);
  const float _t7 = (1.0f - (2.0f * _t3));
  const float _t13 = sqrtf((((x * x) + (y * y)) + (_t7 * _t7)));
  return vec3_float{(x / _t13), (y / _t13), (_t7 / _t13)};
}
float reflectance(const float cos_theta, const float refract_idx) {
  const float _t2 = ((1.0f - refract_idx) / (1.0f + refract_idx));
  const float r1 = (_t2 * _t2);
  return (r1 + ((1.0f - r1) * powf((1.0f - cos_theta), 5.0f)));
}
vec3_float refract(const vec3_float uv, const vec3_float n, const float etai_over_etat) {
  const float cos_theta = min(dot((-uv), n), 1.0f);
  const vec3_float _t2 = (vec3_float{etai_over_etat} * (uv + (vec3_float{cos_theta} * n)));
  const vec3_float r_out_parallel = (vec3_float{(-sqrtf(abs((1.0f - reduce_add((_t2 * _t2))))))} * n);
  return (_t2 + r_out_parallel);
}
Scatter_record scatter(const Ray* __restrict__ ray, const MaterialSphere* __restrict__ ms) {
  const Hit_record _t1 = get_hit_record(ray, (&(*ms).s));
  const uint32_t _t20 = (*ms).material;
  if (_t20 == 0) {
    const vec3_float _t2 = _t1.normal;
    vec3_float scatter_dir = (_t2 + random_unit_vector());
    if ((((abs(scatter_dir[0]) < 0.00000001f) & (abs(scatter_dir[1]) < 0.00000001f)) & (abs(scatter_dir[2]) < 0.00000001f))) {
      scatter_dir = _t2;
    }
    const Ray l_scattered = Ray{.o=_t1.p, .d=scatter_dir, .tmax=std::numeric_limits<float>::infinity()};
    return Scatter_record{.attenuation=(*ms).albedo, .ray=l_scattered, .hit=true};
  } else if (_t20 == 1) {
    const vec3_float ref = ((*ray).d - (vec3_float{(2.0f * dot((*ray).d, _t1.normal))} * _t1.normal));
    const vec3_float reflected = ((ref / vec3_float{norm(ref)}) + (vec3_float{(*ms).fuzz} * random_unit_vector()));
    const Ray m_scattered = Ray{.o=_t1.p, .d=reflected, .tmax=std::numeric_limits<float>::infinity()};
    return Scatter_record{.attenuation=(*ms).albedo, .ray=m_scattered, .hit=true};
  } else {
    const float _t11 = (*ms).fuzz;
    const float ri = (_t1.front_face ? (1.0f / _t11) : _t11);
    const vec3_float _t14 = ((*ray).d / vec3_float{norm((*ray).d)});
    const vec3_float _t15 = _t1.normal;
    const float cos_theta = min(dot((-_t14), _t15), 1.0f);
    const float sin_theta = sqrtf((1.0f - (cos_theta * cos_theta)));
    const bool cannot_refract = ((1.0f < (ri * sin_theta)) | (random_float<float>() < reflectance(cos_theta, ri)));
    const vec3_float direction = (cannot_refract ? (_t14 - (vec3_float{(2.0f * dot(_t14, _t15))} * _t15)) : refract(_t14, _t15, ri));
    const Ray d_scattered = Ray{.o=_t1.p, .d=direction, .tmax=std::numeric_limits<float>::infinity()};
    return Scatter_record{.attenuation=vec3_float{1.0f, 1.0f, 1.0f}, .ray=d_scattered, .hit=true};
  }
}
vec3_float sample(const Ray* __restrict__ r, const int32_t depth, const vec3_float mult, const Spheres* __restrict__ spheres) {
  if ((depth <= 0)) {
    return vec3_float{0.0f, 0.0f, 0.0f};
  }
  const std::optional<MaterialSphere> isect = _traverse_tree0(r, spheres);
  if (isect.has_value()) {
    const MaterialSphere _lv0 = *isect;
    const Scatter_record data = scatter(r, (&_lv0));
    if (data.hit) {
      return sample((&data.ray), (depth - 1), (mult * data.attenuation), spheres);
    } else {
      return vec3_float{0.0f, 0.0f, 0.0f};
    }
  }
  const vec3_float _t1 = (*r).d;
  const vec3_float unit_direction = (_t1 / vec3_float{norm(_t1)});
  const float a = (0.50000000f * (unit_direction[1] + 1.0f));
  return (mult * ((vec3_float{(1.0f - a)} * vec3_float{1.0f, 1.0f, 1.0f}) + (vec3_float{a} * vec3_float{0.50000000f, 0.69999999f, 1.0f})));
}
vec3_float _traverse_array1(const int32_t i, const int32_t j, const Camera* __restrict__ c, const Spheres* __restrict__ spheres) {
  vec3_float _alloc1 = vec3_float{0.0f};
  for (uint32_t _i0 = 0; _i0 < (*c).samples_per_pixel; _i0 += 1) {
    const Ray _lv0 = build_ray(i, j, c);
    _alloc1 += sample((&_lv0), (*c).max_depth, vec3_float{1.0f, 1.0f, 1.0f}, spheres);
  }
  return _alloc1;
}
float linear_to_gamma_f(const float l) {
  if ((0.0f < l)) {
    return sqrtf(l);
  }
  return 0.0f;
}
vec3_int32_t to_rgb(const vec3_float v) {
  const vec3_float corrected = vec3_float{linear_to_gamma_f(v[0]), linear_to_gamma_f(v[1]), linear_to_gamma_f(v[2])};
  return (vec3_int32_t)((vec3_float{256.0f} * min(max(corrected, vec3_float{0.0f}), vec3_float{0.99900001f})));
}
vec3_int32_t** _traverse_array0(const Camera* __restrict__ c, const int32_t height, const Spheres* __restrict__ spheres) {
  int32_t* _alloc0 = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * ((height * (*c).width) * 3)));
  for (int32_t _i0 = 0; _i0 < height; _i0 += 1) {
    for (int32_t _i1 = 0; _i1 < (*c).width; _i1 += 1) {
      const vec3_int32_t _t0 = to_rgb((_traverse_array1(_i1, _i0, c, spheres) / vec3_float{(float)((*c).samples_per_pixel)}));
      _alloc0[(((_i0 * (*c).width) + _i1) * 3)] = _t0[0];
      _alloc0[((((_i0 * (*c).width) + _i1) * 3) + 1)] = _t0[1];
      _alloc0[((((_i0 * (*c).width) + _i1) * 3) + 2)] = _t0[2];
    }
  }
  return reinterpret_cast<vec3_int32_t**>(_alloc0);
}
vec3_float at(const Ray* __restrict__ r, const float t) {
  return ((*r).o + (vec3_float{t} * (*r).d));
}
bool axis(const vec3_float A, const vec3_float extents, const vec3_float v0, const vec3_float v1, const vec3_float v2) {
  const float R = dot(extents, abs(A));
  const vec3_float _t3 = vec3_float{dot(v0, A), dot(v1, A), dot(v2, A)};
  return reduce_and(((_t3 <= vec3_float{R}) & (vec3_float{(-R)} <= _t3)));
}
Sphere bounding_sphere(const Sphere* __restrict__ a, const Sphere* __restrict__ b) {
  const vec3_float _t1 = (*a).center;
  const vec3_float _t2 = ((*b).center - _t1);
  const float dist_sq = reduce_add((_t2 * _t2));
  const float dist = sqrtf(dist_sq);
  const float _t7 = (*b).radius;
  const float _t8 = (*a).radius;
  if (((dist + _t7) <= _t8)) {
    return (*a);
  } else if (((dist + _t8) <= _t7)) {
    return (*b);
  }
  const float _t13 = (0.50000000f * ((dist + _t8) + _t7));
  const vec3_float direction = ((0.0f < dist) ? (_t2 / vec3_float{dist}) : vec3_float{1.0f, 0.0f, 0.0f});
  const vec3_float new_center = (_t1 + (direction * vec3_float{(_t13 - _t8)}));
  return Sphere{.center=new_center, .radius=_t13};
}
uint32_t rec_build_spheres(const BVH* __restrict__ node, Spheres* __restrict__ ST, size_t* __restrict__ nodes_index, size_t* __restrict__ primitives_index) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].center = node.center;
      (*ST).nodes[this_index].radius = node.radius;
      (*ST).nodes[this_index].nprims = 0;
      const uint32_t left_index = rec_build_spheres(node.left, ST, nodes_index, primitives_index);
      const uint32_t right_index = rec_build_spheres(node.right, ST, nodes_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index) += 1;
      (*ST).nodes[this_index].center = node.center;
      (*ST).nodes[this_index].radius = node.radius;
      (*ST).nodes[this_index].nprims = node.nprims;
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint8_t __p = 0; __p < node.nprims; __p += 1) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index) += node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_spheres(const BVH* __restrict__ node, Spheres* __restrict__ ST) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_spheres(node.left, ST);
      rec_count_spheres(node.right, ST);
      (*ST).node_count += 1;
    },
    [&](const Leaf& node) {
      (*ST).primitive_count += node.nprims;
      (*ST).node_count += 1;
    }
  }, *node);
}
Spheres build_spheres(const BVH* __restrict__ CT) {
  Spheres ST;
  size_t primitives_index = 0;
  size_t nodes_index = 0;
  rec_count_spheres(CT, (&ST));
  MaterialSphere* primitives = reinterpret_cast<MaterialSphere*>(malloc(sizeof(MaterialSphere) * ST.primitive_count));
  ST.primitives = primitives;
  Nodes* nodes = reinterpret_cast<Nodes*>(malloc(sizeof(Nodes) * ST.node_count));
  ST.nodes = nodes;
  rec_build_spheres(CT, (&ST), (&nodes_index), (&primitives_index));
  return ST;
}
vec3_float clamp(const vec3_float x, const float low, const float high) {
  return min(max(x, vec3_float{low}), vec3_float{high});
}
std::tuple<Point, Point> closestPointonTriangle(const Point* __restrict__ pt, const Triangle* __restrict__ tri) {
  const vec3_float p = (*pt).vec;
  const vec3_float a = (*tri).p0;
  const vec3_float b = (*tri).p1;
  const vec3_float c = (*tri).p2;
  const vec3_float ab = (b - a);
  const vec3_float ac = (c - a);
  const vec3_float ap = (p - a);
  const float d1 = dot(ab, ap);
  const float d2 = dot(ac, ap);
  if ((d1 <= 0.0f)) {
    if ((d2 <= 0.0f)) {
      return std::tuple<Point, Point>{Point{.vec=a}, Point{.vec=vec3_float{1.0f, 0.0f, 0.0f}}};
    }
  }
  const vec3_float bp = (p - b);
  const float d3 = dot(ab, bp);
  const float d4 = dot(ac, bp);
  if ((0.0f <= d3)) {
    if ((d4 <= d3)) {
      return std::tuple<Point, Point>{Point{.vec=b}, Point{.vec=vec3_float{0.0f, 1.0f, 0.0f}}};
    }
  }
  const float _t2 = ((d1 * d4) - (d3 * d2));
  if ((_t2 <= 0.0f)) {
    if ((0.0f <= d1)) {
      if ((d3 <= 0.0f)) {
        const float _t4 = (d1 / (d1 - d3));
        return std::tuple<Point, Point>{Point{.vec=(a + (vec3_float{_t4} * ab))}, Point{.vec=vec3_float{(1.0f - _t4), _t4, 0.0f}}};
      }
    }
  }
  const vec3_float cp = (p - c);
  const float d5 = dot(ab, cp);
  const float d6 = dot(ac, cp);
  if ((0.0f <= d6)) {
    if ((d5 <= d6)) {
      return std::tuple<Point, Point>{Point{.vec=c}, Point{.vec=vec3_float{0.0f, 0.0f, 1.0f}}};
    }
  }
  const float _t7 = ((d5 * d2) - (d1 * d6));
  if ((_t7 <= 0.0f)) {
    if ((0.0f <= d2)) {
      if ((d6 <= 0.0f)) {
        const float _t9 = (d2 / (d2 - d6));
        return std::tuple<Point, Point>{Point{.vec=(a + (vec3_float{_t9} * ac))}, Point{.vec=vec3_float{(1.0f - _t9), 0.0f, _t9}}};
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
        return std::tuple<Point, Point>{Point{.vec=(b + (vec3_float{_t17} * (c - b)))}, Point{.vec=vec3_float{0.0f, (1.0f - _t17), _t17}}};
      }
    }
  }
  const float _t22 = (1.0f / ((_t12 + _t7) + _t2));
  const float v = (_t7 * _t22);
  const float w = (_t2 * _t22);
  const float u = (_t12 * _t22);
  return std::tuple<Point, Point>{Point{.vec=((a + (ab * vec3_float{v})) + (ac * vec3_float{w}))}, Point{.vec=vec3_float{u, v, w}}};
}
vec3_float cross_(const vec3_float v0, const vec3_float v1) {
  const float _t0 = v0[1];
  const float _t1 = v1[2];
  const float _t2 = v0[2];
  const float _t3 = v1[1];
  const float _t5 = v1[0];
  const float _t6 = v0[0];
  return vec3_float{__prod_diff_f32(_t0, _t1, _t2, _t3), __prod_diff_f32(_t2, _t5, _t6, _t1), __prod_diff_f32(_t6, _t3, _t0, _t5)};
}
float degrees_to_radians(const float degrees) {
  return ((degrees * 3.14159274f) / 180.0f);
}
float gamma(const int32_t n) {
  const float _t1 = ((float)(n) * 0.00000006f);
  return (_t1 / (1.0f - _t1));
}
std::optional<FInterval> intersectsp_ray_aabb(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const vec3_float _t1 = (vec3_float{1.0f} / (*r).d);
  const vec3_bool dirIsNeg = (_t1 < vec3_float{0.0f});
  const vec3_float _t2 = (*b).high;
  const vec3_float _t3 = (*b).low;
  const vec3_float low_parts = select(dirIsNeg, _t2, _t3);
  const vec3_float high_parts = select(dirIsNeg, _t3, _t2);
  const vec3_float _t6 = (*r).o;
  vec3_float tMin = ((low_parts - _t6) * _t1);
  vec3_float tMax = ((high_parts - _t6) * _t1);
  tMax *= (1.0f + (2.0f * gamma(3)));
  if (((tMax[1] < tMin[0]) || (tMax[0] < tMin[1]))) {
    return std::nullopt;
  }
  float tmin = max(tMin[0], tMin[1]);
  float tmax = min(tMax[0], tMax[1]);
  if (((tMax[2] < tmin) || (tmax < tMin[2]))) {
    return std::nullopt;
  }
  tmin = max(tmin, tMin[2]);
  tmax = min(tmax, tMax[2]);
  return (std::optional<FInterval>)(FInterval{.low=tmin, .high=tmax});
}
float distmax_Ray_AABB(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.high;
  }
  return std::numeric_limits<float>::infinity();
}
float distmax_Ray_MaterialSphere(const Ray* __restrict__ r, const MaterialSphere* __restrict__ ms) {
  return distmax_Ray_Sphere(r, (&(*ms).s));
}
std::optional<TriangleIntersection> intersectsp_ray_tri(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  const vec3_float _t0 = (*tri).p2;
  const vec3_float _t1 = (*tri).p0;
  const vec3_float _t2 = (*tri).p1;
  if (reduce_add((cross_((_t0 - _t1), (_t2 - _t1)) * cross_((_t0 - _t1), (_t2 - _t1)))) == 0.0f) {
    return std::nullopt;
  }
  const vec3_float _t5 = (*ray).o;
  vec3_float p0t = (_t1 - _t5);
  vec3_float p1t = (_t2 - _t5);
  vec3_float p2t = (_t0 - _t5);
  const vec3_float _t10 = (*ray).d;
  const uint32_t kz = reduce_idxmax(abs(_t10));
  const uint32_t kx = ((kz + 1) % 3);
  const uint32_t ky = ((kx + 1) % 3);
  const vec3_float d = shuffle(_t10, {kx, ky, kz});
  p0t = shuffle(p0t, {kx, ky, kz});
  p1t = shuffle(p1t, {kx, ky, kz});
  p2t = shuffle(p2t, {kx, ky, kz});
  const float _t13 = d[2];
  const float _t14 = ((-d[0]) / _t13);
  const float _t17 = ((-d[1]) / _t13);
  const float Sz = (1.0f / _t13);
  p0t[0] += (_t14 * p0t[2]);
  p0t[1] += (_t17 * p0t[2]);
  p1t[0] += (_t14 * p1t[2]);
  p1t[1] += (_t17 * p1t[2]);
  p2t[0] += (_t14 * p2t[2]);
  p2t[1] += (_t17 * p2t[2]);
  const float _t20 = __prod_diff_f32(p1t[0], p2t[1], p1t[1], p2t[0]);
  const float _t21 = __prod_diff_f32(p2t[0], p0t[1], p2t[1], p0t[0]);
  const float _t22 = __prod_diff_f32(p0t[0], p1t[1], p0t[1], p1t[0]);
  if ((((_t20 < 0.0f) || (_t21 < 0.0f)) || (_t22 < 0.0f))) {
    if ((((0.0f < _t20) || (0.0f < _t21)) || (0.0f < _t22))) {
      return std::nullopt;
    }
  }
  const float _t24 = ((_t20 + _t21) + _t22);
  if (_t24 == 0.0f) {
    return std::nullopt;
  }
  p0t[2] *= Sz;
  p1t[2] *= Sz;
  p2t[2] *= Sz;
  const float tScaled = (((_t20 * p0t[2]) + (_t21 * p1t[2])) + (_t22 * p2t[2]));
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
  const float maxZt = reduce_max(abs(vec3_float{p0t[2], p1t[2], p2t[2]}));
  const float _t25 = gamma(3);
  const float deltaZ = (_t25 * maxZt);
  const float maxXt = reduce_max(abs(vec3_float{p0t[0], p1t[0], p2t[0]}));
  const float maxYt = reduce_max(abs(vec3_float{p0t[1], p1t[1], p2t[1]}));
  const float _t26 = gamma(5);
  const float deltaX = (_t26 * (maxXt + maxZt));
  const float deltaY = (_t26 * (maxYt + maxZt));
  const float deltaE = (2.0f * ((((gamma(2) * maxXt) * maxYt) + (deltaY * maxXt)) + (deltaX * maxYt)));
  const float maxE = reduce_max(abs(vec3_float{_t20, _t21, _t22}));
  const float deltaT = ((3.0f * ((((_t25 * maxE) * maxZt) + (deltaE * maxZt)) + (deltaZ * maxE))) * abs(invDet));
  if ((t <= deltaT)) {
    return std::nullopt;
  }
  return (std::optional<TriangleIntersection>)(TriangleIntersection{.b0=b0, .b1=b1, .b2=b2, .t=t});
}
float distmax_Ray_Triangle(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  const std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    const TriangleIntersection isect_ = *isect;
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
float distmin_Ray_AABB(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.low;
  }
  return (-std::numeric_limits<float>::infinity());
}
float distmin_Ray_MaterialSphere(const Ray* __restrict__ r, const MaterialSphere* __restrict__ ms) {
  return distmin_Ray_Sphere(r, (&(*ms).s));
}
float distmin_Ray_Triangle(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  const std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    const TriangleIntersection isect_ = *isect;
    return isect_.t;
  } else {
    return std::numeric_limits<float>::infinity();
  }
}
vec3_int32_t** image(const Camera* __restrict__ c, const Spheres* __restrict__ spheres) {
  int32_t height = (int32_t)(((float)((*c).width) / (*c).aspect_ratio));
  height = ((height < 1) ? 1 : height);
  return _traverse_array0(c, height, spheres);
}
bool intersects_AABB_AABB(const AABB* __restrict__ a, const AABB* __restrict__ b) {
  const vec3_float low = max((*a).low, (*b).low);
  const vec3_float high = min((*a).high, (*b).high);
  return reduce_and((low <= high));
}
bool intersects_AABB_Sphere(const AABB* __restrict__ a, const Sphere* __restrict__ s) {
  const vec3_float _t1 = (*s).center;
  const vec3_float _t6 = (_t1 - max((*a).low, min(_t1, (*a).high)));
  const float _t7 = (*s).radius;
  return (dot(_t6, _t6) <= (_t7 * _t7));
}
bool intersects_AABB_Triangle(const AABB* __restrict__ a, const Triangle* __restrict__ b) {
  const vec3_float _t0 = (*a).low;
  const vec3_float _t1 = (*a).high;
  const vec3_float _t3 = ((_t0 + _t1) * vec3_float{0.50000000f});
  const vec3_float _t7 = ((_t1 - _t0) * vec3_float{0.50000000f});
  const vec3_float _t9 = ((*b).p0 - _t3);
  const vec3_float _t11 = ((*b).p1 - _t3);
  const vec3_float _t13 = ((*b).p2 - _t3);
  const vec3_float f0 = (_t11 - _t9);
  const vec3_float f1 = (_t13 - _t11);
  const vec3_float f2 = (_t9 - _t13);
  const vec3_float tri_min = min(min(_t9, _t11), _t13);
  const vec3_float tri_max = max(max(_t9, _t11), _t13);
  if ((!reduce_and(((tri_min <= _t7) & ((-_t7) <= tri_max))))) {
    return false;
  }
  const vec3_float A0 = cross(f0, vec3_float{1.0f, 0.0f, 0.0f});
  const vec3_float A1 = cross(f0, vec3_float{0.0f, 1.0f, 0.0f});
  const vec3_float A2 = cross(f0, vec3_float{0.0f, 0.0f, 1.0f});
  const vec3_float A3 = cross(f1, vec3_float{1.0f, 0.0f, 0.0f});
  const vec3_float A4 = cross(f1, vec3_float{0.0f, 1.0f, 0.0f});
  const vec3_float A5 = cross(f1, vec3_float{0.0f, 0.0f, 1.0f});
  const vec3_float A6 = cross(f2, vec3_float{1.0f, 0.0f, 0.0f});
  const vec3_float A7 = cross(f2, vec3_float{0.0f, 1.0f, 0.0f});
  const vec3_float A8 = cross(f2, vec3_float{0.0f, 0.0f, 1.0f});
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
  const vec3_float N = cross(f0, f1);
  const float Rn = dot(_t7, abs(N));
  const float dist = dot(_t9, N);
  if (((Rn < dist) || (dist < (-Rn)))) {
    return false;
  }
  return true;
}
bool intersects_Ray_AABB(const Ray* __restrict__ r, const AABB* __restrict__ b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return ((extract.low < (*r).tmax) & (0.0f < extract.high));
  }
  return false;
}
bool intersects_Ray_MaterialSphere(const Ray* __restrict__ r, const MaterialSphere* __restrict__ ms) {
  return intersects_Ray_Sphere(r, (&(*ms).s));
}
bool intersects_Ray_Triangle(const Ray* __restrict__ ray, const Triangle* __restrict__ tri) {
  return intersectsp_ray_tri(ray, tri).has_value();
}
bool tri_tri_axis(const vec3_float A, const vec3_float a0, const vec3_float a1, const vec3_float a2, const vec3_float b0, const vec3_float b1, const vec3_float b2) {
  if (reduce_and(A == vec3_float{0.0f})) {
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
  const vec3_float _t0 = (*a).p1;
  const vec3_float _t1 = (*a).p0;
  const vec3_float _t2 = (_t0 - _t1);
  const vec3_float _t3 = (*a).p2;
  const vec3_float _t5 = (_t3 - _t0);
  const vec3_float _t8 = (_t1 - _t3);
  const vec3_float _t9 = (*b).p1;
  const vec3_float _t10 = (*b).p0;
  const vec3_float _t11 = (_t9 - _t10);
  const vec3_float _t12 = (*b).p2;
  const vec3_float _t14 = (_t12 - _t9);
  const vec3_float _t17 = (_t10 - _t12);
  const vec3_float N0 = cross(_t2, _t5);
  const vec3_float N1 = cross(_t11, _t14);
  if ((!tri_tri_axis(N0, _t1, _t0, _t3, _t10, _t9, _t12))) {
    return false;
  }
  if ((!tri_tri_axis(N1, _t1, _t0, _t3, _t10, _t9, _t12))) {
    return false;
  }
  const vec3_float E00 = cross(_t2, _t11);
  const vec3_float E01 = cross(_t2, _t14);
  const vec3_float E02 = cross(_t2, _t17);
  const vec3_float E10 = cross(_t5, _t11);
  const vec3_float E11 = cross(_t5, _t14);
  const vec3_float E12 = cross(_t5, _t17);
  const vec3_float E20 = cross(_t8, _t11);
  const vec3_float E21 = cross(_t8, _t14);
  const vec3_float E22 = cross(_t8, _t17);
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
float len_squared(const vec3_float v) {
  return reduce_add((v * v));
}
vec3_float linear_to_gamma_v(const vec3_float l) {
  return vec3_float{linear_to_gamma_f(l[0]), linear_to_gamma_f(l[1]), linear_to_gamma_f(l[2])};
}
bool near_zero(const vec3_float v) {
  return (((abs(v[0]) < 0.00000001f) & (abs(v[1]) < 0.00000001f)) & (abs(v[2]) < 0.00000001f));
}
vec3_float pixel(const int32_t i, const int32_t j, const Camera* __restrict__ c, const Spheres* __restrict__ spheres) {
  return (_traverse_array1(i, j, c, spheres) / vec3_float{(float)((*c).samples_per_pixel)});
}
float random_float(const float low, const float high) {
  return (low + ((high - low) * random_float<float>()));
}
vec3_float random_on_hemisphere(const vec3_float normal) {
  const vec3_float on_unit_sphere = random_unit_vector();
  if ((0.0f < dot(on_unit_sphere, normal))) {
    return on_unit_sphere;
  } else {
    return (-on_unit_sphere);
  }
}
vec3_float random_vec3f() {
  const float _t0 = random_float<float>();
  return vec3_float{_t0, _t0, _t0};
}
vec3_float random_vec3f_in(const float low, const float high) {
  return vec3_float{(low + ((high - low) * random_float<float>())), (low + ((high - low) * random_float<float>())), (low + ((high - low) * random_float<float>()))};
}
vec3_float reflect(const vec3_float v, const vec3_float n) {
  return (v - (vec3_float{(2.0f * dot(v, n))} * n));
}
std::optional<MaterialSphere> trace(const Ray* __restrict__ r, const Spheres* __restrict__ spheres) {
  return _traverse_tree0(r, spheres);
}
vec3_float unit_vector(const vec3_float v) {
  return (v / vec3_float{norm(v)});
}
