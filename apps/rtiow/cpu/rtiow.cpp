#include "apps/rtiow/cpu/rtiow.h"
Point ClosestPtPointAABB(const Point* pt, const AABB* a) {
  return Point{.vec=min(max((*pt).vec, (*a).low), (*a).high)};
}
float SqDistPointAABB(const Point* pt, const AABB* a) {
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
  const float r = sqrtf(random_float<float>());
  const float theta = ((2.0f * 3.14159274f) * random_float<float>());
  return vec3_float{(r * cosf(theta)), (r * sinf(theta)), 0.0f};
}
vec3_float defocus_disk_sample(const vec3_float center, const vec3_float defocus_disk_u, const vec3_float defocus_disk_v) {
  const vec3_float p = random_in_unit_disk();
  return ((center + (vec3_float{p[0]} * defocus_disk_u)) + (vec3_float{p[1]} * defocus_disk_v));
}
Ray build_ray(const int32_t i, const int32_t j, const Camera* cam) {
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
  const vec3_float offset = vec3_float{(random_float<float>() - 0.50000000f), (random_float<float>() - 0.50000000f), 0.0f};
  const vec3_float pixel_sample = ((pixel00_loc + (vec3_float{((float)(i) + offset[0])} * pixel_delta_u)) + (vec3_float{((float)(j) + offset[1])} * pixel_delta_v));
  vec3_float ray_origin = camera_center;
  if ((0.0f < (*cam).defocus_angle)) {
    ray_origin = defocus_disk_sample(camera_center, defocus_disk_u, defocus_disk_v);
  }
  const vec3_float ray_direction = (pixel_sample - ray_origin);
  return Ray{.o=ray_origin, .d=ray_direction, .tmax=std::numeric_limits<float>::infinity()};
}
std::optional<FInterval> intersectsp_ray_aabb(const Ray* r, const AABB* b) {
  const vec3_float invDir = (vec3_float{1.0f} / (*r).d);
  const vec3_bool dirIsNeg = (invDir < vec3_float{0.0f});
  const vec3_float low_parts = select(dirIsNeg, (*b).high, (*b).low);
  const vec3_float high_parts = select(dirIsNeg, (*b).low, (*b).high);
  vec3_float tMin = ((low_parts - (*r).o) * invDir);
  vec3_float tMax = ((high_parts - (*r).o) * invDir);
  tMax*= (1.0f + (2.0f * (((float)(3) * 0.00000006f) / (1.0f - ((float)(3) * 0.00000006f)))));
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
float distmax_Ray_AABB(const Ray* r, const AABB* b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.high;
  }
  return std::numeric_limits<float>::infinity();
}
float distmin_Ray_AABB(const Ray* r, const AABB* b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.low;
  }
  return (-std::numeric_limits<float>::infinity());
}
std::optional<FInterval> intersectsp_ray_sphere(const Ray* r, const Sphere* s) {
  const vec3_float oc = ((*s).center - (*r).o);
  const float a = reduce_add(((*r).d * (*r).d));
  const float h = dot((*r).d, oc);
  const float c = (reduce_add((oc * oc)) - ((*s).radius * (*s).radius));
  const float disc = ((h * h) - (a * c));
  if ((disc < 0.0f)) {
    return std::nullopt;
  }
  const float sqrtd = sqrtf(disc);
  const float root0 = ((h - sqrtd) / a);
  const float root1 = ((h + sqrtd) / a);
  const FInterval interval = FInterval{.low=min(root0, root1), .high=max(root0, root1)};
  return (std::optional<FInterval>)(interval);
}
float distmin_Ray_Sphere(const Ray* r, const Sphere* s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.low;
  }
  return (-std::numeric_limits<float>::infinity());
}
bool intersects_Ray_AABB(const Ray* r, const AABB* b) {
  const std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return ((extract.low < (*r).tmax) & (0.0f < extract.high));
  }
  return false;
}
bool intersects_Ray_Sphere(const Ray* ray, const Sphere* s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(ray, s);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return ((extract.low < (*ray).tmax) & (0.0f < extract.high));
  }
  return false;
}
std::optional<MaterialSphere> _traverse_tree0(const Ray* r, const Spheres* spheres) {
  std::tuple<float, MaterialSphere> _best0 = std::tuple<float, MaterialSphere>{std::numeric_limits<float>::infinity(), MaterialSphere{}};
  int32_t _queue_count0 = 1;
  uint32_t* _queue0 = reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t) * 64));
  _queue0[0] = 0;
  do {
    _queue_count0-= 1;
    const uint32_t index = _queue0[_queue_count0];
    const AABB _lv2 = AABB{.low=fadd_rd((*spheres).wlow, vec3_float{fmul_rd((float)((((*spheres).nodes[index].q_min >> 20) & 1023)), (*spheres).bins[0]), fmul_rd((float)((((*spheres).nodes[index].q_min >> 10) & 1023)), (*spheres).bins[1]), fmul_rd((float)((((*spheres).nodes[index].q_min >> 0) & 1023)), (*spheres).bins[2])}), .high=fsub_ru((*spheres).whigh, vec3_float{fmul_rd((float)((((*spheres).nodes[index].q_max >> 20) & 1023)), (*spheres).bins[0]), fmul_rd((float)((((*spheres).nodes[index].q_max >> 10) & 1023)), (*spheres).bins[1]), fmul_rd((float)((((*spheres).nodes[index].q_max >> 0) & 1023)), (*spheres).bins[2])})};
    if (intersects_Ray_AABB(r, (&_lv2))) {
      const AABB _lv1 = AABB{.low=fadd_rd((*spheres).wlow, vec3_float{fmul_rd((float)((((*spheres).nodes[index].q_min >> 20) & 1023)), (*spheres).bins[0]), fmul_rd((float)((((*spheres).nodes[index].q_min >> 10) & 1023)), (*spheres).bins[1]), fmul_rd((float)((((*spheres).nodes[index].q_min >> 0) & 1023)), (*spheres).bins[2])}), .high=fsub_ru((*spheres).whigh, vec3_float{fmul_rd((float)((((*spheres).nodes[index].q_max >> 20) & 1023)), (*spheres).bins[0]), fmul_rd((float)((((*spheres).nodes[index].q_max >> 10) & 1023)), (*spheres).bins[1]), fmul_rd((float)((((*spheres).nodes[index].q_max >> 0) & 1023)), (*spheres).bins[2])})};
      if ((0.00100000f < distmax_Ray_AABB(r, (&_lv1)))) {
        const AABB _lv0 = AABB{.low=fadd_rd((*spheres).wlow, vec3_float{fmul_rd((float)((((*spheres).nodes[index].q_min >> 20) & 1023)), (*spheres).bins[0]), fmul_rd((float)((((*spheres).nodes[index].q_min >> 10) & 1023)), (*spheres).bins[1]), fmul_rd((float)((((*spheres).nodes[index].q_min >> 0) & 1023)), (*spheres).bins[2])}), .high=fsub_ru((*spheres).whigh, vec3_float{fmul_rd((float)((((*spheres).nodes[index].q_max >> 20) & 1023)), (*spheres).bins[0]), fmul_rd((float)((((*spheres).nodes[index].q_max >> 10) & 1023)), (*spheres).bins[1]), fmul_rd((float)((((*spheres).nodes[index].q_max >> 0) & 1023)), (*spheres).bins[2])})};
        if ((distmin_Ray_AABB(r, (&_lv0)) < std::get<0>(_best0))) {
          if ((*spheres).nodes[index].nprims == 0) {
            _queue0[_queue_count0] = (index + 1);
            _queue0[(_queue_count0 + 1)] = (index + reinterpret<Arm_Interior>((*spheres).nodes[index].split0on_nprims).offset);
            _queue_count0+= 2;
          } else {
            for (uint32_t _idx0 = reinterpret<Arm_Leaf>((*spheres).nodes[index].split0on_nprims).poffset; _idx0 < (reinterpret<Arm_Leaf>((*spheres).nodes[index].split0on_nprims).poffset + (uint32_t)((*spheres).nodes[index].nprims)); _idx0 += 1) {
              if (intersects_Ray_Sphere(r, (&(*spheres).primitives[_idx0].s))) {
                if ((0.00100000f < distmin_Ray_Sphere(r, (&(*spheres).primitives[_idx0].s)))) {
                  if ((distmin_Ray_Sphere(r, (&(*spheres).primitives[_idx0].s)) < std::get<0>(_best0))) {
                    _best0 = argmin<float, MaterialSphere>(_best0, std::tuple<float, MaterialSphere>{distmin_Ray_Sphere(r, (&(*spheres).primitives[_idx0].s)), (*spheres).primitives[_idx0]});
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
Hit_record get_hit_record(const Ray* r, const Sphere* s) {
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
  const float s = (((x1 * x1) + (x2 * x2)) + 0.00000001f);
  const float factor = sqrtf((2.0f / s));
  const float x = (factor * x1);
  const float y = (factor * x2);
  const float z = (1.0f - (2.0f * s));
  const float len = sqrtf((((x * x) + (y * y)) + (z * z)));
  return vec3_float{(x / len), (y / len), (z / len)};
}
float reflectance(const float cos_theta, const float refract_idx) {
  const float r0 = ((1.0f - refract_idx) / (1.0f + refract_idx));
  const float r1 = (r0 * r0);
  return (r1 + ((1.0f - r1) * powf((1.0f - cos_theta), 5.0f)));
}
vec3_float refract(const vec3_float uv, const vec3_float n, const float etai_over_etat) {
  const float cos_theta = min(dot((-uv), n), 1.0f);
  const vec3_float r_out_perp = (vec3_float{etai_over_etat} * (uv + (vec3_float{cos_theta} * n)));
  const vec3_float r_out_parallel = (vec3_float{(-sqrtf(abs((1.0f - reduce_add((r_out_perp * r_out_perp))))))} * n);
  return (r_out_perp + r_out_parallel);
}
Scatter_record scatter(const Ray* ray, const MaterialSphere* ms) {
  const Hit_record hit = get_hit_record(ray, (&(*ms).s));
  if ((*ms).material == 0) {
    vec3_float scatter_dir = (hit.normal + random_unit_vector());
    if ((((abs(scatter_dir[0]) < 0.00000001f) & (abs(scatter_dir[1]) < 0.00000001f)) & (abs(scatter_dir[2]) < 0.00000001f))) {
      scatter_dir = hit.normal;
    }
    const Ray l_scattered = Ray{.o=hit.p, .d=scatter_dir, .tmax=std::numeric_limits<float>::infinity()};
    return Scatter_record{.attenuation=(*ms).albedo, .ray=l_scattered, .hit=true};
  } else if ((*ms).material == 1) {
    const vec3_float ref = ((*ray).d - (vec3_float{(2.0f * dot((*ray).d, hit.normal))} * hit.normal));
    const vec3_float reflected = ((ref / vec3_float{norm(ref)}) + (vec3_float{(*ms).fuzz} * random_unit_vector()));
    const Ray m_scattered = Ray{.o=hit.p, .d=reflected, .tmax=std::numeric_limits<float>::infinity()};
    return Scatter_record{.attenuation=(*ms).albedo, .ray=m_scattered, .hit=true};
  } else {
    const float ri = (hit.front_face ? (1.0f / (*ms).fuzz) : (*ms).fuzz);
    const vec3_float unit_dir = ((*ray).d / vec3_float{norm((*ray).d)});
    const float cos_theta = min(dot((-unit_dir), hit.normal), 1.0f);
    const float sin_theta = sqrtf((1.0f - (cos_theta * cos_theta)));
    const bool cannot_refract = ((1.0f < (ri * sin_theta)) | (random_float<float>() < reflectance(cos_theta, ri)));
    const vec3_float direction = (cannot_refract ? (unit_dir - (vec3_float{(2.0f * dot(unit_dir, hit.normal))} * hit.normal)) : refract(unit_dir, hit.normal, ri));
    const Ray d_scattered = Ray{.o=hit.p, .d=direction, .tmax=std::numeric_limits<float>::infinity()};
    return Scatter_record{.attenuation=vec3_float{1.0f, 1.0f, 1.0f}, .ray=d_scattered, .hit=true};
  }
}
vec3_float sample(const Ray* r, const int32_t depth, const vec3_float mult, const Spheres* spheres) {
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
  const vec3_float unit_direction = ((*r).d / vec3_float{norm((*r).d)});
  const float a = (0.50000000f * (unit_direction[1] + 1.0f));
  return (mult * ((vec3_float{(1.0f - a)} * vec3_float{1.0f, 1.0f, 1.0f}) + (vec3_float{a} * vec3_float{0.50000000f, 0.69999999f, 1.0f})));
}
vec3_float _traverse_array1(const int32_t i, const int32_t j, const Camera* c, const Spheres* spheres) {
  vec3_float _alloc1 = vec3_float{0.0f};
  for (uint32_t _i0 = 0; _i0 < (*c).samples_per_pixel; _i0 += 1) {
    const Ray _lv0 = build_ray(i, j, c);
    _alloc1+= sample((&_lv0), (*c).max_depth, vec3_float{1.0f, 1.0f, 1.0f}, spheres);
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
vec3_int32_t** _traverse_array0(const Camera* c, const int32_t height, const Spheres* spheres) {
  int32_t* _alloc0 = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * ((height * (*c).width) * 3)));
  for (int32_t _i0 = 0; _i0 < height; _i0 += 1) {
    for (int32_t _i1 = 0; _i1 < (*c).width; _i1 += 1) {
      const vec3_int32_t __temp = to_rgb((_traverse_array1(_i1, _i0, c, spheres) / vec3_float{(float)((*c).samples_per_pixel)}));
      _alloc0[(((_i0 * (*c).width) + _i1) * 3)] = __temp[0];
      _alloc0[((((_i0 * (*c).width) + _i1) * 3) + 1)] = __temp[1];
      _alloc0[((((_i0 * (*c).width) + _i1) * 3) + 2)] = __temp[2];
    }
  }
  return reinterpret_cast<vec3_int32_t**>(_alloc0);
}
vec3_float at(const Ray* r, const float t) {
  return ((*r).o + (vec3_float{t} * (*r).d));
}
bool axis(const vec3_float A, const vec3_float extents, const vec3_float v0, const vec3_float v1, const vec3_float v2) {
  const float R = dot(extents, abs(A));
  const vec3_float P = vec3_float{dot(v0, A), dot(v1, A), dot(v2, A)};
  return reduce_and(((P <= vec3_float{R}) & (vec3_float{(-R)} <= P)));
}
Sphere bounding_sphere(const Sphere* a, const Sphere* b) {
  const vec3_float d = ((*b).center - (*a).center);
  const float dist_sq = reduce_add((d * d));
  const float dist = sqrtf(dist_sq);
  if (((dist + (*b).radius) <= (*a).radius)) {
    return (*a);
  } else if (((dist + (*a).radius) <= (*b).radius)) {
    return (*b);
  }
  const float new_radius = (0.50000000f * ((dist + (*a).radius) + (*b).radius));
  const vec3_float direction = ((0.0f < dist) ? (d / vec3_float{dist}) : vec3_float{1.0f, 0.0f, 0.0f});
  const vec3_float new_center = ((*a).center + (direction * vec3_float{(new_radius - (*a).radius)}));
  return Sphere{.center=new_center, .radius=new_radius};
}
vec3_float build_bins_inverse(const vec3_float low, const vec3_float high) {
  const vec3_float L1 = vec3_float{fsub_ru(high[0], low[0]), fsub_ru(high[1], low[1]), fsub_ru(high[2], low[2])};
  const vec3_float L2 = select((L1 <= vec3_float{0.0f}), vec3_float{1.0f}, L1);
  return vec3_float{fdiv_rd(1023.0f, L2[0]), fdiv_rd(1023.0f, L2[1]), fdiv_rd(1023.0f, L2[2])};
}
vec3_float build_bins(const vec3_float low, const vec3_float high) {
  const vec3_float bins_inverse = build_bins_inverse(low, high);
  return vec3_float{frcp_rd(bins_inverse[0]), frcp_rd(bins_inverse[1]), frcp_rd(bins_inverse[2])};
}
int32_t quantize(const vec3_float current, const vec3_float world, const vec3_float bin_inverse) {
  const uint32_t x = (uint32_t)(floor(fmul_rd(fsub_rd(current[0], world[0]), bin_inverse[0])));
  const uint32_t y = (uint32_t)(floor(fmul_rd(fsub_rd(current[1], world[1]), bin_inverse[1])));
  const uint32_t z = (uint32_t)(floor(fmul_rd(fsub_rd(current[2], world[2]), bin_inverse[2])));
  return (int32_t)((((x << 20) | (y << 10)) | z));
}
uint32_t rec_build_spheres(const BVH* node, Spheres* ST, size_t* nodes_index, size_t* primitives_index) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index)+= 1;
      if (this_index == 0) {
        (*ST).wlow = node.low;
        (*ST).whigh = node.high;
        (*ST).bins_inv = build_bins_inverse(node.low, node.high);
        (*ST).bins = build_bins(node.low, node.high);
      }
      (*ST).nodes[this_index].q_min = quantize(node.low, (*ST).wlow, (*ST).bins_inv);
      (*ST).nodes[this_index].q_max = quantize(node.high, (*ST).whigh, (*ST).bins_inv);
      (*ST).nodes[this_index].nprims = 0;
      const uint32_t left_index = rec_build_spheres(node.left, ST, nodes_index, primitives_index);
      const uint32_t right_index = rec_build_spheres(node.right, ST, nodes_index, primitives_index);
      reinterpret_cast<Arm_Interior *>(&(*ST).nodes[this_index].split0on_nprims)->offset = (right_index - this_index);
      return this_index;
    },
    [&](const Leaf& node) {
      const size_t this_index = (*nodes_index);
      (*nodes_index)+= 1;
      (*ST).nodes[this_index].q_min = quantize(node.low, (*ST).wlow, (*ST).bins_inv);
      (*ST).nodes[this_index].q_max = quantize(node.high, (*ST).whigh, (*ST).bins_inv);
      (*ST).nodes[this_index].nprims = node.nprims;
      reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)->poffset = (*primitives_index);
      for (uint16_t __p = 0; __p < node.nprims; __p += 1) {
        (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
      }
      (*primitives_index)+= node.nprims;
      return this_index;
    }
  }, *node);
}
void rec_count_spheres(const BVH* node, Spheres* ST) {
  return std::visit(overloaded{
    [&](const Interior& node) {
      rec_count_spheres(node.left, ST);
      rec_count_spheres(node.right, ST);
    },
    [&](const Leaf& node) {
      (*ST).node_count+= 1;
    }
  }, *node);
}
Spheres build_spheres(const BVH* CT) {
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
std::tuple<Point, Point> closestPointonTriangle(const Point* pt, const Triangle* tri) {
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
  const float vc = ((d1 * d4) - (d3 * d2));
  if ((vc <= 0.0f)) {
    if ((0.0f <= d1)) {
      if ((d3 <= 0.0f)) {
        const float v0 = (d1 / (d1 - d3));
        return std::tuple<Point, Point>{Point{.vec=(a + (vec3_float{v0} * ab))}, Point{.vec=vec3_float{(1.0f - v0), v0, 0.0f}}};
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
  const float vb = ((d5 * d2) - (d1 * d6));
  if ((vb <= 0.0f)) {
    if ((0.0f <= d2)) {
      if ((d6 <= 0.0f)) {
        const float w0 = (d2 / (d2 - d6));
        return std::tuple<Point, Point>{Point{.vec=(a + (vec3_float{w0} * ac))}, Point{.vec=vec3_float{(1.0f - w0), 0.0f, w0}}};
      }
    }
  }
  const float va = ((d3 * d6) - (d5 * d4));
  if ((va <= 0.0f)) {
    if ((0.0f <= (d4 - d3))) {
      if ((0.0f <= (d5 - d6))) {
        const float w1 = ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        return std::tuple<Point, Point>{Point{.vec=(b + (vec3_float{w1} * (c - b)))}, Point{.vec=vec3_float{0.0f, (1.0f - w1), w1}}};
      }
    }
  }
  const float denom = (1.0f / ((va + vb) + vc));
  const float v = (vb * denom);
  const float w = (vc * denom);
  const float u = (va * denom);
  return std::tuple<Point, Point>{Point{.vec=((a + (ab * vec3_float{v})) + (ac * vec3_float{w}))}, Point{.vec=vec3_float{u, v, w}}};
}
vec3_float cross_(const vec3_float v0, const vec3_float v1) {
  return vec3_float{__prod_diff_f32(v0[1], v1[2], v0[2], v1[1]), __prod_diff_f32(v0[2], v1[0], v0[0], v1[2]), __prod_diff_f32(v0[0], v1[1], v0[1], v1[0])};
}
float degrees_to_radians(const float degrees) {
  return ((degrees * 3.14159274f) / 180.0f);
}
vec3_float dequantize(const int32_t v, const vec3_float bins) {
  return vec3_float{fmul_rd((float)(((v >> 20) & 1023)), bins[0]), fmul_rd((float)(((v >> 10) & 1023)), bins[1]), fmul_rd((float)(((v >> 0) & 1023)), bins[2])};
}
float distmax_Ray_Sphere(const Ray* r, const Sphere* s) {
  const std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
  if (interval.has_value()) {
    const FInterval extract = *interval;
    return extract.high;
  }
  return std::numeric_limits<float>::infinity();
}
float distmax_Ray_MaterialSphere(const Ray* r, const MaterialSphere* ms) {
  return distmax_Ray_Sphere(r, (&(*ms).s));
}
std::optional<TriangleIntersection> intersectsp_ray_tri(const Ray* ray, const Triangle* tri) {
  if (reduce_add((cross_(((*tri).p2 - (*tri).p0), ((*tri).p1 - (*tri).p0)) * cross_(((*tri).p2 - (*tri).p0), ((*tri).p1 - (*tri).p0)))) == 0.0f) {
    return std::nullopt;
  }
  vec3_float p0t = ((*tri).p0 - (*ray).o);
  vec3_float p1t = ((*tri).p1 - (*ray).o);
  vec3_float p2t = ((*tri).p2 - (*ray).o);
  const uint32_t kz = reduce_idxmax(abs((*ray).d));
  const uint32_t kx = ((kz + 1) % 3);
  const uint32_t ky = ((kx + 1) % 3);
  const vec3_float d = shuffle((*ray).d, {kx, ky, kz});
  p0t = shuffle(p0t, {kx, ky, kz});
  p1t = shuffle(p1t, {kx, ky, kz});
  p2t = shuffle(p2t, {kx, ky, kz});
  const float Sx = ((-d[0]) / d[2]);
  const float Sy = ((-d[1]) / d[2]);
  const float Sz = (1.0f / d[2]);
  p0t[0]+= (Sx * p0t[2]);
  p0t[1]+= (Sy * p0t[2]);
  p1t[0]+= (Sx * p1t[2]);
  p1t[1]+= (Sy * p1t[2]);
  p2t[0]+= (Sx * p2t[2]);
  p2t[1]+= (Sy * p2t[2]);
  const float e0 = __prod_diff_f32(p1t[0], p2t[1], p1t[1], p2t[0]);
  const float e1 = __prod_diff_f32(p2t[0], p0t[1], p2t[1], p0t[0]);
  const float e2 = __prod_diff_f32(p0t[0], p1t[1], p0t[1], p1t[0]);
  if ((((e0 < 0.0f) || (e1 < 0.0f)) || (e2 < 0.0f))) {
    if ((((0.0f < e0) || (0.0f < e1)) || (0.0f < e2))) {
      return std::nullopt;
    }
  }
  const float det = ((e0 + e1) + e2);
  if (det == 0.0f) {
    return std::nullopt;
  }
  p0t[2]*= Sz;
  p1t[2]*= Sz;
  p2t[2]*= Sz;
  const float tScaled = (((e0 * p0t[2]) + (e1 * p1t[2])) + (e2 * p2t[2]));
  if (((det < 0.0f) && ((0.0f <= tScaled) || (tScaled < ((*ray).tmax * det))))) {
    return std::nullopt;
  } else if (((0.0f < det) && ((tScaled <= 0.0f) || (((*ray).tmax * det) < tScaled)))) {
    return std::nullopt;
  }
  const float invDet = (1.0f / det);
  const float b0 = (e0 * invDet);
  const float b1 = (e1 * invDet);
  const float b2 = (e2 * invDet);
  const float t = (tScaled * invDet);
  const float maxZt = reduce_max(abs(vec3_float{p0t[2], p1t[2], p2t[2]}));
  const float deltaZ = ((((float)(3) * 0.00000006f) / (1.0f - ((float)(3) * 0.00000006f))) * maxZt);
  const float maxXt = reduce_max(abs(vec3_float{p0t[0], p1t[0], p2t[0]}));
  const float maxYt = reduce_max(abs(vec3_float{p0t[1], p1t[1], p2t[1]}));
  const float deltaX = ((((float)(5) * 0.00000006f) / (1.0f - ((float)(5) * 0.00000006f))) * (maxXt + maxZt));
  const float deltaY = ((((float)(5) * 0.00000006f) / (1.0f - ((float)(5) * 0.00000006f))) * (maxYt + maxZt));
  const float deltaE = (2.0f * (((((((float)(2) * 0.00000006f) / (1.0f - ((float)(2) * 0.00000006f))) * maxXt) * maxYt) + (deltaY * maxXt)) + (deltaX * maxYt)));
  const float maxE = reduce_max(abs(vec3_float{e0, e1, e2}));
  const float deltaT = ((3.0f * (((((((float)(3) * 0.00000006f) / (1.0f - ((float)(3) * 0.00000006f))) * maxE) * maxZt) + (deltaE * maxZt)) + (deltaZ * maxE))) * abs(invDet));
  if ((t <= deltaT)) {
    return std::nullopt;
  }
  return (std::optional<TriangleIntersection>)(TriangleIntersection{.b0=b0, .b1=b1, .b2=b2, .t=t});
}
float distmax_Ray_Triangle(const Ray* ray, const Triangle* tri) {
  const std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    const TriangleIntersection isect_ = *isect;
    return isect_.t;
  } else {
    return (-std::numeric_limits<float>::infinity());
  }
}
float distmin_Point_AABB(const Point* pt, const AABB* a) {
  return sqrtf(SqDistPointAABB(pt, a));
}
float distmin_Point_Triangle(const Point* p, const Triangle* tri) {
  const std::tuple<Point, Point> pts = closestPointonTriangle(p, tri);
  return norm(((*p).vec - std::get<0>(pts).vec));
}
float distmin_Ray_MaterialSphere(const Ray* r, const MaterialSphere* ms) {
  return distmin_Ray_Sphere(r, (&(*ms).s));
}
float distmin_Ray_Triangle(const Ray* ray, const Triangle* tri) {
  const std::optional<TriangleIntersection> isect = intersectsp_ray_tri(ray, tri);
  if (isect.has_value()) {
    const TriangleIntersection isect_ = *isect;
    return isect_.t;
  } else {
    return std::numeric_limits<float>::infinity();
  }
}
float gamma(const int32_t n) {
  return (((float)(n) * 0.00000006f) / (1.0f - ((float)(n) * 0.00000006f)));
}
vec3_int32_t** image(const Camera* c, const Spheres* spheres) {
  int32_t height = (int32_t)(((float)((*c).width) / (*c).aspect_ratio));
  height = ((height < 1) ? 1 : height);
  return _traverse_array0(c, height, spheres);
}
bool intersects_AABB_AABB(const AABB* a, const AABB* b) {
  const vec3_float low = max((*a).low, (*b).low);
  const vec3_float high = min((*a).high, (*b).high);
  return reduce_and((low <= high));
}
bool intersects_AABB_Sphere(const AABB* a, const Sphere* s) {
  const vec3_float closest = max((*a).low, min((*s).center, (*a).high));
  const vec3_float d = ((*s).center - closest);
  return (dot(d, d) <= ((*s).radius * (*s).radius));
}
bool intersects_AABB_Triangle(const AABB* a, const Triangle* b) {
  const vec3_float center = (((*a).low + (*a).high) * vec3_float{0.50000000f});
  const vec3_float extents = (((*a).high - (*a).low) * vec3_float{0.50000000f});
  const vec3_float v0 = ((*b).p0 - center);
  const vec3_float v1 = ((*b).p1 - center);
  const vec3_float v2 = ((*b).p2 - center);
  const vec3_float f0 = (v1 - v0);
  const vec3_float f1 = (v2 - v1);
  const vec3_float f2 = (v0 - v2);
  const vec3_float tri_min = min(min(v0, v1), v2);
  const vec3_float tri_max = max(max(v0, v1), v2);
  if ((!reduce_and(((tri_min <= extents) & ((-extents) <= tri_max))))) {
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
  if ((!axis(A0, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A1, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A2, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A3, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A4, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A5, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A6, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A7, extents, v0, v1, v2))) {
    return false;
  }
  if ((!axis(A8, extents, v0, v1, v2))) {
    return false;
  }
  const vec3_float N = cross(f0, f1);
  const float Rn = dot(extents, abs(N));
  const float dist = dot(v0, N);
  if (((Rn < dist) || (dist < (-Rn)))) {
    return false;
  }
  return true;
}
bool intersects_Ray_MaterialSphere(const Ray* r, const MaterialSphere* ms) {
  return intersects_Ray_Sphere(r, (&(*ms).s));
}
bool intersects_Ray_Triangle(const Ray* ray, const Triangle* tri) {
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
bool intersects_Triangle_Triangle(const Triangle* a, const Triangle* b) {
  const vec3_float eA0 = ((*a).p1 - (*a).p0);
  const vec3_float eA1 = ((*a).p2 - (*a).p1);
  const vec3_float eA2 = ((*a).p0 - (*a).p2);
  const vec3_float eB0 = ((*b).p1 - (*b).p0);
  const vec3_float eB1 = ((*b).p2 - (*b).p1);
  const vec3_float eB2 = ((*b).p0 - (*b).p2);
  const vec3_float N0 = cross(eA0, eA1);
  const vec3_float N1 = cross(eB0, eB1);
  const vec3_float a0 = (*a).p0;
  const vec3_float a1 = (*a).p1;
  const vec3_float a2 = (*a).p2;
  const vec3_float b0 = (*b).p0;
  const vec3_float b1 = (*b).p1;
  const vec3_float b2 = (*b).p2;
  if ((!tri_tri_axis(N0, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(N1, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  const vec3_float E00 = cross(eA0, eB0);
  const vec3_float E01 = cross(eA0, eB1);
  const vec3_float E02 = cross(eA0, eB2);
  const vec3_float E10 = cross(eA1, eB0);
  const vec3_float E11 = cross(eA1, eB1);
  const vec3_float E12 = cross(eA1, eB2);
  const vec3_float E20 = cross(eA2, eB0);
  const vec3_float E21 = cross(eA2, eB1);
  const vec3_float E22 = cross(eA2, eB2);
  if ((!tri_tri_axis(E00, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E01, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E02, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E10, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E11, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E12, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E20, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E21, a0, a1, a2, b0, b1, b2))) {
    return false;
  }
  if ((!tri_tri_axis(E22, a0, a1, a2, b0, b1, b2))) {
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
vec3_float pixel(const int32_t i, const int32_t j, const Camera* c, const Spheres* spheres) {
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
  return vec3_float{random_float<float>(), random_float<float>(), random_float<float>()};
}
vec3_float random_vec3f_in(const float low, const float high) {
  return vec3_float{(low + ((high - low) * random_float<float>())), (low + ((high - low) * random_float<float>())), (low + ((high - low) * random_float<float>()))};
}
vec3_float reflect(const vec3_float v, const vec3_float n) {
  return (v - (vec3_float{(2.0f * dot(v, n))} * n));
}
vec3_float sample_square() {
  return vec3_float{(random_float<float>() - 0.50000000f), (random_float<float>() - 0.50000000f), 0.0f};
}
std::optional<MaterialSphere> trace(const Ray* r, const Spheres* spheres) {
  return _traverse_tree0(r, spheres);
}
vec3_float unit_vector(const vec3_float v) {
  return (v / vec3_float{norm(v)});
}
