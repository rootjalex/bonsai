#include "helpers.h"

#include <stdio.h>

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/universal_vector.h>

#include <variant>

struct Interior;
struct Leaf;
using BVH = std::variant<Interior, Leaf>;

struct AABB {
    float3 low;
    float3 high;
};

struct Camera {
    float aspect_ratio;
    int32_t width;
    uint32_t samples_per_pixel = 100u;
    int32_t max_depth = 10;
    float vfov = (float)90;
    float3 lookfrom = float3{0, 0, 0};
    float3 lookat = float3{0, 0, (float)-1};
    float3 vup = float3{0, 1, 0};
    float defocus_angle = 0;
    float focus_dist = (float)10;
};

struct FInterval {
    float low;
    float high;
};

struct Interior {
    float3 center;
    float radius;
    BVH *left;
    BVH *right;
};

struct Sphere {
    float3 center;
    float radius;
};

struct MaterialSphere {
    Sphere s;
    uint32_t material;
    float3 albedo;
    float fuzz;
};

struct Leaf {
    float3 center;
    float radius;
    uint8_t nprims;
    MaterialSphere *data;
};

struct Point {
    float3 vec;
};

struct Ray {
    float3 o;
    float3 d;
    float tmax = INFINITY;
};

struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
};

struct TriangleIntersection {
    float b0;
    float b1;
    float b2;
    float t;
};

struct Arm_Interior {
    uint16_t offset;
} __attribute__((packed));

struct Arm_Leaf {
    uint16_t poffset;
} __attribute__((packed));

struct Nodes {
    float3 center;
    float radius;
    uint8_t nprims;
    uint8_t pad0;
    uchar2 split0on_nprims;
} __attribute__((packed));

struct Spheres {
    uint32_t primitive_count;
    MaterialSphere *primitives;
    uint32_t node_count;
    Nodes *nodes;
} __attribute__((packed));

struct _ctx0 {
    int32_t height;
    Camera *c;
    Spheres *spheres;
    int32_t *_alloc0;
};

struct Hit_record {
    float3 p;
    float3 normal;
    float t;
    bool front_face;
};

struct Scatter_record {
    float3 attenuation;
    Ray ray;
    bool hit;
};

__device__ float3 random_in_unit_disk(curandState *_rng_state) {
    float _t0 = curand_uniform(_rng_state);
    float _t1 = sqrtf(_t0);
    float _t4 = (((float)2 * (float)3.14159) * _t0);
    return float3{_t1 * cosf(_t4), _t1 * sinf(_t4), 0};
}

__device__ float3 defocus_disk_sample(float3 center, float3 defocus_disk_u,
                                      float3 defocus_disk_v,
                                      curandState *_rng_state) {
    float3 p = random_in_unit_disk(_rng_state);
    return ((center + (make_float3(p.x) * defocus_disk_u)) +
            (make_float3(p.y) * defocus_disk_v));
}

__device__ float3 sample_square(curandState *_rng_state) {
    float _t0 = curand_uniform(_rng_state);
    float _t1 = (curand_uniform(_rng_state) - (float)0.5);
    return float3{_t1, _t1, 0};
}

__device__ Ray build_ray(int32_t i, int32_t j, Camera *cam,
                         curandState *_rng_state) {
    int32_t width = (*cam).width;
    int32_t height = (int32_t)((float)width / (*cam).aspect_ratio);
    height = ((height < 1) ? 1 : height);
    float theta = (((*cam).vfov * (float)3.14159) / (float)180);
    float h = tanf(theta / (float)2);
    float viewport_height = (((float)2 * h) * (*cam).focus_dist);
    float viewport_width = (viewport_height * ((float)width / (float)height));
    float3 camera_center = (*cam).lookfrom;
    float3 w = (((*cam).lookfrom - (*cam).lookat) /
                make_float3(length((*cam).lookfrom - (*cam).lookat)));
    float3 u =
        (cross((*cam).vup, w) / make_float3(length(cross((*cam).vup, w))));
    float3 v = cross(w, u);
    float3 viewport_u = (make_float3(viewport_width) * u);
    float3 viewport_v = (make_float3(viewport_height) * -v);
    float3 pixel_delta_u = (viewport_u / make_float3((float)width));
    float3 pixel_delta_v = (viewport_v / make_float3((float)height));
    float3 viewport_upper_left =
        (((camera_center - (make_float3((*cam).focus_dist) * w)) -
          (viewport_u / make_float3((float)2))) -
         (viewport_v / make_float3((float)2)));
    float3 pixel00_loc =
        (viewport_upper_left +
         (make_float3((float)0.5) * (pixel_delta_u + pixel_delta_v)));
    float defocus_radius =
        ((*cam).focus_dist *
         tanf((((*cam).defocus_angle / (float)2) * (float)3.14159) /
              (float)180));
    float3 defocus_disk_u = (u * make_float3(defocus_radius));
    float3 defocus_disk_v = (v * make_float3(defocus_radius));
    float3 offset = sample_square(_rng_state);
    float3 pixel_sample =
        ((pixel00_loc + (make_float3(((float)i + offset.x)) * pixel_delta_u)) +
         (make_float3(((float)j + offset.y)) * pixel_delta_v));
    float3 ray_origin = camera_center;
    if (0 < (*cam).defocus_angle) {
        ray_origin = defocus_disk_sample(camera_center, defocus_disk_u,
                                         defocus_disk_v, _rng_state);
    }
    float3 ray_direction = (pixel_sample - ray_origin);
    return Ray{ray_origin, ray_direction, INFINITY};
}

__device__ cuda::std::optional<FInterval> intersectsp_ray_sphere(Ray *r,
                                                                 Sphere *s) {
    float3 _t2 = ((*s).center - (*r).o);
    float3 _t3 = (*r).d;
    float a = sum((_t3 * _t3));
    float _t7 = dot(_t3, _t2);
    float _t9 = (*s).radius;
    float _t15 = ((_t7 * _t7) - (a * (sum((_t2 * _t2)) - (_t9 * _t9))));
    if (_t15 < 0) {
        return cuda::std::nullopt;
    }
    float sqrtd = sqrtf(_t15);
    float _t17 = ((_t7 - sqrtd) / a);
    float _t19 = ((_t7 + sqrtd) / a);
    FInterval interval = FInterval{fminf(_t17, _t19), fmaxf(_t17, _t19)};
    return interval;
}

__device__ float distmax_Ray_Sphere(Ray *r, Sphere *s) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return extract.high;
    }
    return INFINITY;
}

__device__ float distmin_Ray_Sphere(Ray *r, Sphere *s) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_sphere(r, s);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return extract.low;
    }
    return -INFINITY;
}

__device__ bool intersects_Ray_Sphere(Ray *ray, Sphere *s) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_sphere(ray, s);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return ((extract.low < (*ray).tmax) & (0 < extract.high));
    }
    return false;
}

__device__ cuda::std::optional<MaterialSphere>
_traverse_tree0(Ray *r, Spheres *spheres) {
    cuda::std::tuple<float, MaterialSphere> _best0 =
        cuda::std::tuple<float, MaterialSphere>{INFINITY, MaterialSphere{}};
    int32_t _queue_count0 = 1;
    uint32_t _queue0[64];
    _queue0[0] = 0u;
    do {
        _queue_count0 -= 1;
        uint32_t index = _queue0[_queue_count0];
        Nodes _t49 = (*spheres).nodes[index];
        Sphere _t54 = Sphere{_t49.center, _t49.radius};
        if (intersects_Ray_Sphere(r, (&_t54))) {
            if ((float)0.001 < distmax_Ray_Sphere(r, (&_t54))) {
                if (distmin_Ray_Sphere(r, (&_t54)) <
                    cuda::std::get<0>(_best0)) {
                    uint8_t _t32 = _t49.nprims;
                    if (_t32 == 0u) {
                        _queue0[_queue_count0] = (index + 1u);
                        _queue0[(_queue_count0 + 1)] =
                            (index + (uint32_t)bonsai_reinterpret<Arm_Interior>(
                                         _t49.split0on_nprims)
                                         .offset);
                        _queue_count0 += 2;
                    } else {
                        uint16_t _t21 =
                            bonsai_reinterpret<Arm_Leaf>(_t49.split0on_nprims)
                                .poffset;
                        for (uint16_t _idx0 = _t21;
                             _idx0 < (_t21 + (uint16_t)_t32); _idx0 += 1u) {
                            MaterialSphere _t16 = (*spheres).primitives[_idx0];
                            if (intersects_Ray_Sphere(r, (&_t16.s))) {
                                float _t14 = distmin_Ray_Sphere(r, (&_t16.s));
                                if ((float)0.001 < _t14) {
                                    if (_t14 < cuda::std::get<0>(_best0)) {
                                        _best0 = argmin(
                                            _best0,
                                            cuda::std::tuple<float,
                                                             MaterialSphere>{
                                                _t14, _t16});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } while (_queue_count0 != 0);
    return ((cuda::std::get<0>(_best0) != INFINITY)
                ? cuda::std::optional<MaterialSphere>{cuda::std::get<1>(_best0)}
                : cuda::std::nullopt);
}

__device__ Hit_record get_hit_record(Ray *r, Sphere *s) {
    float t = distmin_Ray_Sphere(r, s);
    float3 p = ((*r).o + (make_float3(t) * (*r).d));
    float3 outward_normal = ((p - (*s).center) / make_float3((*s).radius));
    bool front_face = (dot((*r).d, outward_normal) < 0);
    float3 normal = (front_face ? outward_normal : -outward_normal);
    Hit_record record = Hit_record{p, normal, t, front_face};
    return record;
}

__device__ float3 random_unit_vector(curandState *_rng_state) {
    float x1 = ((float)-1 + ((1 - (float)-1) * curand_uniform(_rng_state)));
    float x2 = ((float)-1 + ((1 - (float)-1) * curand_uniform(_rng_state)));
    float _t3 = (((x1 * x1) + (x2 * x2)) + (float)1e-08);
    float _t5 = sqrtf((float)2 / _t3);
    float x = (_t5 * x1);
    float y = (_t5 * x2);
    float _t7 = (1 - ((float)2 * _t3));
    float _t13 = sqrtf(((x * x) + (y * y)) + (_t7 * _t7));
    return float3{x / _t13, y / _t13, _t7 / _t13};
}

__device__ float reflectance(float cos_theta, float refract_idx) {
    float _t2 = ((1 - refract_idx) / (1 + refract_idx));
    float r1 = (_t2 * _t2);
    return (r1 + ((1 - r1) * powf(1 - cos_theta, (float)5)));
}

__device__ float3 refract(float3 uv, float3 n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1);
    float3 _t2 =
        (make_float3(etai_over_etat) * (uv + (make_float3(cos_theta) * n)));
    float3 r_out_parallel =
        (make_float3(-sqrtf(fabsf(1 - sum((_t2 * _t2))))) * n);
    return (_t2 + r_out_parallel);
}

__device__ Scatter_record scatter(Ray *ray, MaterialSphere *ms,
                                  curandState *_rng_state) {
    Hit_record _t1 = get_hit_record(ray, (&(*ms).s));
    uint32_t _t20 = (*ms).material;
    if (_t20 == 0u) {
        float3 _t2 = _t1.normal;
        float3 scatter_dir = (_t2 + random_unit_vector(_rng_state));
        if (((fabsf(scatter_dir.x) < (float)1e-08) &
             (fabsf(scatter_dir.y) < (float)1e-08)) &
            (fabsf(scatter_dir.z) < (float)1e-08)) {
            scatter_dir = _t2;
        }
        Ray l_scattered = Ray{_t1.p, scatter_dir, INFINITY};
        return Scatter_record{(*ms).albedo, l_scattered, true};
    } else {
        if (_t20 == 1u) {
            float3 ref = ((*ray).d -
                          (make_float3(((float)2 * dot((*ray).d, _t1.normal))) *
                           _t1.normal));
            float3 reflected =
                ((ref / make_float3(length(ref))) +
                 (make_float3((*ms).fuzz) * random_unit_vector(_rng_state)));
            Ray m_scattered = Ray{_t1.p, reflected, INFINITY};
            return Scatter_record{(*ms).albedo, m_scattered, true};
        } else {
            float _t11 = (*ms).fuzz;
            float ri = (_t1.front_face ? (1 / _t11) : _t11);
            float3 _t14 = ((*ray).d / make_float3(length((*ray).d)));
            float3 _t15 = _t1.normal;
            float cos_theta = fminf(dot(-_t14, _t15), 1);
            float sin_theta = sqrtf(1 - (cos_theta * cos_theta));
            bool cannot_refract =
                ((1 < (ri * sin_theta)) |
                 (curand_uniform(_rng_state) < reflectance(cos_theta, ri)));
            float3 direction =
                (cannot_refract
                     ? (_t14 -
                        (make_float3(((float)2 * dot(_t14, _t15))) * _t15))
                     : refract(_t14, _t15, ri));
            Ray d_scattered = Ray{_t1.p, direction, INFINITY};
            return Scatter_record{make_float3(1, 1, 1), d_scattered, true};
        }
    }
}

__device__ float3 sample(Ray *r, int32_t depth, float3 mult, Spheres *spheres,
                         curandState *_rng_state) {
    Ray S_r = (*r);
    int32_t S_depth = depth;
    float3 S_mult = mult;
    do {
        if (S_depth <= 0) {
            return make_float3(0, 0, 0);
        }
        cuda::std::optional<MaterialSphere> isect =
            _traverse_tree0((&S_r), spheres);
        if (isect.has_value()) {
            MaterialSphere _lv0 = *isect;
            Scatter_record data = scatter((&S_r), (&_lv0), _rng_state);
            if (data.hit) {
                S_r = data.ray;
                S_depth = (S_depth - 1);
                S_mult = (S_mult * data.attenuation);
                continue;
            } else {
                return make_float3(0, 0, 0);
            }
        }
        float3 unit_direction = (S_r.d / make_float3(length(S_r.d)));
        float a = ((float)0.5 * (unit_direction.y + 1));
        return (S_mult *
                ((make_float3((1 - a)) * make_float3(1, 1, 1)) +
                 (make_float3(a) * make_float3((float)0.5, (float)0.7, 1))));
    } while (true);
}

__device__ float3 _traverse_array1(int32_t i, int32_t j, Camera *c,
                                   Spheres *spheres, curandState *_rng_state) {
    float3 _alloc1 = make_float3(0);
    for (uint32_t _i0 = 0u; _i0 < (*c).samples_per_pixel; _i0 += 1u) {
        Ray _lv0 = build_ray(i, j, c, _rng_state);
        _alloc1 += sample((&_lv0), (*c).max_depth, make_float3(1, 1, 1),
                          spheres, _rng_state);
    }
    return _alloc1;
}

__device__ float linear_to_gamma_f(float l) {
    if (0 < l) {
        return sqrtf(l);
    }
    return 0;
}

__device__ int3 to_rgb(float3 v) {
    float3 corrected = float3{linear_to_gamma_f(v.x), linear_to_gamma_f(v.y),
                              linear_to_gamma_f(v.z)};
    return make_int3(
        (make_float3((float)256) *
         min(max(corrected, make_float3(0)), make_float3((float)0.999)))
            .x,
        (make_float3((float)256) *
         min(max(corrected, make_float3(0)), make_float3((float)0.999)))
            .y,
        (make_float3((float)256) *
         min(max(corrected, make_float3(0)), make_float3((float)0.999)))
            .z);
}

__global__ void _parfunc0(_ctx0 ctx0) {
    int32_t tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if ((ctx0.height * (*ctx0.c).width) <= tid) {
        return;
    }
    curandState _rng_state;
    curand_init(tid, 0, 0, &_rng_state);
    int32_t _i0 = (tid / (*ctx0.c).width);
    int32_t _i1 = (tid % (*ctx0.c).width);
    int3 _t0 =
        to_rgb(_traverse_array1(_i1, _i0, ctx0.c, ctx0.spheres, (&_rng_state)) /
               make_float3((float)(*ctx0.c).samples_per_pixel));
    ctx0._alloc0[(((_i0 * (*ctx0.c).width) + _i1) * 3)] = _t0.x;
    ctx0._alloc0[((((_i0 * (*ctx0.c).width) + _i1) * 3) + 1)] = _t0.y;
    ctx0._alloc0[((((_i0 * (*ctx0.c).width) + _i1) * 3) + 2)] = _t0.z;
    return;
}

__host__ int3 *_traverse_array0(Camera *c, int32_t height, Spheres *spheres) {
    int32_t *_alloc0;
    (void)cudaMalloc((void **)&_alloc0,
                     ((height * (*c).width) * 3) * sizeof(int32_t));
    Camera *d_c;
    cudaMallocAndCopyToDevice((void **)&d_c, c, sizeof(Camera));
    MaterialSphere *primitives;
    cudaMallocAndCopyToDevice((void **)&primitives, (*spheres).primitives,
                              (*spheres).primitive_count *
                                  sizeof(MaterialSphere));
    Nodes *nodes;
    cudaMallocAndCopyToDevice((void **)&nodes, (*spheres).nodes,
                              (*spheres).node_count * sizeof(Nodes));
    Spheres h_spheres = *spheres;
    h_spheres.primitives = primitives;
    h_spheres.nodes = nodes;
    Spheres *d_spheres;
    cudaMallocAndCopyToDevice((void **)&d_spheres, &h_spheres, sizeof(Spheres));
    _ctx0 ctx = _ctx0{height, d_c, d_spheres, _alloc0};
    _parfunc0<<<(((height * (*c).width) + 511) / 512), 512>>>(ctx);
    cudaDeviceSynchronize();
    cudaFree(primitives);
    cudaFree(nodes);
    cudaFree(d_c);
    cudaFree(d_spheres);
    int32_t *h__alloc0;
    mallocAndCopyFromDevice((void **)&h__alloc0, _alloc0,
                            ((height * (*c).width) * 3) * sizeof(int32_t));
    cudaFree(_alloc0);
    _alloc0 = h__alloc0;
    return reinterpret_cast<int3 *>(_alloc0);
}

__host__ Sphere bounding_sphere(Sphere *a, Sphere *b) {
    float3 _t1 = (*a).center;
    float3 _t2 = ((*b).center - _t1);
    float dist_sq = sum((_t2 * _t2));
    float dist = sqrtf(dist_sq);
    float _t7 = (*b).radius;
    float _t8 = (*a).radius;
    if ((dist + _t7) <= _t8) {
        return (*a);
    } else {
        if ((dist + _t8) <= _t7) {
            return (*b);
        }
    }
    float _t13 = ((float)0.5 * ((dist + _t8) + _t7));
    float3 direction =
        ((0 < dist) ? (_t2 / make_float3(dist)) : make_float3(1, 0, 0));
    float3 new_center = (_t1 + (direction * make_float3((_t13 - _t8))));
    return Sphere{new_center, _t13};
}

__host__ uint32_t rec_build_spheres(BVH *node_, Spheres *ST,
                                    size_t *nodes_index,
                                    size_t *primitives_index) {
    if (std::holds_alternative<Interior>(*node_)) {
        printf("visiting Interior\n");
        const Interior &node = std::get<Interior>(*node_);
        size_t this_index = (*nodes_index);
        (*nodes_index) += 1;
        (*ST).nodes[this_index].center = node.center;
        (*ST).nodes[this_index].radius = node.radius;
        (*ST).nodes[this_index].nprims = 0;
        uint32_t left_index =
            rec_build_spheres(node.left, ST, nodes_index, primitives_index);
        uint32_t right_index =
            rec_build_spheres(node.right, ST, nodes_index, primitives_index);
        reinterpret_cast<Arm_Interior *>(
            &(*ST).nodes[this_index].split0on_nprims)
            ->offset = (right_index - this_index);
        return this_index;
    } else if (std::holds_alternative<Leaf>(*node_)) {
        printf("visiting Leaf\n");
        const Leaf &node = std::get<Leaf>(*node_);
        size_t this_index = (*nodes_index);
        (*nodes_index) += 1;
        (*ST).nodes[this_index].center = node.center;
        (*ST).nodes[this_index].radius = node.radius;
        (*ST).nodes[this_index].nprims = node.nprims;
        reinterpret_cast<Arm_Leaf *>(&(*ST).nodes[this_index].split0on_nprims)
            ->poffset = (*primitives_index);
        for (uint8_t __p = 0u; __p < node.nprims; __p += 1u) {
            (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
        }
        (*primitives_index) += node.nprims;
        return this_index;
    }
}

__host__ void rec_count_spheres(BVH *node_, Spheres *ST) {
    if (std::holds_alternative<Interior>(*node_)) {
        const Interior &node = std::get<Interior>(*node_);
        rec_count_spheres(node.left, ST);
        rec_count_spheres(node.right, ST);
        (*ST).node_count += 1u;
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        (*ST).primitive_count += node.nprims;
        (*ST).node_count += 1u;
    }
}

__host__ Spheres build_spheres(BVH *CT) {
    Spheres ST;
    size_t primitives_index = 0;
    size_t nodes_index = 0;
    rec_count_spheres(CT, (&ST));
    std::cout << "prim count: " << ST.primitive_count << std::endl;
    std::cout << "node count: " << ST.node_count << std::endl;
    printf("prim count: %d\n", ST.primitive_count);
    printf("node count: %d\n", ST.node_count);
    MaterialSphere *primitives;
    (void)cudaMalloc((void **)&primitives,
                     ST.primitive_count * sizeof(MaterialSphere));
    ST.primitives = primitives;
    Nodes *nodes;
    (void)cudaMalloc((void **)&nodes, ST.node_count * sizeof(Nodes));
    ST.nodes = nodes;
    printf("REACHED\n");
    rec_build_spheres(CT, (&ST), (&nodes_index), (&primitives_index));
    return ST;
}

__host__ int3 *image(Camera *c, Spheres *spheres) {
    int32_t height = (int32_t)((float)(*c).width / (*c).aspect_ratio);
    height = ((height < 1) ? 1 : height);
    return _traverse_array0(c, height, spheres);
}
