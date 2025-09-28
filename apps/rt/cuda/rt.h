#include "helpers.h"

#include <cuda/std/array>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/universal_vector.h>

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
    float3 low;
    float3 high;
    BVH *left;
    BVH *right;
    uint8_t axis;
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
    Triangle *data;
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

struct Node;
struct Arm_Interior {
    Node *left;
    Node *right;
} __attribute__((packed));

struct Arm_Leaf {
    uint32_t poffset;
    uint64_t pad0;
    uint32_t pad1;
} __attribute__((packed));

struct Node {
    float3 low;
    float3 high;
    uint16_t nprims;
    uint8_t axis;
    uint8_t pad0;
    cuda::std::array<uint8_t, 16> split0on_nprims;
} __attribute__((packed));

struct Triangles {
    uint32_t primitive_count;
    Triangle *primitives;
    Node *node;
} __attribute__((packed));

struct _ctx0 {
    int64_t n;
    cuda::std::optional<Triangle> *_alloc0;
    Ray *rays;
    Triangles *triangles;
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

__host__ cuda::std::optional<FInterval> intersectsp_ray_aabb(Ray *r, AABB *b) {
    float3 _t1 = (make_float3(1) / (*r).d);
    bool3 dirIsNeg = (_t1 < make_float3(0));
    float3 _t2 = (*b).high;
    float3 _t3 = (*b).low;
    float3 low_parts =
        make_float3((dirIsNeg.x ? _t2.x : _t3.x), (dirIsNeg.y ? _t2.y : _t3.y),
                    (dirIsNeg.z ? _t2.z : _t3.z));
    float3 high_parts =
        make_float3((dirIsNeg.x ? _t3.x : _t2.x), (dirIsNeg.y ? _t3.y : _t2.y),
                    (dirIsNeg.z ? _t3.z : _t2.z));
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

__host__ float distmin_Ray_AABB(Ray *r, AABB *b) {
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
    return float3{__prod_diff_f32(_t0, _t1, _t2, _t3),
                  __prod_diff_f32(_t2, _t5, _t6, _t1),
                  __prod_diff_f32(_t6, _t3, _t0, _t5)};
}

__host__ cuda::std::optional<TriangleIntersection>
intersectsp_ray_tri(Ray *ray, Triangle *tri) {
    float3 _t0 = (*tri).p2;
    float3 _t1 = (*tri).p0;
    float3 _t2 = (*tri).p1;
    if (sum((cross_(_t0 - _t1, _t2 - _t1) * cross_(_t0 - _t1, _t2 - _t1))) ==
        0) {
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
        if ((0 < _t24) &&
            ((tScaled <= 0) || (((*ray).tmax * _t24) < tScaled))) {
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
    float deltaE =
        ((float)2 * ((((gamma(2) * maxXt) * maxYt) + (deltaY * maxXt)) +
                     (deltaX * maxYt)));
    float maxE = max(abs(float3{_t20, _t21, _t22}));
    float deltaT = (((float)3 * ((((_t25 * maxE) * maxZt) + (deltaE * maxZt)) +
                                 (deltaZ * maxE))) *
                    fabsf(invDet));
    if (t <= deltaT) {
        return cuda::std::nullopt;
    }
    return TriangleIntersection{b0, b1, b2, t};
}

__host__ float distmin_Ray_Triangle(Ray *ray, Triangle *tri) {
    cuda::std::optional<TriangleIntersection> isect =
        intersectsp_ray_tri(ray, tri);
    if (isect.has_value()) {
        TriangleIntersection isect_ = *isect;
        return isect_.t;
    } else {
        return INFINITY;
    }
}

__host__ bool intersects_Ray_AABB(Ray *r, AABB *b) {
    cuda::std::optional<FInterval> interval = intersectsp_ray_aabb(r, b);
    if (interval.has_value()) {
        FInterval extract = *interval;
        return ((extract.low < (*r).tmax) & (0 < extract.high));
    }
    return false;
}

__device__ __host__ cuda::std::optional<Triangle>
_traverse_tree0(Ray *ray, Triangles *triangles) {
    cuda::std::tuple<float, Triangle> _best0 =
        cuda::std::tuple<float, Triangle>{INFINITY, Triangle{}};
    int32_t _queue_count0 = 1;
    Node *_queue0[64];
    _queue0[0] = (*triangles).node;
    do {
        _queue_count0 -= 1;
        Node *root = _queue0[_queue_count0];
        AABB _t29 = AABB{(*root).low, (*root).high};
        if (intersects_Ray_AABB(ray, (&_t29))) {
            if (distmin_Ray_AABB(ray, (&_t29)) < cuda::std::get<0>(_best0)) {
                uint16_t _t22 = (*root).nprims;
                if (_t22 == 0u) {
                    Arm_Interior _t1 = bonsai_reinterpret<Arm_Interior>(
                        (*root).split0on_nprims);
                    _queue0[_queue_count0] = _t1.left;
                    _queue0[(_queue_count0 + 1)] = _t1.right;
                    _queue_count0 += 2;
                } else {
                    uint32_t _t17 =
                        bonsai_reinterpret<Arm_Leaf>((*root).split0on_nprims)
                            .poffset;
                    for (uint32_t _idx0 = _t17; _idx0 < (_t17 + (uint32_t)_t22);
                         _idx0 += 1u) {
                        Triangle _t14 = (*triangles).primitives[_idx0];
                        if (intersectsp_ray_tri(ray, (&_t14)).has_value()) {
                            float _t11 = distmin_Ray_Triangle(ray, (&_t14));
                            if (_t11 < cuda::std::get<0>(_best0)) {
                                _best0 = argmin(
                                    _best0, cuda::std::tuple<float, Triangle>{
                                                _t11, _t14});
                            }
                        }
                    }
                }
            }
        }
    } while (_queue_count0 != 0);
    return ((cuda::std::get<0>(_best0) != INFINITY)
                ? cuda::std::optional<Triangle>{cuda::std::get<1>(_best0)}
                : cuda::std::nullopt);
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

__host__ cuda::std::optional<Triangle> *_traverse_array0(int64_t n, Ray *rays,
                                                         Triangles *triangles) {
    cuda::std::optional<Triangle> *_alloc0;
    (void)cudaMalloc((void **)&_alloc0,
                     n * sizeof(cuda::std::optional<Triangle>));
    Ray *d_rays;
    cudaMallocAndCopyToDevice((void **)&d_rays, rays, n * sizeof(Ray));
    Triangle *__primitives;
    cudaMallocAndCopyToDevice((void **)&__primitives, (*triangles).primitives,
                              (*triangles).primitive_count * sizeof(Triangle));
    Node *__node;
    printf("recursive cuda malloc begin!\n");
    std::function<void(Node **, Node *)> cudaMallocAndCopyToDeviceRecursive =
        [&](Node **device_node_ptr, Node *host_node) {
            if (!host_node) {
                *device_node_ptr = nullptr;
                return;
            }

            Node *d_node;
            cudaMalloc((void **)&d_node, sizeof(Node));
            cudaMemcpy(d_node, host_node, sizeof(Node), cudaMemcpyHostToDevice);

            if (host_node->nprims == 0) {
                Arm_Interior *arm = reinterpret_cast<Arm_Interior *>(
                    host_node->split0on_nprims.data());

                Node *d_left = nullptr;
                Node *d_right = nullptr;

                if (arm->left) {
                    cudaMallocAndCopyToDeviceRecursive(&d_left, arm->left);
                }
                if (arm->right) {
                    cudaMallocAndCopyToDeviceRecursive(&d_right, arm->right);
                }

                Node temp_node = *host_node;
                Arm_Interior *temp_arm = reinterpret_cast<Arm_Interior *>(
                    temp_node.split0on_nprims.data());
                temp_arm->left = d_left;
                temp_arm->right = d_right;
                cudaMemcpy(d_node, &temp_node, sizeof(Node),
                           cudaMemcpyHostToDevice);
            }

            *device_node_ptr = d_node;
        };

    cudaMallocAndCopyToDeviceRecursive(&__node, (*triangles).node);
    printf("recursive cuda malloc complete!\n");
    Triangles h_triangles = *triangles;
    h_triangles.primitives = __primitives;
    h_triangles.node = __node;
    Triangles *d_triangles;
    cudaMallocAndCopyToDevice((void **)&d_triangles, &h_triangles,
                              sizeof(Triangles));
    _ctx0 ctx = _ctx0{n, _alloc0, d_rays, d_triangles};
    _parfunc0<<<((n + 511) / 512), 512>>>(ctx);
    cudaDeviceSynchronize();
    cuda::std::optional<Triangle> *h__alloc0;
    mallocAndCopyFromDevice((void **)&h__alloc0, _alloc0,
                            n * sizeof(cuda::std::optional<Triangle>));
    cudaFree(__primitives);
    cudaFree(_alloc0);
    cudaFree(d_rays);
    cudaFree(d_triangles);
    _alloc0 = h__alloc0;
    return _alloc0;
}

__host__ Node *rec_build_triangles(BVH *node_, Triangles *ST,
                                   size_t *primitives_index) {
    if (std::holds_alternative<Interior>(*node_)) {
        const Interior &node = std::get<Interior>(*node_);
        Node *this_index;
        (*this_index).low = node.low;
        (*this_index).high = node.high;
        (*this_index).nprims = 0;
        (*this_index).axis = argmax(node.high - node.low);
        Node *left_index = rec_build_triangles(node.left, ST, primitives_index);
        reinterpret_cast<Arm_Interior *>(&(*this_index).split0on_nprims)->left =
            left_index;
        Node *right_index =
            rec_build_triangles(node.right, ST, primitives_index);
        reinterpret_cast<Arm_Interior *>(&(*this_index).split0on_nprims)
            ->right = right_index;
        return this_index;
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        Node *this_index;
        (*this_index).low = node.low;
        (*this_index).high = node.high;
        (*this_index).nprims = node.nprims;
        (*this_index).axis = argmax(node.high - node.low);
        reinterpret_cast<Arm_Leaf *>(&(*this_index).split0on_nprims)->poffset =
            (*primitives_index);
        printf("primitives index: %d, nprims: %d\n", *primitives_index,
               nodenprims);
        for (uint16_t __p = 0u; __p < node.nprims; __p += 1u) {
            (*ST).primitives[(__p + (*primitives_index))] = node.data[__p];
        }
        (*primitives_index) += node.nprims;
        return this_index;
    }
}

__host__ void rec_count_triangles(BVH *node_, Triangles *ST) {
    if (std::holds_alternative<Interior>(*node_)) {
        const Interior &node = std::get<Interior>(*node_);
        rec_count_triangles(node.left, ST);
        rec_count_triangles(node.right, ST);
    } else if (std::holds_alternative<Leaf>(*node_)) {
        const Leaf &node = std::get<Leaf>(*node_);
        (*ST).primitive_count += node.nprims;
    }
}

__host__ Triangles build_triangles(BVH *CT) {
    Triangles ST;
    size_t primitives_index = 0;
    ST.primitive_count = 0u;
    rec_count_triangles(CT, (&ST));
    printf("primitive_count: %d\n", ST.primitive_count);
    Triangle *primitives = reinterpret_cast<Triangle *>(
        malloc(sizeof(Triangle) * ST.primitive_count));
    ST.primitives = primitives;
    ST.node = rec_build_triangles(CT, (&ST), (&primitives_index));
    printf("BUILD COMPLETE\n");
    return ST;
}

__host__ cuda::std::optional<Triangle> *chrt(int64_t n, Ray *rays,
                                             Triangles *triangles) {
    return _traverse_array0(n, rays, triangles);
}

__host__ cuda::std::optional<Triangle> trace(Ray *ray, Triangles *triangles) {
    return _traverse_tree0(ray, triangles);
}
