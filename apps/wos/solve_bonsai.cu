#include "helpers.h"
#include "solve_bonsai.cuh"

__device__ void addSolutionEstimate(Statistics *s, float estimate) {
    s->nSolEstimates += 1;
    float delta = (estimate - (*s).solMean);
    s->solMean += (delta / (float)(*s).nSolEstimates);
    float delta2 = (estimate - (*s).solMean);
    s->solMean2 += (delta * delta2);
    return;
}

__device__ float dirichletPDE(float3 x, const PDE *p, const WalkSettings *s) {
    if (((((*s).flags & 1u) != 0u) &&
         !(all(((*s).box.low <= x)) & all((x <= (*s).box.high)))) ||
        (((*s).flags & 16u) != 0u)) {
        return 0;
    } else {
        return (sinf((*p).freq * x.x) * cosf((*p).freq * x.y));
    }
}

__device__ float distmax_Point_AABB(const Point *pt, const AABB *a) {
    float3 u = ((*a).low - (*pt).vec);
    float3 v = ((*pt).vec - (*a).high);
    float3 d = min(u, v);
    return dot(d, d);
}

__device__ float distmin_Point_AABB(const Point *pt, const AABB *a) {
    float3 u = ((*a).low - (*pt).vec);
    float3 v = ((*pt).vec - (*a).high);
    float3 d = max(max(u, v), make_float3(0));
    return dot(d, d);
}

__device__ __tuple_0 closestPointonTriangle(const Point *pt,
                                            const Triangle *tri) {
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
            return __tuple_0{Point{a}, Point{float3{1, 0, 0}}};
        }
    }
    float3 bp = (p - b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (0 <= d3) {
        if (d4 <= d3) {
            return __tuple_0{Point{b}, Point{float3{0, 1, 0}}};
        }
    }
    float vc = ((d1 * d4) - (d3 * d2));
    if (vc <= 0) {
        if (0 <= d1) {
            if (d3 <= 0) {
                float v0 = (d1 / (d1 - d3));
                return __tuple_0{Point{a + (make_float3(v0) * ab)},
                                 Point{float3{1 - v0, v0, 0}}};
            }
        }
    }
    float3 cp = (p - c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (0 <= d6) {
        if (d5 <= d6) {
            return __tuple_0{Point{c}, Point{float3{0, 0, 1}}};
        }
    }
    float vb = ((d5 * d2) - (d1 * d6));
    if (vb <= 0) {
        if (0 <= d2) {
            if (d6 <= 0) {
                float w0 = (d2 / (d2 - d6));
                return __tuple_0{Point{a + (make_float3(w0) * ac)},
                                 Point{float3{1 - w0, 0, w0}}};
            }
        }
    }
    float va = ((d3 * d6) - (d5 * d4));
    if (va <= 0) {
        if (0 <= (d4 - d3)) {
            if (0 <= (d5 - d6)) {
                float w1 = ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
                return __tuple_0{Point{b + (make_float3(w1) * (c - b))},
                                 Point{float3{0, 1 - w1, w1}}};
            }
        }
    }
    float denom = (1 / ((va + vb) + vc));
    float v = (vb * denom);
    float w = (vc * denom);
    float u = (va * denom);
    return __tuple_0{Point{(a + (ab * make_float3(v))) + (ac * make_float3(w))},
                     Point{float3{u, v, w}}};
}

__device__ float distmin_Point_Triangle(const Point *p, const Triangle *tri) {
    const __tuple_0 pts = closestPointonTriangle(p, tri);
    float3 d = ((*p).vec - pts._field0.vec);
    return dot(d, d);
}

__device__ float _traverse_tree0(const Point *p, const _tree_layout0 *tris) {
    float __best0 = INFINITY;
    float *_best0 = &__best0;
    int32_t __queue_count0 = 1;
    int32_t *_queue_count0 = &__queue_count0;
    uint16_t _queue0[16];
    _queue0[0] = 0u;
    do {
        *_queue_count0 -= 1;
        uint16_t tris_index = _queue0[(*_queue_count0)];
        const AABB _lv30 = AABB{(*tris).group0_index[tris_index].low,
                                (*tris).group0_index[tris_index].high};
        if (distmin_Point_AABB(p, (&_lv30)) < (*_best0)) {
            if ((*tris).group0_index[tris_index].nPrims == 0u) {
                const AABB _lv27 =
                    AABB{(*tris).group0_index[tris_index + 1u].low,
                         (*tris).group0_index[tris_index + 1u].high};
                if (distmin_Point_AABB(p, (&_lv27)) < (*_best0)) {
                    const AABB _lv22 = AABB{
                        (*tris)
                            .group0_index[tris_index +
                                          bonsai_reinterpret<_tree_layout2>(
                                              (*tris)
                                                  .group0_index[tris_index]
                                                  .split0on_nPrims)
                                              .offset]
                            .low,
                        (*tris)
                            .group0_index[tris_index +
                                          bonsai_reinterpret<_tree_layout2>(
                                              (*tris)
                                                  .group0_index[tris_index]
                                                  .split0on_nPrims)
                                              .offset]
                            .high};
                    if (distmin_Point_AABB(p, (&_lv22)) < (*_best0)) {
                        const AABB _lv0 =
                            AABB{(*tris).group0_index[tris_index + 1u].low,
                                 (*tris).group0_index[tris_index + 1u].high};
                        const AABB _lv1 = AABB{
                            (*tris)
                                .group0_index[tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                .low,
                            (*tris)
                                .group0_index[tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                .high};
                        *_best0 =
                            min(*_best0, fmaxf(distmax_Point_AABB(p, (&_lv0)),
                                               distmax_Point_AABB(p, (&_lv1))));
                        const AABB _lv16 =
                            AABB{(*tris).group0_index[tris_index + 1u].low,
                                 (*tris).group0_index[tris_index + 1u].high};
                        const AABB _lv17 = AABB{
                            (*tris)
                                .group0_index[tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                .low,
                            (*tris)
                                .group0_index[tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                .high};
                        if (distmax_Point_AABB(p, (&_lv16)) <
                            distmin_Point_AABB(p, (&_lv17))) {
                            _queue0[(*_queue_count0)] = (tris_index + 1u);
                            *_queue_count0 += 1;
                        } else {
                            const AABB _lv14 =
                                AABB{(*tris)
                                         .group0_index
                                             [tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                         .low,
                                     (*tris)
                                         .group0_index
                                             [tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                         .high};
                            const AABB _lv15 = AABB{
                                (*tris).group0_index[tris_index + 1u].low,
                                (*tris).group0_index[tris_index + 1u].high};
                            if (distmax_Point_AABB(p, (&_lv14)) <
                                distmin_Point_AABB(p, (&_lv15))) {
                                _queue0[(*_queue_count0)] =
                                    (tris_index +
                                     bonsai_reinterpret<_tree_layout2>(
                                         (*tris)
                                             .group0_index[tris_index]
                                             .split0on_nPrims)
                                         .offset);
                                *_queue_count0 += 1;
                            } else {
                                const AABB _lv2 = AABB{
                                    (*tris).group0_index[tris_index + 1u].low,
                                    (*tris).group0_index[tris_index + 1u].high};
                                const AABB _lv3 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                const AABB _lv4 = AABB{
                                    (*tris).group0_index[tris_index + 1u].low,
                                    (*tris).group0_index[tris_index + 1u].high};
                                const AABB _lv5 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                const AABB _lv6 = AABB{
                                    (*tris).group0_index[tris_index + 1u].low,
                                    (*tris).group0_index[tris_index + 1u].high};
                                const AABB _lv7 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                _queue0[(*_queue_count0)] =
                                    (((distmin_Point_AABB(p, (&_lv2)) <
                                       distmin_Point_AABB(p, (&_lv3))) |
                                      ((distmin_Point_AABB(p, (&_lv4)) ==
                                        distmin_Point_AABB(p, (&_lv5))) &
                                       (distmax_Point_AABB(p, (&_lv6)) <
                                        distmax_Point_AABB(p, (&_lv7)))))
                                         ? (tris_index +
                                            bonsai_reinterpret<_tree_layout2>(
                                                (*tris)
                                                    .group0_index[tris_index]
                                                    .split0on_nPrims)
                                                .offset)
                                         : (tris_index + 1u));
                                const AABB _lv8 = AABB{
                                    (*tris).group0_index[tris_index + 1u].low,
                                    (*tris).group0_index[tris_index + 1u].high};
                                const AABB _lv9 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                const AABB _lv10 = AABB{
                                    (*tris).group0_index[tris_index + 1u].low,
                                    (*tris).group0_index[tris_index + 1u].high};
                                const AABB _lv11 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                const AABB _lv12 = AABB{
                                    (*tris).group0_index[tris_index + 1u].low,
                                    (*tris).group0_index[tris_index + 1u].high};
                                const AABB _lv13 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                _queue0[(*_queue_count0) + 1] =
                                    (((distmin_Point_AABB(p, (&_lv8)) <
                                       distmin_Point_AABB(p, (&_lv9))) |
                                      ((distmin_Point_AABB(p, (&_lv10)) ==
                                        distmin_Point_AABB(p, (&_lv11))) &
                                       (distmax_Point_AABB(p, (&_lv12)) <
                                        distmax_Point_AABB(p, (&_lv13)))))
                                         ? (tris_index + 1u)
                                         : (tris_index +
                                            bonsai_reinterpret<_tree_layout2>(
                                                (*tris)
                                                    .group0_index[tris_index]
                                                    .split0on_nPrims)
                                                .offset));
                                *_queue_count0 += 2;
                            }
                        }
                    } else {
                        const AABB _lv21 =
                            AABB{(*tris).group0_index[tris_index + 1u].low,
                                 (*tris).group0_index[tris_index + 1u].high};
                        if (distmin_Point_AABB(p, (&_lv21)) < (*_best0)) {
                            const AABB _lv18 = AABB{
                                (*tris).group0_index[tris_index + 1u].low,
                                (*tris).group0_index[tris_index + 1u].high};
                            *_best0 =
                                min(*_best0, distmax_Point_AABB(p, (&_lv18)));
                            _queue0[(*_queue_count0)] = (tris_index + 1u);
                            *_queue_count0 += 1;
                        } else {
                            const AABB _lv20 =
                                AABB{(*tris)
                                         .group0_index
                                             [tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                         .low,
                                     (*tris)
                                         .group0_index
                                             [tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                         .high};
                            if (distmin_Point_AABB(p, (&_lv20)) < (*_best0)) {
                                const AABB _lv19 = AABB{
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .low,
                                    (*tris)
                                        .group0_index
                                            [tris_index +
                                             bonsai_reinterpret<_tree_layout2>(
                                                 (*tris)
                                                     .group0_index[tris_index]
                                                     .split0on_nPrims)
                                                 .offset]
                                        .high};
                                *_best0 = min(*_best0,
                                              distmax_Point_AABB(p, (&_lv19)));
                                _queue0[(*_queue_count0)] =
                                    (tris_index +
                                     bonsai_reinterpret<_tree_layout2>(
                                         (*tris)
                                             .group0_index[tris_index]
                                             .split0on_nPrims)
                                         .offset);
                                *_queue_count0 += 1;
                            }
                        }
                    }
                } else {
                    const AABB _lv26 =
                        AABB{(*tris).group0_index[tris_index + 1u].low,
                             (*tris).group0_index[tris_index + 1u].high};
                    if (distmin_Point_AABB(p, (&_lv26)) < (*_best0)) {
                        const AABB _lv23 =
                            AABB{(*tris).group0_index[tris_index + 1u].low,
                                 (*tris).group0_index[tris_index + 1u].high};
                        *_best0 = min(*_best0, distmax_Point_AABB(p, (&_lv23)));
                        _queue0[(*_queue_count0)] = (tris_index + 1u);
                        *_queue_count0 += 1;
                    } else {
                        const AABB _lv25 = AABB{
                            (*tris)
                                .group0_index[tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                .low,
                            (*tris)
                                .group0_index[tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                .high};
                        if (distmin_Point_AABB(p, (&_lv25)) < (*_best0)) {
                            const AABB _lv24 =
                                AABB{(*tris)
                                         .group0_index
                                             [tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                         .low,
                                     (*tris)
                                         .group0_index
                                             [tris_index +
                                              bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset]
                                         .high};
                            *_best0 =
                                min(*_best0, distmax_Point_AABB(p, (&_lv24)));
                            _queue0[(*_queue_count0)] =
                                (tris_index + bonsai_reinterpret<_tree_layout2>(
                                                  (*tris)
                                                      .group0_index[tris_index]
                                                      .split0on_nPrims)
                                                  .offset);
                            *_queue_count0 += 1;
                        }
                    }
                }
            } else {
                for (uint16_t _idx0 = 0u;
                     _idx0 < (uint16_t)(*tris).group0_index[tris_index].nPrims;
                     _idx0 += 1u) {
                    const Triangle _lv29 =
                        (*tris).prims[bonsai_reinterpret<_tree_layout3>(
                                          (*tris)
                                              .group0_index[tris_index]
                                              .split0on_nPrims)
                                          .pOffset +
                                      _idx0];
                    if (distmin_Point_Triangle(p, (&_lv29)) < (*_best0)) {
                        const Triangle _lv28 =
                            (*tris).prims[bonsai_reinterpret<_tree_layout3>(
                                              (*tris)
                                                  .group0_index[tris_index]
                                                  .split0on_nPrims)
                                              .pOffset +
                                          _idx0];
                        *_best0 =
                            min(*_best0, distmin_Point_Triangle(p, (&_lv28)));
                    }
                }
            }
        }
    } while ((*_queue_count0) != 0);
    return (*_best0);
}

__device__ float computeDistToAbsBoundary(float3 pt,
                                          const _tree_layout0 *tris) {
    const Point p = Point{pt};
    return sqrtf(_traverse_tree0((&p), tris));
}

__device__ __tuple_1 harmonicSampleVolume(const HarmonicGreensBall *h,
                                          float3 dir, curandState *_rng_state) {
    float u1 = curand_uniform(_rng_state);
    float u2 = curand_uniform(_rng_state);
    float phi = (((float)2 * (float)3.14159) * u2);
    float _r = (((1 + (sqrtf(1 - powf(u1 * u1, 1 / (float)3)) * cosf(phi))) *
                 (*h).radius) /
                (float)2);
    float *r = &_r;
    *r = fmaxf((*h).rClamp, (*r));
    if ((*h).radius < (*r)) {
        *r = ((*h).radius / (float)2);
    }
    float pdf =
        ((((1 / (*r)) - (1 / (*h).radius)) / ((float)4 * (float)3.14159)) /
         (((*h).radius * (*h).radius) / (float)6));
    float3 q = ((*h).center + (make_float3((*r)) * dir));
    return __tuple_1{q, (*r), pdf};
}

__device__ float3 sampleUnitSphereUniform(curandState *_rng_state) {
    float u0 = curand_uniform(_rng_state);
    float u1 = curand_uniform(_rng_state);
    float z = (1 - ((float)2 * u0));
    float r = sqrtf(fmaxf(0, 1 - (z * z)));
    float phi = (((float)2 * (float)3.14159) * u1);
    return float3{r * cosf(phi), r * sinf(phi), z};
}

__device__ float sourcePDE(float3 x, const PDE *p, const WalkSettings *s) {
    if (((*s).flags & 64u) != 0u) {
        return 0;
    } else {
        return (((float)5 * sinf((*p).freq * x.y)) * cosf((*p).freq * x.x));
    }
}

__device__ float computeSourceContribution(const PDE *pde,
                                           const WalkSettings *s,
                                           const HarmonicGreensBall *ball,
                                           const WalkResults *res,
                                           curandState *_rng_state) {
    if (!(((*s).flags & 64u) != 0u)) {
        float3 dir = sampleUnitSphereUniform(_rng_state);
        const __tuple_1 sample = harmonicSampleVolume(ball, dir, _rng_state);
        float3 sourcePt = sample._field0;
        float sourceContribution =
            ((((*ball).radius * (*ball).radius) / (float)6) *
             sourcePDE(sourcePt, pde, s));
        return ((*res).throughput * sourceContribution);
    }
    return 0;
}

__device__ float getTerminalContribution(const PDE *pde, const WalkSettings *s,
                                         const WalkResults *res) {
    if (!(((*s).flags & 16u) != 0u)) {
        return dirichletPDE((*res).pt, pde, s);
    } else {
        return 0;
    }
}

__device__ __tuple_2 terminateWalk(const WalkSettings *s, float throughput,
                                   curandState *_rng_state) {
    if (throughput < (*s).russianRouletteThreshold) {
        float survivalProb = (throughput / (*s).russianRouletteThreshold);
        if (survivalProb < curand_uniform(_rng_state)) {
            return __tuple_2{true, 0};
        }
        return __tuple_2{false, (*s).russianRouletteThreshold};
    }
    return __tuple_2{false, throughput};
}

__device__ _option0 walk(const PDE *pde, const WalkSettings *s,
                         const WalkResults *res, const _tree_layout0 *tris,
                         curandState *_rng_state) {
    WalkResults _S_res = (*res);
    WalkResults *S_res = &_S_res;
    do {
        if ((*S_res).distToAbs <= (*s).epsShellAbs) {
            float terminalContribution = getTerminalContribution(pde, s, S_res);
            float totalContribution =
                (((*S_res).throughput * terminalContribution) +
                 (*S_res).totalSourceContribution);
            return _option0{
                WalkResults{(*S_res).pt, totalContribution, (*S_res).throughput,
                            (*S_res).walkLength, (*S_res).distToAbs},
                true};
        } else {
            const HarmonicGreensBall ball = HarmonicGreensBall{
                (*S_res).pt, (*S_res).distToAbs, (float)0.0001};
            float contrib =
                computeSourceContribution(pde, s, (&ball), S_res, _rng_state);
            float totalSourceContribution =
                ((*S_res).totalSourceContribution + contrib);
            float3 dir = sampleUnitSphereUniform(_rng_state);
            float3 pt = ((*S_res).pt + (make_float3((*S_res).distToAbs) * dir));
            if (!(all(((*s).box.low <= pt)) & all((pt <= (*s).box.high)))) {
                return _option0{};
            }
            float t = ((*S_res).throughput * 1);
            const __tuple_2 term = terminateWalk(s, t, _rng_state);
            float throughput = term._field1;
            if (term._field0) {
                return _option0{WalkResults{pt, totalSourceContribution,
                                            throughput, (*S_res).walkLength,
                                            (*S_res).distToAbs},
                                true};
            }
            uint32_t walkLength = ((*S_res).walkLength + 1u);
            if ((*s).maxWalkLength < walkLength) {
                return _option0{WalkResults{pt, totalSourceContribution,
                                            throughput, walkLength,
                                            (*S_res).distToAbs},
                                true};
            }
            float distToAbs = computeDistToAbsBoundary(pt, tris);
            const WalkResults new_res = WalkResults{
                pt, totalSourceContribution, throughput, walkLength, distToAbs};
            *S_res = new_res;
            continue;
        }
    } while (true);
}

__device__ uint32_t doWalk(const PDE *pde, const WalkSettings *s,
                           const SamplePoint *p, uint32_t w, uint32_t nWalks,
                           Statistics *stats, uint32_t walkLength,
                           const _tree_layout0 *tris, curandState *_rng_state) {
    uint32_t _S_w = w;
    uint32_t *S_w = &_S_w;
    uint32_t _S_walkLength = walkLength;
    uint32_t *S_walkLength = &_S_walkLength;
    do {
        if (nWalks <= (*S_w)) {
            return (*S_walkLength);
        }
        const WalkResults starter_res =
            WalkResults{(*p).pt, 0, 1, 0u, (*p).distToAbs};
        const _option0 res = walk(pde, s, (&starter_res), tris, _rng_state);
        if (res.set) {
            const WalkResults results = res.value;
            float totalContribution = results.totalSourceContribution;
            addSolutionEstimate(stats, totalContribution);
            *S_w = ((*S_w) + 1u);
            *S_walkLength = ((*S_walkLength) + results.walkLength);
            continue;
        }
        return (*S_walkLength);
    } while (true);
}

__device__ Statistics sol(const PDE *pde, const WalkSettings *s,
                          const SamplePoint *p, uint32_t nWalks,
                          const _tree_layout0 *tris, curandState *_rng_state) {
    Statistics _stats = Statistics{0, 0, 0, 0u, 0u, 0u, 0};
    Statistics *stats = &_stats;
    if (((*p).type_and_quantity & 3u) == 1u) {
        float _totalContribution = 0;
        float *totalContribution = &_totalContribution;
        if (!(((*s).flags & 16u) != 0u)) {
            *totalContribution = dirichletPDE((*p).pt, pde, s);
        }
        addSolutionEstimate(stats, (*totalContribution));
        stats->firstSphereRadius = 0;
        return (*stats);
    }
    uint32_t n = (((*p).distToAbs <= (*s).epsShellAbs) ? 1u : nWalks);
    stats->firstSphereRadius = (*p).distToAbs;
    uint32_t walkLength = doWalk(pde, s, p, 0u, n, stats, 0u, tris, _rng_state);
    stats->totalWalkLength = walkLength;
    return (*stats);
}

__global__ void _parfunc0(_ctx0 ctx0) {
    uint32_t tid = ((blockIdx.x * blockDim.x) + threadIdx.x);
    uint32_t _i0 = tid;
    if (ctx0.n <= _i0) {
        return;
    }
    curandState _rng_state;
    curand_init(tid, 0, 0, &_rng_state);
    const SamplePoint _lv0 = ctx0.pts[_i0];
    ctx0._alloc0[_i0] =
        sol(ctx0.pde, ctx0.s, (&_lv0), ctx0.nWalks, ctx0.tris, (&_rng_state));
    return;
}

__host__ Statistics *_traverse_array0(const PDE *pde, const WalkSettings *s,
                                      uint32_t nWalks, uint32_t n,
                                      SamplePoint *pts,
                                      const _tree_layout0 *tris) {
    Statistics *_alloc0;
    (void)cudaMalloc((void **)&_alloc0, n * sizeof(Statistics));
    PDE *d_pde;
    cudaMallocAndCopyToDevice((void **)&d_pde, pde, sizeof(PDE));
    WalkSettings *d_s;
    cudaMallocAndCopyToDevice((void **)&d_s, s, sizeof(WalkSettings));
    SamplePoint *d_pts;
    cudaMallocAndCopyToDevice((void **)&d_pts, pts, n * sizeof(SamplePoint));
    Triangle *prims;
    cudaMallocAndCopyToDevice((void **)&prims, (*tris).prims,
                              (*tris).pCount * sizeof(Triangle));
    _tree_layout1 *group0_index;
    cudaMallocAndCopyToDevice((void **)&group0_index, (*tris).group0_index,
                              (*tris).count * sizeof(_tree_layout1));
    _tree_layout0 h_tris = *tris;
    h_tris.prims = prims;
    h_tris.group0_index = group0_index;
    _tree_layout0 *d_tris;
    cudaMallocAndCopyToDevice((void **)&d_tris, &h_tris, sizeof(_tree_layout0));
    _ctx0 ctx = _ctx0{n, _alloc0, d_pde, d_s, d_pts, nWalks, d_tris};
    _parfunc0<<<((n + 511u) / 512u), 512u>>>(ctx);
    cudaDeviceSynchronize();
    Statistics *h__alloc0;
    mallocAndCopyFromDevice((void **)&h__alloc0, _alloc0,
                            n * sizeof(Statistics));
    cudaFree(prims);
    cudaFree(group0_index);
    cudaFree(_alloc0);
    cudaFree(d_pde);
    cudaFree(d_s);
    cudaFree(d_pts);
    cudaFree(d_tris);
    _alloc0 = h__alloc0;
    return _alloc0;
}

__host__ Statistics *solve(const PDE *pde, const WalkSettings *s, uint32_t n,
                           SamplePoint *pts, uint32_t nWalks,
                           const _tree_layout0 *tris) {
    return _traverse_array0(pde, s, nWalks, n, pts, tris);
}
