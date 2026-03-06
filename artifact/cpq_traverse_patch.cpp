// NOTE: The current version of Scion does not yet support the priority-queue
// scheduling optimization used for closest-point queries. We therefore perform
// this transformation manually by replacing the generated _traverse_tree
// function with the version below, which reorders child visits by distance.
// Future versions of Scion will support this scheduling directive natively.
// Regardless, this paper's claims concern data layout polymorphism, and the
// traversal algorithm is an orthogonal input to the system; this manual edit
// does not affect the layout-related results.

Triangle _traverse_tree0(const Point *__restrict__ p,
                         const Triangles *__restrict__ triangles) {
    std::tuple<float, Triangle> _best0 = std::tuple<float, Triangle>{
        std::numeric_limits<float>::infinity(), Triangle{}};
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
        const Nodes &_t195 = (*triangles).nodes[index];
        const AABB _t198 = AABB{.low = _t195.low, .high = _t195.high};
        if ((SqDistPointAABB(p, (&_t198)) < std::get<0>(_best0))) {
            const uint16_t _t35 = _t195.nprims;
            if (_t35 == 0u) {
                _best0 = argmin<float, Triangle>(
                    _best0, std::tuple<float, Triangle>{
                                distmax_Point_AABB(p, (&_t198)), Triangle{}});
                const uint32_t L = index + 1u;
                const uint32_t R =
                    index +
                    reinterpret<Arm_Interior>(_t195.split0on_nprims).offset;

                const Nodes &Ln = (*triangles).nodes[L];
                const float3 Llow = Ln.low;
                const float3 Lhigh = Ln.high;
                const AABB lAABB = AABB{.low = Llow, .high = Lhigh};
                float left_dist = SqDistPointAABB(p, &lAABB);
                const Nodes &Rn = (*triangles).nodes[R];
                const AABB rAABB = AABB{.low = Rn.low, .high = Rn.high};
                float right_dist = SqDistPointAABB(p, &rAABB);
                if (left_dist <= right_dist) {
                    _queue0[_queue_count0] = R;
                    _queue0[(_queue_count0 + 1)] = L;
                } else {
                    _queue0[_queue_count0] = L;
                    _queue0[(_queue_count0 + 1)] = R;
                }
                _queue_count0 += 2;
            } else {
                const uint32_t _t24 =
                    reinterpret<Arm_Leaf>(_t195.split0on_nprims).poffset;
                for (uint32_t _idx0 = _t24; _idx0 < (_t24 + (uint32_t)(_t35));
                     ++_idx0) {
                    const Triangle &_t202 = (*triangles).primitives[_idx0];
                    float _t14 = 0.0f;
                    const std::tuple<Point, Point> __inline1 =
                        closestPointonTriangle(p, (&_t202));
                    const float3 __inline0 =
                        ((*p).vec - std::get<0>(__inline1).vec);
                    _t14 = dot(__inline0, __inline0);
                    bool _t0 = (_t14 < std::get<0>(_best0));
                    if (_t0) {
                        float _t1 = _t14;
                        _best0 = std::tuple<float, Triangle>{_t1, _t202};
                    }
                }
            }
        }
    } while ((_queue_count0 != 0));
    return std::get<1>(_best0);
}
