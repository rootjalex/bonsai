add_bonsai_tests(
    LABELS backends llvm
    # These tests diff generated LLVM IR, which otherwise depends on the host:
    # the triple decides the data layout, and the CPU decides what the
    # vectorizers do. Pinning both is what lets the goldens travel between
    # machines.
    FLAGS --triple x86_64-unknown-linux-gnu --mcpu generic
    FILES
    adt-tagged-index.bonsai
    and-of-compares.bonsai
    any-hit-arms.bonsai
    any-hit-prefetch.bonsai
    compare-on-bits.bonsai
    mirror-derived-loads.bonsai
    foreign.bonsai
    foreign-linked.bonsai
    child-volumes.bonsai
    child-volumes-vectorized.bonsai
    tiled-leaf-vectorized.bonsai
    prefetch-children.bonsai
    prefetch-switch-arm.bonsai
    prefetch-lines.bonsai
    rcp.bonsai
    permute.bonsai
    sort-key-nonnegative.bonsai
    sort-key-through-bits.bonsai
    sort-peel-prefetch.bonsai
    stored-lane.bonsai
    defer.bonsai
    defer-gang.bonsai
    defer-gang-bools.bonsai
    gate-drains.bonsai
    round-guard.bonsai
    demote-atomics.bonsai
    divide_by_constant.bonsai
    dynamic-array.bonsai
    exported-function.bonsai
    ffp-contract.bonsai
    float.bonsai
    float-zero-folds.bonsai
    forall.bonsai
    gpu-launch-buffers.bonsai
    index.bonsai
    isel.bonsai
    ivector.bonsai
    lambda.bonsai
    libm-call.bonsai
    map-func.bonsai
    max-min.bonsai
    mulhi.bonsai
    mut-array-literal.bonsai
    mut-element-field-argument.bonsai
    mut-basic.bonsai
    nested-exported-function.bonsai
    option.bonsai
    packed-struct.bonsai
    parfor-sequential.bonsai
    rand-par.bonsai
    persistent.bonsai
    rand.bonsai
    sub-word-gather.bonsai
    index-widening.bonsai
    short-circuit.bonsai
    simple.bonsai
    sret.bonsai
    struct_defaults.bonsai
    switch-three-way.bonsai
    tail-call.bonsai
    tight-array.bonsai
    tiled-elements.bonsai
    tiled-elements-derived.bonsai
    tiled-elements-where.bonsai
    argmin-tighten-children.bonsai
    argmin-over-map.bonsai
    tree-traversal.bonsai
    tree-traversal-instanced.bonsai
    tree-traversal-mixed.bonsai
    tree-traversal-nested.bonsai
    tree-traversal-recursive.bonsai
    thread_dispatch.bonsai
    type_coercion_primitive.bonsai
    vectorize_add_one.bonsai
    vectorize_call_guard.bonsai
    vectorize_calls.bonsai
    vectorize_pure_call.bonsai
    vectorize_divergent.bonsai
    vectorize_division.bonsai
    vectorize_gather.bonsai
    vectorize_loop.bonsai
    vectorize_masked_accumulate.bonsai
    vectorize_masked_math_call.bonsai
    vector_math.bonsai
    vectorize_packet_traversal.bonsai
    vectorize_scale.bonsai
    vectorize_short_vectors.bonsai
    vectorize_soa.bonsai
    vectorize_skip_arms.bonsai
    vectorize_skip_cursor.bonsai
    vectorize_union.bonsai
    vectorize_varying_memory.bonsai
    vectorize_vector_index.bonsai
)

# The same pinning on a machine with AVX-512 (Zen 4, which has every
# AVX-512 extension the backend asks after), for what the x86 backend emits
# only where the instructions exist. A file's own `--mcpu` would not do: the
# directory's flags come after it on the command line and win.
add_bonsai_tests(
    LABELS backends llvm
    FLAGS --triple x86_64-unknown-linux-gnu --mcpu znver4
    FILES
    permute-avx512.bonsai
    rcp-approx.bonsai
)
