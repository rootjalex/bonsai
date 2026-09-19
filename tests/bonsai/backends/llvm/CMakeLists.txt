add_bonsai_tests(
    LABELS backends llvm
    # These tests diff generated LLVM IR, which otherwise depends on the host:
    # the triple decides the data layout, and the CPU decides what the
    # vectorizers do. Pinning both is what lets the goldens travel between
    # machines.
    FLAGS --triple x86_64-unknown-linux-gnu --mcpu generic
    FILES
    adt-tagged-index.bonsai
    demote-atomics.bonsai
    dynamic-array.bonsai
    exported-function.bonsai
    ffp-contract.bonsai
    float.bonsai
    forall.bonsai
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
    rand.bonsai
    sub-word-gather.bonsai
    short-circuit.bonsai
    simple.bonsai
    sret.bonsai
    struct_defaults.bonsai
    switch-three-way.bonsai
    tail-call.bonsai
    tight-array.bonsai
    tree-traversal.bonsai
    tree-traversal-instanced.bonsai
    tree-traversal-mixed.bonsai
    tree-traversal-nested.bonsai
    tree-traversal-recursive.bonsai
    type_coercion_primitive.bonsai
    vectorize_add_one.bonsai
    vectorize_calls.bonsai
    vectorize_divergent.bonsai
    vectorize_gather.bonsai
    vectorize_loop.bonsai
    vectorize_masked_accumulate.bonsai
    vectorize_packet_traversal.bonsai
    vectorize_scale.bonsai
    vectorize_short_vectors.bonsai
    vectorize_soa.bonsai
    vectorize_skip_arms.bonsai
    vectorize_union.bonsai
    vectorize_varying_memory.bonsai
)
