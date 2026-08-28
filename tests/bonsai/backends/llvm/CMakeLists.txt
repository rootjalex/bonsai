add_bonsai_tests(
    LABELS backends llvm
    # These tests diff generated LLVM IR, which otherwise depends on the host:
    # the triple decides the data layout, and the CPU decides what the
    # vectorizers do. Pinning both is what lets the goldens travel between
    # machines.
    FLAGS --triple x86_64-unknown-linux-gnu --mcpu generic
    FILES
    dynamic-array.bonsai
    exported-function.bonsai
    float.bonsai
    forall.bonsai
    index.bonsai
    isel.bonsai
    ivector.bonsai
    known-limitation-ssa-mut-array-local.bonsai
    lambda.bonsai
    map-func.bonsai
    mut-basic.bonsai
    nested-exported-function.bonsai
    option.bonsai
    packed-struct.bonsai
    parfor-sequential.bonsai
    rand-par.bonsai
    rand.bonsai
    short-circuit.bonsai
    simple.bonsai
    struct_defaults.bonsai
    tail-call.bonsai
    tree-traversal.bonsai
    tree-traversal-recursive.bonsai
    type_coercion_primitive.bonsai
    vectorize_add_one.bonsai
    vectorize_calls.bonsai
    vectorize_divergent.bonsai
    vectorize_gather.bonsai
    vectorize_loop.bonsai
    vectorize_scale.bonsai
    vectorize_soa.bonsai
)
