add_bonsai_tests(
    LABELS backends cuda
    FILES
    build-bvh8.bonsai
    build-collision.bonsai
    build-dop14.bonsai
    build-embree-hair.bonsai
    build-extent-quantization.bonsai
    # build-inner-group.bonsai was deleted; only its golden survives. The CPPX
    # directory still has the source if someone wants the CUDA case back.
    build-pbrt.bonsai
    build-ptr.bonsai
    global.bonsai
    parallel.bonsai
    rand.bonsai
    rtiow-primer.bonsai
    simple.bonsai
)
