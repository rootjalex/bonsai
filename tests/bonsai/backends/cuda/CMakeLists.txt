add_bonsai_tests(
    LABELS backends cuda
    FILES
    build-bvh8.bonsai
    build-collision.bonsai
    build-dop14.bonsai
    build-embree-hair.bonsai
    build-extent-quantization.bonsai
    # build-inner-group.bonsai has no CUDA source; the CPPX directory covers it.
    build-pbrt.bonsai
    build-ptr.bonsai
    global.bonsai
    parallel.bonsai
    rand.bonsai
    rtiow-primer.bonsai
    simple.bonsai
)
