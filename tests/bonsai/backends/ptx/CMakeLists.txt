# The device side of programs with loops bound to the GPU, as `-b ptx` prints
# it: the device module's LLVM IR, then the PTX. Pinned to sm_90 so that the
# goldens do not depend on the GPU in the machine -- none is needed to
# generate them -- and the host pinned as backends/llvm pins it.
#
# CUDA's libdevice is linked into any kernel that calls into it (math.bonsai),
# so the directory needs the CUDA toolkit for that file; where the toolkit is
# not found the tests are registered disabled rather than left out, so that
# the gap shows in the test list instead of passing by absence. The compiler
# looks for the bitcode the same way at run time (see CodeGen_PTX.cpp).
# Cached, so that correctness/gpu (a sibling directory) sees the answer too.
if (DEFINED ENV{BONSAI_LIBDEVICE} AND EXISTS "$ENV{BONSAI_LIBDEVICE}")
    set(BONSAI_LIBDEVICE "$ENV{BONSAI_LIBDEVICE}" CACHE FILEPATH
        "CUDA's libdevice.10.bc, for the PTX tests")
else ()
    find_file(BONSAI_LIBDEVICE libdevice.10.bc
              PATHS ENV CUDA_HOME ENV CUDA_PATH /usr/local/cuda /opt/cuda
              PATH_SUFFIXES nvvm/libdevice
              NO_DEFAULT_PATH
              DOC "CUDA's libdevice.10.bc, for the PTX tests")
endif ()
if (BONSAI_LIBDEVICE)
    set(ptx_disabled)
else ()
    message(STATUS "libdevice.10.bc not found: the backends/ptx tests are "
                   "disabled. Install the CUDA toolkit or set CUDA_HOME.")
    set(ptx_disabled DISABLED)
endif ()

add_bonsai_tests(
    LABELS backends ptx
    ${ptx_disabled}
    FLAGS --gpu-arch sm_90 --triple x86_64-unknown-linux-gnu --mcpu generic
    FILES
    atomic-float.bonsai
    block-effects.bonsai
    blocks.bonsai
    blocks-threads.bonsai
    device-function.bonsai
    dynamic-size.bonsai
    map-thread.bonsai
    math.bonsai
    mut-bound.bonsai
    print.bonsai
    rand.bonsai
    rtiow-primer.bonsai
    struct-capture.bonsai
    threads.bonsai
)
