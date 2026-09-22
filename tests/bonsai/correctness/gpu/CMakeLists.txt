# Programs with loops bound to the GPU, run on it: the same programs as
# backends/ptx, executed through the JIT, whose kernels launch through
# runtime/bonsai_cuda.cpp on whatever GPU the machine has. The goldens are
# what the programs print, which is what the same programs print on the CPU.
#
# Needs a GPU the driver can see, found here with nvidia-smi, and CUDA's
# libdevice for the maths (see backends/ptx). Without either the tests are
# registered disabled, so that a machine without a GPU still lists them.
find_program(BONSAI_NVIDIA_SMI nvidia-smi)
set(BONSAI_HAVE_GPU FALSE)
if (BONSAI_NVIDIA_SMI)
    execute_process(COMMAND "${BONSAI_NVIDIA_SMI}" -L
                    RESULT_VARIABLE smi_result
                    OUTPUT_QUIET ERROR_QUIET)
    if (smi_result EQUAL 0)
        set(BONSAI_HAVE_GPU TRUE)
    endif ()
endif ()
if (BONSAI_HAVE_GPU AND BONSAI_LIBDEVICE)
    set(gpu_disabled)
else ()
    message(STATUS "No GPU (or no libdevice): the correctness/gpu tests are "
                   "disabled.")
    set(gpu_disabled DISABLED)
endif ()

add_bonsai_tests(
    LABELS correctness gpu
    ${gpu_disabled}
    FILES
    atomic-float.bonsai
    block-effects.bonsai
    blocks.bonsai
    blocks-threads.bonsai
    buffers-cpp.bonsai
    device-function.bonsai
    dynamic-size.bonsai
    math.bonsai
    mut-bound.bonsai
    print.bonsai
    rand.bonsai
    struct-capture.bonsai
    threads.bonsai
)
