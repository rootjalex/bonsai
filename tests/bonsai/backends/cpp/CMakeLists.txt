add_bonsai_tests(
    LABELS backends cpp
    # The C++ backend emits an object file through LLVM, so its output is
    # host-dependent for the same reasons as the llvm backend tests.
    FLAGS --triple x86_64-unknown-linux-gnu --mcpu generic
    FILES
    not-exported.bonsai
    simple.bonsai
    struct.bonsai
)
