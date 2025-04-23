add_bonsai_tests(
    LABELS correctness llvm
    FILES
    dce.bonsai
    exported-function.bonsai
    exported-recursive-function.bonsai
    nested-exported-function.bonsai
    option.bonsai
    order-of-operations.bonsai
    print.bonsai
    map.bonsai
    struct.bonsai
)
