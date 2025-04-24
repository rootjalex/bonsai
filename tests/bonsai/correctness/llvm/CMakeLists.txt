add_bonsai_tests(
    LABELS correctness llvm
    FILES
    bitwise.bonsai
    dce.bonsai
    exported-function.bonsai
    exported-recursive-function.bonsai
    nested-exported-function.bonsai
    option.bonsai
    print.bonsai
    map.bonsai
    struct.bonsai
)
