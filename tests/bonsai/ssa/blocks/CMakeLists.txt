# Goldens of the SSA block graph rather than of the relooper's reading of it.
#
# Every scheduling transform is a rewrite of this graph, and the statements
# `-p ssa` prints are a reconstruction of whatever the rewrite left: two
# different graphs can reloop to the same statements, so a golden of the
# statements is a lossy witness for what a schedule did. The tests here ask
# the compiler for the graph, before the schedule and after it.
add_bonsai_tests(
    LABELS ssa blocks
    FILES
    loopify.bonsai
    mixed-traversal.bonsai
    nested-traversal.bonsai
    sort.bonsai
    switch-merge.bonsai
)
