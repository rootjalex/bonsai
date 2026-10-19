// Used in constant_argument_provisional.bonsai.
//
// The light-tree descent's shape, run through the C++ backend: the path
// apps/pbrt takes, and the one on which the 2026-10-19 regression showed
// (the LLVM backend's direct lowering asked the definitions analysis in an
// order that did not meet the flaw). Each call is checked against what the
// walk means: stop at the first of i == n, choosing i, or i == stop,
// choosing nothing.
#include "constant_argument_provisional.h"

#include <iostream>

namespace {

int failures = 0;

void check(const char *what, int got, int expected) {
    std::cout << what << ": " << got
              << (got == expected ? " -- ok" : " -- WRONG") << '\n';
    if (got != expected) {
        failures++;
    }
}

} // namespace

int main() {
    check("descend(5, 10)", descend(5, 10), 5);   // five steps, then the leaf
    check("descend(5, 3)", descend(5, 3), -1);    // the zero-importance arm at 3
    check("descend(0, 0)", descend(0, 0), 0);     // the leaf at once
    check("descend(7, 7)", descend(7, 7), 7);     // the leaf arm is tested first
    // With the latch handing the header `true` every walk took one step:
    // -1, -1, 0, -1.
    return failures == 0 ? 0 : 1;
}
