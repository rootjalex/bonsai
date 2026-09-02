#include "ffp-contract.h"

#include <cstdio>

// Used in ffp-contract.bonsai, which explains the numbers.
//
// 8193 is one rounding and 8192 is two. Nothing else can come out: both are
// exactly representable, and the inputs are exact.
int main() {
    const float a = 1.0f + 1.0f / 4096.0f; // 1 + 2^-12
    const float b = a;
    const float c = -1.0f;
    printf("%.1f\n", scaled(a, b, c));

    // The same call with a product that needs no rounding, so that the fused
    // and unfused answers agree. This is the control: a test that only ever
    // printed the interesting case could not tell "contraction happened" from
    // "this function returns 8193".
    printf("%.1f\n", scaled(2.0f, 0.5f, -0.5f));
}
