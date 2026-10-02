// Used in rcp.bonsai.
//
// The reciprocals the program takes, against the exact quotient. An estimate
// instruction alone (`rcpps`, 12 bits; `vrcp14ps`, 14 bits) is off by about
// one part in four thousand; one Newton step squares that error, to within a
// few units in the last place of a float. The tolerance here, eight ulps
// relative, passes the refined estimate and the exact division and fails
// the raw estimate -- which is what tells a missing Newton step from a
// present one.
#include "rcp.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

int failures = 0;

void check(const char *what, float x, float got) {
    const float want = 1.0f / x;
    // Relative error in units of the exact quotient's last place.
    const float ulp = std::ldexp(1.0f, std::ilogb(want) - 23);
    const float err = std::fabs(got - want) / ulp;
    const bool ok = err <= 8.0f;
    std::cout << what << " rcp(" << x << ") = " << got << " (exact " << want
              << ", " << err << " ulp) -- " << (ok ? "ok" : "WRONG") << "\n";
    if (!ok) {
        failures++;
    }
}

} // namespace

int main() {
    const float xs[] = {1.0f, 2.0f, 3.0f, 0.1f, 7.5f, 1e-6f, 123456.0f, -4.0f};
    for (float x : xs) {
        check("scalar", x, rcp_scalar(x));
    }
    float3 v3;
    v3[0] = 3.0f;
    v3[1] = -0.25f;
    v3[2] = 1e3f;
    const float3 r3 = rcp_vec3(v3);
    for (int k = 0; k < 3; k++) {
        check("vec3", v3[k], r3[k]);
    }
    float4 v4;
    v4[0] = 0.5f;
    v4[1] = 9.0f;
    v4[2] = -1e-3f;
    v4[3] = 42.0f;
    const float4 r4 = rcp_vec4(v4);
    for (int k = 0; k < 4; k++) {
        check("vec4", v4[k], r4[k]);
    }
    if (failures != 0) {
        std::cout << failures << " WRONG\n";
        return EXIT_FAILURE;
    }
    std::cout << "all ok\n";
    return EXIT_SUCCESS;
}
