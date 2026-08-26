#include "vector_ops.h"

#include <iostream>

// Used in vector_ops.bonsai
namespace {

void show(const char *label, float3 v) {
    std::cout << label << ": [" << v.x << ", " << v.y << ", " << v.z << "]\n";
}

} // namespace

int main() {
    // The C++ and generated sides must agree on the type exactly, or the
    // values below come back with a lane missing or a field misread.
    static_assert(sizeof(float3) == 16, "float3 must match <3 x float>");
    static_assert(alignof(float3) == 16, "float3 must match <3 x float>");
    static_assert(offsetof(Bundle, weight) == 16, "vector field must be 16 wide");

    float3 v = {1, -2, 3};
    float3 w = {2, -1, 0};

    show("scale", scale(v, 2));
    show("clamp_each", clamp_each(v, float3{0, 0, 0}, float3{2, 2, 2}));
    show("pick", pick(v, w));

    std::cout << "total: " << total(v) << '\n';
    std::cout << "biggest: " << biggest(v) << '\n';

    Bundle b{v, 42.5f};
    std::cout << "weight_of: " << weight_of(b) << '\n';
    show("shifted", shifted(b, 10));
}
