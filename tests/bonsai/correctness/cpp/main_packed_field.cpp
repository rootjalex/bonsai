#include "packed_field.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

// Used in packed_field.bonsai.
//
// The record's bytes are another system's: what a C struct of
// `float p[3]; float d; uint32_t index;` is -- twenty bytes, the distance at
// twelve, the index at sixteen, aligned to the float -- and not the sixteen
// bytes aligned to sixteen a native three-float vector would take. The
// values go in as floats and come back as floats, shifted by what the
// program computed on the vector in between.
struct Reference {
    float p[3];
    float d;
    uint32_t index;
};
static_assert(sizeof(HitRecord) == sizeof(Reference));
static_assert(sizeof(HitRecord) == 20);
static_assert(alignof(HitRecord) == alignof(float));
static_assert(offsetof(HitRecord, p) == offsetof(Reference, p));
static_assert(offsetof(HitRecord, d) == offsetof(Reference, d));
static_assert(offsetof(HitRecord, index) == offsetof(Reference, index));

int main() {
    const uint32_t n = 3;
    std::vector<Hit> hits(n);
    for (uint32_t i = 0; i < n; i++) {
        hits[i].primID = 10 + i;
        hits[i].d = 0.5f * float(i + 1);
        hits[i].p = {float(i), 2.0f * float(i), 3.0f * float(i)};
    }
    std::vector<HitRecord> recs(n);
    std::vector<Hit> back(n);
    std::vector<float> squares(n);

    round_trip(n, hits.data(), recs.data(), back.data(), squares.data());

    for (uint32_t i = 0; i < n; i++) {
        std::cout << "rec " << recs[i].index << " " << recs[i].d << " "
                  << recs[i].p[0] << " " << recs[i].p[1] << " " << recs[i].p[2]
                  << "\n";
        std::cout << "hit " << back[i].primID << " " << back[i].d << " "
                  << back[i].p.x << " " << back[i].p.y << " " << back[i].p.z
                  << "\n";
        std::cout << "square " << squares[i] << "\n";
    }
    return 0;
}
