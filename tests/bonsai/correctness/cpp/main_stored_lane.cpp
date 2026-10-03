// Used in stored_lane.bonsai.
//
// Three rows of eight, row r lane k holding 10r + k, so that every answer
// says which row and which lane was read. The mask has lanes 2, 4 and 7 on:
// `first` reads the lowest, lane 2; `two` reads the two lowest, 2 and 4.
#include "stored_lane.h"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    std::vector<float8> rows(3);
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 8; k++) {
            rows[r][k] = float(10 * r + k);
        }
    }
    uint32_t8 m;
    for (int k = 0; k < 8; k++) {
        m[k] = (k == 2 || k == 4 || k == 7) ? 1u : 0u;
    }
    // Row 1, lane 5.
    std::cout << pick(1, 5, rows.data()) << '\n';
    // Row 2, lane 0 -- the lane the index names, not the first.
    std::cout << pick(2, 0, rows.data()) << '\n';
    // Row 1, the lowest lane the mask has on: lane 2.
    std::cout << first(1, m, rows.data()) << '\n';
    // Row 0, lanes 2 and 4: 2 + 4.
    std::cout << two(0, m, rows.data()) << '\n';
    // Row 2, lane 2.
    std::cout << third(2, rows.data()) << '\n';
}
