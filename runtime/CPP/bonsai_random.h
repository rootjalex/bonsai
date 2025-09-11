#pragma once

#include <cstdlib>

// Random float in [0, 1)
template <typename T>
inline float random_float() {
    return T(std::rand()) / (T{RAND_MAX} + T{1});
}