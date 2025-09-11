#pragma once

#include <cstdlib>

// Random float in [0, 1)
template <typename T>
inline float random_float() {
    return std::rand() / (RAND_MAX + T{1});
}