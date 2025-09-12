#pragma once

#include <tuple>

template <typename T1, typename T2>
std::tuple<T1, T2> argmin(const std::tuple<T1, T2> &a,
                          const std::tuple<T1, T2> &b) {
    if (std::get<0>(a) < std::get<0>(b)) {
        return a;
    }
    return b;
}

template <typename T1, typename T2>
std::tuple<T1, T2> argmax(const std::tuple<T1, T2> &a,
                          const std::tuple<T1, T2> &b) {
    if (std::get<0>(a) > std::get<0>(b)) {
        return a;
    }
    return b;
}

// Used for std::variant visitor.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};