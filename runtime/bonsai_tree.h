// bonsai_tree.h
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

template <typename... Fs>
struct overload : Fs... {
    using Fs::operator()...;
};

// Deduction guide must be at namespace scope
template <typename... Fs>
overload(Fs...) -> overload<Fs...>;

template <typename... Ts>
struct tree {
    std::variant<Ts...> value;

    tree() = default;

    template <typename T>
    tree(T &&val) : value(std::forward<T>(val)) {}

    // Match function: accepts a set of callables, and invokes the correct one
    template <typename... Fs>
    decltype(auto) match(Fs &&...fs) {
        return std::visit(overload{std::forward<Fs>(fs)...}, value);
    }

    template <typename... Fs>
    decltype(auto) match(Fs &&...fs) const {
        return std::visit(overload{std::forward<Fs>(fs)...}, value);
    }
};
