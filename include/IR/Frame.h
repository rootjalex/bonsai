#pragma once

#include <list>
#include <map>
#include <string>

#include "Error.h"

namespace bonsai {
namespace ir {

// Maintains a stack of scopes, where each frame is a map from some key type K
// to some value type V. This is useful for doing analysis within scopes.
// insertion of duplicates is illegal; it is up to the user to ensure that if
// the stack already contains the key `k` for this scope, then insertion of `k`
// does not occur again. For example,
//
// MapStack<std::string, Expr> fs;             // [{}]
// fs.add_to_frame("x", Var::make(i32, "v"));  // [{"x": Var(i32, v)}]
// fs.push_frame();                            // [{"x": Var(i32, v)}, {}]
// fs.contains("x");                           // true
// fs.pop_frame();                             // [{x: Var(i32, v)}]
// fs.pop_frame();                             // []
// fs.contains("x");                           // false
template <typename K, typename V, typename H = std::less<K>>
struct MapStack {
    // Retrieves the variable from this frame stack if it exists, and
    // {} otherwise.
    std::optional<V> from_frames(const K &k) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(k);
            if (found != frame.cend()) {
                return found->second;
            }
        }
        return {};
    }

    bool contains(const K &k) const { return from_frames(k).has_value(); }

    // There is always at least one frame (the global scope).
    bool empty() const { return frames.size() == 1; }

    // Replaces the value at `k` with `v`. Precondition: `k` must be present in
    // the scope. TODO(cgyurgyik): This double lookup idiom is bad.
    void replace(const K &k, V v) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            auto &frame = *it;
            auto found = frame.find(k);
            if (found != frame.end()) {
                found->second = std::move(v);
                return;
            }
        }
        internal_error << "Key: " << k << " not found";
    }

    void add_to_frame(K k, V v) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(k);
            if (found == frame.end()) {
                continue;
            }
            internal_error << "found duplicate value: " << k;
        }
        frames.back()[std::move(k)] = std::move(v);
    }

    void push_frame() { frames.emplace_back(); }

    void pop_frame() { frames.pop_back(); }

  private:
    std::vector<std::map<K, V, H>> frames = {{}};
};

// Similar to MapStack, but only inserts keys.
template <typename K, typename H = std::less<K>>
struct SetStack {
    bool contains(const K &k) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(k);
            if (found != frame.cend()) {
                return true;
            }
        }
        return false;
    }

    void add_to_frame(K k) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(k);
            if (found == frame.end()) {
                continue;
            }
            internal_error << "found duplicate value: " << k;
        }
        frames.back().insert(k);
    }

    void push_frame() { frames.emplace_back(); }

    void pop_frame() { frames.pop_back(); }

  private:
    std::vector<std::set<K, H, std::allocator<K>>> frames = {{}};
};

} // namespace ir
} // namespace bonsai
