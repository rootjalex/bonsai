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

    // Whether T is an ir::Expr or ir::Stmt.
    template <typename T>
    static constexpr bool I =
        std::is_same_v<T, ir::Expr> || std::is_same_v<T, ir::Stmt>;

    // Returns the value at key `k` if found, and nullptr otherwise. This is a
    // safe way to update the value at key `k`. (We opt against using iterators
    // for simplicity.)
    template <typename T = V>
    typename std::enable_if<!I<T>, V>::type *find(const K &k) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            auto &frame = *it;
            auto found = frame.find(k);
            if (found != frame.end()) {
                return &found->second;
            }
        }
        return nullptr;
    }

    // Returns whether `k` is in this stack.
    bool contains(const K &k) const { return from_frames(k).has_value(); }

    // There is always at least one frame (the global scope).
    bool empty() const { return frames.size() == 1; }

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
    bool contains(const K &k) const { return from_frames(k).has_value(); }

    bool empty() const { return frames.empty(); }

    void add_to_frame(K k) { frames.add_to_frame(std::move(k), false); }

    void push_frame() { frames.push_frame(); }

    void pop_frame() { frames.pop_back(); }

  private:
    // TODO(cgyurgyik): The "value" here is dead. Eventually this should just be
    // a set, but I'm keeping it this way as we iterate on the design of this.
    ir::MapStack<K, bool, H> frames;
};

} // namespace ir
} // namespace bonsai
