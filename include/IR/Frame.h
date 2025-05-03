#pragma once

#include <list>
#include <map>
#include <string>
#include <type_traits>

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

template <typename K, typename V>
struct Window {
    std::map<K, V> map = {};
    // The previous window. If this is -1, then we've reached the global scope.
    int32_t parent;
    std::vector<int32_t> children;
};

template <typename K, typename V>
class History {
  public:
    std::vector<Window<K, V>> windows = {
        Window<K, V>{.map = {}, .parent = -1},
    };

    // (Used when creating the window history.)
    void new_window(int32_t parent, std::vector<int32_t> children) {
        internal_assert(0 <= parent && parent < windows.size());
        windows.push_back({
            .map = {},
            .parent = parent,
            .children = std::move(children),
        });
    }

    void add_to_window(const K &k, V v) {
        for (int i = current_index; i != -1;) {
            internal_assert(0 <= i && i < windows.size()) << i;
            Window<K, V> &window = windows[i];
            auto it = window.map.find(k);
            if (it != window.map.end()) {
                internal_error << "duplicate value found: " << k;
            }
            i = window.parent;
        }
        windows.back().map[k] = std::move(v);
    }

    V &operator[](const K &k) {
        // TODO(cgyurgyik): Yeah buddy, we can do better.
        auto [it, _] = windows.back().map.emplace(k, V{});
        return it->second;
    }

    std::optional<V> look_back(const K &k) const {
        for (int i = current_index; i != -1;) {
            internal_assert(0 <= i && i < windows.size()) << i;
            const Window<K, V> &window = windows[i];
            auto it = window.map.find(k);
            if (it != window.map.end()) {
                return it->second;
            }
            i = window.parent;
        }
        return {};
    }

    bool any_children(const K &k, std::function<bool(V)> f) const {
        const int32_t saved = current_index;
        internal_assert(0 <= saved && saved < windows.size())
            << saved << ", [0, " << windows.size() << ")";
        const Window<K, V> &window = windows[saved];
        if (auto it = window.map.find(k); it != window.map.end()) {
            if (f(it->second)) {
                current_index = saved;
                return true;
            }
        }

        for (const int32_t child_index : window.children) {
            current_index = child_index;
            if (any_children(k, f)) {
                current_index = saved;
                return true;
            }
        }
        current_index = saved;
        return false;
    }

    std::optional<V> from_back(const K &k) const {
        for (int i = windows.size() - 1; i != -1;) {
            internal_assert(0 <= i && i < windows.size()) << i;
            const Window<K, V> &window = windows[i];
            auto it = window.map.find(k);
            if (it != window.map.end()) {
                return it->second;
            }
            i = window.parent;
        }
        return {};
    }

    std::vector<std::pair<K, V>> elements() const {
        std::vector<std::pair<K, V>> elements;
        for (int i = current_index; i != -1;) {
            internal_assert(0 <= i && i < windows.size()) << i;
            for (const auto &[k, v] : windows[i].map) {
                elements.push_back({k, v});
            }
            i = windows[i].parent;
        }
        return elements;
    }

    void erase(const K &k) {
        for (int i = current_index; i != -1;) {
            Window<K, V> &window = windows[i];
            auto it = window.map.find(k);
            if (it != window.map.end()) {
                window.map.erase(it);
                return;
            }
            i = window.parent;
        }
        internal_error << "no key found: " << k;
    }

    // Returns whether this key `k` exists anywhere in the current history. This
    // is necessary for
    bool exists(const K &k) const {
        for (const Window<K, V> &window : windows) {
            if (auto it = window.map.find(k); it == window.map.end()) {
                continue;
            }
            return true;
        }
        return false;
    }

    bool contains(const K &k) const { return look_back(k).has_value(); }
    bool contains_back(const K &k) const { return from_back(k).has_value(); }
    bool size() const { return this->windows.size(); }
    bool empty() const { return size() == 1 && windows.front().map.empty(); }

    // TODO(cgyurgyik): Support this for other value types.
    void dump()
        requires std::is_integral_v<V>
    {
        std::cout << "\n---\n";
        for (int i = 0; i < windows.size(); ++i) {
            const auto &window = windows[i];
            std::cout << "Window " << i << ": " << "{ ";
            for (const auto &[k, v] : window.map) {

                std::cout << k << ":" << v << ", ";
            }
            std::cout << "}, previous: " << window.parent << ", children: [";
            for (const int32_t c : window.children) {
                std::cout << c << ",";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "---\n";
    }

    void dump()
        requires std::is_same_v<V, std::map<std::string, int32_t>>
    {
        std::cout << "\n---\n";
        for (int i = 0; i < windows.size(); ++i) {
            const auto &window = windows[i];
            std::cout << "Window " << i << ": " << "{ ";
            for (const auto &[k, v] : window.map) {

                std::cout << k << ": {";
                for (const auto &[kk, vv] : v) {
                    std::cout << kk << ":" << vv << ",";
                }
                std::cout << "}, ";
            }
            std::cout << "}, previous: " << window.parent << ", children: [";
            for (const int32_t c : window.children) {
                std::cout << c << ",";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "---\n";
    }

    // Used when traversing the window history.
    // TODO(cgyurgyik): Should be private.
    mutable int32_t current_index = 0;

  private:
};

} // namespace ir
} // namespace bonsai
