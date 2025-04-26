#pragma once

#include <list>
#include <map>
#include <string>

#include "Error.h"

namespace bonsai {
namespace ir {

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

    void new_frame() { frames.emplace_back(); }

    void pop_frame() { frames.pop_back(); }

  private:
    std::vector<std::map<K, V, H>> frames = {{}};
};

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

    void new_frame() { frames.emplace_back(); }

    void pop_frame() { frames.pop_back(); }

  private:
    std::vector<std::set<K, H, std::allocator<K>>> frames = {{}};
};

// Mapping from string identifiers, e.g., variables names, to some type T.
template <typename T>
using FrameStack = MapStack<std::string, T>;

} // namespace ir
} // namespace bonsai
