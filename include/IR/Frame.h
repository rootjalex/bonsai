#pragma once

#include <list>
#include <map>
#include <string>

#include "Error.h"

namespace bonsai {
namespace ir {

template <typename T>
struct FrameStack {
    std::list<std::map<std::string, T>> frames;

    T from_frames(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                return found->second;
            }
        }
        internal_error << "Cannot get from frame: " << name;
        return T();
    }

    bool name_in_scope(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                return true;
            }
        }
        return false;
    }

    void add_to_frame(const std::string &name, T value) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            internal_assert(found == frame.cend())
                << name << " shadows another variable (of the same name)";
        }
        frames.back()[name] = value;
    }

    void new_frame() { frames.emplace_back(); }

    void pop_frame() { frames.pop_back(); }
};

} // namespace ir
} // namespace bonsai
