#pragma once

#include <list>
#include <map>
#include <string>

#include "Error.h"

namespace bonsai {
namespace ir {

template <typename T>
struct FrameStack {
    std::list<std::map<std::string, T>> frames = {{}};

    T from_frames(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                return found->second;
            }
        }
        internal_error << "Cannot get from frame: " << name;
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
        internal_assert(!frames.empty());
        frames.back()[name] = value;
    }

    void new_frame() { frames.emplace_back(); }

    void pop_frame() { frames.pop_back(); }
};

// Creates a history of stack frames. This is necessary when we want to do some
// scoped analysis, for example, when computing use counts in DCE.
template <typename T>
struct History {
    bool name_in_scope(const std::string &name) {
        return current_frame().name_in_scope(name);
    }

    void add_to_frame(const std::string &name, T value) {
        current_frame().add_to_frame(name, value);
    }

    // TODO(chrisgyurgyik): This is non-const, while FrameStack is const.
    T &from_frames(const std::string &name) {
        for (auto it = current_frame().frames.rbegin();
             it != current_frame().frames.rend(); it++) {
            auto &frame = *it;
            auto found = frame.find(name);
            if (found != frame.end()) {
                return found->second;
            }
        }
        internal_error << "Cannot get from frame: " << name;
    }

    std::vector<std::pair<std::string, T>> find_all() {
        std::vector<std::pair<std::string, T>> found;
        for (auto it = current_frame().frames.rbegin();
             it != current_frame().frames.rend(); it++) {
            for (const auto &item : *it) {
                found.push_back(item);
            }
        }
        return found;
    }

    T &operator[](std::string name) { return from_frames(name); }

    void erase(const std::string &name) {
        for (auto it = current_frame().frames.rbegin();
             it != current_frame().frames.rend(); it++) {
            auto &frame = *it;
            auto found = frame.find(name);
            if (found != frame.end()) {
                frame.erase(found);
            }
        }
    }

    bool empty() {
        for (auto it = current_frame().frames.crbegin();
             it != current_frame().frames.crend(); it++) {
            if (!it->empty()) {
                return false;
            }
        }
        return true;
    }

    // A ghetto increment function.
    void increment(const std::string &name, T value) {
        for (auto it = current_frame().frames.rbegin();
             it != current_frame().frames.rend(); it++) {
            auto &frame = *it;
            auto found = frame.find(name);
            if (found != frame.end()) {
                found->second = found->second + value;
                return;
            }
        }
        add_to_frame(name, value);
    }

    void new_frame() {
        internal_assert(!history.empty());
        FrameStack<T> frame = history.back();
        frame.new_frame();
        history.push_back(std::move(frame));
    }

    void pop_frame() {
        internal_assert(!history.empty());
        FrameStack<T> frame = history.back();
        frame.pop_frame();
        history.push_back(std::move(frame));
    }

    // These should only be used internally by FrameHistory.
    void __pop_back() { history.pop_back(); }
    void __post_process() { std::reverse(history.begin(), history.end()); }

  private:
    std::vector<FrameStack<T>> history = {FrameStack<T>()};

    // Returns the current frame in this history.
    FrameStack<T> &current_frame() {

        internal_assert(!history.empty());
        return history.back();
    }
};

template <typename T>
struct FrameHistory {
    FrameHistory(History<T> history) : history(std::move(history)) {
        history.__post_process();
    }

    bool name_in_scope(const std::string &name) {
        return history.name_in_scope(name);
    }

    void add_to_frame(const std::string &name, T value) {
        history.add_to_frame(name, value);
    }

    T &from_frames(const std::string &name) {
        return history.from_frames(name);
    }

    std::vector<std::pair<std::string, T>> find_all() {
        return history.find_all();
    }

    T &operator[](std::string name) { return history.from_frames(name); }

    void erase(const std::string &name) { history.erase(name); }

    bool empty() { history.empty(); }

    void increment(const std::string &name, T value) {
        history.increment(name, value);
    }

    void new_frame() { history.__pop_back(); }
    void pop_frame() { history.__pop_back(); }

  private:
    History<T> history;
};

} // namespace ir
} // namespace bonsai
