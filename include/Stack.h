#pragma once

#include <iostream>
#include <map>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// #include "Debug.h"
#include "Error.h"

namespace bonsai {

template <typename T>
struct Stack {
    void push_frame() { frames.emplace_back(Frame{}); }

    void pop_frame() { frames.pop_back(); }

    void insert(const std::string &name, const T &value) {
        internal_assert(!frames.empty());
        frames.back().insert(name, value);
    }

    bool contains(const std::string &name) {
        // TODO: iterate backwards and return if any frame has it?
    }

    bool contains(const std::string &name) {
        // TODO: iterate backwards and return if any frame has it?
    }

  private:
    struct Frame {
        void insert(const std::string &name, const T &value) {
            internal_assert(data.insert(name, value).second);
        }

      private:
        std::map<std::string, T> data;
    };
    std::vector<Frame> frames;
};

} // namespace bonsai
