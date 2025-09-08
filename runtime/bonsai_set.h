// bonsai_set.h
#pragma once

#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <tuple>
#include <vector>

template <typename T, typename i_t>
struct range {
    const T *data;
    i_t offset, count;

    range(const T *_data, i_t _offset, i_t _count)
        : data(_data), offset(_offset), count(_count) {}
};

// basically just a thread-safe atomic std::vector
// can be in read or write mode only.
// does not implement deduplication!
template <typename T>
struct set {
  private:
    // mutable std::shared_mutex mutex;
  public:
    std::vector<T> data;
    set() = default;
    ~set() = default;
    set(std::vector<T> &&_data) : data(std::move(_data)) {}

    // Disable copying
    set(const set &) = default;
    set &operator=(const set &) = default;

    // Enable moving
    set(set &&) = default;
    set &operator=(set &&) = default;

    // Append elements safely
    void push_back(const T &value) {
        // std::unique_lock lock(mutex);
        data.push_back(value);
    }
    void push_back(T &&value) {
        // std::unique_lock lock(mutex);
        data.emplace_back(std::move(value));
    }
    // Vector overload
    template <typename U>
    void push_back(const U &values) {
        static_assert(std::is_trivially_copyable_v<U>,
                      "U must be trivially copyable");
        static_assert(sizeof(U) % sizeof(T) == 0,
                      "U must be a multiple of T in size");

        constexpr size_t count = sizeof(U) / sizeof(T);
        const T *elems = reinterpret_cast<const T *>(&values);

        // std::unique_lock lock(mutex);
        data.insert(data.end(), elems, elems + count);
    }

    // Range overload
    template <typename U>
    void push_back(const range<T, U> &range) {
        // std::unique_lock lock(mutex);
        data.insert(data.end(), range.data + range.offset,
                    range.data + range.offset + range.count);
    }

    size_t size() const {
        // std::unique_lock lock(mutex);
        return data.size();
    }

    // Iterate using a callback. No lock required, assumed read-only.
    void for_each(std::function<void(const T &)> fn) const {
        for (const auto &item : data) {
            fn(item);
        }
    }

    void for_each(std::function<void(T &)> fn) {
        for (auto &item : data) {
            fn(item);
        }
    }
};

template <typename T, typename Predicate>
set<T> filter(Predicate &&predicate, const set<T> &input) {
    std::vector<T> result;
    input.for_each([&](const T &item) {
        if (predicate(item)) {
            result.push_back(item);
        }
    });
    return set<T>(std::move(result));
}

template <typename T, typename U, typename Func>
set<U> map(Func &&f, const set<T> &input) {
    std::vector<U> result;
    result.reserve(input.size());
    input.for_each([&](const T &item) { result.push_back(f(item)); });
    return set<U>(std::move(result));
}

template <typename T, typename U>
U reduce(std::function<U(const U &, const T &)> reducer, const set<T> &input,
         U initial) {
    input.for_each([&](const T &item) { initial = reducer(initial, item); });
    return initial;
}

template <typename T, typename U>
T &argmin(std::function<U(const U &, const T &)> metric, const set<T> &input) {
    if (input.data.empty()) {
        throw std::invalid_argument("Input set must not be empty for argmin.");
    }

    U best = std::numeric_limits<U>::infinity();
    T &result = input.data[0];
    input.for_each([&](T &item) {
        if (metric(item) < best) {
            result = item;
        }
    });
    return result;
}

// TODO: this should never be fairly used, a nested join should be fused!
template <typename T, typename U>
set<std::tuple<T, U>> product(const set<T> &input0, const set<U> &input1) {
    std::vector<std::tuple<T, U>> result;
    result.reserve(input0.size() * input1.size());
    input0.for_each([&](const T &item0) {
        input1.for_each([&](const U &item1) {
            result.push_back(std::make_tuple(item0, item1));
        });
    });
    return set<std::tuple<T, U>>(result);
}

template <typename T, typename U>
set<std::tuple<T, U>>
nested_join(std::function<bool(const T &, const U &)> predicate,
            const set<T> &input0, const set<U> &input1) {
    std::vector<std::tuple<T, U>> result;
    input0.for_each([&](const T &item0) {
        input1.for_each([&](const U &item1) {
            if (predicate(item0, item1)) {
                result.push_back(std::make_tuple(item0, item1));
            }
        });
    });
    return set<std::tuple<T, U>>(result);
}

// Compare two sets for equality (order-agnostic)
template <typename T>
bool operator==(const set<T> &a, const set<T> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    std::set<T> std_a, std_b;
    a.for_each([&](const T &x) { std_a.insert(x); });
    b.for_each([&](const T &x) { std_b.insert(x); });
    return std_a == std_b;
}