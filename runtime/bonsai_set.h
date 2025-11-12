// bonsai_set.h
#pragma once

#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <tuple>
#include <type_traits>
#include <vector>

template <typename T, typename i_t>
struct range {
    const T *data;
    i_t offset, count;

    range(const T *_data, i_t _offset, i_t _count)
        : data(_data), offset(_offset), count(_count) {}
};

template <typename T, typename i_t, typename U, typename j_t>
struct product_range {
    const range<T, i_t> r0;
    const range<U, j_t> r1;
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

    template <typename S, typename i_t, typename U, typename j_t>
    void push_back(const product_range<S, i_t, U, j_t> &pr) {
        // Request space
        const size_t total = pr.r0.count * pr.r1.count;
        data.reserve(data.size() + total);

        for (i_t i = 0; i < pr.r0.count; ++i) {
            const S &elem0 = pr.r0.data[pr.r0.offset + i];
            for (j_t j = 0; j < pr.r1.count; ++j) {
                const U &elem1 = pr.r1.data[pr.r1.offset + j];
                data.push_back(std::make_tuple(elem0, elem1));
            }
        }
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

template <typename T>
struct is_tuple : std::false_type {};

template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool is_tuple_v = is_tuple<std::decay_t<T>>::value;

// ------------------------------------------------------------------

template <typename T, typename Predicate>
set<T> filter(Predicate &&predicate, const set<T> &input) {
    std::vector<T> result;

    const auto &data = input.data; // avoid repeated vector access
    for (size_t i = 0; i < data.size(); ++i) {
        if constexpr (is_tuple_v<T>) {
            // Expand tuple elements when calling predicate
            if (std::apply(predicate, data[i])) {
                result.push_back(data[i]);
            }
        } else {
            // Pass item directly
            if (predicate(data[i])) {
                result.push_back(data[i]);
            }
        }
    }

    return set<T>(std::move(result));
}

template <typename T, typename Func>
auto map(Func &&f, const set<T> &input)
{
    using U = decltype(f(std::declval<T>()));
    std::vector<U> result;
    result.reserve(input.data.size()); // direct access

    const auto &data = input.data; // avoid repeated vector access
    for (size_t i = 0; i < data.size(); ++i) {
        result.emplace_back(f(data[i])); // construct in-place
    }

    return set<U>(std::move(result));
}

template <typename T>
T sum(const set<T> &input) {
    return std::accumulate(input.data.begin(), input.data.end(), T{0});
}

template <typename T, typename Func>
auto sum_map(Func &&f, const set<T> &input) {
    using U = decltype(f(std::declval<T>()));
    U result = 0;

    const auto &data = input.data; // avoid repeated vector access
    for (size_t i = 0; i < data.size(); ++i) {
        result += f(data[i]);
    }

    return result;
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

template <typename T1, typename T2>
std::tuple<T1, T2> argmin(const std::tuple<T1, T2> *a,
                          const std::tuple<T1, T2> &b) {
    if (std::get<0>(*a) < std::get<0>(b)) {
        return *a;
    }
    return b;
}

template <typename T1, typename T2>
std::tuple<T1, T2> argmin(const std::tuple<T1, T2> &a,
                          const std::tuple<T1, T2> &b) {
    if (std::get<0>(a) < std::get<0>(b)) {
        return a;
    }
    return b;
}

template <typename T>
inline uint64_t count(const set<T> &input) {
    return input.size();
}

template <typename T, typename i_t>
inline uint64_t count(const range<T, i_t> &input) {
    return input.count;
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
    return set<std::tuple<T, U>>(std::move(result));
}

template <typename T, typename i_t, typename U, typename j_t>
inline auto product(const range<T, i_t> &input0, const range<U, j_t> &input1) {
    return product_range<T, i_t, U, j_t>{input0, input1};
}

template <typename Predicate, typename T, typename U>
set<std::tuple<T, U>>
nested_join(Predicate &&predicate,
            const set<T> &input0,
            const set<U> &input1) {
    std::vector<std::tuple<T, U>> result;
    input0.for_each([&](const T &item0) {
        input1.for_each([&](const U &item1) {
            if (predicate(item0, item1)) {
                result.push_back(std::make_tuple(item0, item1));
            }
        });
    });
    return set<std::tuple<T, U>>(std::move(result));
}

template <typename Predicate, typename T, typename U>
uint64_t nested_join_count(Predicate &&predicate, const set<T> &input0,
                           const set<U> &input1) {
    uint64_t result = 0;
    input0.for_each([&](const T &item0) {
        input1.for_each([&](const U &item1) {
            if (predicate(item0, item1)) {
                result++;
            }
        });
    });
    return result;
}

// Compare two sets for equality (order-agnostic)
template <typename T>
bool operator==(const set<T> &a, const set<T> &b) {
    if (a.size() != b.size()) {
        std::cout << "Different sizes: " << a.size() << " vs " << b.size() << "\n";
        return false;
    }

    std::set<T> std_a(a.data.begin(), a.data.end()), std_b(b.data.begin(), b.data.end());
    return std_a == std_b;
}

template <typename T>
bool operator==(const set<T> &a, const std::set<T> &b) {
    std::set<T> std_a(a.data.begin(), a.data.end());
    return std_a == b;
}


template <typename T>
void print_set(const set<T> &s) {
    std::cout << "{ ";
    s.for_each([&](const T &i) { std::cout << i << " "; });
    std::cout << "}" << std::endl;
}

template <typename T, typename U>
std::set<std::tuple<T, U>>
flatten(const set<std::tuple<T, set<U>>> &input) {
    std::set<std::tuple<T, U>> result;
    input.for_each([&](const std::tuple<T, set<U>> &p) {
        const T &outer = std::get<0>(p);
        const set<U> &inner = std::get<1>(p);
        inner.for_each([&](const U &u) {
            result.emplace(outer, u);
        });
    });
    return result;
}