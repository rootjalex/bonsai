#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace bonsai {

class Error final : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace detail {

// __FILE__ is an absolute path into whatever directory the repository was
// cloned into. Trim it back to a repository-relative path, so that the
// "[internal] Error: <file>:<line>" lines that tests/bonsai/error/*.expect
// compares byte-for-byte do not depend on where the checkout lives.
inline std::string source_relative_path(const char *file) {
    std::string f(file);

#ifdef BONSAI_SOURCE_DIR
    // The repository root, injected by the top-level CMakeLists.txt.
    constexpr std::string_view root = BONSAI_SOURCE_DIR;
    if (!root.empty() && f.size() > root.size() &&
        f.compare(0, root.size(), root) == 0 && f[root.size()] == '/') {
        return f.substr(root.size() + 1);
    }
#endif

    // For a translation unit built outside our CMake, or one whose __FILE__
    // has been rewritten (-ffile-prefix-map), fall back to the last top-level
    // source directory. Printing the whole path is the honest last resort.
    for (const std::string_view dir : {"/src/", "/include/", "/tests/"}) {
        if (const size_t pos = f.rfind(dir); pos != std::string::npos) {
            return f.substr(pos + 1);
        }
    }
    return f;
}

} // namespace detail

class ErrorReport {
  public:
    ErrorReport(const char *cond_str, const char *file, size_t line) {
        stream << "[internal] Error: ";
        stream << detail::source_relative_path(file) << ":" << line << "\n";
        if (cond_str) {
            stream << "\n--> " << cond_str << "\n";
        }
    }

    ErrorReport &ref() { return *this; }

    template <typename T>
    ErrorReport &operator<<(const T &value) {
        stream << value;
        return *this;
    }

    [[noreturn]]
    ~ErrorReport() noexcept(false) {
        stream << "\n";
        // TODO: debug mode should do this.
        // std::cerr << stream.str();
        // abort();
        throw Error(stream.str());
    }

  private:
    std::ostringstream stream;
};

namespace detail {
// this is a syntax hack that enables placing a << operator after the .ref()
// method of the ErrorReport class. It can be any binary operator with lower
// precedence than << and higher than ?: (ternary). This also changes the
// semantics of `internal_assert(cond) << foo;` so that `foo` is only evaluated
// when the condition is false.
struct Voidifier {
    template <typename T>
    void operator&(T &) const {}
};
} // namespace detail

#define internal_assert(cond)                                                  \
    (cond) ? (void)0                                                           \
           : bonsai::detail::Voidifier() &                                     \
                 bonsai::ErrorReport(#cond, __FILE__, __LINE__).ref()

#define internal_error bonsai::ErrorReport(nullptr, __FILE__, __LINE__)

} // namespace bonsai
