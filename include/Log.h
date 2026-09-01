#pragma once

#include <iostream>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>

namespace bonsai {

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

namespace {
enum class LogType {
    INFO = 0,
    WARN = 1,
    ERROR = 2,
    DEBUG = 3,
};

inline std::string log_level_to_string(LogType level) {
    switch (level) {
    case LogType::INFO:
        return "INFO";
    case LogType::WARN:
        return "WARN";
    case LogType::ERROR:
        return "ERROR";
    case LogType::DEBUG:
        return "DEBUG";
    }
}
} // namespace

// TODO(cgyurgyik): verbosity levels is probably nice to have as well.
class LogStream {
  public:
    LogStream(std::ostream &out, LogType level, std::source_location location)
        : out(out), level(level), location(location) {}

    ~LogStream() {
        out << "[" << log_level_to_string(level) << "] "
            << detail::source_relative_path(location.file_name()) << ":"
            << location.line() << ": " << stream.str() << std::endl;
    }

    template <typename T>
    LogStream &operator<<(T &&value) {
        stream << std::forward<T>(value);
        return *this;
    }

  private:
    std::ostream &out;
    LogType level;
    std::source_location location;
    std::ostringstream stream;
};

#define LOG_ERROR                                                              \
    LogStream(std::cerr, LogType::ERROR, std::source_location::current())
#define LOG_WARN                                                               \
    LogStream(std::cerr, LogType::WARN, std::source_location::current())
#define LOG_INFO                                                               \
    LogStream(std::cerr, LogType::INFO, std::source_location::current())

} // namespace bonsai
