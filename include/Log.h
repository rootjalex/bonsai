#pragma once

#include <iostream>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>

// Set by the build to the absolute path of the repository root, with a
// trailing separator. Empty when the header is used outside that build.
#ifndef BONSAI_SOURCE_DIR
#define BONSAI_SOURCE_DIR ""
#endif

namespace bonsai {

// __FILE__ and std::source_location report absolute paths. Trim the source
// root so diagnostics read the same from any checkout directory.
inline std::string_view source_relative_path(std::string_view file) {
    constexpr std::string_view root = BONSAI_SOURCE_DIR;
    if (file.starts_with(root)) {
        file.remove_prefix(root.size());
    }
    return file;
}

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
            << source_relative_path(location.file_name()) << ":"
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
