#pragma once

#include <iostream>
#include <source_location>
#include <sstream>
#include <string>

namespace bonsai {

namespace {
enum class LogLevel {
    INFO = 0,
    WARN = 1,
    ERROR = 2,
    DEBUG = 3,
};

inline std::string log_level_to_string(LogLevel level) {
    switch (level) {
    case LogLevel::INFO:
        return "I";
    case LogLevel::WARN:
        return "W";
    case LogLevel::ERROR:
        return "E";
    case LogLevel::DEBUG:
        return "D";
    }
}
} // namespace

// TODO(cgyurgyik): verbosity levels is probably nice to have as well.
class LogStream {
  public:
    LogStream(std::ostream &out, LogLevel level, std::source_location location)
        : out(out), level(level), location(location) {}

    ~LogStream() {
        // Print the file path proceeding the root directory (inclusive).
        constexpr std::string_view ROOT_DIRECTORY = "bonsai";
        std::string file(location.file_name());

        // TODO(cgyurgyik): Fix this hack.
        // Finds the last occurrence of `bonsai` to conform with Github Actions,
        // where both the WORKSPACE and the REPOSITORY are named `bonsai`.
        if (size_t pos = file.rfind(ROOT_DIRECTORY); pos != std::string::npos) {
            file = file.substr(pos + ROOT_DIRECTORY.length() + 1);
        }
        out << "[" << log_level_to_string(level) << "] " << file << ":"
            << location.line() << ": " << stream.str() << std::endl;
    }

    template <typename T>
    LogStream &operator<<(T &&value) {
        stream << std::forward<T>(value);
        return *this;
    }

  private:
    std::ostream &out;
    LogLevel level;
    std::source_location location;
    std::ostringstream stream;
};

#define LOG_ERROR                                                              \
    LogStream(std::cerr, LogLevel::ERROR, std::source_location::current())
#define LOG_WARN                                                               \
    LogStream(std::cerr, LogLevel::WARN, std::source_location::current())
#define LOG_INFO                                                               \
    LogStream(std::cerr, LogLevel::INFO, std::source_location::current())

} // namespace bonsai
