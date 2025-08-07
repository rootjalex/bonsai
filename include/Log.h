#pragma once

#include <iostream>
#include <source_location>
#include <sstream>
#include <string>

namespace bonsai {

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
