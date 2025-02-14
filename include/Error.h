#pragma once

#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bonsai {
namespace {

// Produces an appropriate error message and appends it to the `os`.
// `condition_string` is the (optional) code found inside the conditional check,
// `file` is the file in which this error was reported, and `line` is the line
// number.
void error_message(std::ostringstream &os,
                   std::optional<const char *> condition_string,
                   const char *file, size_t line) {
    // Print the file path proceeding the root directory (inclusive).
    constexpr std::string_view rootDirectory = "bonsai";
    std::string F(file);

    // TODO(cgyurgyik): Fix this hack.
    // Finds the last occurrence of `bonsai` to conform with Github Actions,
    // where both the WORKSPACE and the REPOSITORY are named `bonsai`.
    if (size_t pos = F.rfind(rootDirectory); pos != std::string::npos) {
        F = F.substr(pos + rootDirectory.length() + 1);
    }

    os << "[internal] Error: ";
    os << F << ":" << line << "\n";
    if (condition_string.has_value()) {
        os << "\n--> " << *condition_string << "\n";
    }
}

} // namespace

// Conditionally report an error. If `triggered` is true, this will abort the
// program after printing an error message to I/O.
class ConditionalErrorReport {
  public:
    ConditionalErrorReport(bool cond, const char *condition_string,
                           const char *file, size_t line)
        : triggered(!cond) {
        [[likely]] if (!triggered) { return; }
        error_message(stream, condition_string, file, line);
    }

    template <typename T>
    ConditionalErrorReport &operator<<(const T &value) {
        [[unlikely]] if (triggered) { stream << value; }
        return *this;
    }
    ~ConditionalErrorReport() noexcept(false) {
        [[likely]] if (!triggered) { return; }
        stream << "\n";
        std::cerr << stream.str();
        abort();
    }

  private:
    // Whether the failure condition should be triggered.
    bool triggered;
    std::ostringstream stream;
};

// Always report an error. This will abort the program after printing an error
// message to I/O.
class ErrorReport {
  public:
    ErrorReport(const char *file, size_t line) {
        std::optional<const char *> condition_string = std::nullopt;
        error_message(stream, condition_string, file, line);
    }

    template <typename T>
    ErrorReport &operator<<(const T &value) {
        stream << value;
        return *this;
    }

    [[noreturn]] ~ErrorReport() noexcept(false) {
        stream << "\n";
        std::cerr << stream.str();
        abort();
    }

  private:
    std::ostringstream stream;
};

// Prints an error message and aborts if `cond` is false, and does nothing
// otherwise. For example,
//   internal_assert(1 < 2) << "error message";
#define internal_assert(cond)                                                  \
    ConditionalErrorReport((cond), #cond, __FILE__, __LINE__)

// Prints an error message and aborts.  For example,
//   internal_assert(1 < 2) << "error message";
#define internal_error ErrorReport(__FILE__, __LINE__)

} // namespace bonsai
