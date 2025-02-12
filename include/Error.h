#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bonsai {

// An error class for error propagation. For example,
//
// auto E1 = Error::failure() << "failed pre-condition X: " << x;
// if (E1.failed()) { return E1.with_message("additional error message"); }
//
// auto E2 = Error::success();
// if (E2) { return 0; }
struct Error {

    static Error failure() { return Error(/*has_error=*/true); }
    static Error success() { return Error(/*has_error=*/false); }

    template <typename T>
    Error &operator<<(const T &value) {
        internal_assert(has_error);
        std::ostringstream os;
        os << value;
        error_message += os.str();
        return *this;
    }

    // Returns whether this has an error.
    bool failed() const { return has_error; }

    // Returns whether this was successful with no errors.
    bool succeeded() const { return !failed(); }

    // Enables implicit conversion to bool, e.g.,
    // if (auto E = foo()) { abort(); }
    explicit operator bool() const { return has_error; }

    // Returns the message associated with this error.
    std::string message() const { return error_message; }

  private:
    Error(bool has_error) : has_error(has_error) {}

    // Whether this carry an error.
    bool has_error;

    // The error message associated with this class.
    std::string error_message;
};

std::ostream &operator<<(std::ostream &, const Error &);

// TODO: Halide's has some weird magic I don't understand, but I probably should
// try to...

class ErrorReport {
  public:
    ErrorReport(bool cond, const char *cond_str, const char *file, size_t line)
        : triggered(!cond) {
        [[likely]] if (!triggered) { return; }
        // Print the file path proceeding the root directory (inclusive).
        constexpr std::string_view rootDirectory = "bonsai";
        std::string F(file);

        // TODO(cgyurgyik): Fix this hack.
        // Finds the last occurrence of `bonsai` to conform with Github Actions,
        // where both the WORKSPACE and the REPOSITORY are named `bonsai`.
        if (size_t pos = F.rfind(rootDirectory); pos != std::string::npos) {
            F = F.substr(pos + rootDirectory.length() + 1);
        }

        stream << "[internal] Error: ";
        stream << F << ":" << line << "\n";
        if (cond_str == nullptr)
            return;
        stream << "\n--> " << cond_str << "\n";
    }

    template <typename T>
    ErrorReport &operator<<(const T &value) {
        [[unlikely]] if (triggered) { stream << value; }
        return *this;
    }
    ~ErrorReport() noexcept(false) {
        [[likely]] if (!triggered) { return; }
        stream << "\n";
        std::cerr << stream.str();
        abort();
    }

  private:
    bool triggered;
    std::ostringstream stream;
};

#define internal_assert(cond) ErrorReport((cond), #cond, __FILE__, __LINE__)
#define internal_error ErrorReport(false, nullptr, __FILE__, __LINE__)

} // namespace bonsai
