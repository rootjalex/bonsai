#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bonsai {

// TODO: Halide's has some weird magic I don't understand, but I probably should
// try to...

class ErrorReport {
  public:
    ErrorReport(bool cond, const char *cond_str, const char *file, size_t line)
        : triggered(!cond) {
        if (triggered) {
            if (cond_str) {
                stream << "Assertion failed: " << cond_str << " at " << file
                       << ":" << line << "\n";
            } else {
                stream << "Error at: " << file << ":" << line << "\n";
            }
        }
    }

    template <typename T>
    ErrorReport &operator<<(const T &value) {
        if (triggered) {
            stream << value;
        }
        return *this;
    }
    ~ErrorReport() noexcept(false) {
        if (triggered) {
            stream << "\n";
            throw std::runtime_error(stream.str());
        }
    }

  private:
    bool triggered;
    std::ostringstream stream;
};

#define internal_assert(cond) ErrorReport((cond), #cond, __FILE__, __LINE__)
#define internal_error ErrorReport(false, nullptr, __FILE__, __LINE__)

} // namespace bonsai
