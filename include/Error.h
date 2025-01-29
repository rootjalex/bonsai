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
        [[likely]] if (!triggered) { return; }
        // Print the file path proceeding the root directory (inclusive).
        constexpr char rootDirectory[] = "bonsai";
        std::string F(file);
        const size_t pos = F.find(rootDirectory);
        F = F.substr(pos, F.length());
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
        if (triggered) {
            stream << "\n";
            std::cerr << stream.str();
            abort();
        }
    }

  private:
    bool triggered;
    std::ostringstream stream;
};

#define internal_assert(cond) ErrorReport((cond), #cond, __FILE__, __LINE__)
#define internal_error ErrorReport(false, nullptr, __FILE__, __LINE__)

} // namespace bonsai
