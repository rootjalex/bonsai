#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// macOS/clang includes for stack traces
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include "Log.h"

namespace bonsai {

class Error final : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class StackTraceCollector {
  public:
    // Collects the stack traces for macos, skipping the first `skip_frames`
    // symbols.
    static std::vector<std::string> collect_stack_trace(int skip_frames) {
        std::vector<std::string> trace;

        const int max_frames = 128;
        void *buffer[max_frames];

        int frame_count = backtrace(buffer, max_frames);
        if (frame_count == max_frames) {
            LOG_WARN << "Maximum capacity for frame count saturated: "
                     << frame_count << ". This may result in missing symbols.";
        }
        char **symbols = backtrace_symbols(buffer, frame_count);
        if (symbols == nullptr) {
            return trace;
        }

        for (int i = skip_frames; i < frame_count; ++i) {
            std::string frame = demangle_symbol(symbols[i]);
            if (frame.empty()) {
                continue;
            }
            trace.push_back(frame);
        }
        free(symbols);
        return trace;
    }

  private:
    // Attempts to demangle the symbol, and returns the original symbol if it
    // fails.
    static std::string demangle_symbol(const char *symbol) {
        std::string result(symbol);
        // MacOS format: <frame> <binary> <address> <mangled_name> + <offset>
        //
        // e.g.,
        // `4 foo 0x0000000102479860 _ZN6bonsai11FooC1EPKcS2_m + 42`
        size_t address_start = result.find("0x");
        if (address_start == std::string::npos) {
            return result;
        }
        // Find the space after the address.
        size_t name_start = result.find(' ', address_start);
        if (name_start == std::string::npos) {
            return result;
        }
        ++name_start; // ...and skip it.

        // Find the end of the mangled name.
        size_t name_end = result.find(" + ", name_start);
        if (name_end == std::string::npos) {
            name_end = result.length();
        }

        std::string mangled_name =
            result.substr(name_start, name_end - name_start);
        // Remove any trailing whitespace.
        size_t last_nonspace = mangled_name.find_last_not_of(" \t\n\r");
        if (last_nonspace != std::string::npos) {
            mangled_name = mangled_name.substr(0, last_nonspace + 1);
        }

        // https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html
        int status = 0;
        char *demangled_name = abi::__cxa_demangle(mangled_name.c_str(),
                                                   /*output_buffer=*/nullptr,
                                                   /*length=*/nullptr, &status);
        if (status == 0 && demangled_name) {
            return demangled_name;
        }

        // If demangling fails but we found a name, return it.
        if (!mangled_name.empty()) {
            return mangled_name;
        }
        // If all else fails, return the original symbol.
        return result;
    }
};

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
        add_stack_trace();
        throw Error(stream.str());
    }

  private:
    std::ostringstream stream;

    void add_stack_trace() {
        // Skip redundant error-reporting symbols.
        auto stack_trace =
            StackTraceCollector::collect_stack_trace(/*skip_frames=*/4);
        if (!stack_trace.empty()) {
            stream << "[stack trace]\n";
            for (size_t i = 0; i < stack_trace.size(); ++i) {
                size_t index = stack_trace.size() - (i + 1);
                stream << "  #" << index << ": " << stack_trace[i] << "\n";
            }
        }
    }
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