#include "CLI/CLI.h"
#include "Error.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#include <thread>

#include <unistd.h>

// TODO: port to MSVC

enum PIPES { READ, WRITE };

namespace io {

template <size_t Size>
static int read(const int fd, std::array<char, Size> &buffer) {
    return ::read(fd, buffer.data(), Size - 1);
}

static int dup(const int src) {
    int ret;
    if ((ret = ::dup(src)) < 0) {
        throw std::system_error(errno, std::system_category());
    }
    return ret;
}

static void pipe(int *pipes) {
    if (::pipe(pipes) < 0) {
        throw std::system_error(errno, std::system_category());
    }
}

static void dup2(const int src, const int dest) {
    if (::dup2(src, dest) < 0) {
        throw std::system_error(errno, std::system_category());
    }
}

static void close(int &fd) {
    if (::close(fd) < 0) {
        throw std::system_error(errno, std::system_category());
    }
    fd = -1;
}

} // namespace io

class Capture {
  public:
    Capture(FILE *file, std::string &output);
    ~Capture() noexcept(false);

  private:
    std::string &output;
    int pipe[2];
    int fd, old_fd;
};

Capture::Capture(FILE *file, std::string &output) : output(output) {
    setvbuf(file, nullptr, _IONBF, 0);

    fd = fileno(file);

    io::pipe(pipe);

    old_fd = io::dup(fd);
    io::dup2(pipe[WRITE], fd);
    io::close(pipe[WRITE]);
}

Capture::~Capture() noexcept(false) {
    io::dup2(old_fd, fd);

    std::stringstream stream;

    std::array<char, 1025> buffer;
    int bytes_read = 0;
    do {
        bytes_read = io::read(pipe[READ], buffer);

        if (bytes_read < 0) {
            throw std::system_error(errno, std::system_category());
        }

        if (bytes_read > 0) {
            buffer[bytes_read] = 0;
            stream << buffer.data();
        }
    } while (bytes_read == buffer.size() - 1);

    output = stream.str();

    io::close(old_fd);
    io::close(pipe[READ]);
}

namespace {
std::vector<std::string> get_flags_for_file(const std::string &filename) {
    std::vector<std::string> flags = {"-i", filename};
    std::ifstream file(filename);
    if (std::string line; std::getline(file, line)) {
        if (line.starts_with("//! flags:")) {
            std::istringstream stream(line.substr(11));
            std::string flag;
            while (stream >> flag) {
                flags.push_back(flag);
            }
        }
    }
    return flags;
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <actual_output_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_file = argv[1];
    std::string actual_output_file = argv[2];

    std::string stdout_s, stderr_s;
    int code = EXIT_FAILURE;

    {
        Capture capout(stdout, stdout_s);
        Capture caperr(stderr, stderr_s);
        code = run(bonsai::cli::parse(get_flags_for_file(input_file)));
    }

    std::ofstream output(actual_output_file);
    if (!stdout_s.empty()) {
        output << stdout_s;
        if (!stdout_s.ends_with('\n')) {
            output << '\n';
        }
    }
    if (code != EXIT_SUCCESS) {
        output << "---CODE---\n" << code << "\n";
    }
    if (!stderr_s.empty()) {
        output << "---STDERR---\n" << stderr_s;
        if (!stderr_s.ends_with('\n')) {
            output << '\n';
        }
    }

    return 0;
}
