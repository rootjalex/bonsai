#include "CLI/CLI.h"
#include "Error.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#include <thread>

#include <filesystem>
#include <unistd.h>

// TODO: port to MSVC

// In POSIX, the read end of a pipe is numbered 0 and the write end is 1.
enum Pipes { READ, WRITE };

// Wrap common POSIX IO functions to throw errors on failure.
namespace io {

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

// Capture writes to a FILE* (either stdout / stderr) into a std::string.
//
// A pipe only buffers so much -- 64 KiB on Linux -- so the reading has to
// happen while the writing does. Draining afterwards deadlocks the moment a
// test produces more output than fits, which a golden of generated code
// reaches easily.
class Capture {
  public:
    Capture(FILE *file, std::string &output);
    ~Capture() noexcept(false);

  private:
    std::string &output;
    std::stringstream stream;
    std::thread reader;
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

    // `fd` is now the only handle on the write end, so restoring it in the
    // destructor is what gives this thread its EOF.
    reader = std::thread([this] {
        std::array<char, 1025> buffer;
        for (;;) {
            const int bytes_read =
                ::read(this->pipe[READ], buffer.data(), buffer.size() - 1);
            if (bytes_read <= 0) {
                // 0 is EOF; a negative value is an error we cannot report from
                // here, and either way there is nothing further to read.
                return;
            }
            buffer[bytes_read] = 0;
            this->stream << buffer.data();
        }
    });
}

Capture::~Capture() noexcept(false) {
    io::dup2(old_fd, fd);

    reader.join();
    output = stream.str();

    io::close(old_fd);
    io::close(pipe[READ]);
}

namespace {

// Prepare command line arguments for a test case. Check the first line for the
// magic `! flags:` comment.
std::vector<std::string> get_flags_for_file(const std::string &filename) {
    std::vector<std::string> flags = {"-i", filename};
    std::ifstream file(filename);
    constexpr std::string_view TAG = "//! flags:";
    if (std::string line; std::getline(file, line)) {
        if (line.starts_with(TAG)) {
            std::istringstream stream(line.substr(TAG.size()));
            std::string flag;
            while (stream >> flag) {
                flags.push_back(flag);
            }
        }
    }
    return flags;
}

// Removes any spaces before the first non-space character, e.g.,
// left_trim("   ss") => "ss"
void left_trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
}

// Replaces all instances of `from` with `to` in `s`, e.g.,
// replace_all("DABD", "AB", "-") => "D-D"
void replace_all(std::string &s, const std::string &from,
                 const std::string &to) {
    if (from.empty())
        return; // avoid infinite loop
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.length(), to);
        pos += to.length(); // advance past the replacement
    }
}

// Returns the directory from `s`, e.g.,
// get_directory("a/b/c/d.bonsai") => "a/b/c"
std::string get_directory(const std::string &s) {
    std::filesystem::path p{s};
    std::filesystem::path dir = p.parent_path();
    return dir.string(); // portable separator
}

// Retrieves commands from the second line of the file. These are assumed to be
// separated by commas. If your command contains a comma, then god speed.
std::vector<std::string> get_commands_for_file(const std::string &filename) {
    std::vector<std::string> commands;
    std::ifstream file(filename);

    std::string _, line;
    if (!std::getline(file, _) || !std::getline(file, line)) {
        return {};
    }
    constexpr std::string_view TAG = "//! commands:";

    std::string_view sv{line};
    if (!sv.starts_with(TAG)) {
        return {};
    }
    sv.remove_prefix(TAG.size());
    std::string directory = get_directory(filename);
    std::istringstream ss(std::string{sv});
    std::string token;
    while (std::getline(ss, token, ',')) {
        left_trim(token);
        // Replace instances of `$<>` with the relative path of this file.
        replace_all(token, "$<>", directory);
        commands.push_back(token);
    }
    return commands;
}

// Runs the commands using default shell. Any non-`rm` commands are printed.
void run_commands(const std::vector<std::string> &commands) {
    int rc = 0;
    for (const std::string &command : commands) {
        if (!command.starts_with("rm")) {
            // Don't print `rm` commands.
            std::cout << "[test] " << command << '\n';
        }

        // Launch via the default shell.
        rc = std::system(command.c_str());
        if (rc == 0) {
            continue;
        }
        std::cerr << "command failed (exit " << rc << ")\n";
        break;
    }
    if (rc == 0) {
        return;
    }
    // Clean up any (potentially) left-over files.
    for (const std::string &command : commands) {
        if (!command.starts_with("rm")) {
            continue;
        }
        std::system(command.c_str());
    }
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <actual_output_file> [extra flags...]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_file = argv[1];
    std::string actual_output_file = argv[2];

    // Flags a whole directory of tests shares, supplied by CMake rather than
    // repeated in every `//! flags:` header -- see add_bonsai_tests. They are
    // appended after the file's own flags, so a test can still say something
    // different for itself.
    std::vector<std::string> extra_flags(argv + 3, argv + argc);

    std::string stdout_s, stderr_s;
    int code = EXIT_FAILURE;

    try {
        Capture capout(stdout, stdout_s);
        Capture caperr(stderr, stderr_s);
        std::vector<std::string> flags = get_flags_for_file(input_file);
        flags.insert(flags.end(), extra_flags.begin(), extra_flags.end());
        std::vector<std::string> commands = get_commands_for_file(input_file);
        code = run(bonsai::cli::parse(flags));
        run_commands(commands);
    } catch (const std::system_error &e) {
        // This might not work if stderr is half-captured, but might as well
        // try.
        std::cerr << "Error while capturing output:" << e.what() << std::endl;
        return EXIT_FAILURE;
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
