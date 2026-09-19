#include "CLI/CLI.h"
#include "Error.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
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

// The first whitespace-separated token of a command, e.g. the program it runs.
std::string first_token(const std::string &command) {
    std::istringstream stream(command);
    std::string token;
    stream >> token;
    return token;
}

// Does the command contain `flag` as a token of its own?
bool has_token(const std::string &command, const std::string &flag) {
    std::istringstream stream(command);
    std::string token;
    while (stream >> token) {
        if (token == flag) {
            return true;
        }
    }
    return false;
}

// Where the C++ compiler a command names is installed: the directory above its
// `bin`, found by resolving the name on PATH and following the symlink that
// `clang++` usually is. Empty when the command is not a C++ compiler or it
// cannot be found; the shell will then report the latter itself.
std::string compiler_prefix(const std::string &command) {
    const std::string name = first_token(command);
    if (!name.ends_with("clang++") && !name.ends_with("g++") &&
        !name.ends_with("c++")) {
        return "";
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path found;
    if (name.find('/') != std::string::npos) {
        found = name;
    } else if (const char *path = std::getenv("PATH")) {
        std::istringstream dirs(path);
        std::string dir;
        while (std::getline(dirs, dir, ':')) {
            const fs::path candidate = fs::path(dir) / name;
            const fs::file_status status = fs::status(candidate, ec);
            if (!ec && fs::is_regular_file(status) &&
                (status.permissions() & fs::perms::owner_exec) !=
                    fs::perms::none) {
                found = candidate;
                break;
            }
        }
    }
    if (found.empty()) {
        return "";
    }
    const fs::path resolved = fs::canonical(found, ec);
    if (ec) {
        return "";
    }
    return resolved.parent_path().parent_path().string();
}

// What the environment adds to a test's C++ commands: Intel TBB, when it is
// installed beside the compiler -- which is where a conda environment puts it,
// and how apps/pbrt/compare.sh finds it too.
//
// The runtime's parallel loop (runtime/bonsai_parallel.h) runs on TBB wherever
// TBB's header is reachable and on std::thread where it is not, and a driver
// built on TBB has to link the library. Whether the header is reachable
// depends on the compiler's default include path -- conda's clang searches its
// environment's `include`, a system clang does not -- so a test that spelled
// the link flag itself would fail on one machine or the other. The test's
// command line stays as written and as echoed, which is what keeps the goldens
// the same from one machine to the next; what the environment supplies is
// added here when the command runs: the include directory for a step that
// compiles, so that every driver in an environment with TBB is built on it,
// and the library, its search path and its run path for a step that links.
std::string environment_flags(const std::string &command) {
    static std::map<std::string, std::string> prefixes;
    const std::string name = first_token(command);
    auto it = prefixes.find(name);
    if (it == prefixes.end()) {
        it = prefixes.emplace(name, compiler_prefix(command)).first;
    }
    const std::string &prefix = it->second;
    if (prefix.empty()) {
        return "";
    }
    namespace fs = std::filesystem;
    if (!fs::exists(fs::path(prefix) / "include" / "tbb" / "parallel_for.h")) {
        return "";
    }
    const bool compiles_only = has_token(command, "-c") ||
                               has_token(command, "-E") ||
                               has_token(command, "-S") ||
                               has_token(command, "-fsyntax-only");
    if (compiles_only) {
        return " -isystem " + prefix + "/include";
    }
    return " -L" + prefix + "/lib -Wl,-rpath," + prefix + "/lib -ltbb";
}

// Runs the commands using default shell. Any non-`rm` commands are printed, as
// the test wrote them; see environment_flags for what is added when they run.
void run_commands(const std::vector<std::string> &commands) {
    int rc = 0;
    for (const std::string &command : commands) {
        if (!command.starts_with("rm")) {
            // Don't print `rm` commands.
            std::cout << "[test] " << command << '\n';
        }

        // Launch via the default shell.
        rc = std::system((command + environment_flags(command)).c_str());
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
