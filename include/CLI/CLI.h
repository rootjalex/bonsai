#pragma once

#include <string>
#include <vector>

#include "CompilerOptions.h"

namespace bonsai::cli {

struct Flags {
    CompilerOptions options;
    bool display_help = false;
};

Flags parse(int argc, char *argv[]);
Flags parse(const std::vector<std::string> &args);

int run(const Flags &flags);

} // namespace bonsai::cli
