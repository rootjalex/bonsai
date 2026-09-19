#pragma once

#include <string>
#include <vector>

#include "IR/Program.h"

namespace bonsai {
namespace parser {

ir::Program parse(const std::string &filename);

// Parses the files, in order, into one program: each after the first is read
// as if it had been imported at the end of the one before it, so a schedule
// can live in a file of its own beside the program it schedules. Every
// `schedule` block found, in whichever file, contributes to the program's one
// schedule (see Parser::parse_schedule).
ir::Program parse(const std::vector<std::string> &filenames);

} // namespace parser
} // namespace bonsai
