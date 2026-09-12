#pragma once

#include "IR/Stmt.h"

#include <map>
#include <string>

namespace bonsai {
namespace ir {

// Renames variables throughout `stmt`, bindings and uses alike: every `let`
// and allocation of a name in `renames`, the base of every store to it, and
// every read of it. This is what lets a copy of a body sit beside the
// original -- an inlined function's, or a run of statements duplicated into
// both arms of a branch -- without the two binding the same names.
Stmt rename_bindings(const Stmt &stmt,
                     const std::map<std::string, std::string> &renames);

} // namespace ir
} // namespace bonsai
