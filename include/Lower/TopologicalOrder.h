
#pragma once

#include "IR/Program.h"

namespace bonsai {
namespace lower {

// Returns a list of function names in topological order based on calls.
// `undef_calls` is a flag that only builds the call graph for calls
// to untyped-functions. This is used for type inference, and will throw an
// error if it detects a cycle, e.g.
// func foo(i : i32) { return bar(i); }
// func bar(i : i32) { return foo(i); }
// Note that the following is okay if `undef_call` is set:
// func foo(i : i32) -> i32 { return bar(i); }
// func bar(i : i32) { return foo(i); }
// This is because `bar` can inferred to return an `i32`
// If `undef_calls` is not set, and there is a cycle, this returns any DFS
// ordering.
std::vector<std::string> func_topological_order(const ir::FuncMap &funcs,
                                                const bool undef_calls);

std::vector<std::string> type_topological_order(const ir::TypeMap &types);

} // namespace lower
} // namespace bonsai
