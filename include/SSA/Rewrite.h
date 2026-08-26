#pragma once

#include "SSA/SSA.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

using FuncMap = std::map<std::string, std::shared_ptr<ssa::Function>>;

void split(FuncMap &funcs, std::string func, std::string idx, int factor, std::string outer, std::string inner, bool exact);

void loopify(FuncMap &funcs, std::string func, int size = 0);

struct Cursor {
    std::list<std::string> ids;

    std::string to_string() const;
};

struct Queue_t {
    std::string qname;
    Cursor owner;
    std::string storage;
    // TODO: make this accept non-constant sizes!
    int size;
};

void defer(FuncMap &funcs, const std::string &func, const Queue_t &queue_t,
           const std::vector<Cursor> &cursors);

// Vectorizes the parfor loop `idx` into a single SIMD gang, in the manner of
// Pharr & Mark, "ispc: A SPMD Compiler for High-Performance CPU Programming"
// (2012), with control flow handled by the strictly stronger partial
// linearization of Moll & Hack (PLDI 2018) rather than ispc's full
// if-conversion. Implemented in SSA/Vectorize.cpp.
//
// The gang width is the loop's extent, which must be constant: to vectorize a
// wider loop, split() it to the gang width first and vectorize the inner
// loop. The loop then runs exactly once and disappears.
void vectorize(FuncMap &funcs, std::string func, std::string idx);

} // namespace ssa
} // namespace ir
} // namespace bonsai
