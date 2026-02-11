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

} // namespace ssa
} // namespace ir
} // namespace bonsai
