#pragma once

#include "SSA/SSA.h"

namespace bonsai {
namespace ir {
namespace ssa {

using FuncMap = std::map<std::string, std::shared_ptr<ssa::Function>>;

void split(FuncMap &funcs, std::string func, std::string idx, int factor, std::string outer, std::string inner, bool exact);

struct Cursor {
    std::vector<std::string> ids;
};

void defer(FuncMap &funcs, std::string func, std::string qname, std::string owner, std::string storage, ir::Expr size, std::vector<Cursor> cursors);

} // namespace ssa
} // namespace ir
} // namespace bonsai
