#include "IR/Build.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace ir {

/* static */ BuildIR BuildLet::make(ir::Stmt stmt) {
    internal_assert(stmt.defined()) << "BuildLet::make received empty stmt";
    internal_assert(stmt.is<ir::LetStmt>());

    BuildLet *node = new BuildLet;
    node->stmt = std::move(stmt);
    return node;
}

/* static */ BuildIR BuildRule::make(ir::Expr field, ir::Expr expr) {
    internal_assert(field.defined())
        << "BuildRule::make received undefined field";

    BuildRule *node = new BuildRule;
    node->field = std::move(field);
    node->expr = std::move(expr);
    return node;
}

/* static */ BuildIR BuildRoot::make(ir::BuildIR rules) {
    internal_assert(rules.defined())
        << "BuildRule::make received undefined rules";

    BuildRoot *node = new BuildRoot;
    node->rules = std::move(rules);
    return node;
}

/* static */ BuildIR BuildSequence::make(std::vector<BuildIR> sequence) {
    internal_assert(!sequence.empty())
        << "BuildSequence::make received empty sequence";

    BuildSequence *node = new BuildSequence;
    node->sequence = std::move(sequence);
    return node;
}

/* static */ BuildIR BuildRecurse::make(ir::Expr field) {
    internal_assert(field.defined())
        << "BuildRecurse::make received empty field";

    BuildRecurse *node = new BuildRecurse;
    node->field = std::move(field);
    return node;
}

/* static */ BuildIR BuildReturn::make(ir::Expr expr) {
    internal_assert(expr.defined()) << "Return::make received undefined expr";

    BuildReturn *node = new BuildReturn;
    node->expr = std::move(expr);
    return node;
}

} // namespace ir
} // namespace bonsai
