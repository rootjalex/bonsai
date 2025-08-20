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

/* static */ BuildIR BuildRule::make(std::string field, bonsai::ir::Expr expr) {
    internal_assert(!field.empty()) << "BuildRule::make received empty field";

    BuildRule *node = new BuildRule;
    node->field = std::move(field);
    node->expr = std::move(expr);
    return node;
}
/* static */ BuildIR BuildSequence::make(std::vector<BuildIR> sequence) {
    internal_assert(!sequence.empty())
        << "BuildSequence::make received empty sequence";

    BuildSequence *node = new BuildSequence;
    node->sequence = std::move(sequence);
    return node;
}
/* static */ BuildIR BuildRecurse::make(std::string field) {
    internal_assert(!field.empty())
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
