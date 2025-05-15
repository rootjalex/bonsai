#include "Lower/Yields.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

//
struct RewriteYieldToReturn : public ir::Mutator {
    ir::Stmt visit(const ir::Yield *node) override {
        return ir::Return::make(node->value);
    }
};

struct FindSoloYield : public ir::Visitor {
    // count of loops we're in.
    uint32_t in_loop = 0;
    // Count of out-of-loop yields found.
    uint32_t n_yields = 0;
    // Count of in-loop yields found.
    uint32_t n_yields_in_loop = 0;

    // This should have been lowered already.
    RESTRICT_VISITOR(ir::ForEach);
    // TODO: support scans too.
    RESTRICT_VISITOR(ir::Scan);
    // TODO: support Iterates too
    RESTRICT_VISITOR(ir::Iterate);

    void visit(const ir::ForAll *node) override {
        in_loop++;
        ir::Visitor::visit(node);
        in_loop--;
    }

    void visit(const ir::DoWhile *node) override {
        in_loop++;
        ir::Visitor::visit(node);
        in_loop--;
    }

    void visit(const ir::Yield *node) override {
        if (in_loop == 0) {
            n_yields++;
        } else {
            n_yields_in_loop++;
        }
    }
};

ir::Stmt lower_yields_impl(const ir::Stmt &stmt) {
    FindSoloYield finder;
    stmt.accept(&finder);
    internal_assert(finder.n_yields_in_loop == 0)
        << "TODO: handle dynamic-sized outputs of tree traversals: " << stmt;
    internal_assert(finder.n_yields <= 1)
        << "TODO: handle dynamic-sized outputs of tree traversals: " << stmt;
    if (finder.n_yields == 0) {
        return stmt;
    }

    RewriteYieldToReturn mutator;
    return mutator.mutate(stmt);
}

} // namespace

ir::FuncMap LowerYields::run(ir::FuncMap funcs,
                             const CompilerOptions &options) const {
    for (auto &[_, f] : funcs) {
        f->body = lower_yields_impl(f->body);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
