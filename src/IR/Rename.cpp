#include "IR/Rename.h"

#include "IR/Mutator.h"
#include "IR/WriteLoc.h"

namespace bonsai {
namespace ir {

namespace {

struct Renamer : Mutator {
    explicit Renamer(const std::map<std::string, std::string> &renames)
        : renames(renames) {}

    const std::map<std::string, std::string> &renames;

    Expr visit(const Var *node) override {
        auto it = renames.find(node->name);
        if (it == renames.end()) {
            return node;
        }
        return Var::make(node->type, it->second);
    }

    std::pair<WriteLoc, bool>
    mutate_writeloc(const WriteLoc &loc) override {
        auto [renamed, not_changed] = Mutator::mutate_writeloc(loc);
        auto it = renames.find(renamed.base);
        if (it == renames.end()) {
            return {renamed, not_changed};
        }
        renamed.base = it->second;
        return {renamed, false};
    }

    Stmt visit(const LetStmt *node) override {
        auto [loc, _] = mutate_writeloc(node->loc);
        return LetStmt::make(std::move(loc), mutate(node->value));
    }

    Stmt visit(const Allocate *node) override {
        auto [loc, _] = mutate_writeloc(node->loc);
        if (node->value.defined()) {
            return Allocate::make(std::move(loc), mutate(node->value),
                                  node->memory, node->unaliased);
        }
        return Allocate::make(std::move(loc), node->memory);
    }
};

} // namespace

Stmt rename_bindings(const Stmt &stmt,
                     const std::map<std::string, std::string> &renames) {
    if (renames.empty() || !stmt.defined()) {
        return stmt;
    }
    Renamer renamer(renames);
    return renamer.mutate(stmt);
}

} // namespace ir
} // namespace bonsai
