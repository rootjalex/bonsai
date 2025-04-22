#include "Opt/Unswitch.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

using namespace ir;

namespace {

struct UnswitchImpl : public Mutator {
    // TODO: do classic loop unswitching as well.

    // if (a) {
    //   if (b) { foo() }
    // } else {
    //   if (b) { bar() }
    // }
    // ->
    // if (b) {
    //   if (a) { foo() }
    //   else { bar() }
    // }
    Stmt visit(const IfElse *node) override {
        Stmt stmt = Mutator::visit(node);
        node = stmt.as<IfElse>();
        internal_assert(node); // this rewrite shouldn't erase if statements.

        // No else body, can't apply this rewrite.
        if (!node->else_body.defined()) {
            return node;
        }
        const IfElse *then_if = node->then_body.as<IfElse>();
        const IfElse *else_if = node->else_body.as<IfElse>();
        if (then_if == nullptr || else_if == nullptr) {
            return node;
        }
        if (then_if->else_body.defined() || else_if->else_body.defined()) {
            // Rewrite results in same sized IR, no reduction order.
            // TODO(ajr): can we use comparisons of `cond` to make a
            // reduction order?
            return node;
        }
        if (equals(then_if->cond, else_if->cond)) {
            // Success!
            Stmt inner = IfElse::make(node->cond, then_if->then_body,
                                      else_if->then_body);
            return IfElse::make(then_if->cond, std::move(inner));
        }
        return node;
    }

    // if (a) { foo(); }
    // if (a) { bar(); }
    // =>
    // if (a) { foo(); bar(); }
    Stmt visit(const Sequence *node) override {
        Stmt stmt = Mutator::visit(node);
        node = stmt.as<Sequence>();
        internal_assert(node); // this rewrite shouldn't erase sequences;

        std::vector<Stmt> new_stmts;
        new_stmts.reserve(node->stmts.size());
        bool changed = false;

        for (size_t i = 0; i < node->stmts.size(); ++i) {
            Stmt s = mutate(node->stmts[i]);

            // Try to merge with previous statement if both are ifs with same
            // cond.
            if (!new_stmts.empty()) {
                const auto *prev_if = new_stmts.back().as<IfElse>();
                const auto *curr_if = s.as<IfElse>();

                // Equal if statements.
                // Can only merge if iff conditions are equal, and the body of
                // the first does not mutate any variable read in the condition
                // of the second.
                if (prev_if && curr_if &&
                    equals(prev_if->cond, curr_if->cond) &&
                    !reads(curr_if->cond,
                           mutated_variables(new_stmts.back()))) {

                    // Useful helper for merging flat sequences.
                    auto insert_sequence = [](std::vector<Stmt> &acc,
                                              Stmt stmt) {
                        if (!stmt.defined()) {
                            return;
                        }
                        const Sequence *seq = stmt.as<Sequence>();
                        if (seq) {
                            acc.insert(acc.end(), seq->stmts.begin(),
                                       seq->stmts.end());
                        } else {
                            acc.emplace_back(std::move(stmt));
                        }
                    };

                    // Merge the then_bodies into a Sequence
                    std::vector<Stmt> merged_then_body;
                    insert_sequence(merged_then_body, prev_if->then_body);
                    insert_sequence(merged_then_body, curr_if->then_body);

                    // Merge the else_bodies into a sequence.
                    std::vector<Stmt> merged_else_body;
                    insert_sequence(merged_else_body, prev_if->else_body);
                    insert_sequence(merged_else_body, curr_if->else_body);

                    internal_assert(!merged_then_body.empty() &&
                                    merged_then_body.size() > 1);

                    Stmt merged_then =
                        Sequence::make(std::move(merged_then_body));
                    // Try to recursively mutate, this sequence may be
                    // merge-able as well.
                    merged_then = mutate(merged_then);
                    Stmt merged_else;
                    if (!merged_else_body.empty()) {
                        if (merged_else_body.size() == 1) {
                            merged_else = std::move(merged_else_body[0]);
                        } else {
                            merged_else =
                                Sequence::make(std::move(merged_else_body));
                            // Try to recursively mutate, this sequence may be
                            // merge-able as well.
                            merged_else = mutate(merged_else);
                        }
                    }
                    Stmt new_if =
                        IfElse::make(prev_if->cond, std::move(merged_then),
                                     std::move(merged_else));

                    new_stmts.back() = std::move(new_if);
                    changed = true;
                    continue;
                }
            }
            new_stmts.emplace_back(std::move(s));
        }
        if (!changed) {
            return node;
        } else {
            return Sequence::make(std::move(new_stmts));
        }
    }

    // forall var in (...) { if (a) { body } } => if (a) { forall var in (...) {
    // body }} if and only if a does not use var!
    Stmt visit(const ForAll *node) override {
        Stmt stmt = Mutator::visit(node);
        node = stmt.as<ForAll>();
        internal_assert(node);

        // TODO(ajr): partition condition into varying and unvarying conditions.
        if (const IfElse *if_else = node->body.as<IfElse>()) {
            std::set<std::string> varying =
                node->header.defined() ? assigned_variables(node->header)
                                       : std::set<std::string>{};
            varying.insert(node->index);
            std::set<std::string> mutating = mutated_variables(node->body);
            varying.insert(mutating.begin(), mutating.end());

            if (!reads(if_else->cond, varying)) {
                // Perform loop unswitching.
                ir::Stmt body0 = ForAll::make(node->index, node->header,
                                              node->slice, if_else->then_body);
                ir::Stmt body1 =
                    if_else->else_body.defined()
                        ? ForAll::make(node->index, node->header, node->slice,
                                       if_else->else_body)
                        : if_else->else_body;
                return IfElse::make(if_else->cond, std::move(body0),
                                    std::move(body1));
            }
        }

        return stmt;
    }
};

Stmt unswitch_stmt(Stmt stmt) { return UnswitchImpl().mutate(std::move(stmt)); }
} // namespace

FuncMap Unswitch::run(FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        func->body = unswitch_stmt(std::move(func->body));
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
