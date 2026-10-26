#include "Opt/Sink.h"

#include "Opt/ValueClass.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace opt {

using namespace ir;

namespace {

int64_t rename_counter = 0;

std::set<std::string> free_names(const Expr &expr) {
    std::set<std::string> names;
    for (const TypedVar &v : gather_free_vars(expr)) {
        names.insert(v.name);
    }
    return names;
}

std::set<std::string> free_names(const Stmt &stmt) {
    std::set<std::string> names;
    for (const TypedVar &v : gather_free_vars(stmt)) {
        names.insert(v.name);
    }
    return names;
}

// Whether `stmt`'s subtree defines or writes `name` -- the guard against
// renaming a duplicated let's uses through an inner shadow.
bool touches_name(const Stmt &stmt, const std::string &name) {
    Crossing c = crossing_of(stmt);
    return c.barrier || c.defines.count(name) > 0 || c.writes.count(name) > 0;
}

struct SinkImpl : Mutator {
    const std::set<std::string> *side_effect_functions = nullptr;

    // A statement prepended to an arm, which may be empty.
    static Stmt prepend(Stmt head, const Stmt &arm) {
        if (!arm.defined()) {
            return Sequence::make(std::vector<Stmt>{std::move(head)});
        }
        std::vector<Stmt> stmts;
        stmts.push_back(std::move(head));
        if (const Sequence *seq = arm.as<Sequence>()) {
            stmts.insert(stmts.end(), seq->stmts.begin(), seq->stmts.end());
        } else {
            stmts.push_back(arm);
        }
        return Sequence::make(std::move(stmts));
    }

    // The let sunk into the arms of `branch` that use `name` -- moved when
    // one does, duplicated under fresh names when several do. Nothing when
    // the move is not profitable (every arm uses it) or not sound (an arm
    // shadows the name). KRS's criterion: some path must stop computing it,
    // no path may compute it more.
    std::optional<Stmt> into_arms(const Stmt &branch, const LetStmt *let,
                                  const ValueClass &klass) {
        const std::string &name = let->loc.base;
        Expr selector;                   // the branch's own read
        std::vector<Stmt> arms;          // every path, undefined ones too
        if (const IfElse *fork = branch.as<IfElse>()) {
            selector = fork->cond;
            arms = {fork->then_body, fork->else_body};
        } else if (const SwitchStmt *sw = branch.as<SwitchStmt>()) {
            selector = sw->value;
            arms = sw->arms;
        } else {
            return std::nullopt;
        }
        if (free_names(selector).count(name) > 0) {
            return std::nullopt;
        }
        // A read never enters an arm past a selector that could write.
        if (klass.memory &&
            has_side_effects(selector, *side_effect_functions)) {
            return std::nullopt;
        }
        std::vector<size_t> using_arms;
        for (size_t k = 0; k < arms.size(); k++) {
            if (arms[k].defined() && free_names(arms[k]).count(name) > 0) {
                using_arms.push_back(k);
            }
        }
        if (using_arms.empty() || using_arms.size() == arms.size()) {
            // Dead (DCE's business), or on every path (nothing improves).
            return std::nullopt;
        }
        // Moved into one arm, the let goes to the arm's head under its own
        // name: it bound the arm's free reads of the name from above the
        // branch and binds them from the arm's head the same, whatever the
        // arm holds -- a call inside the arm is no obstacle, the value being
        // pure. Duplicated into several, the uses in each arm are renamed,
        // and an arm that could rebind the name (or that holds a statement
        // the walk does not understand) is refused, so no inner shadow is
        // renamed through.
        if (using_arms.size() > 1) {
            for (const size_t k : using_arms) {
                if (touches_name(arms[k], name)) {
                    return std::nullopt;
                }
            }
        }
        // The arm the let enters is sunk again with the let at its head:
        // the arm's own statements settled before the let arrived (visit
        // (Sequence) does children first), and the let's use may be in a
        // branch nested inside the arm, which it reaches only by sliding
        // down this arm in turn -- a field read three branches deep sinks
        // three levels, one per arm entered, as lazy code motion places it
        // at its latest point whatever the nesting.
        if (using_arms.size() == 1) {
            arms[using_arms[0]] =
                mutate(prepend(LetStmt::make(let->loc, let->value),
                               arms[using_arms[0]]));
        } else {
            for (const size_t k : using_arms) {
                const std::string fresh =
                    name + "$sink" + std::to_string(rename_counter++);
                RenameVar rename;
                rename.from = name;
                rename.to = fresh;
                Stmt renamed = rename.mutate(arms[k]);
                arms[k] = mutate(prepend(
                    LetStmt::make(WriteLoc(fresh, let->loc.base_type),
                                  let->value),
                    std::move(renamed)));
            }
        }
        if (const IfElse *fork = branch.as<IfElse>()) {
            return IfElse::make(fork->cond, arms[0], arms[1],
                                fork->provenance);
        }
        const SwitchStmt *sw = branch.as<SwitchStmt>();
        return SwitchStmt::make(sw->value, std::move(arms), sw->provenance);
    }

    Stmt visit(const Sequence *node) override {
        // Children first, so an arm's own lets have settled before anything
        // sinks into it.
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());
        for (const Stmt &s : node->stmts) {
            stmts.push_back(mutate(s));
        }
        bool changed = true;
        for (int round = 0; changed && round < 64; round++) {
            changed = false;
            // Backward, so a chain's last value moves first and frees its
            // producers to follow on the next round.
            for (size_t i = stmts.size(); i-- > 0;) {
                const LetStmt *let = stmts[i].as<LetStmt>();
                if (let == nullptr || !let->loc.accesses.empty()) {
                    continue;
                }
                const ValueClass klass = classify(let->value);
                if (!klass.sinkable) {
                    continue;
                }
                const std::string &name = let->loc.base;
                const std::set<std::string> reads = free_names(let->value);
                // How far down it can go: past statements that do not use
                // it, do not redefine it or anything it reads, and -- for a
                // read -- do not touch memory.
                size_t j = i + 1;
                bool uses_at_j = false;
                while (j < stmts.size()) {
                    if (free_names(stmts[j]).count(name) > 0) {
                        uses_at_j = true;
                        break;
                    }
                    const Crossing c = crossing_of(stmts[j]);
                    if (c.barrier || c.defines.count(name) > 0) {
                        break;
                    }
                    bool blocked = klass.memory && c.memory;
                    for (const std::string &r : reads) {
                        blocked = blocked || c.defines.count(r) > 0 ||
                                  c.writes.count(r) > 0;
                    }
                    if (blocked) {
                        break;
                    }
                    j++;
                }
                if (j == i + 1 && !uses_at_j) {
                    continue; // pinned where it is
                }
                if (!uses_at_j) {
                    if (j == stmts.size()) {
                        continue; // unused here: DCE's business, not ours
                    }
                    // Sink to just above what stopped it.
                    Stmt moved = stmts[i];
                    stmts.erase(stmts.begin() + ptrdiff_t(i));
                    stmts.insert(stmts.begin() + ptrdiff_t(j - 1),
                                 std::move(moved));
                    changed = true;
                    continue;
                }
                // The first use. Adjacent (after the slide above) and a
                // branch whose selector does not read it and that nothing
                // later reads: it can enter the using arms.
                bool used_later = false;
                for (size_t k = j + 1; k < stmts.size(); k++) {
                    used_later =
                        used_later || free_names(stmts[k]).count(name) > 0;
                }
                if (!used_later) {
                    if (std::optional<Stmt> sunk =
                            into_arms(stmts[j], let, klass)) {
                        stmts[j] = std::move(*sunk);
                        stmts.erase(stmts.begin() + ptrdiff_t(i));
                        changed = true;
                        continue;
                    }
                }
                if (j - 1 > i) {
                    // Not into the branch, but up against it.
                    Stmt moved = stmts[i];
                    stmts.erase(stmts.begin() + ptrdiff_t(i));
                    stmts.insert(stmts.begin() + ptrdiff_t(j - 1),
                                 std::move(moved));
                    changed = true;
                }
            }
        }
        return Sequence::make(std::move(stmts));
    }
};

} // namespace

ir::FuncMap Sink::run(ir::FuncMap funcs, const CompilerOptions &options) const {
    const std::set<std::string> side_effects = find_side_effects(funcs);
    for (auto &[name, f] : funcs) {
        SinkImpl impl;
        impl.side_effect_functions = &side_effects;
        f->body = impl.mutate(f->body);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
