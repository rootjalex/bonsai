#include "Lower/Prefetches.h"

#include "Error.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "Utils.h"

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace bonsai {
namespace lower {

using namespace ir;

namespace {

// The cursor as the schedule spelled it, for a message.
std::string spell(const Location &loc) {
    std::string out;
    for (const std::string &name : loc.names) {
        out += (out.empty() ? "" : ".") + name;
    }
    return out;
}

// The `_maskN[i]` the children's loop stores each child's test into
// (Lower/Trees.cpp, from_children), as a read of the same slot: the store in
// the loop's body into an array of bools at the loop's index. Undefined when
// the loop tests nothing, in which case every child is descended into and
// every child is prefetched.
Expr mask_of(const Stmt &body, const std::string &index) {
    struct Find : public Visitor {
        const std::string &index;
        Expr found;
        explicit Find(const std::string &index) : index(index) {}
        void visit(const Store *node) override {
            if (found.defined() || node->loc.accesses.size() != 1) {
                return;
            }
            const auto *array = node->loc.base_type.as<Array_t>();
            const Expr *at = std::get_if<Expr>(&node->loc.accesses[0]);
            const Var *var = at != nullptr ? at->as<Var>() : nullptr;
            if (array != nullptr && array->etype.is_bool() && var != nullptr &&
                var->name == index) {
                found = Extract::make(
                    Var::make(node->loc.base_type, node->loc.base), *at);
            }
        }
    };
    Find find(index);
    body.accept(&find);
    return find.found;
}

// The children's loop of the arm a prefetch names, found as Lower/Sorts.cpp
// finds it for a sort's keys: the match on the tree the cursor names, the
// arm, and inside it the parfor named for the field that holds the children.
// The prefetch goes at the end of that loop's body, behind the test's mask.
struct ApplyPrefetch : public Mutator {
    const Location &loc;
    const std::string &func_name;
    FuncMap &funcs;
    bool found_match = false;
    bool found_arm = false;
    bool found_loop = false;

    // The arm being rewritten, as `Unwrap(i, tree)`, while inside it.
    Expr current_match_arg;
    // The field holding the arm's children, which names their loop.
    std::string children_loop;
    // Local names that stand for a tree held in an element's field, as the
    // schedule spells it: a traversal of one matches on a name lowering
    // invented, `_subtree0`, and what a schedule names is the field it came
    // from, `Instance.blas` (see Lower/Sorts.cpp).
    std::map<std::string, std::string> nested_paths;
    // How many cache lines of each child to fetch, when the schedule said
    // (ir::Prefetch::lines); otherwise the layout lowering fetches the
    // widest arm's bytes.
    std::optional<uint64_t> lines;

    ApplyPrefetch(const Location &loc, std::optional<uint64_t> lines,
                  const std::string &func_name, FuncMap &funcs)
        : loc(loc), func_name(func_name), funcs(funcs), lines(lines) {}

    // The traversal is a function of its own that the scheduled function
    // calls (Lower/Trees.cpp, `_traverse_tree0`), rewritten in place where
    // the call is found, as Lower/Sorts.cpp does for a sort.
    Expr visit(const Call *node) override {
        if (const Var *var = node->func.as<Var>();
            var != nullptr && var->name.starts_with("_traverse_tree") &&
            funcs.contains(var->name)) {
            funcs[var->name]->body = mutate(funcs[var->name]->body);
            return node;
        }
        return Mutator::visit(node);
    }

    // `<tree>.<arm>.<field>`: everything before the arm names the tree.
    std::string wanted_object() const {
        std::string out = loc.names[0];
        for (size_t i = 1; i + 2 < loc.names.size(); i++) {
            out += "." + loc.names[i];
        }
        return out;
    }
    const std::string &wanted_arm() const {
        return loc.names[loc.names.size() - 2];
    }
    const std::string &wanted_field() const { return loc.names.back(); }

    Stmt visit(const LetStmt *node) override {
        if (const Access *access = node->value.as<Access>()) {
            if (const auto *elem = access->value.type().as<Struct_t>()) {
                nested_paths.try_emplace(node->loc.base,
                                         elem->name + "." + access->field);
            }
        }
        return Mutator::visit(node);
    }

    Stmt visit(const Match *node) override {
        const Var *var = node->loc.as<Var>();
        internal_assert(var) << Stmt(node);
        const auto nested = nested_paths.find(var->name);
        const std::string &object =
            nested == nested_paths.cend() ? var->name : nested->second;
        if (object != wanted_object()) {
            return Mutator::visit(node);
        }
        internal_assert(!found_match)
            << func_name << ".prefetch(" << spell(loc) << "): " << func_name
            << " traverses " << object << " twice, and the prefetch names "
            << "one traversal";
        found_match = true;

        const size_t n = node->arms.size();
        Match::Arms new_arms(n);
        for (size_t i = 0; i < n; i++) {
            Stmt stmt = node->arms[i].second;
            if (node->arms[i].first.name() == wanted_arm()) {
                found_arm = true;
                // The field has to hold the tree's references: one, or an
                // array or vector of them, which is what the tree lowering
                // tests in a loop named for the field.
                bool holds = false;
                for (const auto &field : node->arms[i].first.fields()) {
                    if (field.name != wanted_field()) {
                        continue;
                    }
                    const Type *element = &field.type;
                    if (const auto *a = field.type.as<Array_t>()) {
                        element = &a->etype;
                    } else if (const auto *v = field.type.as<Vector_t>()) {
                        element = &v->etype;
                    }
                    holds = element->is<Ref_t>();
                    internal_assert(holds)
                        << func_name << ".prefetch(" << spell(loc) << "): "
                        << wanted_field() << " is a " << field.type
                        << ", not a reference to the tree or an array of "
                        << "them. prefetch names the field holding the "
                        << "children a node tests, whose storage each hit "
                        << "child's is brought into cache.";
                }
                internal_assert(holds)
                    << func_name << ".prefetch(" << spell(loc) << "): "
                    << wanted_arm() << " has no field " << wanted_field();
                current_match_arg = Unwrap::make(i, node->loc);
                children_loop = wanted_field();
                stmt = mutate(stmt);
                current_match_arg = Expr();
                children_loop.clear();
            }
            new_arms[i] = {node->arms[i].first, std::move(stmt)};
        }
        internal_assert(found_arm)
            << func_name << ".prefetch(" << spell(loc) << "): " << object
            << " has no arm " << wanted_arm();
        return Match::make(node->loc, std::move(new_arms), node->volume_map);
    }

    // The children's loop: its body tests child `i` into the mask (and
    // computes its sort key, when there is a sort), and now also prefetches
    // what child `i` refers to when the test passed.
    Stmt visit(const ParFor *node) override {
        if (!current_match_arg.defined() || children_loop.empty() ||
            node->index != children_loop) {
            return Mutator::visit(node);
        }
        internal_assert(!found_loop)
            << "Two loops over the children of one arm: " << Stmt(node);
        found_loop = true;
        const Type index_t = node->slice.begin.type();
        const Expr index = Var::make(index_t, node->index);
        const Expr child =
            Extract::make(Access::make(children_loop, current_match_arg), index);
        static size_t counter = 0;
        // The reference, and the line count when the schedule gave one; the
        // layout lowering turns the pair into an address and a byte count
        // (Lower/Layouts.cpp, LowerReferencePrefetches).
        std::vector<Expr> args{child};
        if (lines.has_value()) {
            args.push_back(UIntImm::make(UInt_t::make(32), *lines));
        }
        Stmt fetch = LetStmt::make(
            WriteLoc("_prefetch" + std::to_string(counter++), Void_t::make()),
            Intrinsic::make(Intrinsic::prefetch, std::move(args)));
        if (const Expr hit = mask_of(node->body, node->index); hit.defined()) {
            fetch = IfElse::make(hit, std::move(fetch));
        }
        return ParFor::make(node->index, node->slice,
                            Sequence::make({node->body, std::move(fetch)}),
                            node->binding);
    }
};

} // namespace

Program LowerPrefetches::run(Program program,
                             const CompilerOptions &options) const {
    TransformMap &transforms = program.schedule.func_transforms;

    for (const auto &[name, ts] : transforms) {
        auto fiter = program.funcs.find(name);
        internal_assert(fiter != program.funcs.end());
        auto &func = fiter->second;
        for (const auto &t : ts) {
            const Prefetch *prefetch = std::get_if<Prefetch>(&t);
            if (prefetch == nullptr) {
                continue;
            }
            internal_assert(prefetch->loc.names.size() >= 3)
                << name << ".prefetch(" << spell(prefetch->loc)
                << "): prefetch names the field of an arm of a traversal, as "
                << "`<tree>.<arm>.<field>` -- the children a node tests.";
            ApplyPrefetch apply(prefetch->loc, prefetch->lines, name,
                                program.funcs);
            func->body = apply.mutate(func->body);
            internal_assert(apply.found_match)
                << name << ".prefetch(" << spell(prefetch->loc) << "): "
                << name << " has no traversal of " << apply.wanted_object();
            internal_assert(apply.found_loop)
                << "[unimplemented] " << name << ".prefetch("
                << spell(prefetch->loc) << "): the children of "
                << apply.wanted_arm() << " are not tested in a loop over "
                << apply.wanted_field()
                << " -- an arm whose children are named one by one has no "
                << "loop to put the prefetch in.";
        }
    }
    return program;
}

} // namespace lower
} // namespace bonsai
