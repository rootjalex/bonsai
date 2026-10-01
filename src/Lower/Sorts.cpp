#include "Lower/Sorts.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

using namespace ir;

std::map<std::string, Type> get_names_in_scope(const Function &func) {
    std::map<std::string, Type> args;
    for (const auto &arg : func.args) {
        args.try_emplace(arg.name, arg.type);
    }
    return args;
}

Stmt apply_sort(const Location &loc, const Expr &cost_func, Stmt stmt,
                FuncMap &funcs, std::map<std::string, Type> names_in_scope) {
    struct ApplySortImpl : public Mutator {
        const Location &loc;
        const Expr &cost_func;
        FuncMap &funcs;
        bool found_match = false;
        bool found_from = false;

        std::map<std::string, Type> names_in_scope;
        Expr current_match_arg;
        // Local names that stand for a tree held in an element's field, as the
        // schedule spells it. A traversal of one matches on a name lowering
        // invented -- `_subtree0` -- which the program cannot write, so what a
        // schedule names is the field it came from: `Instance.blas.Interior`.
        std::map<std::string, std::string> nested_paths;

        ApplySortImpl(const Location &loc, const Expr &cost_func,
                      FuncMap &funcs,
                      std::map<std::string, Type> names_in_scope)
            : loc(loc), cost_func(cost_func), funcs(funcs),
              names_in_scope(std::move(names_in_scope)) {}

        // Everything in the location but the arm, which is what a match on a
        // tree has to be to be the one this sort names.
        std::string wanted_object() const {
            std::string out = loc.names[0];
            for (size_t i = 1; i + 1 < loc.names.size(); i++) {
                out += "." + loc.names[i];
            }
            return out;
        }

        const std::string &wanted_arm() const { return loc.names.back(); }

        Stmt visit(const LetStmt *node) override {
            names_in_scope.try_emplace(node->loc.base, node->loc.base_type);
            if (const Access *access = node->value.as<Access>()) {
                if (const auto *elem = access->value.type().as<Struct_t>()) {
                    nested_paths.try_emplace(node->loc.base,
                                             elem->name + "." + access->field);
                }
            }
            return Mutator::visit(node);
        }

        Stmt visit(const Allocate *node) override {
            names_in_scope.try_emplace(node->loc.base, node->loc.base_type);
            return Mutator::visit(node);
        }

        Expr sort_cost(size_t i) const {
            const Lambda *lambda = cost_func.as<Lambda>();
            internal_assert(lambda) << cost_func;
            internal_assert(!lambda->args.empty()) << cost_func;
            const std::string &idx = lambda->args[0].name;
            Expr value = make_const(lambda->args[0].type, i);

            std::map<std::string, Expr> temp_repls;

            for (size_t i = 1; i < lambda->args.size(); i++) {
                if (const auto &iter =
                        names_in_scope.find(lambda->args[i].name);
                    iter != names_in_scope.cend()) {
                    internal_assert(equals(iter->second, lambda->args[i].type))
                        << "Failure in sort() lowering, argument: "
                        << lambda->args[i].name
                        << " of sort lambda: " << cost_func
                        << " does not match the type in scope: "
                        << iter->second;
                    continue;
                } else {
                    Expr expr =
                        Access::make(lambda->args[i].name, current_match_arg);
                    internal_assert(equals(expr.type(), lambda->args[i].type))
                        << "Failure in sort() lowering, argument: "
                        << lambda->args[i].name
                        << " of sort lambda: " << cost_func
                        << " does not match the type in scope: " << expr.type();
                    temp_repls[lambda->args[i].name] = std::move(expr);
                }
            }
            temp_repls[idx] = std::move(value);
            return opt::Simplify::simplify(replace(temp_repls, lambda->value));
        }

        // TODO(ajr): There should be a way to target only a single YieldFrom...
        Stmt visit(const YieldFrom *node) override {
            // The arm this sort names, rather than any `from` after it was
            // found. With a tree inside a tree the two are different: sorting
            // the inner traversal reaches the outer one's `from` as well, and
            // reordering that would apply a schedule where it was not asked
            // for.
            if (!current_match_arg.defined()) {
                return node;
            }
            internal_assert(!found_from)
                << "Found duplicate YieldFrom when lowering sort(): "
                << Stmt(node);
            std::vector<Expr> exprs = break_tuple(node->value);
            if (exprs.size() < 2) {
                return node; // one branch is already in order
            }
            found_from = true;

            // All this pass does is work out what each branch is worth. The
            // reordering is not here, and deliberately: doing it at this level
            // means selecting between whole subtree *values*, because the
            // layout has not run yet and a branch is still a tree reference
            // rather than an index into one. That is what the previous version
            // did -- a bitonic network of selects over those references, built
            // right here -- and it cost more than the better traversal order
            // saved.
            //
            // The keys ride along on the `from` instead. They are lowered like
            // any other expression, so a key naming the node's split axis
            // becomes the same field load anything else would, and the
            // permutation happens on SSA once a branch is a number.
            std::vector<Expr> keys(exprs.size());
            for (size_t i = 0; i < exprs.size(); i++) {
                keys[i] = sort_cost(i);
            }
            return YieldFrom::make(node->value, std::move(keys), node->conds);
        }

        Stmt visit(const Match *node) override {
            const Var *var = node->loc.as<Var>();
            internal_assert(var) << Stmt(node);

            // What this match walks, by the name a schedule can write: the
            // tree itself when the schedule named it, and the field it was
            // reached through when it is held in an element.
            const auto nested = nested_paths.find(var->name);
            const std::string &object = nested == nested_paths.cend()
                                            ? var->name
                                            : nested->second;
            if (object != wanted_object()) {
                return Mutator::visit(node);
            }

            internal_assert(!found_match)
                << "Found duplicate traversal when lowering sort(): "
                << Stmt(node);
            found_match = true;
            bool found = false;
            const size_t n = node->arms.size();
            Match::Arms new_arms(n);
            for (size_t i = 0; i < n; i++) {
                Stmt stmt = node->arms[i].second;
                if (node->arms[i].first.name() == wanted_arm()) {
                    current_match_arg = Unwrap::make(i, node->loc);
                    stmt = mutate(stmt);
                    found = true;
                    current_match_arg = Expr();
                }
                new_arms[i] = {node->arms[i].first, std::move(stmt)};
            }

            internal_assert(found)
                << "Failed to find match arm: " << wanted_arm()
                << " in match:\n"
                << Stmt(node);

            return Match::make(node->loc, std::move(new_arms),
                               node->volume_map);
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                std::string name = var->name;
                // TODO(ajr): hope to God it's impossible to have
                // self-recursion in these.
                if (name.starts_with("_traverse_tree")) {
                    internal_assert(funcs.contains(name));
                    auto temp = std::move(names_in_scope);
                    names_in_scope = get_names_in_scope(*funcs[name]);
                    funcs[name]->body = mutate(funcs[name]->body);
                    names_in_scope = std::move(temp);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    // TODO(ajr): would be 1 if this is applied to a queue.
    //
    // Two names for a tree the schedule gave a layout to -- `instances` and
    // the arm -- and three for one held in an element's field, which is named
    // by the field rather than by the local lowering bound it to:
    // `Instance.blas.Interior`.
    internal_assert(loc.names.size() >= 2)
        << "sort() names a match arm of a traversal, as `<tree>.<arm>`, but "
           "was given: "
        << loc.names.size() << " name(s)";
    ApplySortImpl mutator(loc, cost_func, funcs, std::move(names_in_scope));
    Stmt change = mutator.mutate(std::move(stmt));
    internal_assert(mutator.found_match && mutator.found_from)
        << "Failed to lower sort(): " << stmt;
    return change;
}

// `f.sort(g, |args| key)`: a sort on the call of `g`, the launch's threads
// reordered by `key` just before the stage `g` is. Shader execution
// reordering (OptiX's `optixReorder`, Ada and later): the threads of a raygen
// launch whose keys agree on their low bits are gathered to run together
// from that point on, so that a stage which branches on the key -- the
// material a hit has -- runs each branch over a full wave rather than over
// the few threads of each wave that happened to take it. The same word as
// the sort on a traversal's children: an ordering of what comes next, with
// no change to what is computed.
//
// The lambda's parameters name parameters of `g`; the key is the lambda's
// value with each replaced by what the call passes for that parameter. What
// counts as the key's bits: an ADT value sorts by which variant it is (the
// variant's index, in as many bits as the variants need -- a material's
// kind is the natural key, and the reason an ADT is accepted whole); a bool
// is one bit; an integer is its low bits, at most sixteen, since the hint's
// bits are taken from the hardware's own sort key (optix_device.h,
// optixReorder). The hint, `rt_reorder(key, bits)`, goes before the
// statement that makes the call, and is nothing anywhere but a raygen
// program (CodeGen_LLVM::codegen_rt_intrinsic).
Stmt apply_reorder(const std::string &caller, const std::string &callee,
                   const Expr &lambda_expr, Stmt body, const FuncMap &funcs) {
    const Lambda *lambda = lambda_expr.as<Lambda>();
    internal_assert(lambda != nullptr)
        << caller << ".sort(" << callee << ", ...) expects a lambda for the key";
    const auto g = funcs.find(callee);
    internal_assert(g != funcs.end());
    // Which parameter of the callee each lambda parameter stands for.
    std::vector<size_t> positions;
    for (const TypedVar &arg : lambda->args) {
        size_t at = g->second->args.size();
        for (size_t i = 0; i < g->second->args.size(); i++) {
            if (g->second->args[i].name == arg.name) {
                at = i;
            }
        }
        internal_assert(at < g->second->args.size())
            << caller << ".sort(" << callee << ", ...): the key's lambda takes `"
            << arg.name << "`, which is not a parameter of " << callee
            << "; the lambda's parameters name the callee's, and the key is "
            << "computed from what the call passes for them.";
        internal_assert(equals(arg.type, g->second->args[at].type))
            << caller << ".sort(" << callee << ", ...): the key's lambda takes `"
            << arg.name << "` as " << arg.type << ", where " << callee
            << " takes it as " << g->second->args[at].type;
        positions.push_back(at);
    }

    struct ReplaceVars : public Mutator {
        const std::map<std::string, Expr> &repls;
        explicit ReplaceVars(const std::map<std::string, Expr> &repls)
            : repls(repls) {}
        using Mutator::visit;
        Expr visit(const Var *node) override {
            const auto found = repls.find(node->name);
            return found != repls.end() ? found->second : Expr(node);
        }
    };

    // The hint for one call, from its arguments.
    const auto hint = [&](const std::vector<Expr> &args) -> Stmt {
        std::map<std::string, Expr> repls;
        for (size_t i = 0; i < lambda->args.size(); i++) {
            internal_assert(positions[i] < args.size())
                << "a call of " << callee << " passes " << args.size()
                << " arguments";
            repls[lambda->args[i].name] = args[positions[i]];
        }
        Expr key = ReplaceVars(repls).mutate(lambda->value);
        const Type u32 = UInt_t::make(32);
        uint32_t bits = 0;
        if (const ADT_t *adt = key.type().as<ADT_t>()) {
            std::vector<MatchExpr::Arm> arms;
            for (size_t v = 0; v < adt->variants.size(); v++) {
                arms.push_back({adt->variant_name(v), UIntImm::make(u32, v)});
            }
            key = MatchExpr::make(std::move(key), std::move(arms));
            while ((size_t(1) << bits) < adt->variants.size()) {
                bits++;
            }
            bits = std::max<uint32_t>(bits, 1);
        } else if (key.type().is<Bool_t>()) {
            key = cast(u32, key);
            bits = 1;
        } else if (key.type().is_int() || key.type().is_uint()) {
            bits = std::min<uint32_t>(key.type().bits(), 16);
            key = cast(u32, key);
        } else {
            internal_error << caller << ".sort(" << callee << ", ...): the key is "
                           << key << " of type " << key.type()
                           << "; a key is a variant (an ADT value), a bool, or "
                           << "an integer.";
        }
        return LetStmt::make(WriteLoc("_rt_reordered", u32),
                             Intrinsic::make(Intrinsic::rt_reorder,
                                             {key, UIntImm::make(u32, bits)}));
    };

    // The calls of the callee in a statement (or an expression): each one's
    // arguments.
    struct FindCalls : public Visitor {
        const std::string &callee;
        std::vector<std::vector<Expr>> found;
        explicit FindCalls(const std::string &callee) : callee(callee) {}
        using Visitor::visit;
        void visit(const Call *node) override {
            if (const Var *f = node->func.as<Var>(); f && f->name == callee) {
                found.push_back(node->args);
            }
            Visitor::visit(node);
        }
        void visit(const CallStmt *node) override {
            if (const Var *f = node->func.as<Var>(); f && f->name == callee) {
                found.push_back(node->args);
            }
            Visitor::visit(node);
        }
    };

    // The hint before the statement that makes the call: whichever simple
    // statement it is -- a call statement, a binding, a store, a return, an
    // accumulate -- found by walking down through the compound ones.
    struct Insert : public Mutator {
        const std::string &callee;
        const std::function<Stmt(const std::vector<Expr> &)> &hint;
        size_t placed = 0;
        Insert(const std::string &callee,
               const std::function<Stmt(const std::vector<Expr> &)> &hint)
            : callee(callee), hint(hint) {}
        using Mutator::mutate;
        Stmt mutate(const Stmt &stmt) override {
            if (!stmt.defined()) {
                return stmt;
            }
            switch (stmt.node_type()) {
            case IRStmtEnum::Sequence:
            case IRStmtEnum::IfElse:
            case IRStmtEnum::SwitchStmt:
            case IRStmtEnum::DoWhile:
            case IRStmtEnum::While:
            case IRStmtEnum::Label:
            case IRStmtEnum::RecLoop:
            case IRStmtEnum::Match:
            case IRStmtEnum::MatchVariant:
            case IRStmtEnum::Iterate:
            case IRStmtEnum::Scan:
            case IRStmtEnum::ForAll:
            case IRStmtEnum::ForEach:
            case IRStmtEnum::ParFor:
            case IRStmtEnum::Launch:
                return Mutator::mutate(stmt);
            default:
                break;
            }
            FindCalls finder(callee);
            stmt.accept(&finder);
            if (finder.found.size() != 1) {
                return Mutator::mutate(stmt);
            }
            placed++;
            return Sequence::make({hint(finder.found[0]), Mutator::mutate(stmt)});
        }
    };
    const std::function<Stmt(const std::vector<Expr> &)> hint_fn = hint;
    Insert inserter(callee, hint_fn);
    body = inserter.mutate(std::move(body));

    // Every call was given its hint: one the walk above did not reach --
    // inside a condition, or two in one statement -- is refused rather than
    // left unsorted.
    FindCalls counter(callee);
    body.accept(&counter);
    internal_assert(counter.found.size() == inserter.placed)
        << caller << ".sort(" << callee << ", ...): " << caller << " calls "
        << callee << " " << counter.found.size() << " time(s), and the hint "
        << "could be placed before " << inserter.placed << " of them; a call "
        << "in a condition, or two in one statement, has no place for it.";
    internal_assert(!counter.found.empty())
        << caller << ".sort(" << callee << ", ...): " << caller
        << " does not call " << callee;
    return body;
}

Program LowerSorts::run(Program program, const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }

    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    TransformMap &transforms = program.schedules[Target::Host].func_transforms;

    if (transforms.empty()) {
        return program;
    }

    for (const auto &[name, ts] : transforms) {
        auto fiter = program.funcs.find(name);
        internal_assert(fiter != program.funcs.end());

        auto &func = fiter->second;

        Stmt body = std::move(func->body);

        size_t counter = 0;
        for (size_t i = 0; i < ts.size(); i++) {
            const auto &t = ts[i];
            if (std::holds_alternative<Sort>(t)) {
                if (counter != i) {
                    internal_error
                        << "Bonsai expects sort() to be applied before "
                           "any other scheduling primitives: "
                        << name;
                }
                counter++;
                const Sort &sort = std::get<Sort>(t);
                // A sort on a call, when the location is a function the
                // program defines: `f.sort(g, |args| key)`.
                if (sort.loc.names.size() == 1 &&
                    program.funcs.contains(sort.loc.names[0])) {
                    body = apply_reorder(name, sort.loc.names[0], sort.lambda,
                                         std::move(body), program.funcs);
                    continue;
                }
                body = apply_sort(sort.loc, sort.lambda, std::move(body),
                                  program.funcs, get_names_in_scope(*func));
            }
        }
        func->body = std::move(body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
