#include "Lower/RecLoops.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

size_t func_counter = 0;
std::string unique_func_name() {
    return "_recloop_func" + std::to_string(func_counter++);
}

// The names of the objects lowering invented in this body, which nothing else
// refers to (see Allocate::unaliased).
std::set<std::string> unaliased_allocations(const Stmt &body) {
    struct Finder : public Visitor {
        std::set<std::string> names;
        void visit(const Allocate *node) override {
            if (node->unaliased) {
                names.insert(node->loc.base);
            }
            Visitor::visit(node);
        }
    };
    Finder finder;
    body.accept(&finder);
    return finder.names;
}

struct LowerRecLoopsImpl : public Mutator {
    FuncMap new_funcs;

    // Unaliased objects of the function being lowered.
    std::set<std::string> unaliased;

    Expr current_func;
    std::vector<Expr> current_args;

    Stmt visit(const RecLoop *node) override {
        std::vector<Expr> call_args(node->args.size());
        // The same arguments as a step of the recursion sees them: whatever
        // this invocation is holding, which a `from` below then replaces at
        // the positions it recurses on. Distinct from the values the first
        // call passes, which are read in the enclosing scope and mean nothing
        // inside the function.
        std::vector<Expr> step_args(node->args.size());
        std::vector<Function::Argument> f_args(node->args.size());
        std::set<std::string> carried;
        for (size_t i = 0; i < node->args.size(); i++) {
            // Where the walk begins. Evaluated here, in the scope the
            // recursion was written in, which is the only place the names it
            // is built from are bound.
            call_args[i] = mutate(node->args[i].init);
            step_args[i] =
                Var::make(node->args[i].var.type, node->args[i].var.name);
            f_args[i].name = node->args[i].var.name;
            f_args[i].type = node->args[i].var.type;
            f_args[i].mutating = false;
            carried.insert(node->args[i].var.name);
        }
        // Everything else the body reads becomes a parameter too. Taken from
        // the body rather than from the whole node: a starting value is read
        // by the caller, so a name that only appears there is not something
        // this function needs to be given.
        std::vector<TypedVar> vars;
        for (auto &var : gather_free_vars(node->body)) {
            if (carried.insert(var.name).second) {
                vars.push_back(std::move(var));
            }
        }
        auto mutables = mutated_variables(node->body);
        for (const auto &var : vars) {
            call_args.push_back(Var::make(var.type, var.name));
            step_args.push_back(Var::make(var.type, var.name));
            // A free variable that lowering invented for this traversal --
            // the accumulator it folds into -- is passed in as its own
            // argument, and nothing else the traversal can reach refers to
            // it. Saying so is what keeps the traversal from re-reading
            // everything else it was given after each write (see
            // Allocate::unaliased).
            f_args.emplace_back(var.name, var.type, Expr(),
                                mutables.contains(var.name),
                                unaliased.contains(var.name));
        }

        std::string func_name = unique_func_name();
        std::shared_ptr<Function> func = std::make_shared<Function>(
            func_name, std::move(f_args), Void_t::make(), Stmt(),
            Function::InterfaceList{}, std::vector<Function::Attribute>{});

        Expr fexpr = Var::make(func->call_type(), func_name);
        // Recursions nest: a query over a tree held in an element walks the
        // outer tree and, at each element, walks that element's own. Each
        // becomes its own function, so the one being built is put aside while
        // the inner one is extracted and picked up again after -- a `from` in
        // the outer body after that point still means the outer function.
        Expr outer_func = std::move(current_func);
        std::vector<Expr> outer_args = std::move(current_args);
        current_func = fexpr;
        current_args = std::move(step_args);
        func->body = Sequence::make({mutate(node->body), Return::make()});
        current_func = std::move(outer_func);
        current_args = std::move(outer_args);

        new_funcs[func_name] = func;

        return CallStmt::make(std::move(fexpr), std::move(call_args));
    }

    Stmt visit(const YieldFrom *node) override {
        internal_assert(current_func.defined());
        auto ids = break_tuple(node->value);
        internal_assert(!ids.empty()) << "`from` with nothing to recurse into";

        // How many of the leading arguments each recursion replaces: the whole
        // tuple where the recursion carries one, and otherwise just the node.
        // Every branch of a `from` has the same type, so measuring the first
        // measures all of them, but the rest are checked below rather than
        // assumed -- getting this wrong builds a call with the arguments of
        // two different functions interleaved, which type checking downstream
        // would not obviously catch.
        size_t width = 1;
        if (const Tuple_t *tuple_t = ids.front().type().as<Tuple_t>()) {
            internal_assert(tuple_t->etypes.size() < current_args.size());
            width = tuple_t->etypes.size();
        }

        std::vector<size_t> varying_at;
        varying_at.reserve(width);
        for (size_t i = 0; i < width; i++) {
            varying_at.push_back(i);
        }

        std::vector<std::vector<Expr>> varying;
        varying.reserve(ids.size());
        for (const auto &id : ids) {
            std::vector<Expr> vs;
            vs.reserve(width);
            if (const Tuple_t *tuple_t = id.type().as<Tuple_t>()) {
                internal_assert(tuple_t->etypes.size() == width)
                    << "`from` given recursions of differing arity: " << width
                    << " and " << tuple_t->etypes.size();
                for (size_t i = 0; i < width; i++) {
                    vs.push_back(Extract::make(id, i));
                }
            } else {
                internal_assert(width == 1)
                    << "`from` given recursions of differing arity: " << width
                    << " and 1";
                vs.push_back(id);
            }
            varying.push_back(std::move(vs));
        }

        internal_assert(node->keys.empty() || node->keys.size() == ids.size())
            << "A sort() gave " << node->keys.size() << " keys to a `from` "
            << "with " << ids.size() << " branches";

        // One node rather than N calls, so that a schedule still has something
        // to reorder. See IR/Stmt.h; the backends are what split it up.
        return MultiRecurse::make(current_func, current_args,
                                  std::move(varying_at), std::move(varying),
                                  node->keys);
    }
};

} // namespace

ir::FuncMap LowerRecLoops::run(ir::FuncMap funcs,
                               const CompilerOptions &options) const {
    LowerRecLoopsImpl lowerer;
    for (const auto &[name, func] : funcs) {
        lowerer.unaliased = unaliased_allocations(func->body);
        func->body = lowerer.mutate(func->body);
    }

    for (auto &[name, func] : lowerer.new_funcs) {
        auto [_, inserted] = funcs.try_emplace(name, std::move(func));
        internal_assert(inserted)
            << "Failed to insert recursive lowering: " << name;
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
