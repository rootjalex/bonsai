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
        std::vector<Function::Argument> f_args(node->args.size());
        for (size_t i = 0; i < node->args.size(); i++) {
            call_args[i] = node->args[i].type.is_numeric()
                               ? make_zero(node->args[i].type)
                               : node->args[i];
            f_args[i].name = node->args[i].name;
            f_args[i].type = node->args[i].type;
            f_args[i].mutating = false;
        }
        std::vector<TypedVar> vars = gather_free_vars(node);
        auto mutables = mutated_variables(node->body);
        for (const auto &var : vars) {
            call_args.push_back(Var::make(var.type, var.name));
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
        internal_assert(!current_func.defined());
        current_func = fexpr;
        current_args = call_args;
        func->body = Sequence::make({mutate(node->body), Return::make()});
        current_func = Expr();
        current_args.clear();

        new_funcs[func_name] = func;

        return CallStmt::make(std::move(fexpr), std::move(call_args));
    }

    Stmt visit(const YieldFrom *node) override {
        internal_assert(current_func.defined());
        // TODO(ajr): handle sorting and compression.
        auto ids = break_tuple(node->value);
        std::vector<Stmt> stmts;
        stmts.reserve(ids.size());

        // Make n recursive calls.
        for (const auto &id : ids) {
            std::vector<Expr> call_args = current_args;
            if (const Tuple_t *tuple_t = id.type().as<Tuple_t>()) {
                internal_assert(tuple_t->etypes.size() < current_args.size());
                for (size_t i = 0; i < tuple_t->etypes.size(); i++) {
                    call_args[i] = Extract::make(id, i);
                }
            } else {
                call_args[0] = id;
            }
            stmts.push_back(CallStmt::make(current_func, std::move(call_args)));
        }
        return Sequence::make(std::move(stmts));
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
