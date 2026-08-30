#include "Lower/DynamicSets.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "Lower/TopologicalOrder.h"

#include "Opt/Simplify.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

using namespace ir;

namespace {

static constexpr char DYNAMIC_ALLOCATION[] = "_dyn_alloc";

// func _traverse_tree0(...) -> set<T> {
//   rec(...) { ... }
// }
// ->
// func _traverse_tree0(...) -> {T, n} {
//   alloc _dyn_alloc0 : mut T{n};
//   rec(...) { ... }
//   return _dyn_alloc0;
// }
class LowerDynamicSetImpl : public ir::Mutator {
  public:
    LowerDynamicSetImpl(Type dynamic_array_t)
        : dynamic_array_t(std::move(dynamic_array_t)) {}

    Stmt visit(const Yield *node) override {
        WriteLoc loc(DYNAMIC_ALLOCATION, dynamic_array_t);
        return AppendStmt::make(std::move(loc), node->value);
    }

    // Yielding a whole iterable -- a leaf's payload, or everything a scanned
    // subtree produced -- is one append rather than an element at a time.
    Stmt visit(const Iterate *node) override {
        WriteLoc loc(DYNAMIC_ALLOCATION, dynamic_array_t);
        return AppendStmt::make(std::move(loc), node->value);
    }

    Stmt mutate(const Stmt &stmt) override {
        if (!entry) {
            return ir::Mutator::mutate(stmt);
        }
        entry = false;
        std::vector<Stmt> stmts;
        WriteLoc loc(DYNAMIC_ALLOCATION, dynamic_array_t);
        stmts.push_back(Allocate::make(std::move(loc), Allocate::Memory::Heap));
        stmts.push_back(ir::Mutator::mutate(stmt));
        stmts.push_back(
            Return::make(Var::make(dynamic_array_t, DYNAMIC_ALLOCATION)));
        return Sequence::make(std::move(stmts));
    }

  private:
    Type dynamic_array_t;
    // Whether we are at the top level set of statements.
    bool entry = true;
};

// Retypes calls to the functions whose return type this pass changed.
class LowerDynamicSetCall : public ir::Mutator {
  public:
    LowerDynamicSetCall(const std::map<std::string, Type> &converted)
        : converted(converted) {}

    Expr visit(const Call *node) override {
        const auto retyped = retype(node->func);
        if (!retyped.defined()) {
            return ir::Mutator::visit(node);
        }
        return Call::make(retyped, node->args);
    }

    Stmt visit(const CallStmt *node) override {
        const auto retyped = retype(node->func);
        if (!retyped.defined()) {
            return ir::Mutator::visit(node);
        }
        return CallStmt::make(retyped, node->args);
    }

  private:
    Expr retype(const Expr &callee) const {
        const auto *func = callee.as<Var>();
        if (func == nullptr) {
            return Expr();
        }
        const auto it = converted.find(func->name);
        if (it == converted.cend()) {
            return Expr();
        }
        const Function_t *ftype = func->type.as<Function_t>();
        internal_assert(ftype) << func->type;
        return Var::make(Function_t::make(it->second, ftype->arg_types),
                         func->name);
    }

    const std::map<std::string, Type> &converted;
};

// Whether a function produces its set by yielding, rather than by returning
// one it built itself. Only the former needs an allocation to yield into.
bool yields(const Stmt &body) {
    struct FindYield : ir::Visitor {
        bool found = false;
        void visit(const Yield *) override { found = true; }
        void visit(const Iterate *) override { found = true; }
    };
    FindYield f;
    body.accept(&f);
    return f.found;
}

} // namespace

Program LowerDynamicSets::run(Program program,
                              const CompilerOptions &options) const {
    // Give every function that yields a set somewhere to yield into. Which
    // functions those are is a property of what they return and how they
    // produce it, not of what they are called: a traversal and the scan of a
    // whole subtree lifted out of it have the same shape and the same need.
    std::map<std::string, Type> converted;
    for (auto &[name, func] : program.funcs) {
        const auto *set_t = func->ret_type.as<Set_t>();
        if (set_t == nullptr) {
            continue;
        }
        // TODO(cgyurgyik): Add schedule support for dynamic array size.
        // Type dynamic_array_t = DynArray_t::make(set_t->etype);
        Type dynamic_array_t = Set_t::make(set_t->etype);
        func->ret_type = dynamic_array_t;
        converted.emplace(name, dynamic_array_t);
        if (yields(func->body)) {
            // Canonicalize this into a sequence so we only need to handle a
            // single case.
            LowerDynamicSetImpl lower(dynamic_array_t);
            func->body = lower.mutate(std::move(func->body));
        }
    }

    LowerDynamicSetCall lower(converted);
    for (auto &[name, func] : program.funcs) {
        func->body = lower.mutate(std::move(func->body));
    }
    return program;
}

} // namespace lower
} // namespace bonsai
