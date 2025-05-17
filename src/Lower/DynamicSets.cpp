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
        return Append::make(std::move(loc), node->value);
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

} // namespace

Program LowerDynamicSets::run(Program program,
                              const CompilerOptions &options) const {
    std::vector<std::string> topological_order =
        func_topological_order(program.funcs);
    for (const std::string &name : topological_order) {
        auto &func = program.funcs[name];
        const auto *set_t = func->ret_type.as<Set_t>();
        if (set_t == nullptr) {
            continue;
        }
        // TODO(cgyurgyik): Add schedule support for dynamic array size.
        Type dynamic_array_t = DynArray_t::make(set_t->etype);
        func->ret_type = dynamic_array_t;
        if (name.starts_with("_traverse_tree")) {
            LowerDynamicSetImpl lower(dynamic_array_t);
            // Canonicalize this into a sequence so we only need to handle a
            // single case.
            func->body = lower.mutate(func->body);
        }
    }
    return program;
}

} // namespace lower
} // namespace bonsai
