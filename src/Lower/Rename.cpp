#include "Lower/Rename.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

struct RenameVariable : public ir::Mutator {
    RenameVariable(const std::set<std::string> &mutable_function_arguments)
        : mutable_function_arguments(mutable_function_arguments) {}

    const std::set<std::string> &mutable_function_arguments;
    // Tracks the old variable name to the new name.
    std::unordered_map<std::string, std::string> old_to_new_name;

    std::pair<std::string, bool> rename(std::string name) {
        auto it = old_to_new_name.find(name);
        if (should_rename) {
            std::string new_name = "_" + std::to_string(counter++) + name;
            old_to_new_name[name] = new_name;
            return {new_name, true};
        }
        if (it != old_to_new_name.end()) {
            return {it->second, true};
        }
        return {name, false};
    }

    ir::Expr visit(const ir::Var *node) override {
        auto it = old_to_new_name.find(node->name);
        if (it == old_to_new_name.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

  private:
    // Whether the variable should be given a fresh name.
    bool should_rename = true;
    // For unique variable renaming.
    int64_t counter = 0;
};

} // namespace

ir::FuncMap Rename::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        std::set<std::string> args = get_mutable_arguments(*func);
        RenameVariable rename(args);
        func->body = rename.mutate(std::move(func->body));
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
