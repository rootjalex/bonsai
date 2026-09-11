#include "Lower/SetFunctions.h"

#include "Error.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

// Replaces a call to a set-valued function with that function's expression,
// its parameters standing for the call's arguments. What comes back is
// expanded again, so a set-valued function may be written in terms of another.
struct ExpandSetCalls : public ir::Mutator {
    const ir::FuncMap &named;
    // The functions put back somewhere, which is what decides their fate.
    std::set<std::string> expanded;
    // Calls opened so far along the current path, to notice one that never
    // bottoms out. A set-valued function cannot be recursive: there is no set
    // it could return that is not already written down.
    size_t depth = 0;

    ExpandSetCalls(const ir::FuncMap &named) : named(named) {}

    using ir::Mutator::visit;

    ir::Expr visit(const ir::Call *node) override {
        const ir::Var *callee = node->func.as<ir::Var>();
        if (callee == nullptr) {
            return ir::Mutator::visit(node);
        }
        const auto found = named.find(callee->name);
        if (found == named.cend()) {
            return ir::Mutator::visit(node);
        }
        const ir::Function &func = *found->second;
        const ir::Return *body = func.body.as<ir::Return>();
        internal_assert(body && body->value.defined())
            << callee->name
            << " returns a set and is called inside a query, so it names part "
               "of that query, and a query is one expression: write it as "
               "`func "
            << callee->name << "(..) -> set[..] = <set>;`.";
        internal_assert(func.args.size() == node->args.size())
            << callee->name << " takes " << func.args.size()
            << " arguments and is called with " << node->args.size();
        internal_assert(depth <= named.size())
            << callee->name
            << " returns a set and reaches itself. A set-valued function "
               "names part of a query, and a query is finite: write the set "
               "out.";

        std::map<std::string, ir::Expr> args;
        for (size_t i = 0; i < node->args.size(); i++) {
            args[func.args[i].name] = mutate(node->args[i]);
        }
        expanded.insert(callee->name);

        depth++;
        ir::Expr result = mutate(replace(args, body->value));
        depth--;
        return result;
    }
};

} // namespace

ir::Program LowerSetFunctions::run(ir::Program program,
                                   const CompilerOptions &options) const {
    ir::FuncMap named;
    for (const auto &[name, func] : program.funcs) {
        if (func->ret_type.is<ir::Set_t>()) {
            named[name] = func;
        }
    }
    if (named.empty()) {
        return program;
    }

    ExpandSetCalls expand(named);
    for (auto &[name, func] : program.funcs) {
        func->body = expand.mutate(func->body);
    }
    for (auto &[name, extent] : program.extents) {
        extent = expand.mutate(extent);
    }
    for (const std::string &name : expand.expanded) {
        if (!named.at(name)->is_exported()) {
            program.funcs.erase(name);
        }
    }
    return program;
}

} // namespace lower
} // namespace bonsai
