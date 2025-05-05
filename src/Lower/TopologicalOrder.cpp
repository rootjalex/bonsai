#include "Lower/TypeInference.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Scope.h"
#include "Utils.h"

#include <functional>
#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

using CallGraph = std::map<std::string, std::set<std::string>>;

struct CallGraphBuilder : public ir::Visitor {
    // TODO: this does NOT work for interfaces!
    std::set<std::string> calls;
    bool in_call = false;
    const bool undef_calls;

    CallGraphBuilder(const bool undef_calls) : undef_calls(undef_calls) {}

    void clear() {
        calls.clear();
        in_call = false;
    }

    void visit(const ir::Call *node) override {
        internal_assert(!in_call)
            << "Nested call, how can that happen?" << node;
        in_call = true;
        node->func.accept(this);
        internal_assert(in_call)
            << "Somehow un-nested call, how can that happen?" << node;
        in_call = false;

        // possibly gather call values from arguments.
        for (const auto &a : node->args) {
            a.accept(this);
        }
    }

    void visit(const ir::Access *node) override {
        internal_assert(!in_call)
            << "TODO: support call graph through interface" << node;
        Visitor::visit(node);
    }

    void visit(const ir::Var *node) override {
        if (in_call) {
            internal_assert(!node->type.defined() ||
                            node->type.is<ir::Function_t>())
                << "Somehow in call but Var is not a function type: " << node;
            if (node->type.is<ir::Function_t>()) {
                internal_assert(
                    node->type.as<ir::Function_t>()->ret_type.defined())
                    << "TODO: must have changed Function_t to allow undefined "
                       "ret_type, need to "
                       "update call graph builder!"
                    << node;
            }
            if (!undef_calls || !node->type.defined()) {
                calls.insert(node->name);
            }
        }
    }
};

CallGraph build_call_graph(const ir::FuncMap &funcs, const bool undef_calls) {
    CallGraphBuilder builder(undef_calls);
    CallGraph call_graph;
    for (const auto &f : funcs) {
        // TODO: do we need this for funcs with defined ret_types? probably not.
        if (undef_calls && f.second->ret_type.defined()) {
            call_graph[f.first] = {}; // can be evaluated in any order.
        } else {
            f.second->body.accept(&builder);
            call_graph[f.first] = std::move(builder.calls);
            builder.clear();
        }
    }
    return call_graph;
}

} // namespace

std::vector<std::string> func_topological_order(const ir::FuncMap &funcs,
                                                const bool undef_calls) {
    // Return the order that type inference should run in.
    const CallGraph call_graph = build_call_graph(funcs, undef_calls);
    // DFS-based topological sorting,
    // https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search

    std::vector<std::string> order;
    std::set<std::string> seen;
    std::set<std::string> visiting;

    std::function<void(const std::string &)> visit =
        [&](const std::string &fname) -> void {
        if (seen.contains(fname))
            return;
        if (undef_calls) {
            internal_assert(!visiting.contains(fname))
                << "Type inference found a cycle containing function: " << fname
                << "\nYou may need to specify return types on one or more "
                   "functions to break the cycle";
        } else if (visiting.contains(fname)) {
            return;
        }
        visiting.insert(fname);
        if (!call_graph.contains(fname)) {
            // This is a function argument.
            seen.insert(fname);
            visiting.erase(fname);
            return;
        }
        for (const auto &gname : call_graph.at(fname)) {
            visit(gname);
        }
        visiting.erase(fname);
        seen.insert(fname);
        order.push_back(fname);
    };

    for (const auto &f : funcs) {
        visit(f.first);
    }

    return order;
}

} // namespace lower
} // namespace bonsai
