#include "Lower/RTCoreReads.h"

#include "IR/Analysis.h"
#include "IR/Expr.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "Error.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

// The context struct type, out of a type that is one or addresses one;
// null otherwise.
const ir::Struct_t *context_struct(const ir::Type &type) {
    static const std::string prefix = "_RTContext_";
    const ir::Ptr_t *ptr = type.as<ir::Ptr_t>();
    const ir::Struct_t *s =
        ptr != nullptr ? ptr->etype.as<ir::Struct_t>() : type.as<ir::Struct_t>();
    return s != nullptr && s->name.rfind(prefix, 0) == 0 ? s : nullptr;
}

// The query a trace is of, from the type of the context it traces with:
// `_RTContext_<query>` (Lower/Trees.cpp) -- by address (the memory form) or
// by value (the promoted form, the context in the payload registers).
std::string query_of(const ir::Expr &context) {
    static const std::string prefix = "_RTContext_";
    const ir::Struct_t *s = context_struct(context.type());
    internal_assert(s != nullptr)
        << "rt_trace with a context that is not a query's: " << context
        << " of type " << context.type();
    return s->name.substr(prefix.size());
}

struct CarryReads : public ir::Mutator {
    // Per query, what its programs read: name to type, in a fixed order.
    const std::map<std::string, std::map<std::string, ir::Type>> &reads;

    explicit CarryReads(
        const std::map<std::string, std::map<std::string, ir::Type>> &reads)
        : reads(reads) {}

    using ir::Mutator::visit;

    ir::Expr visit(const ir::Intrinsic *node) override {
        if (node->op != ir::Intrinsic::rt_trace) {
            return ir::Mutator::visit(node);
        }
        internal_assert(node->args.size() >= 7) << "rt_trace: " << ir::Expr(node);
        const std::string query = query_of(node->args[6]);
        const auto found = reads.find(query);
        internal_assert(found != reads.end())
            << "rt_trace of the query `" << query << "`, which has no programs";
        internal_assert(node->args.size() == 7)
            << "rt_trace already carries its programs' reads: " << ir::Expr(node);
        // The whole list, in its order, whether or not the seven refer to
        // some of it already (the tree is the traversable's too): the
        // position is what the code generator goes by.
        std::vector<ir::Expr> args = node->args;
        for (const auto &[name, type] : found->second) {
            args.push_back(ir::Var::make(type, name));
        }
        return ir::Intrinsic::make(node->op, std::move(args));
    }
};

} // namespace

ir::Program LowerRTCoreReads::run(ir::Program program,
                                  const CompilerOptions &) const {
    // What each query's programs read: their parameters other than the
    // context, every extern they reach having been made one by LowerExterns.
    std::map<std::string, std::map<std::string, ir::Type>> reads;
    for (const auto &[_, func] : program.funcs) {
        if (!func->optix_program.has_value()) {
            continue;
        }
        std::map<std::string, ir::Type> &of = reads[func->optix_program->of];
        // Every parameter but the context is a read. A program of a
        // promoted-context query takes no context at all (Lower/Trees.cpp),
        // so the context is known by its type, not its place.
        for (const ir::Function::Argument &arg : func->args) {
            if (context_struct(arg.type) != nullptr) {
                continue;
            }
            of.emplace(arg.name, arg.type);
        }
    }
    if (reads.empty()) {
        return program;
    }
    // The list, on every program of the query, so that a program's
    // parameter finds its place in it.
    for (auto &[_, func] : program.funcs) {
        if (!func->optix_program.has_value()) {
            continue;
        }
        std::vector<std::string> &list = func->optix_program->reads;
        list.clear();
        for (const auto &[name, _type] : reads.at(func->optix_program->of)) {
            list.push_back(name);
        }
    }
    CarryReads carry(reads);
    for (auto &[_, func] : program.funcs) {
        func->body = carry.mutate(func->body);
    }
    return program;
}

} // namespace lower
} // namespace bonsai
