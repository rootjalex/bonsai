#include "Lower/Geometrics.h"

#include "IR/Equality.h"
#include "IR/Function.h"
#include "IR/Printer.h"

#include "Error.h"
#include "Utils.h"

#include <set>
#include <string>

namespace bonsai {
namespace lower {

using namespace ir;

namespace {

struct LowerGeomOps : public Mutator {
    const FuncMap &funcs;

    LowerGeomOps(const FuncMap &funcs) : funcs(funcs) {}

    Expr visit(const GeomOp *node) {
        // TODO: can there be a recursive geometric op?
        // e.g. distance being used to build a sphere?
        // better safe than sorry.
        Expr a = mutate(node->a);
        Expr b = mutate(node->b);

        // TODO: break this out of Parser.cpp so we don't
        // have to keep track of two versions of this.
        const std::string name =
            GeomOp::intrinsic_name(node->op); // reused below.
        std::string typed_name = name;
        internal_assert(a.type().is<Struct_t>());
        typed_name += "_" + a.type().as<Struct_t>()->name;
        internal_assert(b.type().is<Struct_t>());
        typed_name += "_" + b.type().as<Struct_t>()->name;

        const auto &func = funcs.find(typed_name);
        internal_assert(func != funcs.cend())
            << "Lowering needs an implementation of: " << name
            << " with types: " << a.type() << " and " << b.type();

        // Above shouldn't be possible without this.
        const auto &arg_types = func->second->args;
        internal_assert((arg_types.size() == 2) &&
                        equals(arg_types[0].type, a.type()) &&
                        equals(arg_types[1].type, b.type()));

        internal_assert(func->second->interfaces.empty()); // TODO: ?

        Type call_type = func->second->call_type();

        Expr f = Var::make(std::move(call_type), std::move(typed_name));
        Expr call = Call::make(std::move(f), {std::move(a), std::move(b)});

        if (is_geometric_predicate(name)) {
            // some macro error with inlining this into internal_assert
            const bool truthy_type = call.type().is<Option_t, Bool_t>();
            internal_assert(truthy_type);
        } else {
            internal_assert(is_geometric_metric(name));
            internal_assert(call.type().is_numeric());
        }
        return call;
    }
};

} // namespace

FuncMap LowerGeometrics::run(FuncMap funcs) const {
    LowerGeomOps lower(funcs);
    // TODO: what happens with nested geometric ops...?
    for (const auto &[f, func] : funcs) {
        func->body = lower.mutate(func->body);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
