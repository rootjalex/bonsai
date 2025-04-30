#include "Opt/Fusion.h"

#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace opt {

namespace {

using namespace ir;

TypedVar get_arg(Expr func, const FuncMap &funcs) {
    // TODO: handle many vars, e.g. tuple fusion!
    if (const Lambda *l = func.as<Lambda>()) {
        internal_assert(l->args.size() == 1)
            << "[unimplemented] fusion for tuple set operations\n";
        return l->args[0];
    } else if (const Var *v = func.as<Var>()) {
        const auto &iter = funcs.find(v->name);
        internal_assert(iter != funcs.cend())
            << "Can't find func in table for fusion: " << func;
        const auto &f = iter->second;
        internal_assert(f->args.size() == 1)
            << "[unimplemented] fusion for tuple set operations\n";
        return TypedVar{f->args[0].name, f->args[0].type};
    }
    internal_error << "Unknown function type in fusion: " << func;
}

// Produce g(f(x))
Expr fuse_lambdas(Expr f, Expr g, const FuncMap &funcs) {
    TypedVar arg = get_arg(f, funcs);
    Expr fx = call(f, Var::make(arg.type, arg.name));
    Expr gfx = call(g, fx);
    return Lambda::make({std::move(arg)}, std::move(gfx));
}

// Produce f(x) && g(x)
Expr conjunct_lambdas(Expr f, Expr g, const FuncMap &funcs) {
    TypedVar arg = get_arg(f, funcs);
    Expr var = Var::make(arg.type, arg.name);
    Expr fx = call(f, var);
    Expr gx = call(g, var);
    Expr conj = fx && gx;
    return Lambda::make({std::move(arg)}, std::move(conj));
}

struct FuseWithinStmt : public Mutator {
    const FuncMap &funcs;

    FuseWithinStmt(const FuncMap &funcs) : funcs(funcs) {}

    Expr visit(const SetOp *setop) override {
        Expr expr = Mutator::visit(setop);
        if (auto map0 = as_map(expr)) {
            Expr lambda0 = map0->a, array0 = map0->b;
            if (auto map1 = as_map(array0)) {
                Expr lambda1 = map1->a, array1 = map1->b;
                Expr lambda2 = fuse_lambdas(lambda1, lambda0, funcs);
                return map(std::move(lambda2), std::move(array1));
            }
        } else if (auto filter0 = as_filter(expr)) {
            Expr lambda0 = filter0->a, set0 = filter0->b;
            if (auto filter1 = as_map(set0)) {
                Expr lambda1 = filter1->a, set1 = filter1->b;
                Expr lambda2 = conjunct_lambdas(lambda1, lambda0, funcs);
                return filter(std::move(lambda2), std::move(set1));
            }
        }
        return expr;
    }
};

} // namespace

ir::FuncMap Fusion::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
        func->body = fuse_within_stmt(func->body, funcs);
    }
    return funcs;
}

/*static*/
ir::Stmt Fusion::fuse_within_stmt(const ir::Stmt &stmt,
                                  const ir::FuncMap &funcs) {
    ir::Stmt ret = FuseWithinStmt(funcs).mutate(stmt);
    return ret;
}

} // namespace opt
} // namespace bonsai
