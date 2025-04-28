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
        internal_assert(iter != funcs.cend()) << "Can't func func in table for fusion: " << func;
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

    struct Map {
        Expr lambda;
        Expr array;
    };
    std::optional<Map> as_map(const Expr &expr) {
        if (const SetOp *setop = expr.as<SetOp>()) {
            if (setop->op == SetOp::map) {
                return Map{setop->a, setop->b};
            }
        }
        return {};
    }

    struct Filter {
        Expr lambda;
        Expr set;
    };
    std::optional<Filter> as_filter(const Expr &expr) {
        if (const SetOp *setop = expr.as<SetOp>()) {
            if (setop->op == SetOp::filter) {
                return Filter{setop->a, setop->b};
            }
        }
        return {};
    }

    Expr visit(const SetOp *setop) override {
        Expr expr = Mutator::visit(setop);
        if (auto map0 = as_map(expr)) {
            auto [lambda0, array0] = *map0;
            if (auto map1 = as_map(array0)) {
                auto [lambda1, array1] = *map1;
                Expr lambda2 = fuse_lambdas(lambda1, lambda0, funcs);
                return map(std::move(lambda2), std::move(array1));
            }
        } else if (auto filter0 = as_filter(expr)) {
            auto [lambda0, set0] = *filter0;
            if (auto filter1 = as_map(set0)) {
                auto [lambda1, set1] = *filter1;
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
ir::Stmt Fusion::fuse_within_stmt(const ir::Stmt &stmt, const ir::FuncMap &funcs) {
    ir::Stmt ret = FuseWithinStmt(funcs).mutate(stmt);
    if (!ret.same_as(stmt)) {
        std::cout << "Fusion of: " << stmt << "=>\n" << ret << "\n";
    }
    return ret;
}

} // namespace opt
} // namespace bonsai
