#include "Lower/ToSSA.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

std::string get_ssa_func_name(const std::string &fname) {
    return "_ssa_" + fname;
}

struct ToSSAImpl : public Mutator {
    size_t counter = 0;

    std::string get_id() { return "_t" + std::to_string(counter++); }

    // Will be turned into a chain of Lets.
    struct Temporary {
        std::string id;
        Expr value;
    };
    std::vector<Temporary> temps;

    Expr push_temp(Expr expr) {
        std::string id = get_id();
        Type t = expr.type();
        temps.push_back({id, std::move(expr)});
        return Var::make(std::move(t), std::move(id));
    }

    Expr visit(const BinOp *node) override {
        Expr a = mutate(node->a);
        Expr b = mutate(node->b);
        return push_temp(BinOp::make(node->op, std::move(a), std::move(b)));
    }

    Expr visit(const UnOp *node) override {
        Expr a = mutate(node->a);
        return push_temp(UnOp::make(node->op, std::move(a)));
    }

    Expr visit(const Select *node) override {
        Expr cond = mutate(node->cond);
        Expr tval = mutate(node->tvalue);
        Expr fval = mutate(node->fvalue);
        return push_temp(
            Select::make(std::move(cond), std::move(tval), std::move(fval)));
    }

    Expr visit(const Cast *node) override {
        Expr value = mutate(node->value);
        return push_temp(Cast::make(node->type, std::move(value)));
    }

    Expr visit(const Broadcast *node) override {
        Expr value = mutate(node->value);
        return push_temp(Broadcast::make(node->lanes, std::move(value)));
    }

    Expr visit(const VectorReduce *node) override {
        Expr value = mutate(node->value);
        return push_temp(VectorReduce::make(node->op, std::move(value)));
    }

    Expr visit(const VectorShuffle *node) override {
        Expr value = mutate(node->value);
        std::vector<Expr> idxs;
        idxs.reserve(node->idxs.size());
        for (const auto &i : node->idxs) {
            idxs.push_back(mutate(i));
        }
        return push_temp(
            VectorShuffle::make(std::move(value), std::move(idxs)));
    }

    Expr visit(const Extract *node) override {
        Expr vec = mutate(node->vec);
        Expr idx = mutate(node->idx);
        return push_temp(Extract::make(std::move(vec), std::move(idx)));
    }

    Expr visit(const Build *node) override {
        std::vector<Expr> values;
        values.reserve(node->values.size());
        for (const auto &v : node->values) {
            values.push_back(mutate(v));
        }
        return push_temp(Build::make(node->type, std::move(values)));
    }

    Expr visit(const Access *node) override {
        Expr value = mutate(node->value);
        return push_temp(Access::make(node->field, std::move(value)));
    }

    Expr visit(const Intrinsic *node) override {
        std::vector<Expr> args;
        args.reserve(node->args.size());
        for (const auto &a : node->args) {
            args.push_back(mutate(a));
        }
        return push_temp(Intrinsic::make(node->op, std::move(args)));
    }

    Expr visit(const Call *node) override {
        // TODO: RtoP?
        Expr func = mutate(node->func);
        std::vector<Expr> args;
        for (const Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        return push_temp(Call::make(std::move(func), std::move(args)));
    }

    RESTRICT_MUTATOR(Expr, Ramp);
    RESTRICT_MUTATOR(Expr, Unwrap);
    RESTRICT_MUTATOR(Expr, Generator);
    RESTRICT_MUTATOR(Expr, Lambda);
    RESTRICT_MUTATOR(Expr, GeomOp);
    RESTRICT_MUTATOR(Expr, SetOp);
    RESTRICT_MUTATOR(Expr, Instantiate);
    RESTRICT_MUTATOR(Expr, PtrTo);
    RESTRICT_MUTATOR(Expr, Deref);

    // Need to flatten sequences.
    Stmt visit(const Sequence *node) override {
        bool changed = false;
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());

        auto flatten = [&](const Stmt &stmt) {
            Stmt mut = mutate(stmt);
            changed = changed || !mut.same_as(stmt);
            internal_assert(mut.defined()) << stmt;
            if (const ir::Sequence *seq = mut.as<ir::Sequence>()) {
                stmts.insert(stmts.end(), seq->stmts.begin(), seq->stmts.end());
                changed = true;
            } else {
                stmts.emplace_back(std::move(mut));
            }
        };

        for (const auto &stmt : node->stmts) {
            flatten(stmt);
        }

        if (!changed) {
            return node;
        }
        internal_assert(!stmts.empty());
        return ir::Sequence::make(std::move(stmts));
    }

    Stmt make_lets(Stmt stmt) {
        if (temps.empty()) {
            return stmt;
        }
        std::vector<Stmt> stmts;
        stmts.reserve(temps.size() + 1);
        for (auto &[name, value] : temps) {
            // TODO: mutable RtoP for calls?
            WriteLoc loc(std::move(name), value.type());
            stmts.emplace_back(LetStmt::make(std::move(loc), std::move(value)));
        }
        stmts.emplace_back(std::move(stmt));
        temps.clear();
        return Sequence::make(std::move(stmts));
    }

    Stmt visit(const CallStmt *node) override {
        Expr func = mutate(node->func);
        std::vector<Expr> args;
        for (const Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        return make_lets(CallStmt::make(std::move(func), std::move(args)));
    }

    Stmt visit(const Print *node) override {
        Expr value = mutate(node->value);
        return make_lets(Print::make(std::move(value)));
    }

    Stmt visit(const Return *node) override {
        Expr value = mutate(node->value);
        return make_lets(Return::make(std::move(value)));
    }

    Stmt visit(const LetStmt *node) override {
        Expr value = mutate(node->value);
        return make_lets(LetStmt::make(node->loc, std::move(value)));
    }

    Stmt visit(const IfElse *node) override {
        Stmt then_body = mutate(node->then_body);
        Stmt else_body = mutate(node->else_body);
        // Must be after then/else so temps doesn't get clobbered.
        Expr cond = mutate(node->cond);
        return make_lets(IfElse::make(std::move(cond), std::move(then_body),
                                      std::move(else_body)));
    }

    Stmt visit(const DoWhile *node) override {
        Stmt body = mutate(node->body);
        Expr cond = mutate(node->cond);
        if (temps.empty()) {
            return DoWhile::make(std::move(body), std::move(cond));
        }
        // This one is tricky, all Lets for the cond need to be shoved
        // onto the back of body.
        // TODO(ajr): should really make a flatten_sequence helper func.
        std::vector<Stmt> stmts;

        if (const Sequence *seq = body.as<Sequence>()) {
            stmts = seq->stmts;
        } else {
            stmts.emplace_back(std::move(body));
        }

        for (auto &[name, value] : temps) {
            // TODO: mutable RtoP for calls?
            WriteLoc loc(std::move(name), value.type());
            stmts.emplace_back(LetStmt::make(std::move(loc), std::move(value)));
        }
        body = Sequence::make(std::move(stmts));
        return DoWhile::make(std::move(body), std::move(cond));
    }

    Stmt visit(const Allocate *node) override {
        internal_assert(node->loc.accesses.empty()) << Stmt(node);
        Expr value = mutate(node->value);
        return make_lets(
            Allocate::make(node->loc, std::move(value), node->memory));
    }

    Stmt visit(const Store *node) override {
        // LLVM codegen does rhs then loc then mask, so do we.
        Expr value = mutate(node->value);
        auto [loc, not_changed] = mutate_writeloc(node->loc);
        Expr mask = mutate(node->mask);
        return make_lets(
            Store::make(std::move(loc), std::move(value), std::move(mask)));
    }

    Stmt visit(const Accumulate *node) override {
        // LLVM codegen does loc then rhs, so do we.
        auto [loc, not_changed] = mutate_writeloc(node->loc);
        Expr value = mutate(node->value);
        return make_lets(
            Accumulate::make(std::move(loc), node->op, std::move(value)));
    }

    Stmt visit(const ForAll *node) override {
        // Mutate body first so it doesn't grab the
        // Lets from begin/end/stride
        Stmt body = mutate(node->body);
        Expr begin = mutate(node->slice.begin);
        Expr end = mutate(node->slice.end);
        Expr stride = mutate(node->slice.stride);
        return make_lets(ForAll::make(
            node->index,
            ForAll::Slice{std::move(begin), std::move(end), std::move(stride)},
            std::move(body)));
    }

    // Label default mutate is fine.
    // Stmt visit(const Label *node) override {}
    RESTRICT_MUTATOR(Stmt, RecLoop);
    RESTRICT_MUTATOR(Stmt, Match);
    RESTRICT_MUTATOR(Stmt, Yield);
    RESTRICT_MUTATOR(Stmt, Scan);
    RESTRICT_MUTATOR(Stmt, YieldFrom);
    RESTRICT_MUTATOR(Stmt, ForEach);
    RESTRICT_MUTATOR(Stmt, Continue);
    RESTRICT_MUTATOR(Stmt, Launch);
};

} // namespace

std::shared_ptr<Function> to_ssa(const Function &func) {
    // TODO(ajr): this isn't quite SSA, it has Stores still.
    ToSSAImpl lowerer;
    Stmt body = lowerer.mutate(func.body);

    // TODO: RtoP?

    return std::make_shared<Function>(get_ssa_func_name(func.name), func.args,
                                      func.ret_type, std::move(body),
                                      func.interfaces, func.attributes);
}

} // namespace lower
} // namespace bonsai
