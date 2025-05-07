#include "Opt/Packetize.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
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

static const std::string mask_name = "__mask";

Stmt packetize_impl(std::string idx, Expr repl, Stmt body, FuncMap &funcs,
                    TypeMap &types) {
    struct PacketizeImpl : public Mutator {
        FuncMap &funcs;
        TypeMap &types;
        std::map<Expr, Expr, ExprLessThan> varying;

        PacketizeImpl(FuncMap &funcs, TypeMap &types, std::string idx,
                      Expr repl)
            : funcs(funcs), types(types) {
            Expr idx_expr = Var::make(repl.type().element_of(), idx);
            varying[std::move(idx_expr)] = std::move(repl);
        }

        RESTRICT_MUTATOR(Stmt, CallStmt);
        RESTRICT_MUTATOR(Stmt, Print);
        RESTRICT_MUTATOR(Stmt, Return);
        RESTRICT_MUTATOR(Stmt, LetStmt);
        RESTRICT_MUTATOR(Stmt, IfElse);
        RESTRICT_MUTATOR(Stmt, DoWhile);
        // RESTRICT_MUTATOR(Stmt, Sequence);
        RESTRICT_MUTATOR(Stmt, Allocate);
        RESTRICT_MUTATOR(Stmt, Store);
        RESTRICT_MUTATOR(Stmt, Accumulate);
        // RESTRICT_MUTATOR(Stmt, Label);
        RESTRICT_MUTATOR(Stmt, RecLoop);
        RESTRICT_MUTATOR(Stmt, Match);
        RESTRICT_MUTATOR(Stmt, Yield);
        RESTRICT_MUTATOR(Stmt, Scan);
        RESTRICT_MUTATOR(Stmt, YieldFrom);
        RESTRICT_MUTATOR(Stmt, ForAll);
        RESTRICT_MUTATOR(Stmt, ForEach);
        RESTRICT_MUTATOR(Stmt, Continue);
        RESTRICT_MUTATOR(Stmt, Launch);
    };

    PacketizeImpl rewriter(funcs, types, std::move(idx), std::move(repl));
    return rewriter.mutate(std::move(body));
}

} // namespace

Stmt packetize_forall(const std::string &loop_idx, Stmt body, FuncMap &funcs,
                      TypeMap &types) {
    struct PacketizeForAll : public Mutator {
        const std::string &loop_idx;
        FuncMap &funcs;
        TypeMap &types;

        PacketizeForAll(const std::string &loop_idx, FuncMap &funcs,
                        TypeMap &types)
            : loop_idx(loop_idx), funcs(funcs), types(types) {}

        Stmt visit(const ForAll *node) override {
            if (node->index != loop_idx) {
                return Mutator::visit(node);
            }

            Expr lanes = opt::Simplify::simplify(
                ((node->slice.end - node->slice.begin) +
                 (node->slice.stride - make_one(node->slice.stride.type()))) /
                node->slice.stride);

            auto lane_count = get_constant_value(lanes);
            internal_assert(lane_count.has_value())
                << "[unimplemented] packetize with non-constant size: " << lanes
                << " of loop " << loop_idx;

            Expr repl =
                Ramp::make(node->slice.begin, node->slice.stride, *lane_count);

            return packetize_impl(node->index, std::move(repl), node->body,
                                  funcs, types);
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                // TODO(ajr): hope to God it's impossible to have self-recursion
                // in these.
                if (var->name.starts_with("_traverse_array")) {
                    funcs[var->name]->body = packetize_forall(
                        loop_idx, std::move(funcs[var->name]->body), funcs,
                        types);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    PacketizeForAll pac(loop_idx, funcs, types);
    return pac.mutate(std::move(body));
}

} // namespace opt
} // namespace bonsai
