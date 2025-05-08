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

Type broadcast_type(const Type &base, const size_t lanes) {
    if (base.is<Int_t, Float_t, UInt_t, Bool_t>()) {
        return Vector_t::make(base, lanes);
    }
    internal_error << "[unimplemented] broadcast_type: " << base;
}

Stmt packetize_impl(std::string idx, Expr repl, Stmt body, FuncMap &funcs,
                    TypeMap &types) {
    struct PacketizeImpl : public Mutator {
        FuncMap &funcs;
        TypeMap &types;
        std::map<Expr, Expr, ExprLessThan> varying;
        std::set<std::string> broadcasted;
        size_t lanes;

        PacketizeImpl(FuncMap &funcs, TypeMap &types, std::string idx,
                      Expr repl)
            : funcs(funcs), types(types) {
            Expr idx_expr = Var::make(repl.type().element_of(), idx);
            varying[std::move(idx_expr)] = repl;
            broadcasted.insert(idx);
            lanes = repl.type().lanes();
        }

        bool is_varying(const Expr &expr) const {
            struct Finder : public Mutator {
                const std::map<Expr, Expr, ExprLessThan> &varying;
                bool found = false;

                Finder(const std::map<Expr, Expr, ExprLessThan> &varying)
                    : varying(varying) {}

                Expr mutate(const Expr &expr) override {
                    if (found || varying.contains(expr)) {
                        found = true;
                        return expr;
                    }
                    return Mutator::mutate(expr);
                }
            };
            Finder finder(varying);
            finder.mutate(expr);
            return finder.found;
        }

        Expr mutate(const Expr &expr) override {
            if (const auto iter = varying.find(expr); iter != varying.cend()) {
                return iter->second;
            }
            return Mutator::mutate(expr);
        }
        using Mutator::mutate; // for mutate(Stmt)

        // Relevant Exprs
        Expr visit(const Var *node) override {
            if (broadcasted.contains(node->name)) {
                Type type = broadcast_type(node->type, lanes);
                return Var::make(std::move(type), node->name);
            }
            return node;
        }

        RESTRICT_MUTATOR(Expr, Cast);
        RESTRICT_MUTATOR(Expr, Broadcast);
        RESTRICT_MUTATOR(Expr, VectorReduce);
        RESTRICT_MUTATOR(Expr, VectorShuffle);
        RESTRICT_MUTATOR(Expr, Ramp);
        // RESTRICT_MUTATOR(Expr, Extract);
        RESTRICT_MUTATOR(Expr, Build);
        RESTRICT_MUTATOR(Expr, Access);
        RESTRICT_MUTATOR(Expr, Call);

        // not relevant; not supported
        RESTRICT_MUTATOR(Expr, Unwrap);
        RESTRICT_MUTATOR(Expr, Generator);
        RESTRICT_MUTATOR(Expr, Lambda);
        RESTRICT_MUTATOR(Expr, GeomOp);
        RESTRICT_MUTATOR(Expr, SetOp);
        RESTRICT_MUTATOR(Expr, Instantiate);
        RESTRICT_MUTATOR(Expr, PtrTo);
        RESTRICT_MUTATOR(Expr, Deref);

        // Stmts.
        RESTRICT_MUTATOR(Stmt, CallStmt);
        RESTRICT_MUTATOR(Stmt, Print);
        RESTRICT_MUTATOR(Stmt, Return);
        RESTRICT_MUTATOR(Stmt, LetStmt);
        RESTRICT_MUTATOR(Stmt, IfElse);
        RESTRICT_MUTATOR(Stmt, DoWhile);
        // RESTRICT_MUTATOR(Stmt, Sequence);
        RESTRICT_MUTATOR(Stmt, Allocate);

        Stmt visit(const Store *node) override {
            Expr value = mutate(node->value);
            if (value.same_as(node->value)) {
                // This is a uniform value, no need to mutate anything.
                // Just to be safe, will check that the store location is not
                // varying.
                Expr read = writeloc_to_read(node->loc);
                internal_assert(is_varying(read))
                    << "[unimplemented] packetized uniform store to varying "
                       "location: "
                    << Stmt(node);
                return node;
            }
            bool varies = false;

            // TODO: can the base type ever change?
            WriteLoc loc(node->loc.base, node->loc.base_type);
            for (const auto &access : node->loc.accesses) {
                internal_assert(std::holds_alternative<Expr>(access))
                    << "[unimplemented] fields in packetized store "
                    << Stmt(node);
                Expr expr = mutate(std::get<Expr>(access));
                // only care about the last access.
                varies = !expr.same_as(std::get<Expr>(access));
                loc.add_index_access(expr);
            }
            internal_assert(varies)
                << "[unimplemented] packetized store to uniform location: "
                << Stmt(node);
            return Store::make(std::move(loc), std::move(value));
        }

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

    body = opt::Simplify::simplify(body);
    PacketizeForAll pac(loop_idx, funcs, types);
    return pac.mutate(std::move(body));
}

} // namespace opt
} // namespace bonsai
