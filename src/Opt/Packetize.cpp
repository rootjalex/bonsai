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
static const std::string result_name = "__result";
static const std::string result_mask_name = "__result_mask";

Type broadcast_type(const Type &base, const size_t lanes) {
    if (base.is<Int_t, Float_t, UInt_t, Bool_t>()) {
        return Vector_t::make(base, lanes);
    }
    internal_error << "[unimplemented] broadcast_type: " << base;
}

static const std::string MASK_NAME = "__mask";

Stmt packetize_impl(Type scalar_ret_type,
                    std::map<Expr, Expr, ExprLessThan> varying,
                    std::set<std::string> broadcasted, Expr mask, Stmt body,
                    FuncMap &funcs, TypeMap &types) {
    struct PacketizeImpl : public Mutator {
        FuncMap &funcs;
        TypeMap &types;
        std::map<Expr, Expr, ExprLessThan> varying;
        std::set<std::string> broadcasted;
        // Which lanes are currently active
        Expr mask;
        // Which lanes must be filled of result in order to return.
        Expr ret_mask;
        // A Load of the result
        Expr result;
        // The loc to write results to
        WriteLoc result_loc;
        // A Load of the result mask
        Expr result_mask;
        // The loc to write the result mask to.
        WriteLoc result_mask_loc;

        PacketizeImpl(FuncMap &funcs, TypeMap &types,
                      std::map<Expr, Expr, ExprLessThan> varying,
                      std::set<std::string> broadcasted, Expr mask)
            : funcs(funcs), types(types), varying(std::move(varying)),
              broadcasted(std::move(broadcasted)), mask(std::move(mask)) {}

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
                Type type = broadcast_type(node->type, mask.type().lanes());
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

        struct MutateArgs {
            std::vector<Expr> args;
            std::vector<bool> mutated;
            bool any_changed = false;
        };

        MutateArgs mutate_list(const std::vector<Expr> &exprs) {
            MutateArgs ret;
            ret.args.resize(exprs.size());
            ret.mutated.resize(exprs.size());
            for (size_t i = 0; i < exprs.size(); i++) {
                ret.args[i] = mutate(exprs[i]);
                ret.mutated[i] = !ret.args[i].same_as(exprs[i]);
                ret.any_changed = ret.any_changed || ret.mutated[i];
            }
            return ret;
        }

        Expr visit(const Call *node) override {
            MutateArgs rec = mutate_list(node->args);
            internal_assert(rec.any_changed)
                << "Packetized found uniform call: " << Expr(node);
            const Var *func = node->func.as<Var>();
            internal_assert(func && funcs.contains(func->name))
                << "Cannot packetize call to unknown func: " << Expr(node);

            // Generate a packetized implementation of the func body.
            // New function must accept a mask
            // TODO: do we want to RtoP this? Maybe.
            const auto &scalar_func = funcs.at(func->name);
            std::string new_name = "_packetized_" + func->name;
            if (const auto iter = funcs.find(new_name); iter != funcs.cend()) {
                Expr cached = Var::make(iter->second->call_type(), new_name);
                rec.args.push_back(mask);
                return Call::make(cached, rec.args);
            }

            std::vector<Function::Argument> arg_params(rec.args.size() + 1);
            std::vector<Function_t::ArgSig> arg_sigs(rec.args.size() + 1);
            for (size_t i = 0; i < rec.args.size(); i++) {
                arg_params[i].name = scalar_func->args[i].name;
                arg_params[i].type = rec.args[i].type();
                arg_params[i].mutating = scalar_func->args[i].mutating;
                internal_assert(!scalar_func->args[i].default_value.defined())
                    << "[unimplemented] packetization of function with default "
                       "arg: "
                    << Expr(node);

                arg_sigs[i].type = rec.args[i].type();
                arg_sigs[i].is_mutable = scalar_func->args[i].mutating;
            }
            // Also add mask
            arg_params.back().name = MASK_NAME;
            arg_params.back().type = mask.type();
            arg_params.back().mutating =
                false; // TODO(ajr): should it mutate the mask?
            arg_sigs.back().type = mask.type();
            arg_sigs.back().is_mutable = false;

            // Presumably, a vector return type.
            Type ret_type =
                broadcast_type(scalar_func->ret_type, mask.type().lanes());

            std::map<Expr, Expr, ExprLessThan> call_varying;
            std::set<std::string> call_broadcasted;

            for (size_t i = 0; i < rec.args.size(); i++) {
                if (rec.mutated[i]) {
                    call_broadcasted.insert(arg_params[i].name);
                    Expr old_var = Var::make(scalar_func->args[i].type,
                                             scalar_func->args[i].name);
                    call_varying[old_var] = Var::make(
                        rec.args[i].type(), scalar_func->args[i].name);
                }
            }

            Expr call_mask = Var::make(mask.type(), mask_name);

            // Handle recursive calls by inserting and then mutating body.
            Stmt body;
            std::shared_ptr<Function> packet_func = std::make_shared<Function>(
                new_name, arg_params, ret_type, body, scalar_func->interfaces,
                scalar_func->attributes);

            funcs[new_name] = packet_func;

            packet_func->body = packetize_impl(
                scalar_func->ret_type, call_varying, call_broadcasted,
                call_mask, scalar_func->body, funcs, types);

            Expr new_func = Var::make(packet_func->call_type(), new_name);

            rec.args.push_back(mask);
            return Call::make(std::move(new_func), std::move(rec.args));
        }

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

        Stmt visit(const Return *node) override {
            internal_assert(result_loc.defined());
            Expr value = mutate(node->value);

            // TODO(ajr): try to statically figure out when we can
            // always-return?

            // Need to update some masks and stuff.
            value = value.same_as(node->value)
                        ? Broadcast::make(mask.type().lanes(), std::move(value))
                        : value;
            // TODO: should the mask be cleared in all lanes that have returned
            // already?
            // I think mask must also be dynamic.
            value = Select::make(mask, value, result);
            Expr done = mask | result_mask;
            return IfElse::make(
                all(done == ret_mask), Return::make(value),
                Sequence::make({Store::make(result_loc, value),
                                Store::make(result_mask_loc, done)}));
        }

        RESTRICT_MUTATOR(Stmt, LetStmt);

        Stmt visit(const IfElse *node) override {
            // TODO(ajr): another place were I wish this was just a CFG
            if (const BinOp *binop = node->cond.as<BinOp>()) {
                if (binop->op == BinOp::LAnd) {
                    Stmt repl =
                        IfElse::make(binop->a,
                                     IfElse::make(binop->b, node->then_body,
                                                  node->else_body),
                                     node->else_body);
                    return mutate(repl);
                } else if (binop->op == BinOp::LOr) {
                    Stmt repl =
                        IfElse::make(binop->a, node->then_body,
                                     IfElse::make(binop->b, node->then_body,
                                                  node->else_body));
                    return mutate(repl);
                }
            } else if (const UnOp *unop = node->cond.as<UnOp>()) {
                internal_assert(unop->op != UnOp::Neg)
                    << "[unimplemented] packetization through logical negated "
                       "IfElse: "
                    << Stmt(node);
            }

            Expr cond = mutate(node->cond);

            if (cond.same_as(node->cond)) {
                // Uniform control flow! No need to update the mask.
                return IfElse::make(std::move(cond), mutate(node->then_body),
                                    node->else_body);
            }

            // I don't think even CFGs save us from this horrendous explosion.
            Stmt all_case = mutate(node->then_body);
            Stmt none_case = mutate(node->else_body);
            Expr old_mask = std::move(mask);
            mask = old_mask & cond;
            Stmt any_case0 = mutate(node->then_body);
            mask = old_mask & ~cond;
            Stmt any_case1 = mutate(node->else_body);
            mask = old_mask;

            Stmt body = IfElse::make(any((old_mask & cond) == ret_mask),
                                     std::move(any_case0));

            if (any_case1.defined()) {
                body = Sequence::make(
                    {std::move(body),
                     IfElse::make(any((old_mask & ~cond) == ret_mask),
                                  std::move(any_case1))});
            }

            body = none_case.defined()
                       ? IfElse::make(all((old_mask & ~cond) == ret_mask),
                                      std::move(none_case), std::move(body))
                       : std::move(body);

            // TODO: if, at the end of this, it's possible to return, then
            // do so!
            return IfElse::make(all((old_mask & cond) == ret_mask),
                                std::move(all_case), std::move(body));
        }

        RESTRICT_MUTATOR(Stmt, DoWhile);
        // RESTRICT_MUTATOR(Stmt, Sequence);
        RESTRICT_MUTATOR(Stmt, Allocate);

        Stmt visit(const Store *node) override {
            internal_assert(is_const_one(mask))
                << "[unimplemented] packetized store: " << Stmt(node)
                << " with mask: " << mask;
            Expr value = mutate(node->value);
            if (value.same_as(node->value)) {
                // This is a uniform value, no need to mutate anything.
                // Just to be safe, will check that the store location is
                // not varying.
                Expr read = writeloc_to_read(node->loc);
                internal_assert(is_varying(read))
                    << "[unimplemented] packetized uniform store to "
                       "varying "
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

    body = opt::Simplify::simplify(std::move(body));

    PacketizeImpl rewriter(funcs, types, std::move(varying),
                           std::move(broadcasted), mask);

    if (!scalar_ret_type.defined()) {
        // This Stmt does not return.
        return rewriter.mutate(std::move(body));
    }

    // This stmt does return, allocate a stack variable to hold and return
    // the result.
    // TODO(ajr): a result mask?
    Type ret_type = broadcast_type(scalar_ret_type, mask.type().lanes());
    Expr result = Var::make(ret_type, result_name);
    WriteLoc result_loc(result_name, ret_type);
    Expr result_mask = Var::make(mask.type(), result_mask_name);
    WriteLoc result_mask_loc(result_mask_name, mask.type());

    rewriter.result = result;
    rewriter.ret_mask = mask;
    rewriter.result_loc = result_loc;
    rewriter.result_mask = result_mask;
    rewriter.result_mask_loc = result_mask_loc;

    std::vector<Stmt> stmts = {
        // TODO: ideally this would be undef, which would let LLVM be smarter.
        Allocate::make(std::move(result_loc), make_zero(ret_type),
                       Allocate::Memory::Stack),
        Allocate::make(std::move(result_mask_loc), make_zero(mask.type()),
                       Allocate::Memory::Stack),
        rewriter.mutate(std::move(body)),
        // TODO(ajr): We shouldn't need this, for well-formed code.
        Return::make(std::move(result)),
    };

    body = Sequence::make(std::move(stmts));
    return opt::Simplify::simplify(std::move(body));
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

            // TODO: support mask for non-perfect splitting.
            Expr mask = make_one(Vector_t::make(Bool_t::make(), *lane_count));

            Expr idx = Var::make(node->slice.end.type(), loop_idx);
            std::map<Expr, Expr, ExprLessThan> varying = {{idx, repl}};
            std::set<std::string> broadcasted = {loop_idx};

            return packetize_impl(/*scalar_ret_type=*/Type(), varying,
                                  broadcasted, std::move(mask), node->body,
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

    body = opt::Simplify::simplify(std::move(body));
    PacketizeForAll pac(loop_idx, funcs, types);
    return pac.mutate(std::move(body));
}

} // namespace opt
} // namespace bonsai
