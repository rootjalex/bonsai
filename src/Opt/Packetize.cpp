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

static const std::string call_mask_name = "__callmask";
static const std::string dead_mask_name = "__deadmask";
static const std::string result_name = "__result";

Type broadcast_type(const Type &base, const size_t lanes) {
    if (base.is<Int_t, Float_t, UInt_t, Bool_t>()) {
        return Vector_t::make(base, lanes);
    } else if (const Struct_t *struct_t = base.as<Struct_t>()) {
        // TODO(ajr): what if some fields are varying, and some are uniform?
        // TODO(ajr): can't do this if it's a return type...
        // Turn this into a SoA format, for vectorized loads and stores.
        std::string name = "__soa_" + struct_t->name;
        Struct_t::Map fields(struct_t->fields.size());
        for (size_t i = 0; i < struct_t->fields.size(); i++) {
            fields[i].name = struct_t->fields[i].name;
            fields[i].type = broadcast_type(struct_t->fields[i].type, lanes);
        }
        Struct_t::DefMap defaults;
        for (const auto &[field, value] : struct_t->defaults) {
            defaults[field] = Broadcast::make(lanes, value);
        }
        return Struct_t::make(std::move(name), std::move(fields),
                              std::move(defaults), struct_t->attributes);
    } else if (const Vector_t *vector_t = base.as<Vector_t>()) {
        // TODO: LLVM codegen will not support vectors of vectors, will need to
        // flatten after this.
        // Also follow the SoA, with the packeted size on the inside
        return Vector_t::make(Vector_t::make(vector_t->etype, lanes),
                              vector_t->lanes);
    }
    internal_error << "[unimplemented] broadcast_type: " << base;
}

Expr broadcast_expr(const Expr &base, const size_t lanes) {
    if (base.is<IntImm, UIntImm, IdxImm, FloatImm, BoolImm, Var, Infinity,
                BinOp, UnOp, Select, Cast>()) {
        return Broadcast::make(lanes, base);
    } else if (const Build *build = base.as<Build>()) {
        Type t = broadcast_type(build->type, lanes);
        if (build->values.empty()) {
            return Build::make(std::move(t));
        }
        std::vector<Expr> values(build->values.size());
        for (size_t i = 0; i < build->values.size(); i++) {
            values[i] = broadcast_expr(build->values[i], lanes);
        }
        return Build::make(std::move(t), std::move(values));
    }
    internal_error << "[unimplemented] broadcast_expr: " << base;
}

Stmt packetize_impl(Type scalar_ret_type,
                    std::map<Expr, Expr, ExprLessThan> varying,
                    std::set<Expr, ExprLessThan> uniform,
                    std::set<std::string> broadcasted, Expr call_mask,
                    Stmt body, FuncMap &funcs, TypeMap &types) {
    struct PacketizeImpl : public Mutator {
        FuncMap &funcs;
        TypeMap &types;
        std::map<Expr, Expr, ExprLessThan> varying;
        std::set<Expr, ExprLessThan> uniform;
        std::map<WriteLoc, WriteLoc, WriteLocLessThan> varying_locs;
        std::set<WriteLoc, WriteLocLessThan> uniform_locs;
        std::set<std::string> broadcasted;
        // Number of mask lanes
        const size_t lanes;
        // Which lanes need to complete to be done.
        Expr call_mask;
        // Which lanes have completed (initially !call_mask). This is a mutable
        // var, for simplicity. this must always be at least !call_mask
        Expr dead_mask;
        // The loc to write the dead mask to.
        WriteLoc dead_mask_loc;
        // Which lanes are currently active and whose effects should be seen.
        // Initially = call_mask, this must always be a subset of !dead_mask,
        Expr active_mask;
        // A Load of the result
        Expr result;
        // The loc to write results to
        WriteLoc result_loc;
        // Final statement in the mutated block, for a return optimization.
        Stmt final_stmt;
        bool final_stmt_return_opt = false;

        // Whether active_mask == !dead_mask, currently.
        bool in_finish_case = true;
        bool broadcast_builds = false;

        PacketizeImpl(FuncMap &funcs, TypeMap &types,
                      std::map<Expr, Expr, ExprLessThan> varying,
                      std::set<Expr, ExprLessThan> uniform,
                      std::set<std::string> broadcasted, Expr call_mask)
            : funcs(funcs), types(types), varying(std::move(varying)),
              uniform(std::move(uniform)), broadcasted(std::move(broadcasted)),
              lanes(call_mask.type().lanes()), call_mask(std::move(call_mask)) {
        }

        bool is_uniform(Expr e) {
            if (varying.contains(e)) {
                return false;
            } else if (uniform.contains(e)) {
                return true;
            } else if (is_const(e)) {
                return true;
            } else if (const BinOp *bin = e.as<BinOp>()) {
                return is_uniform(bin->a) && is_uniform(bin->b);
            } else if (const UnOp *un = e.as<UnOp>()) {
                return is_uniform(un->a);
            } else if (const Select *select = e.as<Select>()) {
                return is_uniform(select->cond) && is_uniform(select->tvalue) &&
                       is_uniform(select->fvalue);
            } else if (const Cast *cast = e.as<Cast>()) {
                return is_uniform(cast->value);
            }
            internal_error << "[unimplemented] is_uniform on: " << e;
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

        Expr visit(const Cast *node) override {
            Expr value = mutate(node->value);

            if (value.same_as(node->value)) {
                return node;
            }
            Type t = broadcast_type(node->type, lanes);
            return Cast::make(std::move(t), std::move(value));
        }

        RESTRICT_MUTATOR(Expr, Broadcast);
        RESTRICT_MUTATOR(Expr, VectorReduce);
        RESTRICT_MUTATOR(Expr, VectorShuffle);
        RESTRICT_MUTATOR(Expr, Ramp);
        // Default behavior seems fine?
        // RESTRICT_MUTATOR(Expr, Extract);
        Expr visit(const Build *node) override {
            if (const Struct_t *struct_t = node->type.as<Struct_t>()) {
                if (node->values.empty()) {
                    return node;
                }
                internal_assert(struct_t->defaults.empty())
                    << "[unimplemented] handle defaults in packetized Build of "
                       "struct.";
                std::vector<Expr> values(node->values.size());
                Struct_t::Map fields(node->values.size());

                for (size_t i = 0; i < node->values.size(); i++) {
                    values[i] = mutate(node->values[i]);
                    // TODO: figure out which are varying and which are uniform!
                    if (values[i].same_as(node->values[i]) &&
                        broadcast_builds) {
                        values[i] = broadcast_expr(std::move(values[i]), lanes);
                    }
                    fields[i].name = struct_t->fields[i].name;
                    fields[i].type = values[i].type();
                }
                Type type =
                    Struct_t::make("__packed" + struct_t->name,
                                   std::move(fields), struct_t->attributes);
                return Build::make(type, std::move(values));
            }
            internal_error << "[unimplemented] packetized Build: "
                           << Expr(node);
        }

        // RESTRICT_MUTATOR(Expr, Access);

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
            std::string new_name = "__packetized_" + func->name;
            if (const auto iter = funcs.find(new_name); iter != funcs.cend()) {
                Expr cached = Var::make(iter->second->call_type(), new_name);
                rec.args.push_back(active_mask);
                return Call::make(cached, rec.args);
            }

            std::vector<Function::Argument> arg_params(rec.args.size() + 1);
            std::vector<Function_t::ArgSig> arg_sigs(rec.args.size() + 1);
            for (size_t i = 0; i < rec.args.size(); i++) {
                arg_params[i].name = scalar_func->args[i].name;
                arg_params[i].type = rec.args[i].type();
                arg_params[i].mutating = scalar_func->args[i].mutating;
                internal_assert(!scalar_func->args[i].default_value.defined())
                    << "[unimplemented] packetization of function with "
                       "default "
                       "arg: "
                    << Expr(node);

                arg_sigs[i].type = rec.args[i].type();
                arg_sigs[i].is_mutable = scalar_func->args[i].mutating;
            }
            // Also add mask
            arg_params.back().name = call_mask_name;
            arg_params.back().type = call_mask.type();
            arg_params.back().mutating = false;
            arg_sigs.back().type = call_mask.type();
            arg_sigs.back().is_mutable = false;

            // Presumably, a vector return type.
            Type ret_type = broadcast_type(scalar_func->ret_type, lanes);

            std::map<Expr, Expr, ExprLessThan> call_varying;
            std::set<std::string> call_broadcasted;
            std::set<Expr, ExprLessThan> call_uniform;

            for (size_t i = 0; i < rec.args.size(); i++) {
                Expr old_var = Var::make(scalar_func->args[i].type,
                                         scalar_func->args[i].name);
                if (rec.mutated[i]) {
                    call_broadcasted.insert(arg_params[i].name);
                    call_varying[old_var] = Var::make(
                        rec.args[i].type(), scalar_func->args[i].name);
                } else {
                    call_uniform.insert(old_var);
                }
            }

            // Handle recursive calls by inserting and then mutating body.
            Stmt body;
            std::shared_ptr<Function> packet_func = std::make_shared<Function>(
                new_name, arg_params, ret_type, body, scalar_func->interfaces,
                scalar_func->attributes);

            funcs[new_name] = packet_func;

            Expr func_call_mask = Var::make(call_mask.type(), call_mask_name);

            packet_func->body =
                packetize_impl(scalar_func->ret_type, call_varying,
                               call_uniform, call_broadcasted, func_call_mask,
                               scalar_func->body, funcs, types);

            Expr new_func = Var::make(packet_func->call_type(), new_name);

            rec.args.push_back(active_mask);
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
                        ? Broadcast::make(lanes, std::move(value))
                        : value;

            // We want this return value for every active thread,
            // and the current thing for all dead threads.
            // TODO: what about the threads that are neither?
            // For now, use the more-easily-statically-analyzed
            // thing, because dead_mask is a dynamically update
            // memory location.
            value = Select::make(active_mask, value, result);

            Expr now_dead = dead_mask | active_mask;

            // if call_mask == new_dead, we're done!
            // otherwise, just update the mutable return value
            // and the dead_mask location.
            Stmt return_value = Return::make(value);

            // But first, we try a few nice optimizations.
            if (final_stmt.same_as(node)) {
                internal_assert(!final_stmt_return_opt);
                // This is the final return, so just return value.
                final_stmt_return_opt = true;
                return return_value;
            }

            if (in_finish_case) {
                // We statically know that active == !dead, so just return.
                return return_value;
            }

            // We don't use predicated stores, instead opting for the select
            // above, because LLVM seems really bad at optimizing predicated
            // stores.
            // TODO(ajr): investigate this further, maybe will be good on x86.
            Stmt store_results = Sequence::make({
                Store::make(result_loc, value,
                            /*mask=*/Expr()),
                Store::make(dead_mask_loc, now_dead,
                            /*mask=*/Expr()),
            });

            return IfElse::make(all(now_dead == call_mask),
                                std::move(return_value),
                                std::move(store_results));
        }

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
                    << "[unimplemented] packetization through logical "
                       "negated "
                       "IfElse: "
                    << Stmt(node);
            }

            Expr cond = mutate(node->cond);

            if (cond.same_as(node->cond)) {
                // Uniform control flow! No need to update masks.
                return IfElse::make(std::move(cond), mutate(node->then_body),
                                    node->else_body);
            }

            // I don't think even CFGs save us from this horrendous
            // explosion.

            // If we statically know that all !dead threads are currently
            // active, then we can make some special cases of when all threads
            // go one way. However, if we don't statically know that, just give
            // up and test both branches.

            Expr old_active_mask = std::move(active_mask);
            bool old_in_finish_case = in_finish_case;

            in_finish_case = false;
            active_mask = old_active_mask & cond;
            Stmt some_true = mutate(node->then_body);
            some_true =
                IfElse::make(any(~dead_mask == (old_active_mask & cond)),
                             std::move(some_true));
            active_mask = old_active_mask & ~cond;

            Stmt any_check = std::move(some_true);
            if (node->else_body.defined()) {
                Stmt some_false = mutate(node->else_body);
                some_false =
                    IfElse::make(any(~dead_mask == (old_active_mask & ~cond)),
                                 std::move(some_false));
                any_check = Sequence::make(
                    {std::move(any_check), std::move(some_false)});
            }

            in_finish_case = old_in_finish_case;
            active_mask = old_active_mask;

            if (!in_finish_case) {
                // Don't bother testing all/none
                return any_check;
            }

            // Here, we statically know that all !dead threads are active,
            // so test for when they all take one of the branches.

            // No need to update active mask
            if (node->else_body.defined()) {
                Stmt none_case = mutate(node->else_body);
                any_check =
                    IfElse::make(all(~dead_mask == (active_mask & ~cond)),
                                 std::move(none_case), std::move(any_check));
            }

            Stmt all_case = mutate(node->then_body);

            return IfElse::make(all(~dead_mask == (active_mask & cond)),
                                std::move(all_case), std::move(any_check));
        }

        RESTRICT_MUTATOR(Stmt, DoWhile);
        // RESTRICT_MUTATOR(Stmt, Sequence);

        Stmt visit(const LetStmt *node) override {
            Expr value = mutate(node->value);
            if (is_uniform(node->value)) {
                internal_assert(value.same_as(node->value))
                    << "Analysis found uniform location, but write is varying: "
                    << Stmt(node) << " mutated to store: " << value;
                uniform_locs.insert(node->loc);
                uniform.insert(writeloc_to_read(node->loc));
                return node;
            }
            // Otherwise is mutating.
            internal_assert(!value.same_as(node->value))
                << "Analysis found varying location, but write is uniform: "
                << Stmt(node) << " mutated to store: " << value;

            Type write_type = value.type();
            Expr old_var = writeloc_to_read(node->loc);
            WriteLoc new_loc(node->loc.base, std::move(write_type));
            Expr new_var = writeloc_to_read(new_loc);
            varying[old_var] = new_var;
            varying_locs[node->loc] = new_loc;
            return LetStmt::make(new_loc, std::move(value));
        }

        Stmt visit(const Allocate *node) override {
            // Mutable things are always broadcasted, because we don't
            // necessarily know if they'll be updated synchronously or
            // not.
            // TODO(ajr): This ^ is a limitation we should fix
            WriteLoc new_loc(node->loc.base,
                             broadcast_type(node->loc.base_type, lanes));

            varying_locs[node->loc] = new_loc;

            if (!node->value.defined()) {
                broadcasted.insert(node->loc.base);
                return Allocate::make(new_loc, node->memory);
            }

            ScopedValue<bool> _(broadcast_builds, true);
            Expr value = mutate(node->value);

            if (value.same_as(node->value)) {
                value = broadcast_expr(std::move(value), lanes);
            }

            // Broadcast reads to this.
            varying[writeloc_to_read(node->loc)] =
                Var::make(value.type(), node->loc.base);

            return Allocate::make(new_loc, std::move(value), node->memory);
        }

        Stmt visit(const Store *node) override {
            Expr value = mutate(node->value);

            // If the base is varying, all is good to write
            WriteLoc loc(node->loc.base, node->loc.base_type);
            if (auto iter = varying_locs.find(loc);
                iter != varying_locs.cend()) {
                internal_assert(node->loc.accesses.empty())
                    << "[unimplemented] accessed write to varying base: "
                    << Stmt(node) << " mutated to " << value;

                internal_assert(!value.same_as(node->value))
                    << "[unimplemented] uniform write to varying location: "
                    << Stmt(node) << " mutated to " << value;

                return Store::make(iter->second, std::move(value), active_mask);
            }

            // Write is to a uniform base, must be a varying index or uniform
            // value. Look for a varying index.
            bool innermost_varies = false;
            size_t varies_count = 0;

            for (const auto &access : node->loc.accesses) {
                internal_assert(std::holds_alternative<Expr>(access))
                    << "[unimplemented] fields in packetized store "
                    << Stmt(node);
                Expr expr = mutate(std::get<Expr>(access));
                // only care about the last access.
                innermost_varies = !expr.same_as(std::get<Expr>(access));
                varies_count += innermost_varies;
                loc.add_index_access(expr);
            }
            internal_assert(innermost_varies || value.same_as(node->value))
                << "[unimplemented] packetized store to uniform location: "
                << Stmt(node);

            if (!innermost_varies && value.same_as(node->value)) {
                internal_assert(varies_count == 0)
                    << "[unimplemented] packetized store to nested varying"
                    << " location: " << Stmt(node);
                // Perform a single write. It is assumed that the mask is
                // non-empty if we are here, so any(mask) == true
                return Store::make(loc, std::move(value), /*mask=*/Expr());
            }
            internal_assert(innermost_varies);
            internal_assert(varies_count == 1)
                << "[unimplemented] packetized store to nested varying"
                << " location: " << Stmt(node);

            // This must be a vectorized write.
            if (value.same_as(node->value)) {
                // Broadcast the value to match expected return type.
                value = Broadcast::make(lanes, std::move(value));
            }
            return Store::make(loc, std::move(value), active_mask);
        }

        RESTRICT_MUTATOR(Stmt, Accumulate);
        // RESTRICT_MUTATOR(Stmt, Label);
        // RESTRICT_MUTATOR(Stmt, RecLoop);
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

    PacketizeImpl rewriter(funcs, types, std::move(varying), std::move(uniform),
                           std::move(broadcasted), call_mask);

    rewriter.active_mask = call_mask;
    rewriter.dead_mask = ~call_mask;

    if (!scalar_ret_type.defined()) {
        // This Stmt does not return.
        return rewriter.mutate(std::move(body));
    }

    // This stmt does return, allocate a stack variable to hold and return
    // the result.
    const size_t lanes = call_mask.type().lanes();
    Type ret_type = broadcast_type(scalar_ret_type, lanes);
    Expr result = Var::make(ret_type, result_name);
    WriteLoc result_loc(result_name, ret_type);
    Expr dead_mask = Var::make(call_mask.type(), dead_mask_name);
    WriteLoc dead_mask_loc(dead_mask_name, call_mask.type());

    rewriter.result = result;
    rewriter.dead_mask = dead_mask;
    rewriter.result_loc = result_loc;
    rewriter.dead_mask_loc = dead_mask_loc;
    if (const Sequence *seq = body.as<Sequence>()) {
        rewriter.final_stmt = seq->stmts.back();
    }

    std::vector<Stmt> stmts = {
        // TODO: ideally this would be undef, which would let LLVM be
        // smarter.
        Allocate::make(std::move(result_loc), make_zero(ret_type),
                       Allocate::Memory::Stack),
        Allocate::make(std::move(dead_mask_loc), ~call_mask,
                       Allocate::Memory::Stack),
        rewriter.mutate(std::move(body)),
    };

    if (!rewriter.final_stmt_return_opt) {
        // TODO(ajr): We shouldn't need this, for well-formed code.
        stmts.emplace_back(Return::make(std::move(result)));
    }

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
            // TODO(ajr): is this initially filled with anything?
            std::set<Expr, ExprLessThan> uniform;

            return packetize_impl(/*scalar_ret_type=*/Type(), varying, uniform,
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
