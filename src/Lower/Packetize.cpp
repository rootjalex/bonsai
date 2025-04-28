#include "Lower/Packetize.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Opt/Simplify.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

using namespace ir;

namespace {

Stmt concat_stmts(const Stmt &header, const Stmt &body) {
    if (!header.defined()) {
        return body;
    }
    std::vector<Stmt> stmts;

    auto append = [&stmts](const Stmt &stmt) {
        if (const Sequence *seq = stmt.as<Sequence>()) {
            stmts.insert(stmts.end(), seq->stmts.cbegin(), seq->stmts.cend());
        } else {
            stmts.push_back(stmt);
        }
    };

    append(header);
    append(body);
    return Sequence::make(std::move(stmts));
}

struct PacketizeImpl : public Mutator {
    const std::string &idx_name;
    // TODO(ajr): support non-constant-sized packetization.
    const size_t lanes;
    // To insert new packetized funcs into.
    FuncMap &funcs;

    SetStack<std::string> varying;

    PacketizeImpl(const std::string &idx_name, const size_t lanes, FuncMap &funcs)
        : idx_name(idx_name), lanes(lanes), funcs(funcs) {
        push_all_true_mask();
        returned = Broadcast::make(lanes, BoolImm::make(false));
        varying.new_frame();
        varying.add_to_frame(idx_name);
    }

    std::vector<Expr> masks;
    // Mask of lanes that have returned.
    Expr returned;

    void push_all_true_mask() {
        masks.push_back(Broadcast::make(lanes, BoolImm::make(true)));
    }

    void push_mask(Expr mask) {
        masks.emplace_back(std::move(mask));
    }

    void swap_mask(Expr mask) {
        internal_assert(!masks.empty());
        masks.back() = std::move(mask);
    }

    void pop_mask() {
        masks.pop_back();
    }

    Expr get_mask() const {
        internal_assert(!masks.empty());
        return masks.back() & ~returned;
    }

    bool is_varying(const Expr &expr) const {
        struct Checker : public Visitor {
            const SetStack<std::string> &varying;
            bool found = false;

            Checker(const SetStack<std::string> &varying)
                : varying(varying) {}

            // TODO: profile and support early-out if necessary.

            void visit(const Var *node) override {
                if (found) {
                    return;
                }
                found = varying.contains(node->name);
            }
        };
        Checker checker(varying);
        expr.accept(&checker);
        return checker.found;
    }

    Expr visit(const Var *node) override {
        if (varying.contains(node->name)) {
            Expr wide_var = Var::make(Vector_t::make(node->type, lanes), node->name);
            return wide_var;
        }
        return node;
    }

    RESTRICT_MUTATOR(Stmt, CallStmt);
    RESTRICT_MUTATOR(Stmt, Print);

    Stmt visit(const Store *node) override {
        if (!varying.contains(node->name)) {
            varying.add_to_frame(node->name);
        }

        Expr index = mutate(node->index);
        internal_assert(!index.same_as(node->index));
        internal_assert(is_varying(index));

        Expr value = mutate(node->value);

        // TODO: this is unnecesary if Stores with true predicates are codegening correctly.
        std::cout << "Store: " << ir::Stmt(node) << " with mask: " << get_mask() << "\n";
        Expr mask = opt::Simplify::simplify(get_mask());
        std::cout << "  -> " << mask << "\n";

        if (is_const_one(mask)) {
            // No control flow yet.
            return Store::make(node->name, std::move(index), std::move(value), /*predicate=*/Expr());
        } else {
            return Store::make(node->name, std::move(index), std::move(value), /*predicate=*/std::move(mask));
        }
    }

    Stmt visit(const LetStmt *node) override {
        Expr value = mutate(node->value);

        if (value.same_as(node->value)) {
            // Uniform
            return node;
        } else if (is_varying(value)) {
            varying.add_to_frame(node->loc.base);
            WriteLoc loc(node->loc.base, Vector_t::make(node->loc.type, lanes));
            return LetStmt::make(std::move(loc), std::move(value));
        }

        // Uniform?
        internal_error << "Mutated LetStmt value but is not varying: " << Stmt(node) << " -> " << value;
    }

    Expr all_with_short_circuit(Expr cond) {
        if (const BinOp *op = cond.as<BinOp>()) {
            if (op->op == BinOp::LAnd) {
                // all(a && b) = all(a) && all(b)
                return all_with_short_circuit(op->a) && all_with_short_circuit(op->b);
            } else if (op->op == BinOp::LOr) {
                // all(a || b) = ???
                internal_error << "[unimplemented] all(a || b) = ??? for " << cond;
            }
        } else if (const UnOp *op = cond.as<UnOp>()) {
            if (op->op == UnOp::Not) {
                // all(~a) = ~any(a)
                return ~any_with_short_circuit(op->a);
            }
        }
        // Base case, no logical operator.
        return all(mutate(cond));
    }

    Expr any_with_short_circuit(Expr cond) {
        if (const BinOp *op = cond.as<BinOp>()) {
            if (op->op == BinOp::LAnd) {
                // any(a && b) = ???
                internal_error << "[unimplemented] any(a && b) = ??? for " << cond;
            } else if (op->op == BinOp::LOr) {
                // any(a || b) = any(a) || any(b)
                return any_with_short_circuit(op->a) && any_with_short_circuit(op->b);
            }
        } else if (const UnOp *op = cond.as<UnOp>()) {
            if (op->op == UnOp::Not) {
                // any(~a) = ~all(a)
                return ~all_with_short_circuit(op->a);
            }
        }
        // Base case, no logical operator.
        return any(mutate(cond));
    }

    Expr build_short_circuit(Expr cond) {
        if (const BinOp *op = cond.as<BinOp>()) {
            if (op->op == BinOp::LAnd) {
                // If a is false, don't compute that lane of b.
                Expr a = build_short_circuit(op->a);
                Expr mask = get_mask();
                push_mask(mask & a);
                Expr b = build_short_circuit(op->b);
                pop_mask();
                return a & b;
            } else if (op->op == BinOp::LOr) {
                // If a is true, don't compute that lane of b.
                Expr a = build_short_circuit(op->a);
                Expr mask = get_mask();
                push_mask(mask & ~a);
                Expr b = build_short_circuit(op->b);
                pop_mask();
                return a | b;
            }
        } else if (const UnOp *op = cond.as<UnOp>()) {
            if (op->op == UnOp::Not) {
                // Recurse in case of nested short-circuiting
                return ~build_short_circuit(op->a);
            }
        }
        // Base case, no logical operator.
        return mutate(cond);
    }

    Stmt visit(const IfElse *node) override {
        if (is_varying(node->cond)) {
            Expr mask = get_mask();
            // TODO(ajr): do we always want the all() case?
            Expr all_lanes = all_with_short_circuit(mask) && all_with_short_circuit(node->cond);
            Expr no_lanes = all_with_short_circuit(mask) && all_with_short_circuit(~node->cond);

            // If all_lanes do unmasked then_body
            // If no lanes do unmasked else_body
            // Else (some lanes) do masked then + else

            Expr mask_then = mask & build_short_circuit(node->cond);
            Expr mask_else = mask & build_short_circuit(~node->cond);

            /*
            internal_error << "all_lanes of: " << node->cond << " -> " << all_lanes << "\n"
            << "no_lanes of: " << node->cond << " -> " << no_lanes << "\n"
            << "then_mask of: " << node->cond << " -> " << mask_then << "\n"
            << "else_mask of: " << node->cond << " -> " << mask_else << "\n";
            */

            push_all_true_mask();
            varying.new_frame();
            Stmt unmasked_then = mutate(node->then_body);

            Stmt unmasked_else;
            if (node->else_body.defined()) {
                varying.pop_frame();
                varying.new_frame();
                unmasked_else = mutate(node->else_body);
            }

            varying.pop_frame();
            varying.new_frame();
            swap_mask(mask_then);
            Stmt masked_then = mutate(node->then_body);
            swap_mask(mask_else);

            Stmt masked_else;
            if (node->else_body.defined()) {
                varying.pop_frame();
                varying.new_frame();
                masked_else = mutate(node->else_body);
            }

            varying.pop_frame();
            pop_mask();

            if (node->else_body.defined()) {
                Stmt seq = concat_stmts(masked_then, masked_else);
                return IfElse::make(std::move(all_lanes),
                                    std::move(unmasked_then),
                                    IfElse::make(std::move(no_lanes),
                                                 std::move(unmasked_else),
                                                 std::move(seq)));
            } else {
                return IfElse::make(std::move(all_lanes),
                                    std::move(unmasked_then),
                                    IfElse::make(~no_lanes,
                                                 std::move(masked_then)));
            }
        } else {
            Stmt then_body = mutate(node->then_body);
            Stmt else_body = node->else_body.defined() ? mutate(node->else_body) : node->else_body;
            return IfElse::make(node->cond, std::move(then_body), std::move(else_body));
        }
    }

    RESTRICT_MUTATOR(Stmt, DoWhile); // TODO

    // Only overriding in order to remove dead Continue (and Return???) nodes.
    Stmt visit(const Sequence *node) override {
        bool not_changed = true;
        std::vector<Stmt> stmts;
        for (const auto &s : node->stmts) {
            Stmt ns = mutate(s);
            if (ns.defined()) {
                stmts.emplace_back(std::move(ns));
            }
            not_changed = not_changed && stmts.back().same_as(s);
        }
        if (not_changed) {
            return node;
        } else if (stmts.size() == 0) {
            return Stmt();
        } else if (stmts.size() == 1) {
            return stmts.back();
        }
        return Sequence::make(std::move(stmts));
    }

    RESTRICT_MUTATOR(Stmt, Assign); // TODO
    RESTRICT_MUTATOR(Stmt, Accumulate); // TODO
    RESTRICT_MUTATOR(Stmt, Allocate);
    RESTRICT_MUTATOR(Stmt, Label);

    RESTRICT_MUTATOR(Stmt, RecLoop);
    RESTRICT_MUTATOR(Stmt, Match);
    RESTRICT_MUTATOR(Stmt, Yield);
    RESTRICT_MUTATOR(Stmt, Scan);
    RESTRICT_MUTATOR(Stmt, YieldFrom);
    RESTRICT_MUTATOR(Stmt, ForAll); // TODO

    // Need a "returned" mask that clobbers all lanes that have escaped so far.
    Stmt visit(const Continue *node) override {
        // TODO: is there more than this?
        // Anything active right now becomes inactive for the rest of the body.
        // TODO: in the first `continue` case, we actually want to convert to a return...
        if (!is_const_one(masks.back())) {
            returned = returned | masks.back();
            std::cout << "Set returned: " << returned << "\n";
            std::cout << "-> " << opt::Simplify::simplify(returned) << "\n";
        }
        return ir::Stmt();
    }
    RESTRICT_MUTATOR(Stmt, Return); // TODO
};

bool loads_from_array(const Stmt &header, const std::string &loop) {
    if (!header.defined()) {
        return false;
    }
    struct FindArrayLoad : public Visitor {
        const std::string &loop;

        bool found = false;

        FindArrayLoad(const std::string &loop) : loop(loop) {}

        // TODO(ajr): this may be super brittle. It needs to be battle-tested...
        void visit(const LetStmt *node) override {
            if (found) {
                return;
            }
            if (const Extract *extract = node->value.as<Extract>()) {
                if (const Var *var = extract->vec.as<Var>()) {
                    found = var->name == loop;
                }
            }
        }
    };

    FindArrayLoad finder(loop);
    header.accept(&finder);
    return finder.found;
}

struct FindPacketizeLoops : public Mutator {
    const std::string &loop;
    FuncMap &funcs;

    bool found = false;

    FindPacketizeLoops(const std::string &loop, FuncMap &funcs)
        : loop(loop), funcs(funcs) {}

    Stmt visit(const ForAll *node) override {
        // Can packetize explicitly-named loop or loop over array.
        if (node->index == loop || loads_from_array(node->header, loop)) {
            found = true;
            // Compute vector size.
            Expr size = opt::Simplify::simplify(((node->slice.end - node->slice.begin) + (node->slice.stride - 1)) / node->slice.stride);
            internal_assert(is_const(size)) << "[unimplemented] packetize of dynamic-sized loop: " << Stmt(node);
            int64_t lanes = *as_const_int(size);
            Stmt loop_body = concat_stmts(node->header, node->body);
            Stmt repl = PacketizeImpl(node->index, lanes, funcs).mutate(std::move(loop_body));

            Expr index = Ramp::make(node->slice.begin, node->slice.stride, lanes);
            Stmt header = LetStmt::make(ir::WriteLoc(node->index, index.type()), std::move(index));

            std::cout << "Made:\n" << repl << "\n";

            return concat_stmts(header, repl);
        } else {
            return Mutator::visit(node);
        }
    }
};

void packetize(const std::string &fname, const std::string &loop, FuncMap &funcs) {
    auto fiter = funcs.find(fname);
    internal_assert(fiter != funcs.end())
        << "Packetize cannot find function: " << fname;

    auto &func = fiter->second;

    std::cout << "packetize() on:\n" << *func << "\n";

    if (!contains<ForAll>(func->body)) {
        const Return *as_return = func->body.as<Return>();
        internal_assert(as_return)
            << "Packetize cannot find forall in function: " << fname << " : " << func->body;
        const Call *as_call = as_return->value.as<Call>();
        internal_assert(as_call)
            << "Packetize cannot find indirect forall in function: " << fname << " : " << func->body;
        const Var *as_var = as_call->func.as<Var>();
        internal_assert(as_var)
            << "Packetize cannot find indirect forall in function: " << fname << " : " << func->body;
        packetize(as_var->name, loop, funcs);
    } else {
        FindPacketizeLoops finder(loop, funcs);
        func->body = finder.mutate(std::move(func->body));
        internal_assert(finder.found)
            << "Packetize did not find forall in function: " << fname << " : " << func->body;
    }
}

} // namespace

Program Packetize::run(Program program) const {
    // TODO(ajr): get this from the schedule.
    // std::string func = "array_abs";
    // std::string array = "a";

    // packetize(func, array, program.funcs);

    return program;
}

/*static*/ ir::Stmt packetize_stmt(const std::string &index, const size_t lanes, ir::FuncMap &funcs, ir::Stmt stmt) {
    return PacketizeImpl(index, lanes, funcs).mutate(std::move(stmt));
}

} // namespace lower
} // namespace bonsai
