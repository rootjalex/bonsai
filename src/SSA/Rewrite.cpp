#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/InsertPreheader.h"
#include "SSA/QueueRecursion.h"
#include "SSA/Rewrite.h"
#include "SSA/LoopArithmetic.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <limits>

#include <functional>
#include <iostream>
#include <optional>
#include <sstream>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::tuple;
using std::vector;

namespace {

// The split, with the chunk either a constant `factor_n` or the value of
// `func` named `factor_name` (see the two overloads in SSA/Rewrite.h).
void split_impl(FuncMap &funcs, string func, string idx,
                std::optional<int> factor_n, std::optional<Expr> factor_e,
                string outer, string inner, bool exact) {
    internal_assert(funcs.contains(func))
        << "split applied to unknown func:" << func;
    auto f = funcs[func];
    internal_assert(factor_n.has_value() != factor_e.has_value())
        << "split(" << idx << ") on " << func
        << ": a constant chunk or one the program sizes";
    // How the chunk is written in a message.
    const string factor_text = [&] {
        std::ostringstream os;
        if (factor_n.has_value()) {
            os << *factor_n;
        } else {
            os << *factor_e;
        }
        return os.str();
    }();

    vector<shared_ptr<Block>> blocks;

    for (auto &block : f->blocks) {
        blocks.push_back(block);
        if (!std::holds_alternative<Terminator::ParFor>(
                block->terminator.data)) {

            continue;
        }
        Terminator::ParFor parfor =
            std::get<Terminator::ParFor>(block->terminator.data);
        if (parfor.index != idx) {
            continue;
        }

        // Whether an index this loop would have visited is one the split
        // still visits. Without a tail the two loops together cover
        // start, start+factor, start+2*factor, ... and each chunk walks
        // `factor` wide by the original stride, so the cover is exact only if
        // the chunk is a whole number of strides and the range is a whole
        // number of chunks. Get either wrong and the split runs off the end of
        // what it was asked to iterate, which is a wrong answer rather than a
        // slow one -- it writes past the range the program reasoned about.
        const auto as_int =
            [](const std::shared_ptr<Value> &v) -> std::optional<int64_t> {
            const auto *c = std::get_if<Constant>(&v->data);
            if (c == nullptr) {
                return std::nullopt;
            }
            if (const auto *i = std::get_if<int64_t>(&c->data)) {
                return *i;
            }
            // An unsigned loop index makes an unsigned constant, and reading
            // only the signed one meant a `parfor i in 0u:n` could not be
            // split at all -- its stride is the constant one, spelled `1u`.
            // Strides are small and positive; anything that would not survive
            // the narrowing is not a stride.
            if (const auto *u = std::get_if<uint64_t>(&c->data)) {
                if (*u <= uint64_t(std::numeric_limits<int64_t>::max())) {
                    return int64_t(*u);
                }
            }
            return std::nullopt;
        };

        const auto stride_n = as_int(parfor.stride);
        internal_assert(stride_n.has_value())
            << "split(" << idx << ") on " << func
            << " needs a constant stride to know that its chunks line up with "
               "the steps the loop takes";
        if (factor_n.has_value()) {
            internal_assert(*stride_n > 0 && *factor_n % *stride_n == 0)
                << "split(" << idx << ", " << *factor_n << ") on " << func
                << " does not divide the loop's stride of " << *stride_n
                << ", so a chunk would start part way through a step";
        } else {
            internal_assert(*stride_n == 1)
                << "split(" << idx << ", " << factor_text << ") on " << func
                << ": a chunk the program sizes cannot be checked against a "
                << "stride of " << *stride_n << "; only a loop of stride one "
                << "is split by a run-time chunk";
        }

        // Whether the range is a whole number of chunks, known when both its
        // ends are constants. Without a tail, a range known only at run time
        // is the caller's assertion to make -- that is what asking for no
        // tail means -- and one known here is checked.
        const auto start_n = as_int(parfor.start);
        const auto end_n = as_int(parfor.end);
        const bool divisible = factor_n.has_value() && start_n.has_value() &&
                               end_n.has_value() &&
                               (*end_n - *start_n) % *factor_n == 0;
        if (exact && factor_n.has_value() && start_n.has_value() && end_n.has_value()) {
            internal_assert(divisible)
                << "split(" << idx << ", " << *factor_n << ") on " << func
                << " does not divide the loop's range of [" << *start_n << ":"
                << *end_n << "), so without a tail it would run "
                << (*factor_n - (*end_n - *start_n) % *factor_n)
                << " iterations past the end";
        }
        // The tail, asked for and needed: the outer loop's last chunk runs
        // past a range that is not a whole number of chunks, and the body
        // goes behind a test that the index it was handed is still short of
        // the end -- Halide's GuardWithIf tail strategy, which is the one a
        // loop about to be vectorized wants, since the test becomes the
        // gang's mask and the last gang runs partly full rather than the
        // range being trimmed to whole gangs. A test on a constant range
        // known to divide would always pass, and is not made.
        const bool guard = !exact && !divisible;

        // The body's index becomes the inner loop's, and is renamed to say so.
        // parfor i in start:end:stride body(i) cont()
        // ->
        // parfor outer in start:end:factor inner_loop(outer) cont()
        // block inner_loop(o):
        //   parfor inner in 0:factor:stride step(inner, o) inner_cont()
        // block step(n, o): body(o + n)
        // block inner_cont(): yield
        //
        // The inner loop counts from zero and the body is handed the sum,
        // rather than the inner loop counting from `outer` and the body being
        // handed its index directly. The second is one instruction cheaper,
        // but it makes the inner range depend on the outer index, and a range
        // that does has no single trip count -- so nothing could collapse the
        // two loops afterwards, or reason about the inner one on its own. The
        // Stmt-level split makes the same choice.
        Type itype = parfor.start->get_type();
        // A constant has to go into the arm its type will be read back out
        // of: an unsigned index becomes a UIntImm, and a signed constant
        // carrying an unsigned type asserts on the way out rather than here.
        // Both of these are small and non-negative whatever the index is.
        const auto index_constant = [&](int64_t v) {
            return itype.is_uint()
                       ? std::make_shared<Value>(Constant{itype, uint64_t(v)})
                       : std::make_shared<Value>(Constant{itype, v});
        };
        // The chunk: the constant, or the program's value, threaded to this
        // block as any value is and made the index's type. A named chunk is
        // handed to the inner loop's block as an argument of its own, the way
        // the end of the range is below.
        shared_ptr<Value> split_factor;
        std::optional<Argument> factor_arg;
        if (factor_n.has_value()) {
            split_factor = index_constant(*factor_n);
        } else {
            // The expression, emitted in this block: a name is the program's
            // value of that name, threaded here from where it is defined; a
            // constant is one of the index's type; arithmetic is an
            // instruction, folded where its operands are constants (arith
            // in SSA/LoopArithmetic.h).
            const auto type_of_name = [&](const string &name) -> std::optional<Type> {
                std::optional<Type> found;
                for (const Argument &a : f->blocks.front()->args) {
                    if (a.name == name) {
                        found = a.type;
                    }
                }
                for (const auto &b : f->blocks) {
                    for (const auto &in : b->instrs) {
                        if (in->name == name) {
                            found = in->type;
                        }
                    }
                }
                return found;
            };
            std::function<shared_ptr<Value>(const Expr &)> emit =
                [&](const Expr &e) -> shared_ptr<Value> {
                if (const auto *i = e.as<IntImm>()) {
                    return index_constant(i->value);
                }
                if (const auto *u = e.as<UIntImm>()) {
                    return index_constant(int64_t(u->value));
                }
                if (const auto *v = e.as<Var>()) {
                    const auto named_type = type_of_name(v->name);
                    internal_assert(named_type.has_value())
                        << "split(" << idx << ", " << factor_text << ") on " << func
                        << ": no value of the function is named " << v->name
                        << ". The chunk names parameters, or values computed "
                        << "before the loop.";
                    internal_assert(named_type->is_int_or_uint())
                        << "split(" << idx << ", " << factor_text << ") on " << func
                        << ": " << v->name << " is a " << *named_type
                        << ", not an integer";
                    shared_ptr<Value> value = block->get_value(v->name, *named_type);
                    if (!equals(*named_type, itype)) {
                        value = block->make_instruction(itype, Instruction::Op::Cast,
                                                        {value});
                    }
                    return value;
                }
                if (const auto *b = e.as<BinOp>()) {
                    Instruction::Op op;
                    switch (b->op) {
                    case BinOp::Add: op = Instruction::Op::Add; break;
                    case BinOp::Sub: op = Instruction::Op::Sub; break;
                    case BinOp::Mul: op = Instruction::Op::Mul; break;
                    case BinOp::Div: op = Instruction::Op::Div; break;
                    case BinOp::Mod: op = Instruction::Op::Mod; break;
                    default:
                        internal_error
                            << "split(" << idx << ", " << factor_text << ") on "
                            << func << ": [unimplemented] the chunk's expression "
                            << "uses an operation other than + - * / %";
                    }
                    return arith(*block, itype, op, emit(b->a), emit(b->b));
                }
                internal_error << "split(" << idx << ", " << factor_text << ") on "
                               << func << ": [unimplemented] the chunk is an "
                               << "expression of a kind not handled: " << e;
                return nullptr;
            };
            split_factor = emit(*factor_e);
            factor_arg = Argument{itype, f->get_unique_name()};
        }
        auto zero = index_constant(0);

        // TODO: truly unique name generation?
        shared_ptr<Block> outer_yield = std::make_shared<Block>();
        outer_yield->name = parfor.body.name + "_yield_" + outer;
        outer_yield->terminator.data = Terminator::Yield{};
        outer_yield->owner = f;

        Argument outer_arg{itype, outer};
        auto v_outer_arg = std::make_shared<Value>(outer_arg);

        // The arguments the loop was already threading into its body, which
        // both new blocks have to carry so that they still arrive. Taken from
        // what the body block declares rather than from what the jump passes:
        // a jump may pass a constant or an instruction's result, which has no
        // Argument to copy, and dropping those left the blocks below with
        // fewer parameters than their callers supply.
        const BlockMap bmap = make_block_map(f);
        internal_assert(bmap.contains(parfor.body.name))
            << func << " has no block " << parfor.body.name;
        const auto &loop_body = bmap.at(parfor.body.name);
        internal_assert(!loop_body->args.empty())
            << parfor.body.name << " has no index argument";
        const std::vector<Argument> carried(loop_body->args.begin() + 1,
                                            loop_body->args.end());
        internal_assert(carried.size() == parfor.body.args.size())
            << parfor.body.name << " takes " << loop_body->args.size()
            << " arguments but the loop passes it "
            << (parfor.body.args.size() + 1);

        // The loops made here are new loops, and a bind names a loop: bind
        // `outer` or `inner`, after splitting.
        internal_assert(!parfor.binding.has_value())
            << "split(" << idx << ") on " << func << ": " << idx
            << " is bound to " << to_string(*parfor.binding)
            << ", and the split makes two new loops of it. Bind " << outer
            << " or " << inner << " after the split instead.";

        internal_assert(std::holds_alternative<Constant>(parfor.stride->data))
            << "TODO: handle non-Constant strides in split mining";

        // Where the two indices become the one the body expects. With a
        // guard, the end of the range comes along too, threaded through both
        // new blocks as an argument the way the carried values are -- a
        // block refers only to its own values -- unless it is a constant.
        std::optional<Argument> end_arg;
        if (guard && !std::holds_alternative<Constant>(parfor.end->data)) {
            end_arg = Argument{itype, f->get_unique_name()};
        }
        shared_ptr<Block> step = std::make_shared<Block>();
        step->name = parfor.body.name + "_step_" + inner;
        step->owner = f; // This *MUST* exist before make_instruction
        const Argument inner_arg{itype, inner};
        auto v_inner = step->add_argument(inner_arg);
        auto v_outer = step->add_argument(outer_arg);
        for (const Argument &arg : carried) {
            step->add_argument(arg);
        }
        shared_ptr<Value> v_end;
        if (guard) {
            v_end = end_arg ? step->add_argument(*end_arg) : parfor.end;
        }
        auto absolute = step->make_instruction(itype, Instruction::Op::Add,
                                               {v_outer, v_inner});
        std::vector<shared_ptr<Value>> to_body = {absolute};
        for (const Argument &arg : carried) {
            to_body.push_back(std::make_shared<Value>(arg));
        }
        shared_ptr<Block> tail;
        if (guard) {
            // The body runs for an index short of the end; the guard's other
            // arm ends the iteration, and is what the last chunk's steps past
            // the range take.
            auto in_range = step->make_instruction(
                Bool_t::make(), Instruction::Op::Lt, {absolute, v_end});
            tail = std::make_shared<Block>();
            tail->name = parfor.body.name + "_tail_" + inner;
            tail->owner = f;
            tail->terminator.data = Terminator::Yield{};
            step->terminator.data = Terminator::Dispatch{
                in_range,
                {Terminator::Jump{tail->name},
                 Terminator::Jump{parfor.body.name, std::move(to_body)}}};
        } else {
            step->terminator.data =
                Terminator::Jump{parfor.body.name, std::move(to_body)};
        }

        shared_ptr<Block> inner_loop = std::make_shared<Block>();
        inner_loop->name = parfor.body.name + "_split_" + outer;
        inner_loop->owner = f;
        inner_loop->add_argument(outer_arg);
        for (const Argument &arg : carried) {
            inner_loop->add_argument(arg);
        }
        if (end_arg) {
            inner_loop->add_argument(*end_arg);
        }
        // A chunk the program sizes reaches the inner loop as an argument
        // too; a constant is written where it is used.
        shared_ptr<Value> inner_factor = split_factor;
        if (factor_arg) {
            inner_factor = inner_loop->add_argument(*factor_arg);
        }

        std::vector<shared_ptr<Value>> to_step = {v_outer_arg};
        for (const Argument &arg : carried) {
            to_step.push_back(std::make_shared<Value>(arg));
        }
        if (end_arg) {
            to_step.push_back(std::make_shared<Value>(*end_arg));
        }
        inner_loop->terminator.data =
            Terminator::ParFor{inner,
                               zero,
                               inner_factor,
                               parfor.stride,
                               Terminator::Jump{step->name, std::move(to_step)},
                               Terminator::Jump{outer_yield->name}};

        blocks.push_back(inner_loop);
        blocks.push_back(step);
        if (tail) {
            blocks.push_back(tail);
        }
        blocks.push_back(outer_yield);

        std::vector<shared_ptr<Value>> to_inner = parfor.body.args;
        if (end_arg) {
            to_inner.push_back(parfor.end);
        }
        if (factor_arg) {
            to_inner.push_back(split_factor);
        }
        Terminator::ParFor outer_loop{
            outer,
            parfor.start,
            parfor.end,
            split_factor,
            Terminator::Jump{inner_loop->name, std::move(to_inner)},
            parfor.cont};
        // A drain's launch values (Terminator::ParFor::capacity: the queue's
        // size, its header and the slot drained) go to the outer loop, which
        // is the loop a GPU bind launches; the bind then sets the guard on
        // the step above, where the global index is formed (Bind.cpp).
        outer_loop.capacity = parfor.capacity;
        outer_loop.queue_base = parfor.queue_base;
        outer_loop.queue_slot = parfor.queue_slot;
        block->terminator.data = std::move(outer_loop);
    }
    if (blocks.size() == f->blocks.size()) {
        internal_error << "Did not find loop: " << idx
                       << " in function: " << func;
    }
    f->blocks = std::move(blocks);
    // The new blocks' predecessors, and the body's, which now has the step
    // rather than the loop before it.
    refresh_preds(*f);
}

} // namespace

void split(FuncMap &funcs, string func, string idx, int factor, string outer,
           string inner, bool exact) {
    split_impl(funcs, std::move(func), std::move(idx), factor, std::nullopt,
               std::move(outer), std::move(inner), exact);
}

void split(FuncMap &funcs, string func, string idx, const Expr &factor,
           string outer, string inner, bool exact) {
    // A constant expression is the constant split, checks and all.
    if (const auto n = get_constant_value<int64_t>(factor)) {
        internal_assert(*n > 0 && *n <= std::numeric_limits<int>::max())
            << "split(" << idx << ", " << factor << ") on " << func
            << " needs a positive chunk";
        split_impl(funcs, std::move(func), std::move(idx), int(*n), std::nullopt,
                   std::move(outer), std::move(inner), exact);
        return;
    }
    split_impl(funcs, std::move(func), std::move(idx), std::nullopt, factor,
               std::move(outer), std::move(inner), exact);
}

namespace {

// The functions that call themselves, among those `start` can reach.
//
// The schedule names the function the programmer wrote, but the recursion is
// not always in it. A tree query is lowered into a traversal function of its
// own before any of this runs, so `trace.loopify(64)` names a function whose
// body is a call to the traversal, and it is the traversal that recurses.
//
// Nor is it always in a function the walk reaches by calls: a vectorize()
// written earlier in the schedule has copied functions for the gang that
// calls them (Function::specialized_from), and the gang's copies call each
// other, not the originals -- `hit_z_all`'s gang calls the gang's copy of the
// traversal, which the scalar `trace` never mentions. Every copy of a
// function the walk reaches is as much what `trace.loopify(64)` names as the
// function itself, so the walk takes them in as it goes.
std::set<std::string> recursive_functions_from(const FuncMap &funcs,
                                               const std::string &start) {
    std::map<std::string, std::vector<std::string>> copies_of;
    for (const auto &[name, f] : funcs) {
        if (!f->specialized_from.empty()) {
            copies_of[f->specialized_from].push_back(name);
        }
    }
    std::set<std::string> found;
    std::set<std::string> seen;
    std::vector<std::string> work{start};
    while (!work.empty()) {
        const std::string name = work.back();
        work.pop_back();
        if (!seen.insert(name).second) {
            continue;
        }
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue; // an extern, or something not compiled here
        }
        if (is_recursive(*it->second)) {
            found.insert(name);
        }
        for (const auto &block : it->second->blocks) {
            if (const auto *call = block->terminator.callee()) {
                work.push_back(call->name);
            }
        }
        if (const auto copies = copies_of.find(name);
            copies != copies_of.end()) {
            for (const std::string &copy : copies->second) {
                work.push_back(copy);
            }
        }
    }
    return found;
}

} // namespace

namespace {

// Whether `from` calls `to`, directly or through what it calls.
bool calls_through(const FuncMap &funcs, const string &from, const string &to) {
    set<string> seen;
    vector<string> work{from};
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (!seen.insert(name).second) {
            continue;
        }
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue;
        }
        for (const auto &block : it->second->blocks) {
            if (const auto *call = block->terminator.callee()) {
                if (call->name == to) {
                    return true;
                }
                work.push_back(call->name);
            }
        }
    }
    return false;
}

size_t inline_counter = 0;

} // namespace

void inline_call(FuncMap &funcs, const shared_ptr<Function> &caller,
                 const shared_ptr<Block> &site) {
    const auto *at = std::get_if<Terminator::Call>(&site->terminator.data);
    internal_assert(at != nullptr) << "inline_call of " << site->name
                                   << ", which does not end in a call";
    const Terminator::Call call = *at;
    const auto callee_it = funcs.find(call.call.name);
    internal_assert(callee_it != funcs.end())
        << "inline_call of `" << call.call.name << "`, which has no SSA form";
    const Function &callee = *callee_it->second;
    internal_assert(!callee.blocks.empty());
    const size_t n = inline_counter++;
    const string suffix = "!inl" + std::to_string(n);

    // The callee's blocks, renamed, as blocks of the caller.
    map<string, shared_ptr<Block>> copies =
        clone_region(*caller, callee.blocks, suffix);
    // And every name the callee's blocks take as an argument -- its
    // parameters first of all -- renamed too: the copy lives beside the
    // caller's own blocks, where a value is threaded by its name, and a
    // callee whose parameter is called `acc` like the caller's would have
    // its `acc` taken for the caller's. The copy is the whole callee, so a
    // name an argument carries is defined by an argument of the copy.
    //
    // With one exception, for the same reason. A parameter the call binds
    // to a value the caller passes by name -- its `out` handed to the
    // callee's `out`, or to a `buffer` -- is that value: the callee threads
    // it on under its own name, and the copy has to thread it on under the
    // caller's, since the name is what says two references are one
    // definition (the divergence analysis compares definitions by it,
    // SSA/AnalyzeDivergence.cpp value_key; the loop folding asks whether an
    // exit hands a header argument back by it, SSA/UniformizeLoops.cpp).
    // Under a fresh name the same array comes round the back edge as a
    // different value, and a loop that carries it is read as varying it. A
    // caller's name that two parameters would take stays with neither.
    map<string, string> renamed_to;
    {
        set<string> defined;
        for (const auto &block : callee.blocks) {
            for (const Argument &arg : copies.at(block->name)->args) {
                defined.insert(arg.name);
            }
        }
        {
            const vector<Argument> &params = callee.blocks.front()->args;
            map<string, size_t> takers;
            for (size_t i = 0; i < params.size() && i < call.call.args.size(); i++) {
                if (const auto *a = std::get_if<Argument>(&call.call.args[i]->data)) {
                    takers[a->name]++;
                }
            }
            for (size_t i = 0; i < params.size() && i < call.call.args.size(); i++) {
                const auto *a = std::get_if<Argument>(&call.call.args[i]->data);
                if (a != nullptr && takers.at(a->name) == 1) {
                    renamed_to[params[i].name] = a->name;
                }
            }
        }
        const auto new_name = [&](const string &old) {
            const auto alias = renamed_to.find(old);
            return alias != renamed_to.end() ? alias->second : old + suffix;
        };
        set<const Value *> renamed_values;
        const auto rename_value = [&](const shared_ptr<Value> &v) {
            if (!v || !renamed_values.insert(v.get()).second) {
                return;
            }
            if (auto *a = std::get_if<Argument>(&v->data);
                a != nullptr && defined.contains(a->name)) {
                a->name = new_name(a->name);
            }
        };
        const auto rename_jump = [&](Terminator::Jump &j) {
            for (auto &v : j.args) {
                rename_value(v);
            }
        };
        for (const auto &block : callee.blocks) {
            auto copy = copies.at(block->name);
            for (Argument &arg : copy->args) {
                arg.name = new_name(arg.name);
            }
            for (auto &instr : copy->instrs) {
                for (auto &v : instr->operands) {
                    rename_value(v);
                }
            }
            std::visit(
                overloads{
                    [&](std::monostate &) {},
                    [&](Terminator::Jump &j) { rename_jump(j); },
                    [&](Terminator::Dispatch &d) {
                        rename_value(d.cond);
                        for (auto &t : d.targets) {
                            rename_jump(t);
                        }
                    },
                    [&](Terminator::Return &r) { rename_value(r.value); },
                    [&](Terminator::ParFor &p) {
                        rename_value(p.start);
                        rename_value(p.end);
                        rename_value(p.stride);
                        rename_jump(p.body);
                        rename_jump(p.cont);
                    },
                    [&](Terminator::Yield &) {},
                    [&](Terminator::Call &c) {
                        rename_jump(c.call);
                        rename_jump(c.cont);
                    },
                    [&](Terminator::MultiCall &c) {
                        rename_jump(c.call);
                        rename_jump(c.cont);
                        for (auto &row : c.varying) {
                            for (auto &v : row) {
                                rename_value(v);
                            }
                        }
                        for (auto &v : c.keys) {
                            rename_value(v);
                        }
                        for (auto &v : c.conds) {
                            rename_value(v);
                        }
                    },
                },
                copy->terminator.data);
            map<string, shared_ptr<Value>> lookups;
            for (auto &[name, v] : copy->lookups) {
                rename_value(v);
                lookups[defined.contains(name) ? new_name(name) : name] = v;
            }
            copy->lookups = std::move(lookups);
        }
    }
    for (const auto &block : callee.blocks) {
        auto copy = copies.at(block->name);
        copy->owner = caller;
        caller->blocks.push_back(copy);
    }
    BlockMap bmap = make_block_map(*caller);
    for (const auto &block : callee.blocks) {
        auto copy = copies.at(block->name);
        for (const string &succ : successors(*copy)) {
            const auto target = bmap.find(succ);
            internal_assert(target != bmap.end())
                << "the copy of " << block->name << " jumps to " << succ
                << ", which " << caller->blocks.front()->name << " has no block of";
            target->second->preds.push_back(copy);
        }
    }

    // The continuation's values, aliased under fresh names in the site block,
    // so that threading them through the copy to each return meets no name
    // the callee used for something else (a block threads a value by name).
    vector<pair<string, shared_ptr<Value>>> threaded;
    for (size_t i = 0; i < call.cont.args.size(); i++) {
        const shared_ptr<Value> &v = call.cont.args[i];
        if (std::holds_alternative<Constant>(v->data)) {
            threaded.emplace_back("", v);
            continue;
        }
        const string fresh =
            "_inl" + std::to_string(n) + "_" + std::to_string(i);
        site->lookups[fresh] = v;
        threaded.emplace_back(fresh, v);
    }

    // The site jumps into the copy with the call's arguments, and leaves the
    // continuation, which the copy's returns reach instead.
    const shared_ptr<Block> entry = copies.at(callee.blocks.front()->name);
    site->terminator.data = Terminator::Jump{entry->name, call.call.args};
    entry->preds.push_back(site);
    const auto cont = bmap.find(call.cont.name);
    internal_assert(cont != bmap.end())
        << "the call in " << site->name << " continues at " << call.cont.name
        << ", which is no block of " << caller->blocks.front()->name;
    std::erase_if(cont->second->preds, [&](const auto &p) {
        const auto ptr = p.lock();
        return ptr && ptr.get() == site.get();
    });
    for (const auto &block : callee.blocks) {
        auto copy = copies.at(block->name);
        const auto *ret = std::get_if<Terminator::Return>(&copy->terminator.data);
        if (ret == nullptr) {
            continue;
        }
        vector<shared_ptr<Value>> args;
        if (!call.drop) {
            internal_assert(ret->value != nullptr)
                << "the call in " << site->name << " keeps a value, and `"
                << call.call.name << "` returns none";
            args.push_back(ret->value);
        }
        for (const auto &[fresh, v] : threaded) {
            args.push_back(fresh.empty() ? v
                                         : copy->get_value(fresh, v->get_type()));
        }
        copy->terminator.data = Terminator::Jump{call.cont.name, std::move(args)};
        cont->second->preds.push_back(copy);
    }
}

void loopify(FuncMap &funcs, std::string func, int size) {
    internal_assert(funcs.contains(func))
        << "loopify applied to unknown func:" << func;
    auto f = funcs[func];

    if (size > 0) {
        // A recursion that branches cannot become a loop by itself, since the
        // second call has to happen after the first one comes back. It is put
        // on an explicit stack instead (see SSA/QueueRecursion.h).
        const std::set<std::string> targets =
            recursive_functions_from(funcs, func);
        internal_assert(!targets.empty())
            << "loopify(" << size << ") on " << func << ": neither it nor "
            << "anything it calls is recursive, so there is nothing to put on "
            << "a stack";
        for (const std::string &target : targets) {
            queue_recursion(*funcs[target], size_t(size));

            // The target is a loop now, and it is only a function at all
            // because Lower/RecLoops.cpp had to extract the recursion into
            // one. Left standing it costs more than the call: what the
            // traversal folds into is the caller's, so every update to it
            // becomes a store through a pointer and reading the answer back
            // becomes a load per field. Putting it back where it came from is
            // what lets that stay in registers, and the recursion that stopped
            // it from happening is exactly what was just removed.
            auto &attrs = funcs[target]->attributes;
            if (std::find(attrs.begin(), attrs.end(),
                          ir::Function::Attribute::always_inlined) ==
                attrs.end()) {
                attrs.push_back(ir::Function::Attribute::always_inlined);
            }
        }
        return;
    }

    // A recursion that runs through other functions is made direct first:
    // a callee on a path back here, unless the schedule holds it as a
    // function of its own (a deferral's or a stage's callee, `[[noinline]]`),
    // is copied in at its call, and again for what the copy calls, until
    // every path back is a call of this function. Bounded: a cycle that does
    // not run through this function would be copied without end.
    const auto marked = [](const Function &fn, ir::Function::Attribute a) {
        return std::find(fn.attributes.begin(), fn.attributes.end(), a) !=
               fn.attributes.end();
    };
    for (size_t round = 0;; round++) {
        shared_ptr<Block> site;
        for (auto &block : f->blocks) {
            const auto *call =
                std::get_if<Terminator::Call>(&block->terminator.data);
            if (call == nullptr || call->call.name == func) {
                continue;
            }
            const auto g = funcs.find(call->call.name);
            if (g == funcs.end() ||
                marked(*g->second, ir::Function::Attribute::held) ||
                marked(*g->second, ir::Function::Attribute::noinline) ||
                !calls_through(funcs, call->call.name, func)) {
                continue;
            }
            site = block;
            break;
        }
        if (!site) {
            break;
        }
        internal_assert(round < 256)
            << "loopify(" << func << "): the recursion through `"
            << std::get<Terminator::Call>(site->terminator.data).call.name
            << "` does not come back through " << func
            << " -- a cycle among its callees that copying in never ends.";
        inline_call(funcs, f, site);
    }

    // Find any tail calls with empty return continuations and convert them into
    // jumps.

    BlockMap bmap = make_block_map(f);

    // The blocks whose tail call becomes a back edge, i.e. the latches of the
    // loop this leaves behind.
    std::set<std::string> latches;

    for (auto &block : f->blocks) {
        if (!std::holds_alternative<Terminator::Call>(block->terminator.data)) {
            continue;
        }

        Terminator::Call call =
            std::get<Terminator::Call>(block->terminator.data);
        if (call.call.name != func) {
            // Not a tail call.
            continue;
        }

        internal_assert(bmap.contains(call.cont.name))
            << "BlockMap for " << func
            << " does not contain continutation target: " << call.cont.name;

        // The call has to be the last thing the function does, so that going
        // round the loop again is the same as making it. That means a
        // continuation which does nothing but return -- either returning
        // nothing, or returning exactly what the call produced, which is
        // `return f(...)` and is just as much a tail call -- reached either
        // at once or through blocks that only forward it (what a copied-in
        // callee's return became: a jump to its caller's continuation,
        // carrying the value and nothing else the way).
        shared_ptr<Block> tail = bmap.at(call.cont.name);
        std::optional<string> value_name;
        if (!call.drop) {
            if (tail->args.empty()) {
                internal_error << "Cannot loopify the call to " << func << " in "
                               << block->name << ": its value is kept, and its "
                               << "continuation " << call.cont.name
                               << " takes no argument for it";
            }
            value_name = tail->args[0].name;
        }
        for (int hops = 0; hops < 64 && tail->instrs.empty(); hops++) {
            const auto *jump = std::get_if<Terminator::Jump>(&tail->terminator.data);
            if (jump == nullptr) {
                break;
            }
            const auto next = bmap.find(jump->name);
            internal_assert(next != bmap.end());
            std::optional<string> forwarded;
            if (value_name.has_value()) {
                for (size_t i = 0; i < jump->args.size() && i < next->second->args.size(); i++) {
                    const auto *a = std::get_if<Argument>(&jump->args[i]->data);
                    if (a != nullptr && a->name == *value_name) {
                        forwarded = next->second->args[i].name;
                    }
                }
                if (!forwarded.has_value()) {
                    break;
                }
            }
            tail = next->second;
            value_name = forwarded;
        }
        const auto *returns =
            std::get_if<Terminator::Return>(&tail->terminator.data);
        bool is_tail = tail->instrs.empty() && returns != nullptr;
        if (is_tail && !call.drop) {
            // A call whose result is kept hands it to the continuation as a
            // leading argument; returning that argument, and nothing else, is
            // what makes this a tail call rather than a use of the result.
            is_tail = value_name.has_value() && returns->value != nullptr &&
                      std::holds_alternative<Argument>(returns->value->data) &&
                      std::get<Argument>(returns->value->data).name ==
                          *value_name;
        }
        internal_assert(is_tail)
            << "Cannot loopify the call to " << func << " in " << block->name
            << ": its continuation " << call.cont.name << " does more than "
            << "return, so the call is not in tail position";
        const auto cont = bmap.at(call.cont.name);

        // Replace call terminator with direct jump.
        block->terminator.data = call.call;
        latches.insert(block->name);

        // Remove block as predecessor to continuation block.
        std::erase_if(cont->preds, [&](const auto &p) {
            const auto ptr = p.lock();
            internal_assert(ptr);
            return ptr->name == block->name;
        });

        // Add block as predecessor to entry block.
        f->blocks[0]->preds.push_back(block);
    }

    if (latches.empty()) {
        return;
    }

    // The back edges close on the entry block, which makes the entry the loop
    // header -- and then the values carried around the loop are the function's
    // own parameters, reassigned on every iteration. A parameter is not
    // storage, so nothing can be assigned to it; the loop needs a header of
    // its own, entered from a preheader (see SSA/InsertPreheader.h).
    insert_preheader(*f, f->blocks[0]->name, latches);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
