#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/InsertPreheader.h"
#include "SSA/QueueRecursion.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <limits>

#include <functional>
#include <iostream>
#include <optional>

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

void split(FuncMap &funcs, string func, string idx, int factor, string outer,
           string inner, bool exact) {
    internal_assert(funcs.contains(func))
        << "split applied to unknown func:" << func;
    auto f = funcs[func];

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
        internal_assert(*stride_n > 0 && factor % *stride_n == 0)
            << "split(" << idx << ", " << factor << ") on " << func
            << " does not divide the loop's stride of " << *stride_n
            << ", so a chunk would start part way through a step";

        // Whether the range is a whole number of chunks, known when both its
        // ends are constants. Without a tail, a range known only at run time
        // is the caller's assertion to make -- that is what asking for no
        // tail means -- and one known here is checked.
        const auto start_n = as_int(parfor.start);
        const auto end_n = as_int(parfor.end);
        const bool divisible = start_n.has_value() && end_n.has_value() &&
                               (*end_n - *start_n) % factor == 0;
        if (exact && start_n.has_value() && end_n.has_value()) {
            internal_assert(divisible)
                << "split(" << idx << ", " << factor << ") on " << func
                << " does not divide the loop's range of [" << *start_n << ":"
                << *end_n << "), so without a tail it would run "
                << (factor - (*end_n - *start_n) % factor)
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
        auto split_factor = index_constant(factor);
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
                               split_factor,
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
        block->terminator.data = Terminator::ParFor{
            outer,
            parfor.start,
            parfor.end,
            split_factor,
            Terminator::Jump{inner_loop->name, std::move(to_inner)},
            parfor.cont};
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

        internal_assert(call.cont.args.empty())
            << "Cannot loopify tail-call in: " << func
            << ", has continuation arguments to: " << call.cont.name;

        internal_assert(bmap.contains(call.cont.name))
            << "BlockMap for " << func
            << " does not contain continutation target: " << call.cont.name;

        const auto cont = bmap.at(call.cont.name);

        // The call has to be the last thing the function does, so that going
        // round the loop again is the same as making it. That means a
        // continuation which does nothing but return -- either returning
        // nothing, or returning exactly what the call produced, which is
        // `return f(...)` and is just as much a tail call.
        const auto *returns =
            std::get_if<Terminator::Return>(&cont->terminator.data);
        bool is_tail = cont->instrs.empty() && returns != nullptr;
        if (is_tail && !call.drop) {
            // A call whose result is kept hands it to the continuation as a
            // leading argument; returning that argument, and nothing else, is
            // what makes this a tail call rather than a use of the result.
            is_tail = cont->args.size() == 1 && returns->value != nullptr &&
                      std::holds_alternative<Argument>(returns->value->data) &&
                      std::get<Argument>(returns->value->data).name ==
                          cont->args[0].name;
        }
        internal_assert(is_tail)
            << "Cannot loopify the call to " << func << " in " << block->name
            << ": its continuation " << call.cont.name << " does more than "
            << "return, so the call is not in tail position";

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
