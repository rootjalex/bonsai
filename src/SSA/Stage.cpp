#include "SSA/Stage.h"

#include "Error.h"
#include "IR/Printer.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/Defer.h"
#include "SSA/Definitions.h"
#include "SSA/Reach.h"
#include "SSA/Storage.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

// Every place a block holds a value: its instructions' operands and its
// terminator's, whatever kind it is (as SSA/Specialize.cpp walks them).
template <typename F>
void for_each_value(Block &block, const F &fn) {
    const auto all = [&](vector<shared_ptr<Value>> &vs) {
        for (auto &v : vs) {
            fn(v);
        }
    };
    for (auto &instr : block.instrs) {
        all(instr->operands);
    }
    std::visit(overloads{
                   [](std::monostate &) {},
                   [&](Terminator::Jump &j) { all(j.args); },
                   [&](Terminator::Dispatch &d) {
                       fn(d.cond);
                       for (auto &t : d.targets) {
                           all(t.args);
                       }
                   },
                   [&](Terminator::Return &r) { fn(r.value); },
                   [&](Terminator::ParFor &p) {
                       fn(p.start);
                       fn(p.end);
                       fn(p.stride);
                       all(p.body.args);
                       all(p.cont.args);
                   },
                   [](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       all(c.call.args);
                       all(c.cont.args);
                   },
                   [&](Terminator::MultiCall &c) {
                       all(c.call.args);
                       all(c.cont.args);
                       for (auto &vs : c.varying) {
                           all(vs);
                       }
                       all(c.keys);
                       all(c.conds);
                   },
               },
               block.terminator.data);
}

// Whether `call` is in tail position: its continuation returns its value and
// nothing else (the test SSA/Defer.cpp makes).
bool in_tail_position(const Terminator::Call &call, const BlockMap &bmap) {
    const auto it = bmap.find(call.cont.name);
    internal_assert(it != bmap.end()) << call.cont.name;
    const Block &cont = *it->second;
    const auto *returns = std::get_if<Terminator::Return>(&cont.terminator.data);
    if (returns == nullptr || !cont.instrs.empty() || !call.cont.args.empty()) {
        return false;
    }
    if (call.drop) {
        return cont.args.empty() && returns->value == nullptr;
    }
    return cont.args.size() == 1 && returns->value != nullptr &&
           std::holds_alternative<Argument>(returns->value->data) &&
           std::get<Argument>(returns->value->data).name == cont.args[0].name;
}

// The one call to `callee` in `F`, or null when there is none -- or, when
// `refuse_many`, an error for several (a run of them is a branching
// recursion); null for several otherwise.
shared_ptr<Block> one_call_site(const Function &F, const string &callee_name,
                                const string &what, bool refuse_many = true) {
    vector<shared_ptr<Block>> sites;
    for (const auto &block : F.blocks) {
        if (const auto *multi =
                std::get_if<Terminator::MultiCall>(&block->terminator.data)) {
            if (multi->call.name == callee_name && !refuse_many) {
                return nullptr;
            }
            internal_assert(multi->call.name != callee_name)
                << what << ": the calls to " << callee_name << " in "
                << block->name << " are a run of " << multi->varying.size()
                << ", a branching recursion; a boundary is one point after "
                << "one call.";
            continue;
        }
        const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call != nullptr && call->call.name == callee_name) {
            sites.push_back(block);
        }
    }
    if (sites.empty() || (sites.size() > 1 && !refuse_many)) {
        return nullptr;
    }
    internal_assert(sites.size() == 1)
        << what << ": " << F.blocks.front()->name << " makes " << sites.size()
        << " calls to " << callee_name << "; a boundary is one point after one "
        << "call.";
    return sites.front();
}

// Whether `from` calls `to`, directly or through other functions.
bool reaches(const FuncMap &funcs, const string &from, const string &to) {
    set<string> seen;
    vector<string> pending = {from};
    while (!pending.empty()) {
        const string f = pending.back();
        pending.pop_back();
        if (!seen.insert(f).second || !funcs.contains(f)) {
            continue;
        }
        for (const auto &block : funcs.at(f)->blocks) {
            const auto *callee = block->terminator.callee();
            if (callee == nullptr) {
                continue;
            }
            if (callee->name == to) {
                return true;
            }
            pending.push_back(callee->name);
        }
    }
    return false;
}

// `func` factored at its one call to `callee`: the rest of the function as
// a function of its own, `func!after`, and what the site has to hand it.
struct Split {
    shared_ptr<Function> F;         // the function, its site still calling
    shared_ptr<Block> site;         // the block whose call it is
    shared_ptr<Block> cont;         // the call's continuation, now the rest's entry
    shared_ptr<Function> after;     // the rest, `func!after`
    string after_name;
    // What the rest takes after the call's own continuation arguments: the
    // values of `func` live after the call, as the site can pass them.
    vector<shared_ptr<Value>> live;
};

// Factors `func` at its one call to `callee`, as LLVM's coroutine splitting
// factors a function at a suspend point: the blocks from the call's
// continuation on become `func!after`, whose parameters are the
// continuation's arguments (the call's value first, when it has one) and
// then the names those blocks use but do not define -- the values of `func`
// live after the call. `func` itself is left with the call; the caller
// finishes the site as it wants the rest reached.
Split split_at_call(FuncMap &funcs, const string &func_name,
                    const string &callee_name, const string &what) {
    internal_assert(funcs.contains(func_name))
        << what << ": " << func_name << " is not a function of the program";
    internal_assert(funcs.contains(callee_name))
        << what << ": " << callee_name << " is not a function of the program";
    const shared_ptr<Function> F = funcs.at(func_name);

    // The one call to the callee, not in tail position.
    const shared_ptr<Block> site = one_call_site(*F, callee_name, what);
    internal_assert(site)
        << what << ": " << func_name << " makes no call to " << callee_name;
    const BlockMap bmap = make_block_map(*F);
    Terminator::Call &call = std::get<Terminator::Call>(site->terminator.data);
    internal_assert(!in_tail_position(call, bmap))
        << what << ": the call to " << callee_name << " in " << site->name
        << " is in tail position -- nothing of " << func_name
        << " follows it. Queue the call itself: " << func_name
        << ".defer(" << callee_name << ", <queue>).";

    // The rest of the function: the blocks from the call's continuation on.
    const shared_ptr<Block> cont = bmap.at(call.cont.name);
    const Cfg rest_cfg(*F, cont->name);
    const vector<shared_ptr<Block>> rest = rest_cfg.blocks();
    set<const Block *> in_rest;
    for (const auto &block : rest) {
        in_rest.insert(block.get());
    }

    // What the rest uses but does not define -- the values of `func` live
    // after the call. Names: an argument the rest names that no block of it
    // declares and no instruction of it defines (a value from before the
    // call, named where it dominates). Instructions: one referred to by
    // pointer whose block is not in the rest. Each becomes a parameter of the
    // function the rest becomes, under its own name.
    set<string> defined;
    for (const auto &block : rest) {
        for (const Argument &a : block->args) {
            defined.insert(a.name);
        }
        for (const auto &instr : block->instrs) {
            if (!instr->name.empty()) {
                defined.insert(instr->name);
            }
        }
    }
    vector<Argument> free_names;
    set<string> seen_free;
    vector<shared_ptr<Instruction>> free_instrs;
    set<const Instruction *> seen_instrs;
    for (const auto &block : rest) {
        for_each_value(*block, [&](shared_ptr<Value> &v) {
            if (!v) {
                return;
            }
            if (const auto *a = std::get_if<Argument>(&v->data)) {
                if (!defined.count(a->name) && seen_free.insert(a->name).second) {
                    free_names.push_back(*a);
                }
            } else if (const auto *ip = std::get_if<shared_ptr<Instruction>>(&v->data)) {
                const auto owner = (*ip)->owner.lock();
                if ((!owner || !in_rest.count(owner.get())) &&
                    seen_instrs.insert(ip->get()).second) {
                    free_instrs.push_back(*ip);
                }
            }
        });
    }
    for (const auto &instr : free_instrs) {
        internal_assert(!instr->name.empty())
            << what << ": a nameless instruction of " << func_name
            << " is live after the call to " << callee_name;
    }

    // The rest as a function of its own: the continuation block its entry,
    // taking what the call's continuation took and then the live values.
    const string after_name = func_name + "!after";
    internal_assert(!funcs.contains(after_name))
        << what << ": " << after_name << " exists already; " << func_name
        << " is staged twice";
    auto A = std::make_shared<Function>();
    A->ret_type = F->ret_type;
    for (const ir::Function::Attribute attr : F->attributes) {
        if (attr != ir::Function::Attribute::exported) {
            A->attributes.push_back(attr);
        }
    }
    // Held as a function -- it is what a drain runs -- until that drain
    // folds it in (SSA/Defer.cpp).
    if (std::find(A->attributes.begin(), A->attributes.end(),
                  ir::Function::Attribute::held) == A->attributes.end()) {
        A->attributes.push_back(ir::Function::Attribute::held);
    }
    // The rest is a function of its own, so its values keep the program's
    // names for them: a queue's specialize names one (`hits.specialize(
    // material)`).
    const auto copies = clone_region(*F, rest, "!after", /*keep_names=*/true);
    // The entry first, then the others, each now the new function's. A
    // function's entry block carries its name, so the continuation's copy is
    // renamed and whatever jumped to it inside the rest follows.
    const string entry_was = copies.at(cont->name)->name;
    copies.at(cont->name)->name = after_name;
    A->blocks.push_back(copies.at(cont->name));
    for (const auto &block : rest) {
        if (block != cont) {
            A->blocks.push_back(copies.at(block->name));
        }
    }
    for (const auto &block : A->blocks) {
        for (Terminator::Jump *jump : jumps_of(*block)) {
            if (jump->name == entry_was) {
                jump->name = after_name;
            }
        }
    }
    for (const auto &block : A->blocks) {
        block->owner = A;
        for (const auto &instr : block->instrs) {
            instr->owner = block;
            A->reserve_name(instr->name);
        }
        for (const Argument &a : block->args) {
            A->reserve_name(a.name);
        }
    }
    Block &entry = *A->blocks.front();
    for (const Argument &a : free_names) {
        entry.args.push_back(a);
        entry.lookups[a.name] = std::make_shared<Value>(a);
        A->reserve_name(a.name);
    }
    for (const auto &instr : free_instrs) {
        Argument a{instr->type, instr->name};
        entry.args.push_back(a);
        entry.lookups[a.name] = std::make_shared<Value>(a);
        A->reserve_name(a.name);
    }
    // A parameter of the rest that is an address is as writable as what the
    // function hands it. The continuation's arguments say nothing about
    // that -- Argument::mutating means something on a function's entry
    // alone, and a value threaded through a block's arguments loses it on
    // the way -- and a backend takes a pointer parameter without it to be
    // read-only, so that a store through it in the rest would be undefined:
    // the caller then carried what it had stored before the call past the
    // rest's own store (the renderer's visible surface, set by the staged
    // rest and returned from in the same drain, came back unset). So each
    // is looked up where the call's continuation gets it: a parameter of the
    // function, `mut` or not as the program declared it, or an address the
    // function computed -- one of its own locals, a field of one -- which
    // the rest may store through as the function could.
    {
        Definitions defs(*F);
        const Block &fentry = *F->blocks.front();
        const auto is_address = [](const Type &t) {
            return t.is<Ptr_t>() || t.is_reference();
        };
        const auto as_handed = [&](Argument &a, const Argument *param) {
            if (!is_address(a.type)) {
                return;
            }
            a.mutating = param == nullptr || param->mutating;
            a.unaliased = param != nullptr && param->unaliased;
            a.reducer = param != nullptr && param->reducer;
        };
        const size_t result_args = call.drop ? 0 : 1;
        internal_assert(cont->args.size() == result_args + call.cont.args.size())
            << what << ": " << cont->name << " takes " << cont->args.size()
            << " arguments and the call passes " << call.cont.args.size()
            << (call.drop ? "" : " plus its result");
        for (size_t i = 0; i < cont->args.size(); i++) {
            // The call's own result is no parameter of the function's.
            const Argument *param =
                i < result_args
                    ? nullptr
                    : defs.parameter(site->name, call.cont.args[i - result_args]);
            as_handed(entry.args[i], param);
        }
        for (size_t i = 0; i < free_names.size(); i++) {
            const Argument *param = nullptr;
            for (const Argument &p : fentry.args) {
                if (p.name == free_names[i].name) {
                    param = &p;
                }
            }
            as_handed(entry.args[cont->args.size() + i], param);
        }
        for (size_t i = 0; i < free_instrs.size(); i++) {
            as_handed(entry.args[cont->args.size() + free_names.size() + i],
                      nullptr);
        }
    }
    // A live instruction is a parameter now: every reference to it by
    // pointer becomes one by name.
    for (const auto &block : A->blocks) {
        for_each_value(*block, [&](shared_ptr<Value> &v) {
            const auto *ip = v ? std::get_if<shared_ptr<Instruction>>(&v->data) : nullptr;
            if (ip == nullptr || !seen_instrs.count(ip->get())) {
                return;
            }
            v = std::make_shared<Value>(Argument{(*ip)->type, (*ip)->name});
        });
    }
    refresh_preds(*A);
    funcs[after_name] = A;

    Split split;
    split.F = F;
    split.site = site;
    split.cont = cont;
    split.after = A;
    split.after_name = after_name;
    for (const Argument &a : free_names) {
        split.live.push_back(std::make_shared<Value>(a));
    }
    for (const auto &instr : free_instrs) {
        split.live.push_back(std::make_shared<Value>(instr));
    }
    return split;
}

// A block that returns what `F` returns, for a tail call's continuation.
shared_ptr<Block> returning_block(const shared_ptr<Function> &F, const string &name,
                                  const string &result_name) {
    auto returned = std::make_shared<Block>();
    returned->name = name;
    returned->owner = F;
    if (F->ret_type.is<Void_t>()) {
        returned->terminator.data = Terminator::Return{nullptr};
    } else {
        Argument result{F->ret_type, result_name};
        returned->args.push_back(result);
        auto value = std::make_shared<Value>(result);
        returned->lookups[result.name] = value;
        returned->terminator.data = Terminator::Return{value};
    }
    return returned;
}

// `func` cut at its call to `callee`, with the rest a tail call: the block
// of `func` that makes it, and the function it calls -- `func!after` for a
// cut after the call (stage_tail), `func!callee!k` for one before it
// (continuation_tail). What defer() queues, and what the cut on the same
// queue stores instead.
struct Tail {
    shared_ptr<Function> F;
    shared_ptr<Block> block;
    string callee;
};

// The cut after the call. In `func`: the call's continuation is a block of
// the call's own that tail-calls the rest with what it was handed and the
// live values, and returns what comes back.
Tail stage_tail(FuncMap &funcs, const string &func_name,
                const string &callee_name, const string &what) {
    Split split = split_at_call(funcs, func_name, callee_name, what);
    const shared_ptr<Function> F = split.F;
    Terminator::Call &call = std::get<Terminator::Call>(split.site->terminator.data);
    auto staged = std::make_shared<Block>();
    staged->name = split.cont->name + "!stage";
    staged->owner = F;
    staged->args = split.cont->args;
    for (const Argument &a : staged->args) {
        staged->lookups[a.name] = std::make_shared<Value>(a);
    }
    vector<shared_ptr<Value>> passed;
    for (const Argument &a : staged->args) {
        passed.push_back(staged->lookups.at(a.name));
    }
    passed.insert(passed.end(), split.live.begin(), split.live.end());
    auto returned = returning_block(F, split.cont->name + "!staged", "$staged");
    Terminator::Call tail;
    tail.call = Terminator::Jump{split.after_name, std::move(passed)};
    tail.cont = Terminator::Jump{returned->name, {}};
    tail.drop = F->ret_type.is<Void_t>();
    staged->terminator.data = std::move(tail);
    call.cont.name = staged->name;
    F->blocks.push_back(staged);
    F->blocks.push_back(returned);
    refresh_preds(*F);
    remove_unreachable_blocks(*F);
    return Tail{F, staged, split.after_name};
}

// Whether every path from `F`'s entry reaches `site`: no return of `F` lies
// on a path that avoids it. What the cut on the same queue requires of the
// call it cuts at, and of each call of the chain down to it, since its
// second pass runs the rest for every entry.
void every_path_reaches(const Function &F, const Block &site, const string &what,
                        const string &callee_name) {
    const BlockMap bmap = make_block_map(F);
    set<string> seen;
    vector<string> work{F.blocks.front()->name};
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (name == site.name || !seen.insert(name).second) {
            continue;
        }
        const Block &block = *bmap.at(name);
        internal_assert(!std::holds_alternative<Terminator::Return>(block.terminator.data))
            << what << ": " << block.name << " of " << F.blocks.front()->name
            << " returns without reaching the call to " << callee_name
            << ", so not every entry of the first pass would reach the cut. "
            << "[unimplemented] the first pass marking the entries that did, "
            << "and the second pass skipping the rest (docs/schedule-semantics.md, 4.5).";
        for (const string &s : successors(block)) {
            work.push_back(s);
        }
    }
}

// The index of `name` among `F`'s parameters, if it is one.
optional<size_t> parameter_index(const Function &F, const string &name) {
    const Block &entry = *F.blocks.front();
    for (size_t i = 0; i < entry.args.size(); i++) {
        if (entry.args[i].name == name) {
            return i;
        }
    }
    return std::nullopt;
}

Tail continuation_tail(FuncMap &funcs, const string &func_name,
                       const string &callee_name, const string &what);

// The cut on the same queue: see SSA/Stage.h. `after_call` picks the side
// of the call the cut is on.
vector<Type> cut_same_queue(FuncMap &funcs, const string &func_name,
                            const string &callee_name, const QueueSpec &queue,
                            const string &loop, bool after_call) {
    const string what = func_name + (after_call ? ".stage(" : ".defer(") +
                        callee_name + ", " + queue.name + ", " + loop + ")";
    internal_assert(funcs.contains(queue.owner))
        << what << ": " << queue.owner << " is not a function of the program";
    internal_assert(funcs.contains(func_name))
        << what << ": " << func_name << " is not a function of the program";
    const shared_ptr<Function> O = funcs.at(queue.owner);
    const auto owned = O->queue_sizes.find(queue.name);
    internal_assert(owned != O->queue_sizes.end())
        << what << ": " << queue.name << " has no drain yet. The cut on the "
        << "same queue splits that drain, so it is written after the "
        << "deferrals that make it (`<f>.defer(<g>, " << queue.name << ")`).";

    // The drain: the one loop named after the queue, not yet bound.
    shared_ptr<Block> pre;
    for (const auto &block : O->blocks) {
        const auto *pf = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (pf == nullptr || pf->index != queue.name) {
            continue;
        }
        internal_assert(!pre) << what << ": " << queue.owner
                              << " has several loops named " << queue.name;
        pre = block;
    }
    internal_assert(pre)
        << what << ": " << queue.owner << " has no loop named " << queue.name
        << ". The drain is split by a `specialize` into loops of its "
        << "variants' names, or a `split` renamed it; the cut on the same "
        << "queue comes before the directives on the loop.";
    Terminator::ParFor &P = std::get<Terminator::ParFor>(pre->terminator.data);
    internal_assert(!P.binding.has_value())
        << what << ": " << queue.name << " is bound already; the cut on the "
        << "same queue comes before the directives on the loop, since it "
        << "splits that loop in two and names the first.";
    for (const auto &block : O->blocks) {
        const auto *pf = std::get_if<Terminator::ParFor>(&block->terminator.data);
        internal_assert(pf == nullptr || pf->index != loop)
            << what << ": " << queue.owner << " has a loop named " << loop
            << " already; the third argument names the first pass, a new loop.";
    }
    const BlockMap omap = make_block_map(O);
    // The round loop's header, when the drain runs in rounds: the block
    // that dispatches into the drain's block on the count.
    string round_header;
    for (const auto &weak : pre->preds) {
        const shared_ptr<Block> p = weak.lock();
        if (p && std::holds_alternative<Terminator::Dispatch>(p->terminator.data)) {
            round_header = p->name;
        }
    }
    const shared_ptr<Block> body = omap.at(P.body.name);
    auto *bcall = std::get_if<Terminator::Call>(&body->terminator.data);
    internal_assert(bcall != nullptr && bcall->drop)
        << what << ": [unimplemented] the body of the drain of " << queue.name
        << " does not end in the deferred call";
    const shared_ptr<Block> ran = omap.at(bcall->cont.name);
    internal_assert(std::holds_alternative<Terminator::Yield>(ran->terminator.data) &&
                    ran->instrs.empty())
        << what << ": [unimplemented] the drain of " << queue.name
        << " does more after the deferred call than end its iteration";
    const string c0 = bcall->call.name;

    // The chain from the drain's callee down to `func`: one call per
    // function, each in tail position, and no call into it from elsewhere
    // -- the first pass ends at the cut, so a caller that ran a chain
    // function whole would stop there.
    vector<string> chain;
    {
        map<string, string> parent{{c0, ""}};
        vector<string> work{c0};
        while (!work.empty() && !parent.contains(func_name)) {
            const string g = work.back();
            work.pop_back();
            if (!funcs.contains(g)) {
                continue;
            }
            for (const string &h : callees_of(*funcs.at(g))) {
                if (!parent.contains(h)) {
                    parent[h] = g;
                    work.push_back(h);
                }
            }
        }
        internal_assert(parent.contains(func_name))
            << what << ": " << func_name << " is not reached from the drain of "
            << queue.name << " (its callee is " << c0 << ")";
        for (string g = func_name; !g.empty(); g = parent.at(g)) {
            chain.insert(chain.begin(), g);
        }
    }
    struct Link {
        string caller;
        shared_ptr<Block> site;
    };
    vector<Link> links;
    for (size_t i = 0; i + 1 < chain.size(); i++) {
        const shared_ptr<Function> G = funcs.at(chain[i]);
        const shared_ptr<Block> site =
            one_call_site(*G, chain[i + 1], what, /*refuse_many=*/false);
        internal_assert(site) << what << ": " << chain[i] << " calls "
                              << chain[i + 1] << " more than once";
        const auto &call = std::get<Terminator::Call>(site->terminator.data);
        internal_assert(in_tail_position(call, make_block_map(*G)))
            << what << ": the call to " << chain[i + 1] << " in " << chain[i]
            << " is not in tail position, so the chain from the drain of "
            << queue.name << " to " << func_name << " does work after it "
            << "that the second pass would not run";
        links.push_back({chain[i], site});
    }
    for (size_t i = 0; i < chain.size(); i++) {
        for (const auto &[name, G] : funcs) {
            for (const auto &block : G->blocks) {
                const Terminator::Jump *j = block->terminator.callee();
                if (j == nullptr || j->name != chain[i]) {
                    continue;
                }
                const bool from_chain = (i == 0 && block == body) ||
                                        (i > 0 && name == chain[i - 1]);
                internal_assert(from_chain)
                    << what << ": " << chain[i] << " is called from " << name
                    << " (" << block->name << "), outside the drain of "
                    << queue.name << ". The first pass ends at the cut, so a "
                    << "call that ran " << chain[i] << " whole would stop there. "
                    << "Defer that call onto " << queue.name << " too"
                    << (name == queue.owner
                            ? " (`" + name + ".defer(" + chain[i] + ", " +
                                  queue.name + ")`: the owner's call is a push)"
                            : "")
                    << ".";
            }
        }
    }
    const shared_ptr<Function> F = funcs.at(func_name);
    const string slot_param = "_slot_" + queue.name;
    internal_assert(parameter_index(*F, slot_param).has_value())
        << what << ": " << func_name << " carries no slot of " << queue.name
        << "'s record. The entries hold none -- nothing of the producer's "
        << "iteration (a reducer, a mutable local, a value its rest uses) is "
        << "handed down the chain -- and the first pass stores what the rest "
        << "needs by that slot. [unimplemented] storing by the entry's index "
        << "instead.";
    for (size_t i = 0; i + 1 < chain.size(); i++) {
        every_path_reaches(*funcs.at(chain[i]), *links[i].site, what, chain[i + 1]);
    }
    every_path_reaches(*F, *one_call_site(*F, callee_name, what), what, callee_name);
    internal_assert(F->ret_type.is<Void_t>())
        << what << ": " << func_name << " returns " << F->ret_type
        << "; a function on a drained chain returns nothing";

    // The cut, with the rest a tail call; its arguments threaded into the
    // block that makes it, so that each can be followed to its definition.
    Tail tail = after_call ? stage_tail(funcs, func_name, callee_name, what)
                           : continuation_tail(funcs, func_name, callee_name, what);
    Terminator::Call &tcall = std::get<Terminator::Call>(tail.block->terminator.data);
    for (auto &v : tcall.call.args) {
        v = reach(tail.block, v);
    }
    const vector<shared_ptr<Value>> passed = tcall.call.args;
    const shared_ptr<Function> R = funcs.at(tail.callee);
    const Block &rentry = *R->blocks.front();
    internal_assert(rentry.args.size() == passed.size())
        << what << ": " << tail.callee << " takes " << rentry.args.size()
        << " arguments and the cut passes " << passed.size();

    // Which of them the drain's own call hands the chain: a parameter of
    // `func` that is, at each call of the chain, a parameter of the caller,
    // back to the drain's call -- the entry's fields, the uniforms, the slot
    // addresses, the queue. The second pass passes the same values. The
    // others -- the call's value, a value computed before the cut -- are
    // stored by the first pass.
    vector<shared_ptr<Value>> handed(passed.size());
    {
        Definitions fdefs(*F);
        for (size_t i = 0; i < passed.size(); i++) {
            const Argument *param = fdefs.parameter(tail.block->name, passed[i]);
            if (param == nullptr) {
                continue;
            }
            optional<size_t> j = parameter_index(*F, param->name);
            for (size_t k = links.size(); j.has_value() && k-- > 0;) {
                const shared_ptr<Function> G = funcs.at(links[k].caller);
                const auto &call = std::get<Terminator::Call>(links[k].site->terminator.data);
                if (*j >= call.call.args.size()) {
                    j.reset();
                    break;
                }
                Definitions gdefs(*G);
                const Argument *gp = gdefs.parameter(links[k].site->name, call.call.args[*j]);
                j = gp == nullptr ? std::nullopt : parameter_index(*G, gp->name);
            }
            if (j.has_value() && *j < bcall->call.args.size()) {
                handed[i] = bcall->call.args[*j];
            }
        }
    }

    // Storage for the rest: a struct of arrays of the record, each stored
    // value flattened to its scalars by the rule a queue's entry is stored
    // by (QueueLayout, SSA/Defer.h) -- one array per scalar, a short vector
    // one array where no gang touches the queue (stored_vector_type), bools
    // eight to a byte, an ADT's padding nowhere -- one slot per producer
    // iteration,
    // made where the record's arrays are and sized as they are; and a
    // pointer to this entry's slot of each array, handed down the chain as
    // the queue is. pbrt's pixelSampleState has the shape: an SOA, an array
    // per member indexed by the pixel. The second pass, and a drain that
    // reloads a value from here (SSA/Defer.cpp), then read the scalars they
    // use and not the whole.
    struct Stored {
        size_t index;
        string name;
        Type type;
        size_t first_leaf = 0, last_leaf = 0; // its run of `leaves`
    };
    vector<Stored> stored;
    vector<Leaf> leaves;
    for (size_t i = 0; i < passed.size(); i++) {
        if (handed[i]) {
            continue;
        }
        const Type type = passed[i]->get_type();
        internal_assert(!type.is<Ptr_t>() && !type.is_reference())
            << what << ": " << rentry.args[i].name << ", an address (" << type
            << ") live after the cut, is not one the drain hands the chain -- "
            << "it points into " << func_name << "'s own frame, which is gone "
            << "when the second pass runs";
        Stored s;
        s.index = i;
        s.name = after_call && i == 0 ? callee_name : rentry.args[i].name;
        s.type = type;
        s.first_leaf = leaves.size();
        leaves_of(s.name, type, queue.adt_storages, /*in_pad=*/false,
                  queue.vector_leaves, {}, leaves);
        s.last_leaf = leaves.size();
        stored.push_back(std::move(s));
    }
    const size_t value_leaves = pack_bools(leaves);
    const shared_ptr<Block> point = omap.at(owned->second.point);
    vector<shared_ptr<Value>> arrays(leaves.size()); // by leaf; none for a pad or a bit
    for (size_t l = 0; l < leaves.size(); l++) {
        if (leaves[l].kind != Leaf::Kind::Stored) {
            continue;
        }
        arrays[l] = make_alloca(*O, point,
                                Array_t::make(leaves[l].type, as_expr(owned->second.size)),
                                queue.name + "_" + leaves[l].name + "_slots");
        // Written before it is read in every round, so made once outside
        // the loops around the owner, as the record's arrays are
        // (Instruction::scratch; SSA/HoistAllocations.h).
        std::get<shared_ptr<Instruction>>(arrays[l]->data)->scratch = true;
        // Written by the first pass alone, so from the second pass to the
        // end of the round a deferral may read it at its entry's slot
        // rather than store what it was handed (Function::OwnedQueue).
        owned->second.stable_arrays.insert(
            std::get<shared_ptr<Instruction>>(arrays[l]->data).get());
    }
    owned->second.second_pass = pre->name;
    owned->second.round_header = round_header;
    // Each chain function's pointer parameters, by leaf; none for a leaf
    // without an array.
    map<string, vector<shared_ptr<Value>>> slot_ptrs;
    for (const string &g : chain) {
        Function &G = *funcs.at(g);
        Block &entry = *G.blocks.front();
        slot_ptrs[g].resize(leaves.size());
        for (size_t l = 0; l < leaves.size(); l++) {
            if (!arrays[l]) {
                continue;
            }
            const string pname = "_" + leaves[l].name + "_" + queue.name;
            internal_assert(!parameter_index(G, pname).has_value())
                << what << ": " << g << " already has a parameter named " << pname;
            slot_ptrs[g][l] = entry.add_argument(
                Argument{Ptr_t::make(leaves[l].type), pname, /*mutating=*/true});
            G.reserve_name(pname);
        }
    }
    for (const Link &l : links) {
        auto &call = std::get<Terminator::Call>(l.site->terminator.data);
        for (const auto &v : slot_ptrs.at(l.caller)) {
            if (v) {
                call.call.args.push_back(reach(l.site, v));
            }
        }
    }
    // At the cut: each stored value taken apart into its scalars, each into
    // its slot, the bools packed into their bytes; and the first half ends.
    {
        const Emit emit = [&](const Type &t, Instruction::Op op,
                              vector<shared_ptr<Value>> ops) {
            return tail.block->make_instruction(t, op, std::move(ops));
        };
        vector<shared_ptr<Value>> parts;
        size_t l = 0;
        for (const Stored &s : stored) {
            internal_assert(l == s.first_leaf);
            take_apart(reach(tail.block, passed[s.index]), leaves, l, /*lanes=*/1, emit, parts);
            internal_assert(l == s.last_leaf)
                << what << ": " << s.name << " came apart into " << (l - s.first_leaf)
                << " scalars where its type has " << (s.last_leaf - s.first_leaf);
        }
        pack_bits(parts, leaves, value_leaves, UInt_t::make(8), /*lanes=*/1,
                  [](size_t) { return true; }, emit);
        for (size_t k = 0; k < leaves.size(); k++) {
            if (!arrays[k] || parts[k] == nullptr) {
                continue;
            }
            tail.block->make_side_effect(
                Instruction::Op::Store,
                {reach(tail.block, slot_ptrs.at(func_name)[k]), parts[k]});
        }
    }
    tail.block->terminator.data = Terminator::Return{};
    refresh_preds(*F);
    remove_unreachable_blocks(*F);

    // The first pass: a copy of the drain's block and body, named `loop`,
    // that the round enters first and that goes on to the drain. The
    // block's side effects -- the reset of the other buffer's count, at
    // the top of the round -- go with the copy, so that they come before
    // both passes and a push made before the cut is kept; the drain's
    // block keeps the values its loop needs.
    vector<shared_ptr<Block>> region{pre};
    const Cfg body_cfg(*O, body->name);
    for (const auto &block : body_cfg.blocks()) {
        region.push_back(block);
    }
    const auto copies = clone_region(*O, region, "!" + loop);
    const shared_ptr<Block> first = copies.at(pre->name);
    std::erase_if(pre->instrs, [](const auto &instr) { return instr->name.empty(); });
    Terminator::ParFor &P1 = std::get<Terminator::ParFor>(first->terminator.data);
    P1.index = loop;
    P1.capacity = P.capacity;
    P1.queue_base = P.queue_base;
    P1.queue_slot = P.queue_slot;
    P1.count_address = P.count_address;
    {
        vector<shared_ptr<Value>> onward;
        for (const Argument &a : first->args) {
            onward.push_back(first->lookups.at(a.name));
        }
        P1.cont = Terminator::Jump{pre->name, std::move(onward)};
    }
    const shared_ptr<Block> body1 = copies.at(body->name);
    // The body's index under the loop's name.
    {
        const string was = body1->args.front().name;
        for (const auto &block : region) {
            if (block == pre) {
                continue;
            }
            const shared_ptr<Block> copy = copies.at(block->name);
            for (Argument &a : copy->args) {
                if (a.name == was) {
                    a.name = loop;
                }
            }
            if (auto node = copy->lookups.extract(was); !node.empty()) {
                node.key() = loop;
                copy->lookups.insert(std::move(node));
            }
            for_each_value(*copy, [&](shared_ptr<Value> &v) {
                if (auto *a = v ? std::get_if<Argument>(&v->data) : nullptr;
                    a != nullptr && a->name == was) {
                    a->name = loop;
                }
            });
        }
    }
    for (const auto &block : region) {
        const shared_ptr<Block> copy = copies.at(block->name);
        copy->owner = O;
        for (const auto &instr : copy->instrs) {
            instr->owner = copy;
        }
        O->blocks.push_back(copy);
    }
    for (const auto &block : O->blocks) {
        if (block == first) {
            continue;
        }
        for (Terminator::Jump *j : jumps_of(*block)) {
            if (j->name == pre->name) {
                j->name = first->name;
            }
        }
    }
    refresh_preds(*O);
    // Its call hands the chain this entry's slot of each array.
    const optional<size_t> slot_at = parameter_index(*funcs.at(c0), slot_param);
    internal_assert(slot_at.has_value()) << what << ": " << c0 << " has no " << slot_param;
    {
        auto &call1 = std::get<Terminator::Call>(body1->terminator.data);
        internal_assert(call1.call.name == c0);
        const shared_ptr<Value> slot = call1.call.args.at(*slot_at);
        for (size_t l = 0; l < leaves.size(); l++) {
            if (arrays[l]) {
                call1.call.args.push_back(body1->make_instruction(
                    Ptr_t::make(leaves[l].type), Instruction::Op::GEP,
                    {reach(body1, arrays[l]), slot}));
            }
        }
    }

    // The second pass: the drain as it was, its body calling the rest with
    // what it handed the chain and what the first pass stored -- each value
    // put back together from its scalars loaded at the slot, a byte of
    // bools read once, and only the scalars the rest reads of it
    // (read_paths, SSA/Analysis.h): a scalar it never reads is left
    // undefined, and its array is written for the deferrals downstream that
    // reload from it (SSA/Defer.cpp).
    {
        const shared_ptr<Value> slot = bcall->call.args.at(*slot_at);
        const auto rest_reads = read_paths(*R, &funcs);
        const Emit emit = [&](const Type &t, Instruction::Op op,
                              vector<shared_ptr<Value>> ops) {
            return body->make_instruction(t, op, std::move(ops));
        };
        const auto load_leaf = [&](size_t l) {
            auto ptr = body->make_instruction(Ptr_t::make(leaves[l].type), Instruction::Op::GEP,
                                              {reach(body, arrays[l]), slot});
            return body->make_instruction(leaves[l].type, Instruction::Op::Load, {ptr});
        };
        map<size_t, shared_ptr<Value>> words;
        vector<shared_ptr<Value>> args;
        for (size_t i = 0; i < passed.size(); i++) {
            if (handed[i]) {
                args.push_back(handed[i]);
                continue;
            }
            const Stored *s = nullptr;
            for (const Stored &c : stored) {
                if (c.index == i) {
                    s = &c;
                }
            }
            internal_assert(s != nullptr);
            const auto rit = rest_reads.find(rentry.args[i].name);
            const auto wanted = [&](const Leaf &leaf) {
                return rit != rest_reads.end() &&
                       std::any_of(rit->second.begin(), rit->second.end(),
                                   [&](const vector<unsigned> &read) {
                                       return covers(read, leaf.path);
                                   });
            };
            size_t l = s->first_leaf;
            const auto stored_whole = [&](const Type &t) {
                return l < s->last_leaf && keeps_whole(leaves[l], t);
            };
            auto value = rebuild(
                s->type, emit,
                [&](const Type &t) -> shared_ptr<Value> {
                    internal_assert(l < s->last_leaf)
                        << what << ": " << s->name
                        << " has more scalars than the record stores for it";
                    const Leaf &leaf = leaves[l++];
                    if (!wanted(leaf)) {
                        return undef_value(t); // the rest never reads it
                    }
                    switch (leaf.kind) {
                    case Leaf::Kind::Pad:
                        return zero_value(t, *O, body);
                    case Leaf::Kind::Bit: {
                        auto w = words.find(leaf.word);
                        if (w == words.end()) {
                            w = words.emplace(leaf.word, load_leaf(leaf.word)).first;
                        }
                        return unpack_bit(w->second, leaf, t, emit);
                    }
                    case Leaf::Kind::Stored:
                        break;
                    }
                    auto loaded = load_leaf(l - 1);
                    return packs(leaf, t)
                               ? body->make_instruction(t, Instruction::Op::Cast, {loaded})
                               : loaded;
                },
                stored_whole);
            internal_assert(l == s->last_leaf)
                << what << ": " << s->name
                << " has fewer scalars than the record stores for it";
            args.push_back(std::move(value));
        }
        bcall->call = Terminator::Jump{tail.callee, std::move(args)};
        bcall->drop = true;
    }
    // The rest is the second pass's body now, folded in as a drained callee
    // is (SSA/Defer.cpp); its hold, which kept it a function for this
    // directive, comes off.
    std::erase(R->attributes, ir::Function::Attribute::held);
    const auto has = [&](ir::Function::Attribute a) {
        return std::find(R->attributes.begin(), R->attributes.end(), a) != R->attributes.end();
    };
    if (!is_recursive(*R) && !has(ir::Function::Attribute::noinline) &&
        !has(ir::Function::Attribute::always_inlined)) {
        R->attributes.push_back(ir::Function::Attribute::always_inlined);
    }
    if (std::getenv("BONSAI_EXPLAIN_DEFER") != nullptr) {
        std::cerr << "; " << what << ": the first pass " << loop << " stores";
        for (const Stored &s : stored) {
            std::cerr << " " << s.name << " : " << s.type;
        }
        std::cerr << (stored.empty() ? " nothing" : " as");
        for (size_t l = 0; l < leaves.size(); l++) {
            if (arrays[l]) {
                std::cerr << " " << queue.name << "_" << leaves[l].name << "_slots";
            }
        }
        std::cerr << "; the second pass " << queue.name << " calls " << tail.callee << "\n";
    }
    return {};
}

} // namespace

vector<Type> stage(FuncMap &funcs, const string &func_name,
                   const string &callee_name, const QueueSpec &queue) {
    const string what =
        func_name + ".stage(" + callee_name + ", " + queue.name + ")";
    // What follows the call is a tail call now, and a queue of those is what
    // defer() builds.
    const Tail tail = stage_tail(funcs, func_name, callee_name, what);
    return defer(funcs, func_name, tail.callee, queue);
}

vector<Type> stage_same_queue(FuncMap &funcs, const string &func_name,
                              const string &callee_name, const QueueSpec &queue,
                              const string &loop) {
    return cut_same_queue(funcs, func_name, callee_name, queue, loop,
                          /*after_call=*/true);
}

vector<Type> defer_same_queue(FuncMap &funcs, const string &func_name,
                              const string &callee_name, const QueueSpec &queue,
                              const string &loop) {
    return cut_same_queue(funcs, func_name, callee_name, queue, loop,
                          /*after_call=*/false);
}

string call_holder(const FuncMap &funcs, const string &func_name,
                   const string &callee_name) {
    string holder = func_name;
    while (funcs.contains(holder)) {
        for (const auto &block : funcs.at(holder)->blocks) {
            if (const auto *call =
                    std::get_if<Terminator::Call>(&block->terminator.data);
                call != nullptr && call->call.name == callee_name) {
                return holder;
            }
            if (const auto *multi =
                    std::get_if<Terminator::MultiCall>(&block->terminator.data);
                multi != nullptr && multi->call.name == callee_name) {
                return holder;
            }
        }
        holder += "!after";
    }
    return func_name;
}

vector<string> call_holders(const FuncMap &funcs, const string &func_name,
                            const string &callee_name, const string &owner) {
    // The functions the owner reaches by calls, the owner among them.
    set<string> reached;
    vector<string> work{owner};
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (!reached.insert(name).second) {
            continue;
        }
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue;
        }
        for (const string &g : callees_of(*it->second)) {
            work.push_back(g);
        }
    }
    const string prefix = func_name + "!";
    vector<string> holders;
    for (const auto &[name, F] : funcs) {
        if (name != func_name && name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        if (!reached.contains(name)) {
            continue;
        }
        bool calls = false;
        for (const auto &block : F->blocks) {
            if (const auto *call =
                    std::get_if<Terminator::Call>(&block->terminator.data);
                call != nullptr && call->call.name == callee_name) {
                calls = true;
            }
            if (const auto *multi =
                    std::get_if<Terminator::MultiCall>(&block->terminator.data);
                multi != nullptr && multi->call.name == callee_name) {
                calls = true;
            }
        }
        if (calls) {
            holders.push_back(name);
        }
    }
    return holders;
}

vector<string> copies_called(const FuncMap &funcs, const string &func_name,
                             const string &callee_name, const string &owner) {
    set<string> reached;
    vector<string> work{owner};
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (!reached.insert(name).second) {
            continue;
        }
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue;
        }
        for (const string &g : callees_of(*it->second)) {
            work.push_back(g);
        }
    }
    const string prefix = func_name + "!";
    const string callee_prefix = callee_name + "!";
    set<string> copies;
    for (const auto &[name, F] : funcs) {
        if (name != func_name && name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        if (!reached.contains(name)) {
            continue;
        }
        for (const string &g : callees_of(*F)) {
            if (g.compare(0, callee_prefix.size(), callee_prefix) == 0) {
                copies.insert(g);
            }
        }
    }
    return vector<string>(copies.begin(), copies.end());
}

bool has_nontail_call(const FuncMap &funcs, const string &func_name,
                      const string &callee_name) {
    if (!funcs.contains(func_name) || !funcs.contains(callee_name)) {
        return false;
    }
    const Function &func = *funcs.at(func_name);
    // One call: several are a branching recursion, which defer() refuses in
    // its own words. Not a recursion: a call that recurses and then goes on
    // has a stack of continuations, one per level, which no entry carries
    // -- defer() refuses that too, and loopify(N) is what runs it.
    const shared_ptr<Block> site = one_call_site(func, callee_name, func_name,
                                                 /*refuse_many=*/false);
    if (!site || callee_name == func_name || reaches(funcs, callee_name, func_name)) {
        return false;
    }
    const auto &call = std::get<Terminator::Call>(site->terminator.data);
    return !call.spawned && !in_tail_position(call, make_block_map(func));
}

namespace {

// The cut before the call: the call and the rest become one function,
// `func!callee!k`, and the site tail-calls it (see defer_continuation).
Tail continuation_tail(FuncMap &funcs, const string &func_name,
                       const string &callee_name, const string &what) {
    Split split = split_at_call(funcs, func_name, callee_name, what);
    const shared_ptr<Function> F = split.F;
    const shared_ptr<Function> G = funcs.at(callee_name);
    // The call as it is, before the site is rewritten.
    const Terminator::Call call = std::get<Terminator::Call>(split.site->terminator.data);
    const size_t result_args = call.drop ? 0 : 1;
    const Block &aentry = *split.after->blocks.front();

    // The call and its continuation as one function, `func!callee!k`: the
    // callee's arguments and then what the rest takes, less the call's own
    // value, which the body produces.
    const string k_name = func_name + "!" + callee_name + "!k";
    internal_assert(!funcs.contains(k_name))
        << what << ": " << k_name << " exists already; the call is deferred twice";
    auto K = std::make_shared<Function>();
    K->ret_type = F->ret_type;
    for (const ir::Function::Attribute attr : F->attributes) {
        if (attr != ir::Function::Attribute::exported) {
            K->attributes.push_back(attr);
        }
    }
    if (std::find(K->attributes.begin(), K->attributes.end(),
                  ir::Function::Attribute::held) == K->attributes.end()) {
        K->attributes.push_back(ir::Function::Attribute::held);
    }
    auto kentry = std::make_shared<Block>();
    kentry->name = k_name;
    kentry->owner = K;
    set<string> taken;
    for (size_t i = result_args; i < aentry.args.size(); i++) {
        taken.insert(aentry.args[i].name);
    }
    // The callee's parameters, under its names for them, kept apart from
    // the rest's; their `mut`, unaliased and reducer facts are the callee's.
    const Block &gentry = *G->blocks.front();
    internal_assert(gentry.args.size() == call.call.args.size())
        << what << ": " << callee_name << " takes " << gentry.args.size()
        << " arguments and the call passes " << call.call.args.size();
    vector<shared_ptr<Value>> gargs;
    for (const Argument &p : gentry.args) {
        Argument a = p;
        while (!taken.insert(a.name).second) {
            a.name += "_";
        }
        auto v = kentry->add_argument(a);
        K->reserve_name(a.name);
        gargs.push_back(v);
    }
    // Then the rest's parameters after the call's value, as the rest has
    // them (with what the split found out about each address).
    vector<shared_ptr<Value>> onward_params;
    for (size_t i = result_args; i < aentry.args.size(); i++) {
        auto v = kentry->add_argument(aentry.args[i]);
        K->reserve_name(aentry.args[i].name);
        onward_params.push_back(v);
    }
    // The body: the call, then the rest with its value, whose value is the
    // function's.
    auto kran = std::make_shared<Block>();
    kran->name = k_name + "!ran";
    kran->owner = K;
    vector<shared_ptr<Value>> onwards;
    if (result_args == 1) {
        Argument r{G->ret_type, "$" + callee_name};
        onwards.push_back(kran->add_argument(r));
        K->reserve_name(r.name);
    }
    for (size_t i = result_args; i < aentry.args.size(); i++) {
        onwards.push_back(kran->add_argument(aentry.args[i]));
    }
    {
        Terminator::Call g;
        g.call = Terminator::Jump{callee_name, gargs};
        g.cont = Terminator::Jump{kran->name, onward_params};
        g.drop = call.drop;
        kentry->terminator.data = std::move(g);
    }
    auto kdone = returning_block(K, k_name + "!done", "$k");
    {
        Terminator::Call rest;
        rest.call = Terminator::Jump{split.after_name, std::move(onwards)};
        rest.cont = Terminator::Jump{kdone->name, {}};
        rest.drop = K->ret_type.is<Void_t>();
        kran->terminator.data = std::move(rest);
    }
    K->blocks = {kentry, kran, kdone};
    refresh_preds(*K);
    funcs[k_name] = K;

    // In `func`: the site tail-calls the function of the call and its rest,
    // with the call's arguments, what the continuation was handed and the
    // live values.
    vector<shared_ptr<Value>> passed = call.call.args;
    passed.insert(passed.end(), call.cont.args.begin(), call.cont.args.end());
    passed.insert(passed.end(), split.live.begin(), split.live.end());
    auto returned = returning_block(F, split.cont->name + "!deferred", "$deferred");
    Terminator::Call tail;
    tail.call = Terminator::Jump{k_name, std::move(passed)};
    tail.cont = Terminator::Jump{returned->name, {}};
    tail.drop = F->ret_type.is<Void_t>();
    split.site->terminator.data = std::move(tail);
    F->blocks.push_back(returned);
    refresh_preds(*F);
    remove_unreachable_blocks(*F);
    return Tail{F, split.site, k_name};
}

} // namespace

vector<Type> defer_continuation(FuncMap &funcs, const string &func_name,
                                const string &callee_name, const QueueSpec &queue) {
    const string what =
        func_name + ".defer(" + callee_name + ", " + queue.name + ")";
    // A tail call now: the queue holds the call's arguments and the rest's
    // live values, and the drain runs the call and then the rest.
    const Tail tail = continuation_tail(funcs, func_name, callee_name, what);
    return defer(funcs, func_name, tail.callee, queue);
}

bool calls_after(const Function &func, const string &staged,
                 const string &callee) {
    const Block *site = nullptr;
    for (const auto &block : func.blocks) {
        const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call != nullptr && call->call.name == staged) {
            if (site != nullptr) {
                return false;
            }
            site = block.get();
        }
    }
    if (site == nullptr) {
        return false;
    }
    const Cfg rest(func, std::get<Terminator::Call>(site->terminator.data).cont.name);
    bool any = false;
    for (const auto &block : func.blocks) {
        const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call == nullptr || call->call.name != callee) {
            continue;
        }
        any = true;
        if (!rest.contains(*block)) {
            return false;
        }
    }
    return any;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
