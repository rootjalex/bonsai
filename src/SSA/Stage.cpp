#include "SSA/Stage.h"

#include "Error.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/Definitions.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
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
    if (std::find(A->attributes.begin(), A->attributes.end(),
                  ir::Function::Attribute::noinline) == A->attributes.end()) {
        A->attributes.push_back(ir::Function::Attribute::noinline);
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

} // namespace

vector<Type> stage(FuncMap &funcs, const string &func_name,
                   const string &callee_name, const QueueSpec &queue) {
    const string what =
        func_name + ".stage(" + callee_name + ", " + queue.name + ")";
    Split split = split_at_call(funcs, func_name, callee_name, what);
    const shared_ptr<Function> F = split.F;
    Terminator::Call &call = std::get<Terminator::Call>(split.site->terminator.data);

    // In `func`: the call's continuation is a block of the call's own that
    // tail-calls the rest with what it was handed and the live values, and
    // returns what comes back.
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

    // What follows the call is a tail call now, and a queue of those is what
    // defer() builds.
    return defer(funcs, func_name, split.after_name, queue);
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

vector<Type> defer_continuation(FuncMap &funcs, const string &func_name,
                                const string &callee_name, const QueueSpec &queue) {
    const string what =
        func_name + ".defer(" + callee_name + ", " + queue.name + ")";
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
                  ir::Function::Attribute::noinline) == K->attributes.end()) {
        K->attributes.push_back(ir::Function::Attribute::noinline);
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

    // A tail call now: the queue holds the call's arguments and the rest's
    // live values, and the drain runs the call and then the rest.
    return defer(funcs, func_name, k_name, queue);
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
