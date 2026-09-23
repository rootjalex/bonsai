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

} // namespace

vector<Type> stage(FuncMap &funcs, const string &func_name,
                   const string &callee_name, const QueueSpec &queue) {
    const string what =
        func_name + ".stage(" + callee_name + ", " + queue.name + ")";
    internal_assert(funcs.contains(func_name))
        << what << ": " << func_name << " is not a function of the program";
    internal_assert(funcs.contains(callee_name))
        << what << ": " << callee_name << " is not a function of the program";
    const shared_ptr<Function> F = funcs.at(func_name);

    // The one call to the callee, not in tail position.
    vector<shared_ptr<Block>> sites;
    for (const auto &block : F->blocks) {
        if (const auto *multi =
                std::get_if<Terminator::MultiCall>(&block->terminator.data)) {
            internal_assert(multi->call.name != callee_name)
                << what << ": the calls to " << callee_name << " in "
                << block->name << " are a run of " << multi->varying.size()
                << ", a branching recursion; a stage boundary is one point "
                << "after one call.";
            continue;
        }
        const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call != nullptr && call->call.name == callee_name) {
            sites.push_back(block);
        }
    }
    internal_assert(!sites.empty())
        << what << ": " << func_name << " makes no call to " << callee_name;
    internal_assert(sites.size() == 1)
        << what << ": " << func_name << " makes " << sites.size()
        << " calls to " << callee_name
        << "; a stage boundary is one point after one call.";
    const BlockMap bmap = make_block_map(*F);
    const shared_ptr<Block> site = sites.front();
    Terminator::Call &call = std::get<Terminator::Call>(site->terminator.data);
    internal_assert(!in_tail_position(call, bmap))
        << what << ": the call to " << callee_name << " in " << site->name
        << " is in tail position -- nothing of " << func_name
        << " follows it to stage. Queue the call itself: " << func_name
        << ".defer(" << callee_name << ", " << queue.name << ").";

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
    const auto copies = clone_region(*F, rest, "!after");
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

    // In `func`: the call's continuation is a block of the call's own that
    // tail-calls the rest with what it was handed and the live values, and
    // returns what comes back.
    auto staged = std::make_shared<Block>();
    staged->name = cont->name + "!stage";
    staged->owner = F;
    staged->args = cont->args;
    for (const Argument &a : staged->args) {
        staged->lookups[a.name] = std::make_shared<Value>(a);
    }
    auto returned = std::make_shared<Block>();
    returned->name = cont->name + "!staged";
    returned->owner = F;
    const bool is_void = F->ret_type.is<Void_t>();
    vector<shared_ptr<Value>> passed;
    for (const Argument &a : staged->args) {
        passed.push_back(staged->lookups.at(a.name));
    }
    for (const Argument &a : free_names) {
        passed.push_back(std::make_shared<Value>(a));
    }
    for (const auto &instr : free_instrs) {
        passed.push_back(std::make_shared<Value>(instr));
    }
    Terminator::Call tail;
    tail.call = Terminator::Jump{after_name, std::move(passed)};
    tail.cont = Terminator::Jump{returned->name, {}};
    tail.drop = is_void;
    staged->terminator.data = std::move(tail);
    if (is_void) {
        returned->terminator.data = Terminator::Return{nullptr};
    } else {
        Argument result{F->ret_type, "$staged"};
        returned->args.push_back(result);
        auto value = std::make_shared<Value>(result);
        returned->lookups[result.name] = value;
        returned->terminator.data = Terminator::Return{value};
    }
    call.cont.name = staged->name;
    F->blocks.push_back(staged);
    F->blocks.push_back(returned);
    refresh_preds(*F);
    remove_unreachable_blocks(*F);

    // What follows the call is a tail call now, and a queue of those is what
    // defer() builds.
    return defer(funcs, func_name, after_name, queue);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
