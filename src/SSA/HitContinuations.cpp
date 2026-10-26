#include "SSA/HitContinuations.h"

#include "IR/Printer.h"
#include "IR/Resource.h"
#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/Definitions.h"
#include "Error.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

bool explaining() {
    static const bool on =
        std::getenv("BONSAI_EXPLAIN_HIT_CONTINUATIONS") != nullptr;
    return on;
}

void refuse(const string &loop, const string &why) {
    if (explaining()) {
        std::cerr << "hit-continuations: the raygen over `" << loop
                  << "` keeps its continuation: " << why << "\n";
    }
}

const Type &u32() {
    static const Type t = UInt_t::make(32);
    return t;
}

shared_ptr<Value> constant_u32(uint64_t v) {
    return std::make_shared<Value>(Constant{u32(), v});
}

shared_ptr<Value> constant_bool(bool v) {
    return std::make_shared<Value>(Constant{Bool_t::make(), v});
}

shared_ptr<Value> constant_f32(double v) {
    return std::make_shared<Value>(Constant{Float_t::make_f32(), v});
}

shared_ptr<Value> undefined_of(const Type &t) {
    return std::make_shared<Value>(Constant{t, Undefined{}});
}

// Every place a block holds a value (as SSA/Stage.cpp walks them).
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
                   [&](Terminator::Spawn &s) {
                       all(s.call.args);
                       all(s.cont.args);
                   },
                   [&](Terminator::Join &j) { all(j.next.args); },
                   [&](Terminator::Defer &d) {
                       fn(d.queue);
                       fn(d.entry);
                       if (d.answer) {
                           fn(d.answer);
                       }
                   },
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

// The functions the blocks call, by terminator.
set<string> callees_of(const vector<shared_ptr<Block>> &blocks) {
    set<string> out;
    for (const auto &block : blocks) {
        if (const auto *c = std::get_if<Terminator::Call>(&block->terminator.data)) {
            out.insert(c->call.name);
        } else if (const auto *m = std::get_if<Terminator::MultiCall>(
                       &block->terminator.data)) {
            out.insert(m->call.name);
        }
    }
    return out;
}

// A nearest hit's trace function, `_rt_traceN` (Lower/Trees.cpp) -- not
// its hit-side variant `_rt_traceN!hit[$boxes]`, which this pass calls
// from the programs it makes.
bool is_trace_function(const string &name) {
    return name.rfind("_rt_trace", 0) == 0 && name.find('!') == string::npos;
}

// The `rt_trace` instruction of a trace function, or null.
shared_ptr<Instruction> trace_instruction(const Function &f) {
    for (const auto &block : f.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op == Instruction::Op::Intrinsic &&
                instr->intrinsic == Intrinsic::rt_trace) {
                return instr;
            }
        }
    }
    return nullptr;
}

// Whether the address `v` is one of `f`'s own locals: an alloca, through
// whatever field and element arithmetic. `through_param` answers an address
// that is a parameter of `f`: what the caller passed there (the chain's one
// call site), or false.
// Where a function's values are defined, built once per function asked.
using DefCache = map<const Function *, std::unique_ptr<Definitions>>;

Definitions &definitions_of(DefCache &cache, const Function &f) {
    auto &d = cache[&f];
    if (!d) {
        d = std::make_unique<Definitions>(f, /*lenient=*/true);
    }
    return *d;
}

bool local_address(const Function &f, const string &block,
                   const shared_ptr<Value> &v,
                   const std::function<bool(const Argument &)> &through_param,
                   DefCache &cache) {
    shared_ptr<Value> at = v;
    string where = block;
    for (int depth = 0; at && depth < 64; depth++) {
        if (std::get_if<Argument>(&at->data) != nullptr) {
            // A block's argument: the value it carries, followed back
            // through the jumps that pass it (SSA/Definitions.h) -- a
            // function parameter, an instruction, or a merge.
            const Definition d = definitions_of(cache, f).of(where, at);
            if (const auto *a = std::get_if<Argument>(&d.value->data)) {
                if (!f.blocks.empty() && d.block == f.blocks.front()->name) {
                    for (const Argument &param : f.blocks.front()->args) {
                        if (param.name == a->name) {
                            return through_param(*a);
                        }
                    }
                }
                return false; // a merge of addresses, or a loop's
            }
            at = d.value;
            where = d.block;
            if (std::get_if<shared_ptr<Instruction>>(&at->data) == nullptr) {
                return false;
            }
        }
        const auto *ip = std::get_if<shared_ptr<Instruction>>(&at->data);
        if (ip == nullptr) {
            return false;
        }
        const Instruction &i = **ip;
        if (const auto owner = i.owner.lock()) {
            where = owner->name;
        }
        switch (i.op) {
        case Instruction::Op::Alloca:
        case Instruction::Op::Alloc: // a local too: the stack under --no-heap
            return true;
        case Instruction::Op::GEP:
        case Instruction::Op::FieldPtr:
        case Instruction::Op::AddressOf:
        case Instruction::Op::Cast:
        case Instruction::Op::Reinterpret:
            if (i.operands.empty()) {
                return false;
            }
            at = i.operands[0];
            break;
        default:
            return false;
        }
    }
    return false;
}

// The intrinsics a program may not recompute: an effect, a read of state
// that moves, or a trace.
bool effectful_intrinsic(Intrinsic::OpType op) {
    switch (op) {
    case Intrinsic::rt_trace:
    case Intrinsic::rt_set_payload:
    case Intrinsic::rt_report_hit:
    case Intrinsic::rt_ignore_hit:
    case Intrinsic::rt_reorder:
    case Intrinsic::sort_indices:
    case Intrinsic::fill_bytes:
    case Intrinsic::rand:
        return true;
    default:
        return Intrinsic::has_effects(op);
    }
}

// What a function on the chain is, and how its one call site in the caller
// is found.
struct Link {
    string name;              // the function; empty for the raygen's body
    shared_ptr<Function> f;   // the function (the host, for the body)
    vector<shared_ptr<Block>> blocks; // its blocks, or the body's region
    shared_ptr<Block> site;   // the block whose call goes down the chain
};

// Whether every instruction and terminator of `blocks` is one the program
// may run again before the trace: no effect but a store to a local. Calls to
// functions off the chain are checked the same way, whole, memoized.
struct EffectCheck {
    const FuncMap &funcs;
    DefCache cache;
    string why;
    set<string> visiting;

    bool clean_blocks(const Function &f, const vector<shared_ptr<Block>> &blocks,
                      const std::function<bool(const Argument &)> &through_param,
                      const set<const Block *> &skip,
                      const string &chain_callee) {
        for (const auto &block : blocks) {
            if (skip.count(block.get())) {
                continue;
            }
            for (const auto &instr : block->instrs) {
                switch (instr->op) {
                case Instruction::Op::Store:
                    if (instr->atomic ||
                        instr->operands.empty() ||
                        !local_address(f, block->name, instr->operands[0],
                                       through_param, cache)) {
                        // What the address comes down to, for the message.
                        string root = "something computed";
                        shared_ptr<Value> at = instr->operands.empty() ? nullptr : instr->operands[0];
                        for (int depth = 0; at && depth < 64; depth++) {
                            if (const auto *a = std::get_if<Argument>(&at->data)) {
                                root = "the parameter `" + a->name + "`";
                                break;
                            }
                            const auto *ip = std::get_if<shared_ptr<Instruction>>(&at->data);
                            if (ip == nullptr) {
                                break;
                            }
                            const Instruction &i = **ip;
                            if ((i.op == Instruction::Op::GEP || i.op == Instruction::Op::FieldPtr ||
                                 i.op == Instruction::Op::AddressOf || i.op == Instruction::Op::Cast ||
                                 i.op == Instruction::Op::Reinterpret) &&
                                !i.operands.empty()) {
                                at = i.operands[0];
                                continue;
                            }
                            root = "an instruction (op " + std::to_string(int(i.op)) +
                                   (i.name.empty() ? string() : ", `" + i.name + "`") + ")";
                            break;
                        }
                        why = "a store in `" + block->name + "` through " + root +
                              ", memory the launch shares";
                        return false;
                    }
                    break;
                case Instruction::Op::AtomicAdd:
                case Instruction::Op::Push:
                case Instruction::Op::Print:
                case Instruction::Op::Append:
                    why = "an effect (op " + std::to_string(int(instr->op)) +
                          ") in `" + block->name + "`";
                    return false;
                case Instruction::Op::Intrinsic:
                    if (effectful_intrinsic(instr->intrinsic)) {
                        why = "an intrinsic with an effect in `" + block->name + "`";
                        return false;
                    }
                    break;
                default:
                    if (instr->atomic) {
                        why = "an atomic in `" + block->name + "`";
                        return false;
                    }
                    break;
                }
            }
            if (std::holds_alternative<Terminator::ParFor>(block->terminator.data)) {
                why = "a parallel loop in `" + block->name + "`";
                return false;
            }
            if (std::holds_alternative<Terminator::Defer>(block->terminator.data)) {
                why = "a deferral (a push) in `" + block->name + "`";
                return false;
            }
            if (const auto *m = std::get_if<Terminator::MultiCall>(
                    &block->terminator.data)) {
                if (!clean_call(f, block->name, m->call, through_param)) {
                    return false;
                }
            }
            if (const auto *c = std::get_if<Terminator::Call>(&block->terminator.data)) {
                if (c->call.name != chain_callee &&
                    !clean_call(f, block->name, c->call, through_param)) {
                    return false;
                }
            }
        }
        return true;
    }

    // A call to a function off the chain, judged at this site: the callee
    // is clean when it stores only to its own locals or through a
    // parameter the site passes a local's address for -- the sampler's
    // `start_pixel_sample(.., state : mut ..)` on the step's own state --
    // and calls only clean functions, judged the same way. A recursion is
    // clean if its body is.
    bool clean_call(const Function &caller, const string &block,
                    const Terminator::Jump &call,
                    const std::function<bool(const Argument &)> &caller_param) {
        const string &name = call.name;
        const auto f = funcs.find(name);
        if (f == funcs.end()) {
            // A foreign function: nothing is known of it.
            why = "a call to the foreign function `" + name + "`";
            return false;
        }
        if (!visiting.insert(name).second) {
            return true;
        }
        const vector<Argument> &params = f->second->blocks.front()->args;
        const std::function<bool(const Argument &)> through =
            [&](const Argument &a) -> bool {
            for (size_t j = 0; j < params.size() && j < call.args.size(); j++) {
                if (params[j].name == a.name) {
                    return local_address(caller, block, call.args[j], caller_param,
                                         cache);
                }
            }
            return false;
        };
        const bool clean = clean_blocks(*f->second, f->second->blocks, through, {}, "");
        visiting.erase(name);
        if (!clean) {
            why = "the call to `" + name + "`: " + why;
        }
        return clean;
    }
};

// Whether any block of `blocks`, or any function they reach, traces.
bool reaches_trace(const FuncMap &funcs, const vector<shared_ptr<Block>> &blocks,
                   set<string> &seen, bool held_only = false) {
    for (const auto &block : blocks) {
        for (const auto &instr : block->instrs) {
            if (!held_only && instr->op == Instruction::Op::Intrinsic &&
                instr->intrinsic == Intrinsic::rt_trace) {
                return true;
            }
        }
    }
    for (const string &callee : callees_of(blocks)) {
        const auto f = funcs.find(callee);
        if (is_trace_function(callee)) {
            // A held trace function is a nearest hit's, not yet split; an
            // `any`'s is inlined and traces in place.
            if (!held_only) {
                return true;
            }
            if (f != funcs.end() &&
                std::find(f->second->attributes.begin(), f->second->attributes.end(),
                          ir::Function::Attribute::held) != f->second->attributes.end()) {
                return true;
            }
            continue;
        }
        if (!seen.insert(callee).second) {
            continue;
        }
        if (f != funcs.end() &&
            reaches_trace(funcs, f->second->blocks, seen, held_only)) {
            return true;
        }
    }
    return false;
}

// The one block of `blocks` whose terminator calls `callee`, or null; null
// too for several.
shared_ptr<Block> one_call_to(const vector<shared_ptr<Block>> &blocks,
                              const string &callee) {
    shared_ptr<Block> found;
    for (const auto &block : blocks) {
        const auto *c = std::get_if<Terminator::Call>(&block->terminator.data);
        if (c != nullptr && c->call.name == callee) {
            if (found) {
                return nullptr;
            }
            found = block;
        }
    }
    return found;
}

// Whether a call's continuation does nothing but return the call's value
// (or, for the raygen's body, end the iteration).
bool tail_call(const Terminator::Call &call, const BlockMap &bmap, bool in_body) {
    const auto it = bmap.find(call.cont.name);
    if (it == bmap.end()) {
        return false;
    }
    const Block &cont = *it->second;
    if (!cont.instrs.empty() || !call.cont.args.empty()) {
        return false;
    }
    if (in_body) {
        return std::holds_alternative<Terminator::Yield>(cont.terminator.data);
    }
    const auto *returns = std::get_if<Terminator::Return>(&cont.terminator.data);
    if (returns == nullptr) {
        return false;
    }
    if (call.drop) {
        return cont.args.empty() && returns->value == nullptr;
    }
    return cont.args.size() == 1 && returns->value != nullptr &&
           std::holds_alternative<Argument>(returns->value->data) &&
           std::get<Argument>(returns->value->data).name == cont.args[0].name;
}

// An instruction placed at `at` in `block`, named and entered in the lookups.
shared_ptr<Value> insert_at(Function &func, const shared_ptr<Block> &block,
                            size_t at, Type type, Instruction::Op op,
                            vector<shared_ptr<Value>> operands,
                            Intrinsic::OpType intrinsic = Intrinsic::abs) {
    auto instr = std::make_shared<Instruction>(
        func.get_unique_name(), std::move(type), op, std::move(operands), block);
    instr->intrinsic = intrinsic;
    block->instrs.insert(block->instrs.begin() + long(at), instr);
    auto value = std::make_shared<Value>(instr);
    block->lookups[instr->name] = value;
    return value;
}

void drop_attribute(Function &f, ir::Function::Attribute a) {
    f.attributes.erase(std::remove(f.attributes.begin(), f.attributes.end(), a),
                       f.attributes.end());
}

void add_attribute(Function &f, ir::Function::Attribute a) {
    if (std::find(f.attributes.begin(), f.attributes.end(), a) == f.attributes.end()) {
        f.attributes.push_back(a);
    }
}

// A function's entry block carries its name: a copy is renamed there, and
// whatever jumps to the entry inside it follows.
void rename_entry(Function &f, const string &name) {
    const string was = f.blocks.front()->name;
    f.blocks.front()->name = name;
    for (const auto &block : f.blocks) {
        for (Terminator::Jump *jump : jumps_of(*block)) {
            if (jump->name == was) {
                jump->name = name;
            }
        }
    }
}

// Redirects every Call terminator of `f` naming `from` to `to`.
void redirect_calls(Function &f, const string &from, const string &to) {
    for (const auto &block : f.blocks) {
        if (auto *c = std::get_if<Terminator::Call>(&block->terminator.data)) {
            if (c->call.name == from) {
                c->call.name = to;
            }
        }
    }
}

// In `f`, every call of `from` becomes a call of `to`, the arguments
// matched to `to`'s parameters by name: the two were made with one
// signature (Lower/Trees.cpp gives a trace function and its `!hit`
// variants one parameter list), and a pass between may have added to
// either, so the name is what is relied on, not the position.
void redirect_calls_by_name(Function &f, const string &from, const Function &from_f,
                            const string &to, const Function &to_f) {
    const vector<Argument> &from_params = from_f.blocks.front()->args;
    const vector<Argument> &to_params = to_f.blocks.front()->args;
    for (const auto &block : f.blocks) {
        auto *c = std::get_if<Terminator::Call>(&block->terminator.data);
        if (c == nullptr || c->call.name != from) {
            continue;
        }
        internal_assert(c->call.args.size() == from_params.size())
            << block->name << " calls `" << from << "` with "
            << c->call.args.size() << " arguments for " << from_params.size()
            << " parameters";
        vector<shared_ptr<Value>> args;
        for (const Argument &p : to_params) {
            size_t j = from_params.size();
            for (size_t k = 0; k < from_params.size(); k++) {
                if (from_params[k].name == p.name) {
                    j = k;
                }
            }
            internal_assert(j < from_params.size())
                << "`" << to << "` takes `" << p.name << "`, which `" << from
                << "` does not: the trace function and its hit-side variant "
                << "no longer share a signature";
            args.push_back(c->call.args[j]);
        }
        c->call.name = to;
        c->call.args = std::move(args);
    }
}

// In `F` (a copy, the raygen's), the call to the trace function goes on
// only on a miss: its continuation becomes `if hit { return } else { the
// continuation }`. `in_body` says `F`'s blocks are the raygen's own region,
// whose end is a yield.
void keep_miss_path(Function &F, const shared_ptr<Block> &site, bool in_body) {
    auto *call = std::get_if<Terminator::Call>(&site->terminator.data);
    internal_assert(call != nullptr);
    internal_assert(!call->drop)
        << "the trace function returns nothing";
    const BlockMap bmap = make_block_map(F);
    const shared_ptr<Block> cont = bmap.at(call->cont.name);
    internal_assert(!cont->args.empty());
    auto test = std::make_shared<Block>();
    test->name = cont->name + "!miss";
    test->owner = cont->owner;
    test->args = cont->args;
    for (const Argument &a : test->args) {
        test->lookups[a.name] = std::make_shared<Value>(a);
    }
    const shared_ptr<Value> r = test->lookups.at(test->args[0].name);
    // The option's `set` (Lower/Options.cpp: `{ value, set }`).
    const Struct_t *opt = r->get_type().as<Struct_t>();
    internal_assert(opt != nullptr && opt->fields.size() == 2 &&
                    opt->fields[1].name == "set")
        << "the trace function answers a " << r->get_type()
        << ", not a lowered option";
    const shared_ptr<Value> hit = test->make_instruction(
        Bool_t::make(), Instruction::Op::LoadField, {r, constant_u32(1)});
    auto done = std::make_shared<Block>();
    done->name = cont->name + "!hit-done";
    done->owner = cont->owner;
    if (in_body) {
        done->terminator.data = Terminator::Yield{};
    } else {
        Terminator::Return ret;
        if (!F.ret_type.is<Void_t>()) {
            ret.value = undefined_of(F.ret_type);
        }
        done->terminator.data = std::move(ret);
    }
    vector<shared_ptr<Value>> onward;
    for (const Argument &a : test->args) {
        onward.push_back(test->lookups.at(a.name));
    }
    test->terminator.data = Terminator::Dispatch{
        .cond = hit,
        .targets = {Terminator::Jump{cont->name, std::move(onward)},
                    Terminator::Jump{done->name, {}}}};
    call->cont.name = test->name;
    F.blocks.push_back(test);
    F.blocks.push_back(done);
    refresh_preds(F);
}

struct Candidate {
    string host;
    shared_ptr<Function> H;
    shared_ptr<Block> loop_block;
    vector<Link> chain; // the body first, then the functions down to the one that traces
    string trace;       // the trace function
    // The link whose call down the chain is NOT in tail position: the cut.
    // What follows that call is the continuation; the calls below it, down
    // to the trace function, are tail calls (the query function `trace`
    // returning `_rt_traceN`'s answer).
    size_t cut = 0;
    bool has_cut = false;
    string query;
};

// The chain from the raygen's body down to the one call of a held trace
// function, or nothing with the reason.
bool find_chain(const FuncMap &funcs, Candidate &c, string &why) {
    const Terminator::ParFor &loop =
        std::get<Terminator::ParFor>(c.loop_block->terminator.data);
    Link body;
    body.f = c.H;
    body.blocks = Cfg(*c.H, loop.body.name).blocks();
    c.chain.push_back(body);
    set<string> visited;
    for (;;) {
        Link &link = c.chain.back();
        const set<string> callees = callees_of(link.blocks);
        // The trace function itself, or the one callee that reaches one.
        vector<string> onward;
        for (const string &callee : callees) {
            if (is_trace_function(callee)) {
                const auto f = funcs.find(callee);
                if (f == funcs.end() ||
                    std::find(f->second->attributes.begin(),
                              f->second->attributes.end(),
                              ir::Function::Attribute::held) ==
                        f->second->attributes.end()) {
                    continue; // an `any`, or one already split
                }
                onward.push_back(callee);
                continue;
            }
            const auto f = funcs.find(callee);
            if (f == funcs.end()) {
                continue;
            }
            set<string> seen;
            if (reaches_trace(funcs, f->second->blocks, seen, /*held_only=*/true)) {
                onward.push_back(callee);
            }
        }
        if (onward.empty()) {
            why = "nothing it calls traces";
            return false;
        }
        if (onward.size() > 1) {
            why = "`" + (link.name.empty() ? string("the body") : link.name) +
                  "` reaches a trace through more than one call";
            return false;
        }
        const string &next = onward[0];
        link.site = one_call_to(link.blocks, next);
        if (!link.site) {
            why = "`" + next + "` is called more than once, or not by a plain call";
            return false;
        }
        const auto &call = std::get<Terminator::Call>(link.site->terminator.data);
        if (!tail_call(call, make_block_map(*link.f), link.name.empty())) {
            if (c.has_cut) {
                why = "two calls on the way to the trace, `" +
                      c.chain[c.cut].name + "`'s and `" + next +
                      "`'s, go on after they return";
                return false;
            }
            c.cut = c.chain.size() - 1;
            c.has_cut = true;
        }
        if (is_trace_function(next)) {
            c.trace = next;
            if (!c.has_cut) {
                why = "nothing follows the trace";
                return false;
            }
            return true;
        }
        if (!visited.insert(next).second) {
            why = "`" + next + "` recurses";
            return false;
        }
        Link down;
        down.name = next;
        down.f = funcs.at(next);
        down.blocks = down.f->blocks;
        c.chain.push_back(down);
        if (c.chain.size() > 16) {
            why = "the chain to the trace is deeper than 16 calls";
            return false;
        }
    }
}

} // namespace

void split_hit_continuations(FuncMap &funcs) {
    // The raygens: every loop bound to OptixThread.
    vector<Candidate> candidates;
    for (const auto &[name, f] : funcs) {
        for (const auto &block : f->blocks) {
            const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p != nullptr && p->binding.has_value() &&
                *p->binding == Resource::OptixThread) {
                Candidate c;
                c.host = name;
                c.H = f;
                c.loop_block = block;
                candidates.push_back(std::move(c));
            }
        }
    }
    set<string> split_traces;
    for (Candidate &c : candidates) {
        const Terminator::ParFor &loop =
            std::get<Terminator::ParFor>(c.loop_block->terminator.data);
        string why;
        if (!find_chain(funcs, c, why)) {
            refuse(loop.index, why);
            continue;
        }
        const shared_ptr<Function> T = funcs.at(c.trace);
        const shared_ptr<Instruction> trace = trace_instruction(*T);
        if (!trace || trace->operands.size() < 7 ||
            !trace->operands[6]->get_type().is<Struct_t>()) {
            refuse(loop.index, "the trace of `" + c.trace +
                                   "` carries its context in memory");
            continue;
        }
        {
            const string ctx_name = trace->operands[6]->get_type().as<Struct_t>()->name;
            static const string prefix = "_RTContext_";
            internal_assert(ctx_name.rfind(prefix, 0) == 0) << ctx_name;
            c.query = ctx_name.substr(prefix.size());
        }
        // The cut link's continuation after its call down the chain: the
        // rest. It may not trace again nor loop back to the call.
        Link &last = c.chain[c.cut];
        const auto &call = std::get<Terminator::Call>(last.site->terminator.data);
        const vector<shared_ptr<Block>> rest = Cfg(*last.f, call.cont.name).blocks();
        set<const Block *> in_rest;
        for (const auto &b : rest) {
            in_rest.insert(b.get());
        }
        if (in_rest.count(last.site.get())) {
            refuse(loop.index, "the continuation of the trace loops back to it");
            continue;
        }
        {
            set<string> seen;
            if (reaches_trace(funcs, rest, seen)) {
                refuse(loop.index, "the continuation of the trace traces again");
                continue;
            }
        }
        // What the program recomputes before the trace -- every link's
        // blocks but the rest -- has no effect but on locals. A store
        // through a parameter of a link is to what the link above passed
        // there, resolved up the chain.
        EffectCheck check{funcs};
        bool clean = true;
        for (size_t i = 0; i < c.chain.size() && clean; i++) {
            const std::function<bool(const Argument &)> through_param =
                [&, i](const Argument &a) -> bool {
                // Up the chain: the argument the caller's site passes for
                // this parameter, tested in the caller's own terms.
                size_t level = i;
                Argument param = a;
                for (;;) {
                    if (level == 0) {
                        return false; // the body's index or a capture: launch memory
                    }
                    const Link &callee = c.chain[level];
                    const Link &caller = c.chain[level - 1];
                    const vector<Argument> &params = callee.f->blocks.front()->args;
                    size_t j = params.size();
                    for (size_t k = 0; k < params.size(); k++) {
                        if (params[k].name == param.name) {
                            j = k;
                        }
                    }
                    if (j == params.size()) {
                        return false; // not a parameter: a block's own argument
                    }
                    const auto &site_call =
                        std::get<Terminator::Call>(caller.site->terminator.data);
                    if (j >= site_call.call.args.size()) {
                        return false;
                    }
                    const shared_ptr<Value> passed = site_call.call.args[j];
                    // A local of the caller, or the caller's own parameter.
                    bool is_param = false;
                    Argument next_param;
                    const bool local = local_address(
                        *caller.f, caller.site->name, passed,
                        [&](const Argument &pa) {
                            is_param = true;
                            next_param = pa;
                            return false;
                        },
                        check.cache);
                    if (local) {
                        return true;
                    }
                    if (!is_param) {
                        return false;
                    }
                    level--;
                    param = next_param;
                }
            };
            // The cut link's rest is the continuation, not recomputed; the
            // links below the cut are the tail chain to the trace function,
            // checked whole.
            const set<const Block *> skip =
                (i == c.cut) ? in_rest : set<const Block *>{};
            const string chain_callee =
                (i + 1 < c.chain.size()) ? c.chain[i + 1].name : c.trace;
            if (!check.clean_blocks(*c.chain[i].f, c.chain[i].blocks, through_param, skip,
                                    chain_callee)) {
                clean = false;
            }
        }
        if (!clean) {
            refuse(loop.index, "what comes before the trace has " + check.why);
            continue;
        }

        // The split. One flavor of the hit program per hit-side variant
        // the lowering made of the trace function (Lower/Trees.cpp:
        // `!hit`, and `!hit$boxes` where the query carries a witness) --
        // the flavors the generic programs come in.
        vector<string> flavors;
        for (const string &flavor : {string(""), string("$boxes")}) {
            if (funcs.contains(c.trace + "!hit" + flavor)) {
                flavors.push_back(flavor);
            }
        }
        if (flavors.empty()) {
            refuse(loop.index, "`" + c.trace + "` has no hit-side variant");
            continue;
        }
        vector<string> hit_names;
        for (const string &flavor : flavors) {
            // The trace function's hit-side variant, then the chain's
            // copies down to it, each calling the next.
            string below = c.trace + "!hit" + flavor;
            for (size_t i = c.chain.size() - 1; i >= 1; i--) {
                const Link &link = c.chain[i];
                const string copy_name = link.name + "!hit" + flavor;
                auto copy = clone_function(*link.f);
                rename_entry(*copy, copy_name);
                add_attribute(*copy, ir::Function::Attribute::device_only);
                drop_attribute(*copy, ir::Function::Attribute::held);
                if (i + 1 < c.chain.size()) {
                    redirect_calls(*copy, c.chain[i + 1].name, below);
                } else {
                    redirect_calls_by_name(*copy, c.trace, *T, below, *funcs.at(below));
                }
                funcs[copy_name] = copy;
                below = copy_name;
            }
            // The program: the body's region again, calling the copies.
            const string symbol = "__closesthit__" + c.query + flavor;
            const string name = symbol + "$" + loop.index;
            internal_assert(!funcs.contains(name)) << name << " exists already";
            auto P = std::make_shared<Function>();
            P->ret_type = Void_t::make();
            const vector<shared_ptr<Block>> &region = c.chain[0].blocks;
            const auto copies = clone_region(*c.H, region, "!" + name, /*keep_names=*/true);
            const shared_ptr<Block> head = copies.at(loop.body.name);
            const string entry_was = head->name;
            head->name = name;
            P->blocks.push_back(head);
            for (const auto &block : region) {
                if (block->name != loop.body.name) {
                    P->blocks.push_back(copies.at(block->name));
                }
            }
            for (const auto &block : P->blocks) {
                block->owner = P;
                for (const auto &instr : block->instrs) {
                    instr->owner = block;
                    P->reserve_name(instr->name);
                }
                for (const Argument &a : block->args) {
                    P->reserve_name(a.name);
                }
                for (Terminator::Jump *jump : jumps_of(*block)) {
                    if (jump->name == entry_was) {
                        jump->name = name;
                    }
                }
                if (std::holds_alternative<Terminator::Yield>(block->terminator.data)) {
                    block->terminator.data = Terminator::Return{};
                }
            }
            if (c.chain.size() > 1) {
                redirect_calls(*P, c.chain[1].name, below);
            } else {
                redirect_calls_by_name(*P, c.trace, *T, below, *funcs.at(below));
            }
            refresh_preds(*P);
            ir::Function::OptixProgram program;
            program.kind = ir::Function::OptixProgram::Kind::ClosestHit;
            program.of = c.query;
            for (const Argument &a : head->args) {
                program.reads.push_back(a.name);
            }
            program.for_loop = loop.index;
            program.symbol = symbol;
            P->optix_program = std::move(program);
            funcs[name] = P;
            hit_names.push_back(name);
        }
        // The raygen's side: copies of the chain that go on only on a miss,
        // and the body calling the first of them -- or, when the body
        // traces itself, the body's own continuation guarded.
        // Copies down to the cut: the cut's copy guards its call's
        // continuation; the links below it, the tail chain to the trace
        // function, are called as they are.
        if (c.cut == 0) {
            keep_miss_path(*c.H, c.chain[0].site, /*in_body=*/true);
        } else {
            string below;
            for (size_t i = c.cut; i >= 1; i--) {
                const Link &link = c.chain[i];
                const string copy_name = link.name + "!miss";
                auto copy = clone_function(*link.f);
                drop_attribute(*copy, ir::Function::Attribute::held);
                if (i == c.cut) {
                    // The copy's own site: the block of the same name --
                    // found before the entry is renamed, since the site
                    // may be the entry.
                    const BlockMap bmap = make_block_map(*copy);
                    keep_miss_path(*copy, bmap.at(link.site->name), /*in_body=*/false);
                } else {
                    redirect_calls(*copy, c.chain[i + 1].name, below);
                }
                rename_entry(*copy, copy_name);
                funcs[copy_name] = copy;
                below = copy_name;
            }
            // The body's call, in the raygen itself.
            auto &site_call = std::get<Terminator::Call>(c.chain[0].site->terminator.data);
            site_call.call.name = below;
        }
        split_traces.insert(c.trace);
        if (explaining()) {
            std::cerr << "hit-continuations: the raygen over `" << loop.index
                      << "` hands its continuation after `" << c.trace
                      << "` to";
            for (const string &name : hit_names) {
                std::cerr << " " << name;
            }
            std::cerr << " (" << c.chain.size() - 1 << " calls deep)\n";
        }
    }
    // The trace functions are calls no longer held for anything.
    for (const auto &[name, f] : funcs) {
        if (is_trace_function(name)) {
            drop_attribute(*f, ir::Function::Attribute::held);
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
