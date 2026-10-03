#include "SSA/QueueRecursion.h"

#include "SSA/Analysis.h"
#include "SSA/InsertPreheader.h"
#include "SSA/Simplify.h"
#include "SSA/SortRecursion.h"

#include "Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// A fresh block, owned by the same function. Added to the function by the
// caller, which must do so before making the next one: the name is picked by
// looking at the names the function already has.
shared_ptr<Block> new_block(Function &func, const string &stem) {
    set<string> taken;
    for (const auto &block : func.blocks) {
        taken.insert(block->name);
    }
    string name = stem;
    for (size_t i = 0; taken.count(name); i++) {
        name = stem + "_" + std::to_string(i);
    }
    auto block = std::make_shared<Block>();
    block->name = std::move(name);
    block->owner = func.blocks.front()->owner;
    return block;
}

// A name for a piece of storage, distinct from every name already in use.
string new_storage_name(const Function &func, const string &stem) {
    set<string> taken;
    for (const auto &block : func.blocks) {
        for (const Argument &arg : block->args) {
            taken.insert(arg.name);
        }
        for (const auto &instr : block->instrs) {
            taken.insert(instr->name);
        }
    }
    string name = stem;
    for (size_t i = 0; taken.count(name); i++) {
        name = stem + std::to_string(i);
    }
    return name;
}

bool is_named_argument(const Value &v, const string &name) {
    const auto *a = std::get_if<Argument>(&v.data);
    return a != nullptr && a->name == name;
}

// Appends an instruction to `block`, without the operand rethreading
// make_instruction does: everything here refers to values the entry block
// defines, which dominates the lot, so nothing needs threading through
// arguments.
shared_ptr<Value> append(Function &func, const shared_ptr<Block> &block,
                         Type type, Instruction::Op op,
                         vector<shared_ptr<Value>> operands,
                         const string &name = "") {
    auto instr = std::make_shared<Instruction>(
        name.empty() ? func.get_unique_name() : name, std::move(type), op,
        std::move(operands), block);
    block->instrs.push_back(instr);
    return std::make_shared<Value>(std::move(instr));
}

void append_store(const shared_ptr<Block> &block, shared_ptr<Value> dest,
                  shared_ptr<Value> v) {
    block->instrs.push_back(std::make_shared<Instruction>(
        Instruction::Op::Store,
        vector<shared_ptr<Value>>{std::move(dest), std::move(v)}, block));
}

// A boolean the traversal only ever moves one way, and the direction it moves.
//
// `any` raises its accumulator from false and never lowers it; `all` lowers
// from true and never raises. Either way, once it reaches the end of its
// lattice nothing the rest of the traversal does can move it back -- so the
// rest of the traversal need not happen. That is the whole of `any` meaning
// "stop when you find one", and without it a settled query goes on popping its
// stack, reading a node from memory each time, to decide nothing.
struct Monotone {
    shared_ptr<Value> ptr;
    // Rising: only ever set to true, or to `acc | x`. The traversal is done
    // once it is true. Falling is the mirror image, for `all`.
    bool rising = false;
};

// The name a value goes by, if it has one: an argument's, or an
// instruction's. A variable is one instruction where it is made -- or one
// parameter -- and an argument of every block it is threaded through, all
// under this name, which is what identifies it across blocks.
std::optional<string> name_of(const shared_ptr<Value> &v) {
    if (!v) {
        return std::nullopt;
    }
    if (std::optional<Argument> a = v->get_argument()) {
        return a->name;
    }
    if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data);
        in != nullptr && *in != nullptr) {
        return (*in)->name;
    }
    return std::nullopt;
}

// The arguments `pred`'s terminator passes to `target`, and how many of the
// target's arguments come before them -- a call's result, a loop's index.
std::optional<std::pair<const vector<shared_ptr<Value>> *, size_t>>
edge_into(const Block &pred, const string &target) {
    using Found =
        std::optional<std::pair<const vector<shared_ptr<Value>> *, size_t>>;
    return std::visit(
        overloads{
            [&](const Terminator::Jump &j) -> Found {
                if (j.name == target)
                    return {{&j.args, 0}};
                return std::nullopt;
            },
            [&](const Terminator::Dispatch &d) -> Found {
                for (const Terminator::Jump &t : d.targets) {
                    if (t.name == target)
                        return {{&t.args, 0}};
                }
                return std::nullopt;
            },
            [&](const Terminator::Call &c) -> Found {
                if (c.cont.name == target)
                    return {{&c.cont.args, c.drop ? 0u : 1u}};
                return std::nullopt;
            },
            [&](const Terminator::MultiCall &c) -> Found {
                if (c.cont.name == target)
                    return {{&c.cont.args, 0}};
                return std::nullopt;
            },
            [&](const Terminator::ParFor &p) -> Found {
                if (p.body.name == target)
                    return {{&p.body.args, 1}};
                if (p.cont.name == target)
                    return {{&p.cont.args, 0}};
                return std::nullopt;
            },
            [&](const auto &) -> Found { return std::nullopt; }},
        pred.terminator.data);
}

// The value `v` stands for, following a block argument back through the
// predecessors that pass it until it reaches an instruction, a constant, or
// an argument of the function. `at` is the block `v` is used in. A value the
// builder computed before a call arrives in the continuation as an argument,
// and this is what sees the load behind it.
struct Resolver {
    explicit Resolver(const Function &func) : blocks(make_block_map(func)) {
        // From the terminators that exist. This runs while the loop is being
        // built, when the blocks it adds have no terminator yet.
        for (const auto &block : func.blocks) {
            if (!block->terminator.defined()) {
                continue;
            }
            for (const string &target : successors(*block)) {
                preds[target].push_back(block->name);
            }
        }
    }

    const BlockMap blocks;
    // Predecessors by name, in the order the function lists the blocks.
    std::map<string, vector<string>> preds;

    shared_ptr<Value> operator()(shared_ptr<Value> v, const Block *at) const {
        set<const Block *> seen;
        while (v) {
            std::optional<Argument> a = v->get_argument();
            auto pit = preds.find(at->name);
            if (!a || pit == preds.end() || pit->second.empty() ||
                !seen.insert(at).second) {
                return v;
            }
            size_t idx = at->args.size();
            for (size_t i = 0; i < at->args.size(); i++) {
                if (at->args[i].name == a->name) {
                    idx = i;
                    break;
                }
            }
            if (idx == at->args.size()) {
                return v;
            }
            bool followed = false;
            for (const string &pred_name : pit->second) {
                const Block &pred = *blocks.at(pred_name);
                auto found = edge_into(pred, at->name);
                if (!found) {
                    continue;
                }
                const auto &[vals, skip] = *found;
                if (idx < skip) {
                    return v;
                }
                v = (*vals)[idx - skip];
                at = &pred;
                followed = true;
                break;
            }
            if (!followed) {
                return v;
            }
        }
        return v;
    }

    const Block *owner_of(const shared_ptr<Value> &v) const {
        const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data);
        if (in == nullptr || *in == nullptr) {
            return nullptr;
        }
        auto block = (*in)->owner.lock();
        return block.get();
    }
};

// Whether `v`, used in `at`, is a load of the variable called `name`.
bool loads(const shared_ptr<Value> &v, const Block *at, const string &name,
           const Resolver &resolve) {
    shared_ptr<Value> r = resolve(v, at);
    const auto *instr = std::get_if<shared_ptr<Instruction>>(&r->data);
    if (instr == nullptr || *instr == nullptr) {
        return false;
    }
    return (*instr)->op == Instruction::Op::Load &&
           (*instr)->operands.size() == 1 &&
           name_of((*instr)->operands[0]) == name;
}

// Whether `v` is the boolean constant `want` -- or that constant broadcast to
// every lane, which is what a gang stores into a per-lane accumulator.
bool is_bool_const(const shared_ptr<Value> &v, bool want) {
    if (!v) {
        return false;
    }
    if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data);
        in != nullptr && *in != nullptr && (*in)->op == Instruction::Op::Bc &&
        !(*in)->operands.empty()) {
        return is_bool_const((*in)->operands[0], want);
    }
    const auto *constant = std::get_if<Constant>(&v->data);
    if (constant == nullptr) {
        return false;
    }
    const bool *b = std::get_if<bool>(&constant->data);
    return b != nullptr && *b == want;
}

// Whether `v`, used in `at`, is `acc <op> x` for a combining op, with `acc` a
// load of the variable called `name`.
bool combines_with_self(const shared_ptr<Value> &v, const Block *at,
                        const string &name, Instruction::Op logical,
                        Instruction::Op bitwise, const Resolver &resolve) {
    shared_ptr<Value> r = resolve(v, at);
    const auto *held = std::get_if<shared_ptr<Instruction>>(&r->data);
    if (held == nullptr || *held == nullptr) {
        return false;
    }
    const Instruction &instr = **held;
    if (instr.op != logical && instr.op != bitwise) {
        return false;
    }
    const Block *owner = resolve.owner_of(r);
    for (const auto &operand : instr.operands) {
        if (loads(operand, owner ? owner : at, name, resolve)) {
            return true;
        }
    }
    return false;
}

// Find a boolean accumulator this function only ever moves one way.
//
// Deliberately a *check* rather than something the lowering asserts. The
// lowering knows perfectly well that `any` is monotone, but getting that fact
// down here would mean carrying it through every pass that rebuilds a
// function, and a fact carried like that goes missing silently -- it stops
// being applied and nothing says so. Re-deriving it costs one linear scan, and
// the worst a mistake anywhere upstream can do is make this decline.
std::optional<Monotone> find_monotone_accumulator(const Function &func) {
    struct Candidate {
        size_t stores = 0;
        bool rising = false;
        bool falling = false;
        bool disqualified = false;
    };
    // By name. A variable is one instruction where it is made -- or one
    // parameter -- and an argument of every block it is threaded through,
    // all under the same name. Looked at value by value, the stores to it
    // would be counted in some blocks and missed in others, and one `= false`
    // seen on its own passes for `all`; by name every store is seen, and a
    // boolean that is also assigned something arbitrary is disqualified.
    //
    // It also has to outlive an iteration, or it is not accumulating
    // anything: a parameter of the function, or storage allocated in the
    // entry block, before the loop. A boolean a body allocates for itself on
    // every trip -- the result of a small function inlined into it, say -- is
    // fresh each time, and a load of it in the loop's condition would read
    // storage that is not there.
    std::set<std::string> outlives;
    for (const Argument &arg : func.blocks.front()->args) {
        outlives.insert(arg.name);
    }
    for (const auto &instr : func.blocks.front()->instrs) {
        if (instr->op == Instruction::Op::Alloca ||
            instr->op == Instruction::Op::Alloc) {
            outlives.insert(instr->name);
        }
    }
    const Resolver resolve(func);
    std::map<std::string, Candidate> candidates;

    // A pointer that escapes into a call may be moved by the callee in a
    // direction this function cannot see, so it is no longer this function's
    // to reason about.
    std::set<std::string> escaped;
    const auto note_escapes = [&](const std::vector<shared_ptr<Value>> &args) {
        for (const auto &arg : args) {
            if (std::optional<std::string> name = name_of(arg)) {
                escaped.insert(*name);
            }
        }
    };

    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Store) {
                if (instr->op == Instruction::Op::AddressOf) {
                    note_escapes(instr->operands);
                }
                continue;
            }
            // A store made by a gang carries its execution mask as a third
            // operand; the lanes that are off keep what they had, which
            // moves nothing in either direction.
            if (instr->operands.size() != 2 && instr->operands.size() != 3) {
                continue;
            }
            const shared_ptr<Value> &dest = instr->operands[0];
            if (!dest) {
                continue;
            }
            // One bool, or one per lane of a gang -- a traversal vectorized
            // before it was put on a stack accumulates a bool for each lane.
            const Ptr_t *ptr_type = dest->get_type().as<Ptr_t>();
            if (ptr_type == nullptr || !ptr_type->etype.is_bool()) {
                continue;
            }
            std::optional<std::string> name = name_of(dest);
            if (!name.has_value() || !outlives.contains(*name)) {
                continue;
            }

            Candidate &candidate = candidates[*name];
            candidate.stores++;
            const shared_ptr<Value> &value = instr->operands[1];
            if (is_bool_const(value, true) ||
                combines_with_self(value, block.get(), *name,
                                   Instruction::Op::LOr, Instruction::Op::BwOr,
                                   resolve)) {
                candidate.rising = true;
            } else if (is_bool_const(value, false) ||
                       combines_with_self(value, block.get(), *name,
                                          Instruction::Op::LAnd,
                                          Instruction::Op::BwAnd, resolve)) {
                candidate.falling = true;
            } else {
                candidate.disqualified = true;
            }
        }
        if (const auto *call = std::get_if<Terminator::Call>(&block->terminator.data)) {
            note_escapes(call->call.args);
        }
    }

    for (const auto &[name, candidate] : candidates) {
        if (candidate.disqualified || candidate.stores == 0 ||
            candidate.rising == candidate.falling || escaped.contains(name)) {
            continue;
        }
        // Rebuild a handle to the value from the store that named it.
        for (const auto &block : func.blocks) {
            for (const auto &instr : block->instrs) {
                if (instr->op != Instruction::Op::Store ||
                    instr->operands.size() != 2) {
                    continue;
                }
                if (name_of(instr->operands[0]) == name) {
                    return Monotone{instr->operands[0], candidate.rising};
                }
            }
        }
    }
    return std::nullopt;
}

// What a block's terminator calls, if it calls at all: one argument list per
// call it makes, and the continuation reached after all of them.
//
// A Call makes one call and a MultiCall makes as many as the `from` it came
// from had branches. Everything below works off this rather than off the
// terminator, because the only thing here that cares about the difference is
// the stack pushing, which does it once per entry.
struct RecursiveCall {
    string callee;
    vector<vector<shared_ptr<Value>>> args; // one per call, in visit order
    // Whether each call is made (Terminator::MultiCall::conds): one per
    // call, or empty when all are. A call that is not made is not pushed.
    vector<shared_ptr<Value>> conds;
    Terminator::Jump cont;
    bool drop;
    // The run as sorted vectors, when a sort left it so (see
    // SSA/SortRecursion.h): then the pushes are one compacting store each.
    std::optional<SortedRun> sorted;
    // Or as the lanes of vectors with no order asked for -- a node's
    // children under their mask, as the tree lowering makes them: then too.
    std::optional<LaneRun> lanes;
};

std::optional<RecursiveCall> called_by(const Block &block) {
    if (const auto *c = std::get_if<Terminator::Call>(&block.terminator.data)) {
        return RecursiveCall{c->call.name, {c->call.args}, {}, c->cont, c->drop,
                             std::nullopt,   std::nullopt};
    }
    if (const auto *c =
            std::get_if<Terminator::MultiCall>(&block.terminator.data)) {
        vector<vector<shared_ptr<Value>>> args;
        args.reserve(c->varying.size());
        for (size_t i = 0; i < c->varying.size(); i++) {
            args.push_back(c->call_args(i));
        }
        internal_assert(c->conds.empty() ||
                        c->conds.size() == c->varying.size())
            << "The run in " << block.name << " has " << c->conds.size()
            << " conditions for " << c->varying.size() << " calls";
        std::optional<SortedRun> sorted = sorted_run(*c);
        std::optional<LaneRun> lanes =
            sorted.has_value() ? std::nullopt : lane_run(*c);
        return RecursiveCall{c->call.name, std::move(args), c->conds,
                             c->cont,      c->drop,         std::move(sorted),
                             std::move(lanes)};
    }
    return std::nullopt;
}

// Whether `v` is the constant true: a branch with no condition of its own in
// a run where others have one, which needs no test.
bool always(const shared_ptr<Value> &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return false;
    }
    const bool *b = std::get_if<bool>(&c->data);
    return b != nullptr && *b;
}

} // namespace

void queue_recursion(Function &func, size_t size) {
    internal_assert(!func.blocks.empty()) << "Loopifying an empty function";
    const string entry_name = func.blocks.front()->name;

    // Where the function calls itself.
    set<string> recursive;
    {
        const Cfg region(func, entry_name);
        for (const auto &block : region.blocks()) {
            const auto call = called_by(*block);
            if (call && call->callee == entry_name) {
                recursive.insert(block->name);
            }
        }
    }
    if (recursive.empty()) {
        return; // nothing to unrecurse
    }

    // A sorted run votes on its order (see Instruction::Op::Vote), because a
    // run is made once, in one order, by whoever makes it. Once the run is
    // pushes onto a stack there is no run: each child is written down where
    // its own key puts it and visited when it comes off, so the order is each
    // visitor's own -- for a gang, each lane's -- and the comparison stands.
    {
        const BlockMap blocks = make_block_map(func);
        vector<shared_ptr<Instruction>> votes;
        for (const string &name : recursive) {
            for (const auto &instr : blocks.at(name)->instrs) {
                if (instr->op == Instruction::Op::Vote) {
                    votes.push_back(instr);
                }
            }
        }
        for (const auto &vote : votes) {
            internal_assert(vote->operands.size() == 1)
                << "A vote on " << vote->operands.size() << " values in "
                << entry_name;
            replace_uses(func, vote.get(), vote->operands[0]);
        }
    }

    internal_assert(!func.ret_type.defined() || func.ret_type.is<Void_t>())
        << "Cannot put the recursion of " << entry_name << " on a stack: it "
        << "returns a value, and a call that has only been written down has "
        << "not produced one";

    // Which parameters a recursive call changes. The rest are the same at
    // every node of the traversal -- the tree itself, where the answer is
    // being accumulated -- so they stay parameters and only what varies goes
    // on the stack.
    const vector<Argument> params = func.blocks.front()->args;
    // The mask a gang's copy of the function runs under, if it is one: the
    // parameter specialize() adds last, under this name (SSA/Vectorize.cpp).
    std::optional<size_t> mask_param;
    for (size_t i = 0; i < params.size(); i++) {
        if (params[i].name == "!mask") {
            mask_param = i;
        }
    }
    vector<bool> varies(params.size(), false);
    {
        const BlockMap blocks = make_block_map(func);
        for (const string &name : recursive) {
            const auto call = called_by(*blocks.at(name));
            internal_assert(call);
            for (const auto &args : call->args) {
                internal_assert(args.size() == params.size())
                    << "Recursive call in " << name << " passes " << args.size()
                    << " arguments to a function taking " << params.size();
                for (size_t i = 0; i < params.size(); i++) {
                    varies[i] = varies[i] ||
                                !is_named_argument(*args[i], params[i].name);
                }
            }
        }
    }

    //===----------------------------------------------------------------===//
    // The recursion has to be tail-modulo-recursion
    //===----------------------------------------------------------------===//
    //
    // Once a call is only an entry on a stack, whatever the caller did after
    // it has nowhere to happen: the callee's work is deferred to a later turn
    // of the loop, by which time the caller's frame is gone. So everything
    // after a recursive call must be either another recursive call or the
    // return itself.
    {
        const Cfg cfg(func);
        for (const string &name : recursive) {
            const auto call = called_by(cfg[cfg.id(name)]);
            internal_assert(call);
            internal_assert(call->drop)
                << "The recursive call in " << name << " keeps its result, "
                << "which a deferred call has not got";
            for (BlockId reached : reachable_from(cfg, cfg.id(call->cont.name))) {
                const Block &block = cfg[reached];
                const string &after = block.name;
                for (const auto &instr : block.instrs) {
                    internal_assert(instr->op != Instruction::Op::Store &&
                                    instr->op != Instruction::Op::Print &&
                                    instr->op != Instruction::Op::Append &&
                                    instr->op != Instruction::Op::Push)
                        << "Cannot put the recursion of " << entry_name
                        << " on a stack: " << after << " has an effect that "
                        << "happens after the recursive call in " << name
                        << ", which would be deferred past it";
                }
                const auto other = called_by(block);
                internal_assert(!other || other->callee == entry_name)
                    << "Cannot put the recursion of " << entry_name
                    << " on a stack: " << after << " calls " << other->callee
                    << " after the recursive call in " << name
                    << ", which would be deferred past it";
            }
        }
    }

    //===----------------------------------------------------------------===//
    // The stack
    //===----------------------------------------------------------------===//

    // The body is what the function used to be, entered with one node's worth
    // of state instead of the parameters it was called with. Everything the
    // stack needs goes in the block that is left behind, which runs once.
    const string body_name = insert_preheader(func, entry_name, {});
    auto body = make_block_map(func).at(body_name);

    // A recursive call in the entry block moved with it.
    if (recursive.erase(entry_name) > 0) {
        recursive.insert(body_name);
    }

    auto entry = func.blocks.front();
    const Type count_type = UInt_t::make(32);
    auto count_of = [&](uint64_t n) {
        return std::make_shared<Value>(Constant{count_type, n});
    };

    // One stack per varying parameter, all indexed by the same count. Keeping
    // them apart avoids having to invent a struct type to hold a node's worth
    // of state, and comes to the same thing.
    struct Stack {
        size_t param = 0;
        shared_ptr<Value> storage;
    };
    vector<Stack> stacks;
    // A run's push writes the waiting lanes to the stack with one compacting
    // store (below). Given a vector of slack past the stack's capacity, that
    // store may write the whole vector -- the lanes packed in a register and
    // one plain store (ir::Store::slack) -- rather than compact to memory,
    // which is 8 uops at a throughput of 3 cycles on Zen 5 and 88 uops on
    // Zen 4 (uops.info); the slots past the count hold nothing live. So each
    // stack is the widest run longer than the size asked for.
    uint32_t slack = 0;
    {
        const BlockMap blocks = make_block_map(func);
        for (const string &name : recursive) {
            const auto call = called_by(*blocks.at(name));
            if (!call) {
                continue;
            }
            if (call->sorted.has_value()) {
                slack = std::max(slack, call->sorted->lanes);
            }
            if (call->lanes.has_value()) {
                slack = std::max(slack, call->lanes->lanes);
            }
        }
    }
    for (size_t i = 0; i < params.size(); i++) {
        if (!varies[i]) {
            continue;
        }
        const Type array_type = Array_t::make(
            params[i].type, make_const(count_type, size + slack));
        stacks.push_back(
            {i, append(func, entry, array_type, Instruction::Op::Alloca, {},
                       new_storage_name(func, "!stack"))});
    }
    internal_assert(!stacks.empty())
        << "The recursion of " << entry_name << " passes the same arguments "
        << "every time, so it never ends";

    auto count =
        append(func, entry, Ptr_t::make(count_type), Instruction::Op::Alloca,
               {}, new_storage_name(func, "!count"));

    // The traversal starts at whatever the function was called with, held as
    // the current node rather than pushed: the stack is for what is waiting,
    // and nothing is yet.
    append_store(entry, count, count_of(0));

    //===----------------------------------------------------------------===//
    // The loop
    //===----------------------------------------------------------------===//
    //
    // The shape is pbrt's:
    //
    //     current = root; live = true
    //     while (live):
    //       body(current)
    //         a run of calls:  push all but the first; current = the first
    //         anything else:   if the stack is empty, live = false
    //                          else current = pop
    //
    // The current node is an argument of the loop header, and so is `live`,
    // which stands in for the `break` the statement form has no way to write:
    // the header tests it, the edge that has run out of stack passes false,
    // every other edge passes true, and LLVM's jump threading turns the
    // constant edges into direct branches. What this buys over pushing every
    // call and popping the next is a push and a pop fewer at every node with
    // children -- the near child never touches the stack -- and on a
    // traversal that is most of the nodes.

    // insert_preheader only carries what a loop changes, and this one has no
    // back edge yet, so the body takes no arguments: it names the entry's
    // parameters directly. The varying ones are the header's arguments
    // instead, under a new name, since the parameter is one value and the
    // current node is a fresh one every trip.
    auto head = new_block(func, "!visit");
    func.blocks.push_back(head);
    vector<shared_ptr<Value>> current;
    for (const Stack &stack : stacks) {
        const Argument &param = params[stack.param];
        current.push_back(
            head->add_argument(Argument{param.type, param.name + "!top"}));
    }
    auto live = head->add_argument(
        Argument{Bool_t::make(), new_storage_name(func, "!live")});
    auto exit = new_block(func, "!visited");
    func.blocks.push_back(exit);
    exit->terminator.data = Terminator::Return{};
    head->terminator.data = Terminator::Dispatch{
        live, {Terminator::Jump{exit->name}, Terminator::Jump{body_name}}};
    // With the header whole, since the rename walks the graph.
    for (size_t k = 0; k < stacks.size(); k++) {
        rename_argument(func, body_name, params[stacks[k].param].name,
                        head->args[k].name);
    }

    auto pop = new_block(func, "!visit_pop");
    func.blocks.push_back(pop);
    auto pop_next = new_block(func, "!visit_next");
    func.blocks.push_back(pop_next);
    auto pop_done = new_block(func, "!visit_done");
    func.blocks.push_back(pop_done);

    auto bool_of = [](bool b) {
        return std::make_shared<Value>(Constant{Bool_t::make(), b});
    };
    // Round again, with `nodes` as the current node.
    auto visit = [&](vector<shared_ptr<Value>> nodes,
                     shared_ptr<Value> alive) {
        Terminator::Jump jump{head->name, std::move(nodes)};
        jump.args.push_back(std::move(alive));
        return jump;
    };

    // The preheader enters the loop at the root.
    {
        vector<shared_ptr<Value>> root;
        for (const Stack &stack : stacks) {
            root.push_back(std::make_shared<Value>(params[stack.param]));
        }
        entry->terminator.data = visit(std::move(root), bool_of(true));
    }

    // Take the top of the stack, if there is one -- and, if the traversal is
    // accumulating a boolean it only ever moves one way, only while that has
    // not reached the end of its lattice. The second half is what makes `any`
    // mean "stop when you find one" rather than "keep looking but ignore what
    // you find", and `all` likewise.
    const std::optional<Monotone> acc = find_monotone_accumulator(func);
    // Whether the accumulator has yet to settle, read in `block`. A gang's
    // accumulator is one bool per lane, and the gang is done only when every
    // lane is: it goes on while any lane is undecided.
    auto undecided_in = [&](const shared_ptr<Block> &block) {
        const Type held = acc->ptr->get_type().as<Ptr_t>()->etype;
        auto now = append(func, block, held, Instruction::Op::Load, {acc->ptr});
        if (acc->rising) {
            now = append(func, block, held, Instruction::Op::Not, {now});
        }
        if (held.is_vector()) {
            now = append(func, block, Bool_t::make(), Instruction::Op::Any,
                         {now});
        }
        return now;
    };
    auto height = append(func, pop, count_type, Instruction::Op::Load, {count});
    auto more = append(func, pop, Bool_t::make(), Instruction::Op::Ne,
                       {height, count_of(0)});
    if (acc.has_value()) {
        more = append(func, pop, Bool_t::make(), Instruction::Op::LAnd,
                      {more, undecided_in(pop)});
    }
    pop->terminator.data = Terminator::Dispatch{
        more,
        {Terminator::Jump{pop_done->name}, Terminator::Jump{pop_next->name}}};

    // Nothing left to visit: the current node stays what it was, and the loop
    // ends at the header.
    pop_done->terminator.data = visit(current, bool_of(false));

    auto next = append(func, pop_next, count_type, Instruction::Op::Sub,
                       {height, count_of(1)});
    append_store(pop_next, count, next);
    {
        vector<shared_ptr<Value>> popped;
        for (const Stack &stack : stacks) {
            popped.push_back(append(func, pop_next, params[stack.param].type,
                                    Instruction::Op::ExtractIdx,
                                    {stack.storage, next}));
        }
        pop_next->terminator.data = visit(std::move(popped), bool_of(true));
    }

    //===----------------------------------------------------------------===//
    // The calls that are no longer made
    //===----------------------------------------------------------------===//

    const BlockMap blocks = make_block_map(func);

    // Whether anything on a path from the body to `block`, `block` included,
    // stores to the accumulator. Walked before the returns become back edges,
    // so the walk stops where the body begins.
    const Cfg cfg(func);
    const DomTree dom = compute_dominator_tree(cfg);
    const auto stores_accumulator = [&](const Block &block) {
        internal_assert(acc.has_value());
        const std::optional<string> target = name_of(acc->ptr);
        for (const auto &instr : block.instrs) {
            if (instr->op == Instruction::Op::Store &&
                !instr->operands.empty() &&
                name_of(instr->operands[0]) == target) {
                return true;
            }
        }
        return false;
    };
    const auto may_settle_before = [&](const string &name) {
        set<string> seen;
        vector<string> work = {name};
        while (!work.empty()) {
            const string at = work.back();
            work.pop_back();
            if (!seen.insert(at).second) {
                continue;
            }
            if (stores_accumulator(*blocks.at(at))) {
                return true;
            }
            if (at == body_name) {
                continue;
            }
            const BlockId at_id = cfg.find(at);
            if (at_id == NO_BLOCK) {
                continue;
            }
            for (BlockId pred : cfg.preds[at_id]) {
                work.push_back(cfg.name(pred));
            }
        }
        return false;
    };

    for (const string &name : recursive) {
        auto block = blocks.at(name);
        const auto call = called_by(*block);
        internal_assert(call);

        // A run whose continuation only returns is the last thing this node
        // does, so its first call can simply be made: the current node
        // becomes that child, and only the rest wait on the stack. Anything
        // else after the run has to happen before any of the calls, so then
        // all of them wait.
        //
        // "Only returns" through the chain of empty merge blocks the arms of
        // a match leave behind on their way to the function's one return.
        const auto only_returns = [&](string at) {
            set<string> seen;
            for (;;) {
                if (!seen.insert(at).second) {
                    return false;
                }
                const Block &b = *blocks.at(at);
                if (!b.instrs.empty() || !b.args.empty()) {
                    return false;
                }
                if (std::holds_alternative<Terminator::Return>(
                        b.terminator.data)) {
                    return true;
                }
                const auto *j = std::get_if<Terminator::Jump>(&b.terminator.data);
                if (j == nullptr || !j->args.empty()) {
                    return false;
                }
                at = j->name;
            }
        };
        const bool last =
            call->cont.args.empty() && only_returns(call->cont.name);
        const size_t waiting_from = last ? 1 : 0;

        // A gang's recursion is made only if some lane wants it. As a call
        // that was a test in front of the call (see specialize_calls in
        // SSA/Vectorize.cpp), so a run no lane was on was never made. Once
        // the calls are pushes the test is made here: the pushes and the
        // descent go in a block of their own, entered when any lane is on
        // and bypassed straight to what followed the run otherwise. The arm
        // the run sits in is normally behind the linearizer's own test of
        // the same mask (the BOSCC gadget, SSA/Linearize.h), and then the
        // mask is known to have a lane on and no test is made; so is the
        // function's own mask, since a masked variant is only ever called
        // behind a test of the mask it is handed. This one stands for a run
        // the linearizer could not skip. The mask is the argument every call
        // of the run passes for the parameter specialize() added under the
        // name `!mask` (SSA/Vectorize.cpp).
        shared_ptr<Block> into = block;
        const auto nonempty = [&](const Value &mask) {
            return same_value(mask, Value(params[*mask_param])) ||
                   known_nonempty(cfg, dom, cfg.id(name), mask);
        };
        if (mask_param.has_value() && !nonempty(*call->args[0][*mask_param])) {
            const shared_ptr<Value> wanted = call->args[0][*mask_param];
            auto on = append(func, block, Bool_t::make(), Instruction::Op::Any,
                             {wanted});
            into = new_block(func, name + "!push");
            func.blocks.push_back(into);
            // targets[0] is where a false condition goes: past the run, to
            // its continuation, which for a run that was the last thing the
            // node did is the return that becomes the pop below.
            block->terminator.data = Terminator::Dispatch{
                on, {call->cont, Terminator::Jump{into->name}}};
        }

        // Write down what each waiting call would have been, on top of the
        // stack.
        //
        // In reverse, because the stack is last-in-first-out and `args` is in
        // visit order: pushing the earliest waiting call last leaves it on
        // top, so it is the one the next pop takes. That is what makes
        // `from (a, b)` mean "visit a, then b" -- and it is what a sort()
        // before this is relying on, since a sort orders `args` by its key
        // and expects the traversal to follow that order.
        //
        // The old lowering had no say in this. It made one Call per branch in
        // a chain of blocks, so each pushed in turn and the *last* branch came
        // off the stack first -- `from (a, b)` visited b before a, and there
        // was nothing at this level that could have said otherwise.
        //
        // A branch with a condition (RecursiveCall::conds) is pushed only
        // where it holds: the push goes in a block of its own, entered on the
        // condition and bypassed otherwise, so a child whose box the ray
        // missed costs the test and nothing more. This is Embree's traversal
        // of a node whose children's boxes it holds -- test the children,
        // push the hits -- and it is what pruning *at the parent* has to
        // mean once the recursion is a stack: a child that is not visited is
        // one that is never written down.
        //
        // A run sort_recursion() ordered as vectors (SortedRun, see
        // SSA/SortRecursion.h) -- the run of a node that holds its children's
        // boxes, its hits sorted and its children held as one vector per
        // varying parameter, the nearest in the last lane and the children
        // not hit in the lowest -- is written down with one compacting store
        // per stack rather than a conditional push per child: the lanes of
        // the hits that wait, farthest first, go into consecutive slots from
        // the top (packed in a register, `vpcompressq`, and stored whole
        // into the stack's slack; ir::Store::slack), the count advances
        // once by their number, and the nearest is descended into.
        // Where no child is hit there is nothing to write and nothing to
        // descend into, and the next node comes off the stack: Embree's `if
        // (mask == 0) goto pop` in front of its traverseClosestHit.
        if (call->sorted.has_value()) {
            const SortedRun &run = *call->sorted;
            const uint32_t n = run.lanes;
            const Type bool_t = Bool_t::make();
            const Type lanes_t = Vector_t::make(count_type, n);
            const Type mask_t = Vector_t::make(bool_t, n);
            auto made = append(func, into, bool_t, Instruction::Op::Ne,
                               {run.hits, count_of(0)});
            auto push = new_block(func, name + "!push");
            func.blocks.push_back(push);
            into->terminator.data = Terminator::Dispatch{
                made,
                {last ? Terminator::Jump{pop->name} : call->cont,
                 Terminator::Jump{push->name}}};

            auto top =
                append(func, push, count_type, Instruction::Op::Load, {count});
            // The lanes that wait: the hits are the last `hits` lanes, and
            // the nearest of them, the last lane, is not written but visited.
            auto lane = append(func, push, lanes_t, Instruction::Op::Ramp,
                               {count_of(0), count_of(1)});
            auto first = append(func, push, count_type, Instruction::Op::Sub,
                                {count_of(n), run.hits});
            auto from = append(func, push, lanes_t, Instruction::Op::Bc,
                               {first, count_of(n)});
            shared_ptr<Value> waiting =
                append(func, push, mask_t, Instruction::Op::Leq, {from, lane});
            shared_ptr<Value> added = run.hits;
            if (last) {
                vector<shared_ptr<Value>> below;
                for (uint32_t i = 0; i < n; i++) {
                    below.push_back(
                        std::make_shared<Value>(Constant{bool_t, i + 1 < n}));
                }
                auto not_nearest = append(func, push, mask_t,
                                          Instruction::Op::MakeStruct,
                                          std::move(below));
                waiting = append(func, push, mask_t, Instruction::Op::LAnd,
                                 {waiting, not_nearest});
                added = append(func, push, count_type, Instruction::Op::Sub,
                               {run.hits, count_of(1)});
            }
            for (const Stack &stack : stacks) {
                const auto values = run.values.find(stack.param);
                internal_assert(values != run.values.end())
                    << "The sorted run in " << name << " has no vector for "
                    << params[stack.param].name << ", which varies";
                auto slot =
                    append(func, push, Ptr_t::make(params[stack.param].type),
                           Instruction::Op::GEP, {stack.storage, top});
                auto store = std::make_shared<Instruction>(
                    Instruction::Op::Store,
                    vector<shared_ptr<Value>>{slot, values->second, waiting},
                    push);
                store->compact = true;
                // The stack has a vector of slack past its capacity (see
                // its allocation), so the whole vector may be written.
                store->slack = true;
                push->instrs.push_back(store);
            }
            append_store(push, count,
                         append(func, push, count_type, Instruction::Op::Add,
                                {top, added}));
            if (!last) {
                push->terminator.data = call->cont;
                continue;
            }
            shared_ptr<Value> alive = bool_of(true);
            if (acc.has_value() && may_settle_before(name)) {
                alive = undecided_in(push);
            }
            vector<shared_ptr<Value>> nearest;
            for (const Stack &stack : stacks) {
                nearest.push_back(append(
                    func, push, params[stack.param].type,
                    Instruction::Op::ExtractIdx,
                    {run.values.at(stack.param), count_of(n - 1)}));
            }
            push->terminator.data = visit(std::move(nearest), std::move(alive));
            continue;
        }

        // A run that is the lanes of vectors with no order asked for
        // (LaneRun, see SSA/SortRecursion.h): a node's children under their
        // mask, a quantifier's run. Written down the same way, in the order
        // the children are stored: Embree's traverseAnyHit takes the hits in
        // sequence, pushing each but the last and continuing with that one,
        // so the hit in the highest lane is descended into and the others go
        // to the stack by one compacting store, lane order preserved. A run
        // that is not the node's last act pushes every hit.
        if (call->lanes.has_value()) {
            const LaneRun &run = *call->lanes;
            const uint32_t n = run.lanes;
            const Type bool_t = Bool_t::make();
            const Type lanes_t = Vector_t::make(count_type, n);
            const Type mask_t = Vector_t::make(bool_t, n);
            auto mask = lanes_as_vector(func, into, run.mask, n);
            auto hits = append(func, into, count_type, Instruction::Op::Popcount,
                               {mask});
            auto push = new_block(func, name + "!push");
            func.blocks.push_back(push);
            if (last) {
                // A switch on the count rather than a test for any: one hit
                // -- three node tests in five on incoherent rays -- has
                // nothing to write down, and its one child is lane 0 of the
                // vectors packed by the mask (one `vpcompressq` and a
                // `vmovq`; Embree's traverseAnyHit takes it by `tzcnt` and a
                // load), not the highest lane taken out of a vector at a
                // run-time index behind a compacting store of nothing.
                auto one = new_block(func, name + "!one");
                func.blocks.push_back(one);
                vector<shared_ptr<Value>> only;
                for (const Stack &stack : stacks) {
                    const auto source = run.values.find(stack.param);
                    internal_assert(source != run.values.end())
                        << "The run in " << name << " has no lanes for "
                        << params[stack.param].name << ", which varies";
                    auto vec = lanes_as_vector(func, one, source->second, n);
                    auto packed = std::make_shared<Instruction>(
                        func.get_unique_name(), vec->get_type(),
                        Instruction::Op::Intrinsic,
                        vector<shared_ptr<Value>>{vec, mask}, one);
                    packed->intrinsic = ir::Intrinsic::compress;
                    one->instrs.push_back(packed);
                    only.push_back(append(
                        func, one, params[stack.param].type,
                        Instruction::Op::ExtractIdx,
                        {std::make_shared<Value>(packed), count_of(0)}));
                }
                shared_ptr<Value> alive = bool_of(true);
                if (acc.has_value() && may_settle_before(name)) {
                    alive = undecided_in(one);
                }
                one->terminator.data = visit(std::move(only), std::move(alive));
                into->terminator.data = Terminator::Dispatch{
                    hits,
                    {Terminator::Jump{pop->name}, Terminator::Jump{one->name},
                     Terminator::Jump{push->name}}};
            } else {
                auto made = append(func, into, bool_t, Instruction::Op::Ne,
                                   {hits, count_of(0)});
                into->terminator.data = Terminator::Dispatch{
                    made, {call->cont, Terminator::Jump{push->name}}};
            }

            auto top =
                append(func, push, count_type, Instruction::Op::Load, {count});
            shared_ptr<Value> waiting = mask;
            shared_ptr<Value> added = hits;
            shared_ptr<Value> picked; // the lane continued with, if any
            if (last) {
                // The highest lane the mask has on. Where the mask's lanes
                // are the bits of an integer, that is the mask as that
                // integer with its leading zeros counted (`kmov`, `lzcnt`:
                // Embree's `bsr` on its movemask); otherwise the highest of
                // the lane indices the mask keeps.
                auto lane = append(func, push, lanes_t, Instruction::Op::Ramp,
                                   {count_of(0), count_of(1)});
                if (n == 8 || n == 16 || n == 32 || n == 64) {
                    const Type bits_t = UInt_t::make(n);
                    auto bits = append(func, push, bits_t,
                                       Instruction::Op::Reinterpret, {mask});
                    auto zeros = std::make_shared<Instruction>(
                        func.get_unique_name(), bits_t,
                        Instruction::Op::Intrinsic,
                        vector<shared_ptr<Value>>{bits}, push);
                    zeros->intrinsic = ir::Intrinsic::clz;
                    push->instrs.push_back(zeros);
                    auto counted =
                        append(func, push, count_type, Instruction::Op::Cast,
                               {std::make_shared<Value>(zeros)});
                    picked = append(func, push, count_type, Instruction::Op::Sub,
                                    {count_of(n - 1), counted});
                } else {
                    auto none = append(func, push, lanes_t, Instruction::Op::Bc,
                                       {count_of(0), count_of(n)});
                    auto on = append(func, push, lanes_t,
                                     Instruction::Op::Select, {mask, lane, none});
                    auto highest = std::make_shared<Instruction>(
                        func.get_unique_name(), count_type,
                        Instruction::Op::Reduce, vector<shared_ptr<Value>>{on},
                        push);
                    highest->reduce = ir::VectorReduce::Max;
                    push->instrs.push_back(highest);
                    picked = std::make_shared<Value>(highest);
                }
                auto others = append(func, push, mask_t, Instruction::Op::Ne,
                                     {lane, append(func, push, lanes_t,
                                                   Instruction::Op::Bc,
                                                   {picked, count_of(n)})});
                waiting = append(func, push, mask_t, Instruction::Op::LAnd,
                                 {mask, others});
                added = append(func, push, count_type, Instruction::Op::Sub,
                               {hits, count_of(1)});
            }
            std::map<size_t, shared_ptr<Value>> vectors;
            for (const Stack &stack : stacks) {
                const auto source = run.values.find(stack.param);
                internal_assert(source != run.values.end())
                    << "The run in " << name << " has no lanes for "
                    << params[stack.param].name << ", which varies";
                vectors[stack.param] =
                    lanes_as_vector(func, push, source->second, n);
                auto slot =
                    append(func, push, Ptr_t::make(params[stack.param].type),
                           Instruction::Op::GEP, {stack.storage, top});
                auto store = std::make_shared<Instruction>(
                    Instruction::Op::Store,
                    vector<shared_ptr<Value>>{slot, vectors[stack.param],
                                              waiting},
                    push);
                store->compact = true;
                store->slack = true; // as above
                push->instrs.push_back(store);
            }
            append_store(push, count,
                         append(func, push, count_type, Instruction::Op::Add,
                                {top, added}));
            if (!last) {
                push->terminator.data = call->cont;
                continue;
            }
            shared_ptr<Value> alive = bool_of(true);
            if (acc.has_value() && may_settle_before(name)) {
                alive = undecided_in(push);
            }
            vector<shared_ptr<Value>> next;
            for (const Stack &stack : stacks) {
                next.push_back(append(func, push, params[stack.param].type,
                                      Instruction::Op::ExtractIdx,
                                      {vectors.at(stack.param), picked}));
            }
            push->terminator.data = visit(std::move(next), std::move(alive));
            continue;
        }

        const auto condition = [&](size_t i) -> shared_ptr<Value> {
            if (call->conds.empty() || always(call->conds[i])) {
                return nullptr;
            }
            return call->conds[i];
        };
        for (size_t i = call->args.size(); i-- > waiting_from;) {
            shared_ptr<Block> push = into;
            shared_ptr<Block> pushed;
            if (const shared_ptr<Value> cond = condition(i)) {
                push = new_block(func, name + "!push" + std::to_string(i));
                pushed = new_block(func, name + "!pushed" + std::to_string(i));
                func.blocks.push_back(push);
                func.blocks.push_back(pushed);
                into->terminator.data = Terminator::Dispatch{
                    cond,
                    {Terminator::Jump{pushed->name}, Terminator::Jump{push->name}}};
            }
            auto top =
                append(func, push, count_type, Instruction::Op::Load, {count});
            for (const Stack &stack : stacks) {
                auto slot =
                    append(func, push, Ptr_t::make(params[stack.param].type),
                           Instruction::Op::GEP, {stack.storage, top});
                append_store(push, slot, call->args[i][stack.param]);
            }
            append_store(push, count,
                         append(func, push, count_type, Instruction::Op::Add,
                                {top, count_of(1)}));
            if (pushed) {
                push->terminator.data = Terminator::Jump{pushed->name};
                into = pushed;
            }
        }

        if (!last) {
            // ...and carry straight on to what came after them.
            into->terminator.data = call->cont;
            continue;
        }

        // Straight to the first child. This edge does not pass the pop, which
        // is where a settled accumulator ends the traversal; if nothing on
        // the way here could have moved it, its state is what the pop last
        // found and the edge stays live, and otherwise it is asked here.
        shared_ptr<Value> alive = bool_of(true);
        if (acc.has_value() && may_settle_before(name)) {
            alive = undecided_in(into);
        }
        vector<shared_ptr<Value>> child;
        for (const Stack &stack : stacks) {
            child.push_back(call->args[0][stack.param]);
        }
        Terminator::Jump descend = visit(std::move(child), std::move(alive));
        // A first child under a condition is descended into where the
        // condition holds, and where it does not the node is done: the next
        // one comes off the stack, which is where a return went.
        if (const shared_ptr<Value> cond = condition(0)) {
            into->terminator.data = Terminator::Dispatch{
                cond, {Terminator::Jump{pop->name}, std::move(descend)}};
        } else {
            into->terminator.data = std::move(descend);
        }
    }

    // Returning from a visit is the end of that node, not of the traversal:
    // the next one comes off the stack. The exit's own return is the
    // traversal's, and stays.
    const Cfg visit_region(func, body_name);
    for (const auto &reached : visit_region.blocks()) {
        // Only the blocks the visit was made of; the ones added above are
        // the stack's, and end where they mean to.
        const auto found = blocks.find(reached->name);
        const auto block = found == blocks.end() ? nullptr : found->second;
        if (block == nullptr || block == exit) {
            continue;
        }
        if (std::holds_alternative<Terminator::Return>(
                block->terminator.data)) {
            block->terminator.data = Terminator::Jump{pop->name};
        }
    }

    refresh_preds(func);
    // The runs are gone, and with them the only readers of what was made for
    // the calls alone: a sorted run's lanes and conditions (SortedRun).
    remove_dead(func);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
