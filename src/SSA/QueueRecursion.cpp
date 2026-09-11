#include "SSA/QueueRecursion.h"

#include "SSA/Analysis.h"
#include "SSA/InsertPreheader.h"

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
    AdjacencyMap preds;

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

// Whether `v` is the boolean constant `want`.
bool is_bool_const(const shared_ptr<Value> &v, bool want) {
    if (!v) {
        return false;
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
            if (instr->operands.size() != 2) {
                continue;
            }
            const shared_ptr<Value> &dest = instr->operands[0];
            if (!dest) {
                continue;
            }
            const Ptr_t *ptr_type = dest->get_type().as<Ptr_t>();
            if (ptr_type == nullptr || !ptr_type->etype.is<Bool_t>()) {
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
    Terminator::Jump cont;
    bool drop;
};

std::optional<RecursiveCall> called_by(const Block &block) {
    if (const auto *c = std::get_if<Terminator::Call>(&block.terminator.data)) {
        return RecursiveCall{c->call.name, {c->call.args}, c->cont, c->drop};
    }
    if (const auto *c =
            std::get_if<Terminator::MultiCall>(&block.terminator.data)) {
        vector<vector<shared_ptr<Value>>> args;
        args.reserve(c->varying.size());
        for (size_t i = 0; i < c->varying.size(); i++) {
            args.push_back(c->call_args(i));
        }
        return RecursiveCall{c->call.name, std::move(args), c->cont, c->drop};
    }
    return std::nullopt;
}

} // namespace

void queue_recursion(Function &func, size_t size) {
    internal_assert(!func.blocks.empty()) << "Loopifying an empty function";
    const string entry_name = func.blocks.front()->name;

    // Where the function calls itself.
    set<string> recursive;
    {
        const BlockMap blocks = make_block_map(func);
        for (const string &name :
             reachable_from(entry_name, compute_successors(func))) {
            const auto call = called_by(*blocks.at(name));
            if (call && call->callee == entry_name) {
                recursive.insert(name);
            }
        }
    }
    if (recursive.empty()) {
        return; // nothing to unrecurse
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
        const BlockMap blocks = make_block_map(func);
        const AdjacencyMap succs = compute_successors(func);
        for (const string &name : recursive) {
            const auto call = called_by(*blocks.at(name));
            internal_assert(call);
            internal_assert(call->drop)
                << "The recursive call in " << name << " keeps its result, "
                << "which a deferred call has not got";
            for (const string &after : reachable_from(call->cont.name, succs)) {
                const Block &block = *blocks.at(after);
                for (const auto &instr : block.instrs) {
                    internal_assert(instr->op != Instruction::Op::Store &&
                                    instr->op != Instruction::Op::Print &&
                                    instr->op != Instruction::Op::Append)
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
    for (size_t i = 0; i < params.size(); i++) {
        if (!varies[i]) {
            continue;
        }
        const Type array_type =
            Array_t::make(params[i].type, make_const(count_type, size));
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

    // The traversal starts at whatever the function was called with.
    for (const Stack &stack : stacks) {
        auto slot = append(func, entry, Ptr_t::make(params[stack.param].type),
                           Instruction::Op::GEP, {stack.storage, count_of(0)});
        append_store(entry, slot, std::make_shared<Value>(params[stack.param]));
    }
    append_store(entry, count, count_of(1));

    //===----------------------------------------------------------------===//
    // The loop
    //===----------------------------------------------------------------===//

    // insert_preheader only carries what a loop changes, and this one has no
    // back edge yet, so the body takes no arguments: it names the entry's
    // parameters directly. The varying ones have to come off the stack
    // instead, which is what the arguments added here are for.
    vector<Argument> popped;
    for (const Stack &stack : stacks) {
        const Argument &param = params[stack.param];
        const Argument arg{param.type, param.name + "!top"};
        rename_argument(func, body_name, param.name, arg.name);
        body->args.push_back(arg);
        body->lookups[arg.name] = std::make_shared<Value>(arg);
        popped.push_back(arg);
    }

    auto head = new_block(func, "!visit");
    func.blocks.push_back(head);
    auto pop = new_block(func, "!visit_pop");
    func.blocks.push_back(pop);
    auto exit = new_block(func, "!visited");
    func.blocks.push_back(exit);
    exit->terminator.data = Terminator::Return{};

    // The preheader falls into the loop rather than into the body.
    entry->terminator.data = Terminator::Jump{head->name};

    // Go round again as long as anything is left to visit -- and, if the
    // traversal is accumulating a boolean it only ever moves one way, as long
    // as that has not reached the end of its lattice. The second half is what
    // makes `any` mean "stop when you find one" rather than "keep looking but
    // ignore what you find", and `all` likewise.
    auto left = append(func, head, count_type, Instruction::Op::Load, {count});
    auto more = append(func, head, Bool_t::make(), Instruction::Op::Ne,
                       {left, count_of(0)});
    if (std::optional<Monotone> acc = find_monotone_accumulator(func)) {
        auto now =
            append(func, head, Bool_t::make(), Instruction::Op::Load, {acc->ptr});
        auto undecided =
            acc->rising ? append(func, head, Bool_t::make(),
                                 Instruction::Op::Not, {now})
                        : now;
        more = append(func, head, Bool_t::make(), Instruction::Op::LAnd,
                      {more, undecided});
    }
    head->terminator.data = Terminator::Dispatch{
        more, {Terminator::Jump{exit->name}, Terminator::Jump{pop->name}}};

    // Take the top of the stack. Separate from the test above so that the
    // count is only stepped when there is something to step it for.
    auto height = append(func, pop, count_type, Instruction::Op::Load, {count});
    auto next = append(func, pop, count_type, Instruction::Op::Sub,
                       {height, count_of(1)});
    append_store(pop, count, next);
    Terminator::Jump into{body_name};
    for (const Stack &stack : stacks) {
        into.args.push_back(append(func, pop, params[stack.param].type,
                                   Instruction::Op::ExtractIdx,
                                   {stack.storage, next}));
    }
    pop->terminator.data = std::move(into);

    //===----------------------------------------------------------------===//
    // The calls that are no longer made
    //===----------------------------------------------------------------===//

    const BlockMap blocks = make_block_map(func);
    for (const string &name : recursive) {
        auto block = blocks.at(name);
        const auto call = called_by(*block);
        internal_assert(call);

        // Write down what each call would have been, on top of the stack.
        //
        // In reverse, because the stack is last-in-first-out and `args` is in
        // visit order: pushing the first call last leaves it on top, so it is
        // the one the next pop takes. That is what makes `from (a, b)` mean
        // "visit a, then b" -- and it is what a sort() before this is relying
        // on, since a sort orders `args` by its key and expects the traversal
        // to follow that order.
        //
        // The old lowering had no say in this. It made one Call per branch in
        // a chain of blocks, so each pushed in turn and the *last* branch came
        // off the stack first -- `from (a, b)` visited b before a, and there
        // was nothing at this level that could have said otherwise.
        for (size_t i = call->args.size(); i-- > 0;) {
            auto top =
                append(func, block, count_type, Instruction::Op::Load, {count});
            for (const Stack &stack : stacks) {
                auto slot =
                    append(func, block, Ptr_t::make(params[stack.param].type),
                           Instruction::Op::GEP, {stack.storage, top});
                append_store(block, slot, call->args[i][stack.param]);
            }
            append_store(block, count,
                         append(func, block, count_type, Instruction::Op::Add,
                                {top, count_of(1)}));
        }

        // ...and carry straight on to what came after them.
        block->terminator.data = call->cont;
    }

    // Returning from a visit is the end of that node, not of the traversal.
    for (const string &name :
         reachable_from(body_name, compute_successors(func))) {
        auto block = blocks.count(name) ? blocks.at(name) : nullptr;
        if (block == nullptr) {
            continue;
        }
        if (std::holds_alternative<Terminator::Return>(
                block->terminator.data)) {
            block->terminator.data = Terminator::Jump{head->name};
        }
    }

    refresh_preds(func);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
