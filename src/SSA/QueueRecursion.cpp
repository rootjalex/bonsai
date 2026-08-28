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

} // namespace

void queue_recursion(Function &func, size_t size) {
    internal_assert(!func.blocks.empty()) << "Loopifying an empty function";
    const string entry_name = func.blocks.front()->name;

    // Where the function calls itself.
    set<string> recursive;
    {
        const BlockMap blocks = make_block_map(func);
        for (const string &name : reachable_from(entry_name,
                                                 compute_successors(func))) {
            const auto *call =
                std::get_if<Terminator::Call>(&blocks.at(name)->terminator.data);
            if (call != nullptr && call->call.name == entry_name) {
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
            const auto &call =
                std::get<Terminator::Call>(blocks.at(name)->terminator.data);
            internal_assert(call.call.args.size() == params.size())
                << "Recursive call in " << name << " passes "
                << call.call.args.size() << " arguments to a function taking "
                << params.size();
            for (size_t i = 0; i < params.size(); i++) {
                varies[i] = varies[i] ||
                            !is_named_argument(*call.call.args[i],
                                               params[i].name);
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
            const auto &call =
                std::get<Terminator::Call>(blocks.at(name)->terminator.data);
            internal_assert(call.drop)
                << "The recursive call in " << name << " keeps its result, "
                << "which a deferred call has not got";
            for (const string &after : reachable_from(call.cont.name, succs)) {
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
                const auto *other =
                    std::get_if<Terminator::Call>(&block.terminator.data);
                internal_assert(other == nullptr ||
                                other->call.name == entry_name)
                    << "Cannot put the recursion of " << entry_name
                    << " on a stack: " << after << " calls "
                    << other->call.name << " after the recursive call in "
                    << name << ", which would be deferred past it";
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

    auto count = append(func, entry, Ptr_t::make(count_type),
                        Instruction::Op::Alloca, {},
                        new_storage_name(func, "!count"));

    // The traversal starts at whatever the function was called with.
    for (const Stack &stack : stacks) {
        auto slot = append(func, entry, Ptr_t::make(params[stack.param].type),
                           Instruction::Op::GEP, {stack.storage, count_of(0)});
        append_store(entry, slot,
                     std::make_shared<Value>(params[stack.param]));
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

    // Go round again as long as anything is left to visit.
    auto left = append(func, head, count_type, Instruction::Op::Load, {count});
    auto more = append(func, head, Bool_t::make(), Instruction::Op::Ne,
                       {left, count_of(0)});
    head->terminator.data = Terminator::Dispatch{
        more,
        {Terminator::Jump{exit->name}, Terminator::Jump{pop->name}}};

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
        const auto call =
            std::get<Terminator::Call>(block->terminator.data);

        // Write down what the call would have been, on top of the stack.
        auto top = append(func, block, count_type, Instruction::Op::Load,
                          {count});
        for (const Stack &stack : stacks) {
            auto slot =
                append(func, block, Ptr_t::make(params[stack.param].type),
                       Instruction::Op::GEP, {stack.storage, top});
            append_store(block, slot, call.call.args[stack.param]);
        }
        append_store(block, count,
                     append(func, block, count_type, Instruction::Op::Add,
                            {top, count_of(1)}));

        // ...and carry straight on to what came after it.
        block->terminator.data = call.cont;
    }

    // Returning from a visit is the end of that node, not of the traversal.
    for (const string &name : reachable_from(body_name,
                                             compute_successors(func))) {
        auto block = blocks.count(name) ? blocks.at(name) : nullptr;
        if (block == nullptr) {
            continue;
        }
        if (std::holds_alternative<Terminator::Return>(block->terminator.data)) {
            block->terminator.data = Terminator::Jump{head->name};
        }
    }

    refresh_preds(func);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
