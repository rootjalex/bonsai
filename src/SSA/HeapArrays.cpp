#include "SSA/HeapArrays.h"

#include "IR/Analysis.h"
#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::string;
using std::vector;

// Whether a value of `type` holds a reference to storage: a pointer, an
// array handle, or an aggregate with one inside.
bool carries_reference(const Type &type) {
    if (type.is<Ptr_t>() || type.is_reference()) {
        return true;
    }
    if (const auto *s = type.as<Struct_t>()) {
        for (const auto &field : s->fields) {
            if (carries_reference(field.type)) {
                return true;
            }
        }
    }
    if (const auto *t = type.as<Tuple_t>()) {
        for (const auto &etype : t->etypes) {
            if (carries_reference(etype)) {
                return true;
            }
        }
    }
    if (const auto *o = type.as<Option_t>()) {
        return carries_reference(o->etype);
    }
    if (const auto *v = type.as<Vector_t>()) {
        return carries_reference(v->etype);
    }
    return false;
}

// The instruction a value is, as the block named `block` refers to it: the
// instruction itself, or the one an argument is threaded from
// (SSA/Definitions.h). Null for a constant, a parameter of the function, or
// a merge of different values.
const Instruction *instruction_of(Definitions &defs, const string &block,
                                  const shared_ptr<Value> &v) {
    const Definition d = defs.of(block, v);
    if (!d.value) {
        return nullptr;
    }
    const auto *held = std::get_if<shared_ptr<Instruction>>(&d.value->data);
    return held == nullptr ? nullptr : held->get();
}

// Whether a value, as `block` refers to it, is a merge of different values
// -- an argument the block's predecessors disagree about -- rather than a
// parameter or a constant, once it is not an instruction.
bool is_merge(Definitions &defs, const Function &func, const string &block,
              const shared_ptr<Value> &v) {
    const Definition d = defs.of(block, v);
    return d.value && std::holds_alternative<Argument>(d.value->data) &&
           !d.block.empty() && d.block != func.blocks.front()->name;
}

// The allocation of this function a reference-carrying value derives from,
// following the pure instructions that make one value of another -- an
// address into it, an aggregate holding it -- and the arguments that thread
// it, or null when it derives from something else: a parameter, a load, a
// merge, which may be any storage at all.
const Instruction *storage_of(Definitions &defs, const string &block,
                              const shared_ptr<Value> &v) {
    const Instruction *in = instruction_of(defs, block, v);
    if (in == nullptr) {
        return nullptr;
    }
    const string home = in->owner.lock() ? in->owner.lock()->name : block;
    switch (in->op) {
    case Instruction::Op::Alloca:
    case Instruction::Op::Alloc:
        return in;
    case Instruction::Op::GEP:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::AddressOf:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Cast:
        return in->operands.empty() ? nullptr : storage_of(defs, home, in->operands[0]);
    default:
        return nullptr;
    }
}

// Every allocation of this function that a reference-carrying value may
// hold a reference into; `unknown` when it may hold one into storage it
// cannot tell -- a load's value, a call's, a merge of different values. A
// parameter of the function holds references into no storage of its own.
void storages_in(Definitions &defs, const Function &func, const string &block,
                 const shared_ptr<Value> &v, std::set<const Instruction *> &out,
                 bool &unknown, std::set<const Instruction *> &seen) {
    if (!carries_reference(v->get_type())) {
        return;
    }
    const Instruction *in = instruction_of(defs, block, v);
    if (in == nullptr) {
        if (is_merge(defs, func, block, v)) {
            unknown = true;
        }
        return;
    }
    if (!seen.insert(in).second) {
        return;
    }
    const string home = in->owner.lock() ? in->owner.lock()->name : block;
    switch (in->op) {
    case Instruction::Op::Alloca:
    case Instruction::Op::Alloc:
        out.insert(in);
        return;
    case Instruction::Op::GEP:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::AddressOf:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Cast:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Select:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::LoadField:
        for (const auto &operand : in->operands) {
            storages_in(defs, func, home, operand, out, unknown, seen);
        }
        return;
    default:
        unknown = true; // a load, a call's value: any storage
        return;
    }
}

} // namespace

size_t heap_arrays(Function &func, bool in_loop) {
    if (func.blocks.empty() || in_loop) {
        return 0;
    }

    // Where a reference into this function's storage may leave the call: a
    // return of one, or a store of one into memory the function did not
    // allocate (a parameter's buffer, storage reached through a pointer).
    // `escaped` names the allocations such a reference may be into; `any`
    // says one of unknown provenance leaves, which may be into any of them.
    // A reference passed to a callee is trusted to come back, as the
    // promotion of allocas trusts it (SSA/PromoteAllocas.h); one stored into
    // another allocation of this function goes when that does.
    // BONSAI_EXPLAIN_HEAP=1 says what kept each array on the stack.
    const bool explain = std::getenv("BONSAI_EXPLAIN_HEAP") != nullptr;
    const string fname = func.blocks.front()->name;
    std::set<const Instruction *> escaped;
    bool any = false;
    Definitions defs(func, /*lenient=*/true);
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Store || instr->operands.size() < 2 ||
                !carries_reference(instr->operands[1]->get_type())) {
                continue;
            }
            if (storage_of(defs, block->name, instr->operands[0]) != nullptr) {
                continue; // into storage of this function
            }
            std::set<const Instruction *> seen;
            const bool was = any;
            storages_in(defs, func, block->name, instr->operands[1], escaped, any, seen);
            if (explain && any && !was) {
                std::cerr << "; heap_arrays(" << fname << "): a reference of "
                          << "unknown provenance is stored in " << block->name
                          << " (";
                instr->dump(std::cerr);
                std::cerr << "), so no array of the function goes to the heap\n";
            }
        }
        if (const auto *ret =
                std::get_if<Terminator::Return>(&block->terminator.data);
            ret != nullptr && ret->value != nullptr &&
            carries_reference(ret->value->get_type())) {
            std::set<const Instruction *> seen;
            storages_in(defs, func, block->name, ret->value, escaped, any, seen);
        }
    }

    // One block at a time, the graph rebuilt after each: freeing at the
    // edges out of a block's dominance region may split an edge, and a later
    // block's region has to see the block that made.
    std::set<string> done;
    size_t moved = 0;
    while (true) {
        const Cfg cfg(func);
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest loops = compute_loop_forest(cfg, dom);
        BlockSet in_body(cfg.size());
        for (BlockId b : cfg.rpo) {
            if (const auto *p =
                    std::get_if<Terminator::ParFor>(&cfg[b].terminator.data)) {
                for (BlockId inside : reachable_from(cfg, cfg.id(p->body.name))) {
                    in_body.insert(inside);
                }
            }
        }

        // The next block that runs once per call and holds an array to move.
        BlockId at = NO_BLOCK;
        vector<shared_ptr<Instruction>> arrays;
        for (BlockId b : cfg.rpo) {
            if (done.contains(cfg.name(b)) || in_body.contains(b) ||
                loops.innermost(b) != nullptr) {
                continue;
            }
            for (const auto &instr : cfg[b].instrs) {
                if (instr->op != Instruction::Op::Alloca || !instr->type.is_reference()) {
                    continue;
                }
                const auto *array = instr->type.as<Array_t>();
                if (array == nullptr || !array->size.defined() ||
                    gather_free_vars(array->size).empty()) {
                    continue; // a constant size fits the frame
                }
                if (any || escaped.contains(instr.get())) {
                    // A reference into it may outlive the call.
                    if (explain) {
                        std::cerr << "; heap_arrays(" << fname << "): " << instr->name
                                  << " stays on the stack: a reference into it "
                                  << (any ? "may be what is stored" : "is stored")
                                  << " outside the function's own storage\n";
                    }
                    continue;
                }
                arrays.push_back(instr);
            }
            if (!arrays.empty()) {
                at = b;
                break;
            }
        }
        if (at == NO_BLOCK) {
            break;
        }
        done.insert(cfg.name(at));

        // Every path from the block to a return leaves the blocks it
        // dominates exactly once, at a return among them or along an edge
        // out of them; the storage is freed there. An edge out of a block
        // that also has edges staying in is split, so that the free is on
        // that edge alone.
        const vector<BlockId> region = dom.subtree(at);
        BlockSet dominated(cfg.size());
        for (BlockId b : region) {
            dominated.insert(b);
        }
        vector<shared_ptr<Block>> free_in;
        for (BlockId b : region) {
            Block &block = cfg[b];
            if (std::holds_alternative<Terminator::Return>(block.terminator.data)) {
                free_in.push_back(cfg.block(b));
                continue;
            }
            bool leaves = false, stays = false;
            for (BlockId s : cfg.succs[b]) {
                (dominated.contains(s) ? stays : leaves) = true;
            }
            if (!leaves) {
                continue;
            }
            if (!stays) {
                free_in.push_back(cfg.block(b));
                continue;
            }
            size_t made = 0;
            for (Terminator::Jump *jump : jumps_of(block)) {
                const BlockId target = cfg.find(jump->name);
                if (target == NO_BLOCK || dominated.contains(target)) {
                    continue;
                }
                auto split = std::make_shared<Block>();
                split->name = block.name + "!free_" + std::to_string(made++);
                split->owner = func.blocks.front()->owner;
                // The values the edge carried, passed on from here: the
                // block that jumped dominates this one, so it may name them.
                split->terminator.data =
                    Terminator::Jump{jump->name, std::move(jump->args)};
                jump->name = split->name;
                jump->args.clear();
                func.blocks.push_back(split);
                free_in.push_back(split);
            }
        }
        for (const auto &array : arrays) {
            array->op = Instruction::Op::Alloc;
            for (const auto &block : free_in) {
                // The array's value itself rather than a copy threaded in
                // through the blocks between: the allocation's block
                // dominates every block here, and a block may read any value
                // that does (see SSA/CloseBodies.h).
                block->instrs.push_back(std::make_shared<Instruction>(
                    Instruction::Op::Free,
                    vector<shared_ptr<Value>>{std::make_shared<Value>(array)},
                    block));
            }
            moved++;
        }
    }
    if (moved > 0) {
        refresh_preds(func);
    }
    return moved;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
