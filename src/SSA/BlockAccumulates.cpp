#include "SSA/BlockAccumulates.h"

#include "Error.h"
#include "IR/Expr.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/Contention.h"

#include <limits>
#include <map>
#include <set>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

// The accumulates a block can reduce: the ones whose operation the shuffle
// tree can fold. Sub is Add of a negation and could join; Argmin and Argmax
// carry an index beside the value and would need a pair shuffled. Neither
// is what a film does.
bool reducible_op(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
        return true;
    default:
        return false;
    }
}

ir::Intrinsic::OpType reduce_intrinsic(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::AccAdd:
        return ir::Intrinsic::block_reduce_add;
    case Instruction::Op::AccMul:
        return ir::Intrinsic::block_reduce_mul;
    case Instruction::Op::AccMin:
        return ir::Intrinsic::block_reduce_min;
    case Instruction::Op::AccMax:
        return ir::Intrinsic::block_reduce_max;
    default:
        internal_error << "not an accumulate a block reduces";
        return ir::Intrinsic::block_reduce_add;
    }
}

// The word a warp shuffle moves is 32 bits: a float or an integer of that
// width, alone or as the lanes of a short vector (a colour, a normal).
bool reducible_type(const Type &t) {
    if (t.is_vector() && t.lanes() > 4) {
        return false;
    }
    const Type e = t.is_vector() ? t.element_of() : t;
    if (!e.is_scalar() || e.bits() != 32) {
        return false;
    }
    return e.is_float() || e.is_int_or_uint();
}

// What the slot starts at, so that a thread that never reaches the
// accumulate contributes nothing and one that reaches it several times
// contributes their fold.
shared_ptr<Value> identity(Instruction::Op op, const Type &scalar) {
    if (scalar.is_float()) {
        const double inf = std::numeric_limits<double>::infinity();
        double v = 0.0;
        switch (op) {
        case Instruction::Op::AccAdd: v = 0.0; break;
        case Instruction::Op::AccMul: v = 1.0; break;
        case Instruction::Op::AccMin: v = inf; break;
        case Instruction::Op::AccMax: v = -inf; break;
        default: break;
        }
        return std::make_shared<Value>(Constant{scalar, v});
    }
    if (scalar.is_uint()) {
        uint64_t v = 0;
        switch (op) {
        case Instruction::Op::AccAdd: v = 0; break;
        case Instruction::Op::AccMul: v = 1; break;
        case Instruction::Op::AccMin: v = std::numeric_limits<uint32_t>::max(); break;
        case Instruction::Op::AccMax: v = 0; break;
        default: break;
        }
        return std::make_shared<Value>(Constant{scalar, v});
    }
    int64_t v = 0;
    switch (op) {
    case Instruction::Op::AccAdd: v = 0; break;
    case Instruction::Op::AccMul: v = 1; break;
    case Instruction::Op::AccMin: v = std::numeric_limits<int32_t>::max(); break;
    case Instruction::Op::AccMax: v = std::numeric_limits<int32_t>::min(); break;
    default: break;
    }
    return std::make_shared<Value>(Constant{scalar, v});
}

// The thread loop's body: its blocks, its head, its index, and the names
// that merge inside it -- an argument some jump inside the body passes a
// value other than the name itself for, or a result a call hands back --
// which may carry a different value per thread. A name every jump hands on
// as itself is one value throughout the body, whatever redeclares it (the
// test SSA/InsertPreheader.cpp makes for a loop's carried names).
struct Body {
    string head;
    string index;
    set<const Block *> blocks;
    set<string> merged;
};

// Whether `v` is the same value in every thread of the block: a constant, a
// value from before the body, one of the head's uniform arguments, or pure
// arithmetic and addressing over those. A read of memory is not, since the
// memory may differ per thread; the thread index and anything merging
// inside the body are not.
bool uniform(const shared_ptr<Value> &v, const Body &body,
             map<const Instruction *, bool> &memo) {
    if (!v) {
        return false;
    }
    if (std::holds_alternative<Constant>(v->data)) {
        return true;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return a->name != body.index && !body.merged.count(a->name);
    }
    const auto &ip = std::get<shared_ptr<Instruction>>(v->data);
    const Instruction &in = *ip;
    const auto owner = in.owner.lock();
    if (!owner || !body.blocks.count(owner.get())) {
        // Defined before the body: one value for the whole block.
        return true;
    }
    const auto known = memo.find(&in);
    if (known != memo.end()) {
        return known->second;
    }
    bool result = false;
    switch (in.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::GEP:
    case Instruction::Op::Inf:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::SizeOf:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
        result = true;
        for (const auto &operand : in.operands) {
            if (!uniform(operand, body, memo)) {
                result = false;
                break;
            }
        }
        break;
    default:
        result = false;
    }
    memo[&in] = result;
    return result;
}

// `v` as the block `into` can name it: a constant or a value from before
// the body or from the head as itself, a name threaded in through the
// predecessors, and an instruction of an inner body block computed again
// -- its operands first -- since the block it was in need not dominate
// `into`. Only called on values `uniform` accepted, so what is recomputed
// is pure arithmetic over values every thread agrees on.
shared_ptr<Value> materialize(const shared_ptr<Value> &v, Block &into,
                              Function &f, const Body &body,
                              map<const Instruction *, shared_ptr<Value>> &done) {
    if (std::holds_alternative<Constant>(v->data)) {
        return v;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return into.get_value(a->name, a->type);
    }
    const auto &ip = std::get<shared_ptr<Instruction>>(v->data);
    const Instruction &in = *ip;
    const auto owner = in.owner.lock();
    if (!owner || !body.blocks.count(owner.get()) || owner->name == body.head) {
        return v;
    }
    const auto seen = done.find(&in);
    if (seen != done.end()) {
        return seen->second;
    }
    vector<shared_ptr<Value>> operands;
    for (const auto &operand : in.operands) {
        operands.push_back(materialize(operand, into, f, body, done));
    }
    auto copy = std::make_shared<Instruction>(f.get_unique_name(), in.type,
                                              in.op, std::move(operands),
                                              into.shared_from_this());
    copy->queried_type = in.queried_type;
    copy->intrinsic = in.intrinsic;
    copy->reduce = in.reduce;
    copy->shuffle = in.shuffle;
    into.instrs.push_back(copy);
    auto value = std::make_shared<Value>(copy);
    into.lookups[copy->name] = value;
    done[&in] = value;
    return value;
}

shared_ptr<Block> make_block(Function &f, const string &name,
                             const shared_ptr<Block> &like) {
    set<string> taken;
    for (const auto &block : f.blocks) {
        taken.insert(block->name);
    }
    auto block = std::make_shared<Block>();
    block->name = name;
    for (size_t i = 0; taken.count(block->name); i++) {
        block->name = name + std::to_string(i);
    }
    block->owner = like->owner;
    f.blocks.push_back(block);
    return block;
}

// One accumulate the pass rewrites.
struct Candidate {
    shared_ptr<Instruction> instr;
    shared_ptr<Block> block;
    // Whether the leader's accumulate to memory has to be atomic: another
    // block may write the address.
    bool contended = true;
};

void reduce_thread_loop(Function &f, const shared_ptr<Block> &loop_block,
                        const Terminator::ParFor &loop,
                        const map<string, vector<ParallelLoop>> &nests) {
    const BlockMap blocks = make_block_map(f);
    const auto head_it = blocks.find(loop.body.name);
    if (head_it == blocks.end()) {
        return;
    }
    const shared_ptr<Block> head = head_it->second;

    Body body;
    body.head = head->name;
    body.index = loop.index;
    const Cfg region(f, head->name);
    for (const auto &block : region.blocks()) {
        body.blocks.insert(block.get());
    }
    for (const auto &block : region.blocks()) {
        for (Terminator::Jump *jump : jumps_of(*block)) {
            const auto target = blocks.find(jump->name);
            if (target == blocks.end() || !region.contains(*target->second)) {
                continue;
            }
            const Block &to = *target->second;
            if (to.args.size() < jump->args.size()) {
                continue;
            }
            // A continuation's leading arguments are what the call (or an
            // inner loop's index) hands it, which no jump passes.
            const size_t offset = to.args.size() - jump->args.size();
            for (size_t k = 0; k < offset; k++) {
                body.merged.insert(to.args[k].name);
            }
            for (size_t j = 0; j < jump->args.size(); j++) {
                const string &param = to.args[j + offset].name;
                const auto *a = std::get_if<Argument>(&jump->args[j]->data);
                if (a == nullptr || a->name != param) {
                    body.merged.insert(param);
                }
            }
        }
    }

    // The loops around the body other than this one, for whether the
    // leader's accumulate is contended across blocks.
    vector<ParallelLoop> outer;
    if (const auto nest = nests.find(head->name); nest != nests.end()) {
        for (const ParallelLoop &l : nest->second) {
            if (l.index != loop.index) {
                outer.push_back(l);
            }
        }
    }

    vector<Candidate> candidates;
    map<const Instruction *, bool> memo;
    for (const auto &block : region.blocks()) {
        for (const auto &instr : block->instrs) {
            if (!instr->atomic || !reducible_op(instr->op) ||
                instr->operands.size() != 2) {
                continue;
            }
            const Type &value_t = instr->operands[1]->get_type();
            if (!reducible_type(value_t) ||
                !uniform(instr->operands[0], body, memo)) {
                continue;
            }
            Candidate c;
            c.instr = instr;
            c.block = block;
            c.contended = contention_of(instr->operands[0], outer) !=
                          Contention::Disjoint;
            candidates.push_back(std::move(c));
        }
    }
    if (candidates.empty()) {
        return;
    }

    // Where the body ends: every Yield becomes a jump to one block that
    // reduces, so that the barrier inside the reduction is one instruction
    // every thread of the block reaches.
    auto reduce = make_block(f, head->name + "!reduce", head);
    auto leader = make_block(f, head->name + "!leader", head);
    auto done = make_block(f, head->name + "!done", head);
    for (const auto &block : region.blocks()) {
        if (std::holds_alternative<Terminator::Yield>(block->terminator.data)) {
            block->terminator.data = Terminator::Jump{reduce->name, {}};
        }
    }
    done->terminator.data = Terminator::Yield{};
    leader->terminator.data = Terminator::Jump{done->name, {}};
    // The dispatch's targets are false first (see Convert.cpp); the
    // condition is filled in below once the names can be threaded.
    reduce->terminator.data = Terminator::Dispatch{
        nullptr, {Terminator::Jump{done->name, {}}, Terminator::Jump{leader->name, {}}}};
    refresh_preds(f);

    // The places the program named, kept before the slots take their spot
    // in the accumulates: they are what the leader writes at the end.
    vector<shared_ptr<Value>> places;
    for (const Candidate &c : candidates) {
        places.push_back(c.instr->operands[0]);
    }

    // The slots: one per accumulate, at the head, starting at the identity.
    struct Slot {
        shared_ptr<Value> ptr;
        Type type;
    };
    vector<Slot> slots;
    size_t at = 0;
    for (const Candidate &c : candidates) {
        const Type value_t = c.instr->operands[1]->get_type();
        const Type scalar_t = value_t.is_vector() ? value_t.element_of() : value_t;
        auto alloca = std::make_shared<Instruction>(
            f.get_unique_name(), Ptr_t::make(value_t), Instruction::Op::Alloca,
            vector<shared_ptr<Value>>{}, head);
        auto slot = std::make_shared<Value>(alloca);
        head->lookups[alloca->name] = slot;
        head->instrs.insert(head->instrs.begin() + at++, alloca);

        shared_ptr<Value> start = identity(c.instr->op, scalar_t);
        if (value_t.is_vector()) {
            auto bc = std::make_shared<Instruction>(
                f.get_unique_name(), value_t, Instruction::Op::Bc,
                vector<shared_ptr<Value>>{
                    start, std::make_shared<Value>(Constant{
                               UInt_t::make(32), uint64_t(value_t.lanes())})},
                head);
            start = std::make_shared<Value>(bc);
            head->lookups[bc->name] = start;
            head->instrs.insert(head->instrs.begin() + at++, bc);
        }
        auto init = std::make_shared<Instruction>(
            Instruction::Op::Store, vector<shared_ptr<Value>>{slot, start}, head);
        head->instrs.insert(head->instrs.begin() + at++, init);

        // The thread's accumulate goes to its slot, and needs no atomic:
        // the slot is the thread's alone.
        c.instr->operands[0] = slot;
        c.instr->atomic = false;
        slots.push_back(Slot{slot, value_t});
    }

    // The reduction: each slot read, reduced across the block, and -- in
    // the leader -- accumulated into the place the program named.
    const Type index_t = head->args.front().type;
    shared_ptr<Value> index = reduce->get_value(loop.index, index_t);
    shared_ptr<Value> start = loop.start;
    if (const auto *a = std::get_if<Argument>(&start->data)) {
        start = reduce->get_value(a->name, a->type);
    }
    vector<shared_ptr<Value>> totals;
    for (const Slot &slot : slots) {
        shared_ptr<Value> value = reduce->make_instruction(
            slot.type, Instruction::Op::Load, {slot.ptr});
        shared_ptr<Value> total = reduce->make_instruction(
            slot.type, Instruction::Op::Intrinsic, {value});
        std::get<shared_ptr<Instruction>>(total->data)->intrinsic =
            reduce_intrinsic(candidates[totals.size()].instr->op);
        totals.push_back(total);
    }
    shared_ptr<Value> is_leader = reduce->make_instruction(
        Bool_t::make(), Instruction::Op::Eq, {index, start});
    std::get<Terminator::Dispatch>(reduce->terminator.data).cond = is_leader;

    // The leader: each block total into the place the program named, once.
    // The total is a value of the reduce block, reached by name through the
    // dispatch; the place is computed again here from values every thread
    // agrees on (see materialize).
    map<const Instruction *, shared_ptr<Value>> materialized;
    for (size_t i = 0; i < candidates.size(); i++) {
        const Candidate &c = candidates[i];
        shared_ptr<Value> place =
            materialize(places[i], *leader, f, body, materialized);
        const Instruction &total_instr =
            *std::get<shared_ptr<Instruction>>(totals[i]->data);
        shared_ptr<Value> total =
            leader->get_value(total_instr.name, total_instr.type);
        auto acc = std::make_shared<Instruction>(
            c.instr->op, vector<shared_ptr<Value>>{place, total}, leader);
        acc->atomic = c.contended;
        leader->instrs.push_back(acc);
    }
    (void)loop_block;
}

} // namespace

void ReduceBlockAccumulates::run(Function &f) {
    const map<string, vector<ParallelLoop>> nests = parallel_loops_by_block(f);
    // The thread loops, collected first: the rewrite adds blocks.
    vector<std::pair<shared_ptr<Block>, Terminator::ParFor>> loops;
    for (const auto &block : f.blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->binding.has_value() &&
            *parfor->binding == Resource::GPUThread) {
            loops.emplace_back(block, *parfor);
        }
    }
    for (const auto &[block, loop] : loops) {
        reduce_thread_loop(f, block, loop, nests);
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
