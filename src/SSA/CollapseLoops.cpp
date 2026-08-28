#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::string;

shared_ptr<Value> constant(const Type &type, int64_t n) {
    return std::make_shared<Value>(Constant{type, n});
}

// Does `v` change from one step of the loop to the next? An instruction does
// if it reads the index, or reads anything that does.
bool varies_with(const shared_ptr<Value> &v, const string &index,
                 const std::set<string> &region,
                 std::set<const Instruction *> &seen) {
    if (std::holds_alternative<Constant>(v->data)) {
        return false;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return a->name == index;
    }
    const auto &instr = std::get<shared_ptr<Instruction>>(v->data);
    const auto owner = instr->owner.lock();
    if (!owner || region.count(owner->name) == 0) {
        return false; // worked out before the loop began
    }
    if (!seen.insert(instr.get()).second) {
        return false; // already following this one
    }
    for (const auto &operand : instr->operands) {
        if (varies_with(operand, index, region, seen)) {
            return true;
        }
    }
    return false;
}

bool varies_with(const shared_ptr<Value> &v, const string &index,
                 const std::set<string> &region) {
    std::set<const Instruction *> seen;
    return varies_with(v, index, region, seen);
}

// Is `v` the same on every step of the loop whose index is `index`?
//
// The collapsed loop needs the inner loop's trip count before either loop
// starts, so the bounds it is worked out from have to be the same on every
// step of the outer loop. A triangular nest -- `parfor y in 0:n { parfor x in
// 0:y ... }` -- is the case this rules out: its rectangle has a different
// width per row, so there is no single count to collapse against.
//
// Of the body block's arguments only the index varies. A parfor body has no
// back edge -- it ends at a yield -- so the header's jump is the only way in,
// and everything the header passes it was worked out before the loop began.
// Rejecting all of them would refuse the ordinary nested `map` over
// `array[array[f32, n], m]`, where the inner bound `n` is a parameter that
// simply happens to reach the body as an argument.
bool invariant_in(const shared_ptr<Value> &v, const string &index,
                  const std::set<string> &region) {
    if (std::holds_alternative<Constant>(v->data)) {
        return true;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return a->name != index;
    }
    const auto &instr = std::get<shared_ptr<Instruction>>(v->data);
    const auto owner = instr->owner.lock();
    return !owner || region.count(owner->name) == 0;
}

std::optional<int64_t> as_int(const shared_ptr<Value> &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return *i;
    }
    return std::nullopt;
}

// One arithmetic instruction, or the answer if it is already known.
//
// The index arithmetic below is mostly identities -- a loop from zero, by one,
// contributes `- 0`, `* 1` and `/ 1` at every step -- and nothing downstream
// removes them: opt::Simplify runs before the SSA conversion, so it never sees
// what a rewrite builds afterwards. Folding here is what keeps a collapsed
// loop over a plain rectangle down to the two divisions it actually needs.
shared_ptr<Value> emit(Block &block, const Type &type, Instruction::Op op,
                       const shared_ptr<Value> &lhs,
                       const shared_ptr<Value> &rhs) {
    const auto a = as_int(lhs), b = as_int(rhs);
    if (a.has_value() && b.has_value()) {
        switch (op) {
        case Instruction::Op::Add:
            return constant(type, *a + *b);
        case Instruction::Op::Sub:
            return constant(type, *a - *b);
        case Instruction::Op::Mul:
            return constant(type, *a * *b);
        case Instruction::Op::Div:
            if (*b != 0) {
                return constant(type, *a / *b);
            }
            break;
        default:
            break;
        }
    }
    const auto rhs_is = [&](int64_t n) { return b.has_value() && *b == n; };
    if ((op == Instruction::Op::Add || op == Instruction::Op::Sub) &&
        rhs_is(0)) {
        return lhs;
    }
    if ((op == Instruction::Op::Mul || op == Instruction::Op::Div) &&
        rhs_is(1)) {
        return lhs;
    }
    if (op == Instruction::Op::Add && a.has_value() && *a == 0) {
        return rhs;
    }
    return block.make_instruction(type, op, {lhs, rhs});
}

// ceil((end - begin) / stride), as instructions in `block`.
shared_ptr<Value> trip_count(Block &block, const Type &type,
                             const shared_ptr<Value> &begin,
                             const shared_ptr<Value> &end,
                             const shared_ptr<Value> &stride) {
    auto span = emit(block, type, Instruction::Op::Sub, end, begin);
    auto bumped = emit(block, type, Instruction::Op::Add, span, stride);
    auto less_one =
        emit(block, type, Instruction::Op::Sub, bumped, constant(type, 1));
    return emit(block, type, Instruction::Op::Div, less_one, stride);
}

} // namespace

void collapse(FuncMap &funcs, string func, string outer, string inner,
              string collapsed) {
    internal_assert(funcs.contains(func))
        << "collapse applied to unknown func: " << func;
    auto f = funcs[func];
    const BlockMap blocks = make_block_map(*f);

    // The outer loop, and the block its body is.
    shared_ptr<Block> header;
    for (const auto &block : f->blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->index == outer) {
            internal_assert(!header)
                << "Two loops named " << outer << " in " << func;
            header = block;
        }
    }
    internal_assert(header) << "collapse(" << outer << ", " << inner
                            << ") on " << func << ": no parfor named " << outer
                            << ". Only a parfor can be collapsed -- a "
                               "sequential loop has an order to keep.";

    const Terminator::ParFor outer_loop =
        std::get<Terminator::ParFor>(header->terminator.data);
    internal_assert(blocks.contains(outer_loop.body.name))
        << func << " has no block " << outer_loop.body.name;
    const shared_ptr<Block> body = blocks.at(outer_loop.body.name);

    // Whatever the outer body works out before reaching the inner loop stays
    // where it is, and is worked out once per point of the collapsed loop
    // rather than once per row. For the arithmetic that is actually there --
    // the offset a nested `map` computes from the outer index into its output
    // -- that is only redundant, not wrong, and the recovered index it reads
    // is right either way.
    //
    // An *effect* there is a different matter: it would go from happening once
    // a row to once a point, so the collapse is refused rather than silently
    // multiplying it.
    for (const auto &instr : body->instrs) {
        internal_assert(instr->op != Instruction::Op::Store &&
                        instr->op != Instruction::Op::Print &&
                        instr->op != Instruction::Op::Append)
            << "collapse(" << outer << ", " << inner << ") on " << func << ": "
            << outer << " has an effect of its own before running " << inner
            << ", which a collapsed loop would repeat once per " << inner
            << " step instead of once per " << outer << " step";
    }
    const auto *inner_ptr =
        std::get_if<Terminator::ParFor>(&body->terminator.data);
    internal_assert(inner_ptr != nullptr && inner_ptr->index == inner)
        << "collapse(" << outer << ", " << inner << ") on " << func << ": the "
        << "body of " << outer << " is not a parfor named " << inner;
    const Terminator::ParFor inner_loop = *inner_ptr;

    const std::set<string> in_loop =
        reachable_from(body->name, compute_successors(*f));
    for (const auto &bound :
         {inner_loop.start, inner_loop.end, inner_loop.stride}) {
        internal_assert(!varies_with(bound, outer, in_loop))
            << "collapse(" << outer << ", " << inner << ") on " << func
            << ": the range of " << inner << " depends on " << outer
            << ", so it has no one trip count to collapse against";
    }

    // The trip count is worked out before either loop starts, so anything it
    // is worked out from has to be there too. What the outer body computes
    // without reading its index is the same on every step, so it can be --
    // and has to be, since a nested `map` lands the inner array's extent
    // exactly there. What does read the index stays behind.
    {
        std::vector<shared_ptr<Instruction>> stays;
        for (auto &instr : body->instrs) {
            auto as_value = std::make_shared<Value>(instr);
            if (varies_with(as_value, outer, in_loop)) {
                stays.push_back(std::move(instr));
                continue;
            }
            instr->owner = header;
            header->instrs.push_back(std::move(instr));
        }
        body->instrs = std::move(stays);
    }

    const Type itype = outer_loop.start->get_type();

    // co * ci steps, walked one at a time, with the two indices recovered from
    // the step number.
    auto ci = trip_count(*header, itype, inner_loop.start, inner_loop.end,
                         inner_loop.stride);
    auto co = trip_count(*header, itype, outer_loop.start, outer_loop.end,
                         outer_loop.stride);
    auto total = emit(*header, itype, Instruction::Op::Mul, co, ci);

    auto step = std::make_shared<Block>();
    step->name = body->name + "_collapsed_" + collapsed;
    step->owner = f;
    const Argument step_arg{itype, collapsed};
    auto v_step = step->add_argument(step_arg);

    // outer = bo + (c / ci) * so, inner = bi + (c % ci) * si
    auto q = emit(*step, itype, Instruction::Op::Div, v_step, ci);
    auto q_scaled = emit(*step, itype, Instruction::Op::Mul, q,
                         outer_loop.stride);
    auto outer_val =
        emit(*step, itype, Instruction::Op::Add, outer_loop.start, q_scaled);
    auto r = emit(*step, itype, Instruction::Op::Mod, v_step, ci);
    auto r_scaled = emit(*step, itype, Instruction::Op::Mul, r,
                         inner_loop.stride);
    auto inner_val =
        emit(*step, itype, Instruction::Op::Add, inner_loop.start, r_scaled);

    // The body keeps its own index, and gains the outer one it used to read
    // from the block above it -- both now handed over as arguments. Whatever
    // else each loop was already threading into its body is still threaded,
    // behind the two indices: a parfor hands its body the index and then its
    // jump's arguments, and reaching the body by a plain jump means passing
    // all of them.
    const Argument outer_arg{itype, outer};
    const Argument inner_arg{itype, inner};

    const std::vector<Argument> was(body->args.begin() + 1, body->args.end());
    body->args.clear();
    body->add_argument(outer_arg);
    body->add_argument(inner_arg);
    for (const Argument &arg : was) {
        body->add_argument(arg);
    }

    std::vector<shared_ptr<Value>> to_inner = {
        std::make_shared<Value>(inner_arg)};
    for (const auto &arg : inner_loop.body.args) {
        to_inner.push_back(arg);
    }
    body->terminator.data =
        Terminator::Jump{inner_loop.body.name, std::move(to_inner)};

    std::vector<shared_ptr<Value>> to_body = {outer_val, inner_val};
    for (const auto &arg : outer_loop.body.args) {
        to_body.push_back(arg);
    }
    Terminator::Jump enter{body->name, std::move(to_body)};

    // No guard is needed on either index, and deliberately so: a collapsed
    // loop exists to be one flat parallel loop, and a branch in it would be
    // paid on every step. With a trip count of ceil((end - begin) / stride),
    // the last index a loop visits is begin + (count - 1) * stride, and
    // count - 1 < (end - begin) / stride, so that index is always below `end`.
    // Both recovered indices are therefore in range by construction, whether
    // or not the ranges divide by their strides. (The Stmt-level collapse in
    // Lower/LoopTransforms.cpp emits `if (io < eo && ii < ei)` for the
    // non-dividing case; by the same argument that test is always true.)
    step->terminator.data = std::move(enter);
    std::vector<shared_ptr<Block>> added = {step};

    header->terminator.data = Terminator::ParFor{
        collapsed,           constant(itype, 0), total, constant(itype, 1),
        Terminator::Jump{step->name}, outer_loop.cont};

    // The inner loop's continuation only existed to end the outer loop's body,
    // which the body's own yield now does.
    const string dead = inner_loop.cont.name;
    for (auto &block : added) {
        f->blocks.push_back(std::move(block));
    }

    const std::set<string> live =
        reachable_from(f->blocks.front()->name, compute_successors(*f));
    std::vector<shared_ptr<Block>> kept;
    for (auto &block : f->blocks) {
        if (block->name == dead && !live.count(dead)) {
            continue;
        }
        kept.push_back(std::move(block));
    }
    f->blocks = std::move(kept);
    refresh_preds(*f);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
