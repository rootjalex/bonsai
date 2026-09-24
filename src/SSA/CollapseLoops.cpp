#include "SSA/Analysis.h"
#include "SSA/LoopArithmetic.h"
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

using std::shared_ptr;
using std::string;

// The collapsed loop needs the inner loop's trip count before either loop
// starts, so the bounds it is worked out from have to be the same on every
// step of the outer loop (invariant_in, SSA/LoopArithmetic.h). A triangular
// nest -- `parfor y in 0:n { parfor x in 0:y ... }` -- is the case this rules
// out: its rectangle has a different width per row, so there is no single
// count to collapse against.
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
    internal_assert(header) << "collapse(" << outer << ", " << inner << ") on "
                            << func << ": no parfor named " << outer
                            << ". Only a parfor can be collapsed -- a "
                               "sequential loop has an order to keep.";

    const Terminator::ParFor outer_loop =
        std::get<Terminator::ParFor>(header->terminator.data);
    internal_assert(blocks.contains(outer_loop.body.name))
        << func << " has no block " << outer_loop.body.name;
    const shared_ptr<Block> body = blocks.at(outer_loop.body.name);
    // The loop made here is a new loop, and a bind names a loop: bind the
    // collapsed loop, after collapsing.
    internal_assert(!outer_loop.binding.has_value())
        << "collapse(" << outer << ", " << inner << ") on " << func << ": "
        << outer << " is bound to " << to_string(*outer_loop.binding)
        << ", and the collapsed loop is a new loop. Bind " << collapsed
        << " after the collapse instead.";

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
                        instr->op != Instruction::Op::Append &&
                        instr->op != Instruction::Op::Push)
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
    internal_assert(!inner_loop.binding.has_value())
        << "collapse(" << outer << ", " << inner << ") on " << func << ": "
        << inner << " is bound to " << to_string(*inner_loop.binding)
        << ", and the collapsed loop is a new loop. Bind " << collapsed
        << " after the collapse instead.";

    std::set<string> in_loop;
    const Cfg body_region(*f, body->name);
    for (const auto &block : body_region.blocks()) {
        in_loop.insert(block->name);
    }
    // The index as the body knows it: the loop's name may carry a variant's
    // suffix the argument does not (SSA/Specialize.cpp).
    internal_assert(!body->args.empty()) << func << ": " << body->name
                                         << " takes no index";
    const string outer_index = body->args.front().name;
    for (const auto &bound :
         {inner_loop.start, inner_loop.end, inner_loop.stride}) {
        internal_assert(!varies_with(bound, outer_index, in_loop))
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
            if (varies_with(as_value, outer_index, in_loop)) {
                stays.push_back(std::move(instr));
                continue;
            }
            // Moved whole: the body forgets the name and the header learns
            // it, so that a block asking for the value by name finds it where
            // it now is (Block::get_value).
            body->lookups.erase(instr->name);
            header->lookups[instr->name] = as_value;
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
    auto total = arith(*header, itype, Instruction::Op::Mul, co, ci);

    auto step = std::make_shared<Block>();
    step->name = body->name + "_collapsed_" + collapsed;
    step->owner = f;
    const Argument step_arg{itype, collapsed};
    auto v_step = step->add_argument(step_arg);

    // outer = bo + (c / ci) * so, inner = bi + (c % ci) * si
    auto q = arith(*step, itype, Instruction::Op::Div, v_step, ci);
    auto q_scaled =
        arith(*step, itype, Instruction::Op::Mul, q, outer_loop.stride);
    auto outer_val =
        arith(*step, itype, Instruction::Op::Add, outer_loop.start, q_scaled);
    auto r = arith(*step, itype, Instruction::Op::Mod, v_step, ci);
    auto r_scaled =
        arith(*step, itype, Instruction::Op::Mul, r, inner_loop.stride);
    auto inner_val =
        arith(*step, itype, Instruction::Op::Add, inner_loop.start, r_scaled);

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

    // What the outer loop handed its body besides the index -- its captures
    // -- it now hands the step, which passes each on. The step is a block of
    // its own between the header and the body, so a value of the header's
    // reaches the body through an argument of the step, not by name: a body
    // that read its captures from the enclosing scope would be a loop whose
    // body cannot be compiled apart from its function, and a loop bound to
    // the GPU is exactly that -- the kernel has the step's arguments and
    // nothing else. As with the trip count, the step's argument is made
    // here and the header's value for it is supplied by the loop edge below.
    const auto through_step = [&](const shared_ptr<Value> &v) {
        return std::visit(
            overloads{
                [&](const Constant &) { return v; },
                [&](const Argument &a) { return step->get_value(a.name, a.type); },
                [&](const shared_ptr<Instruction> &i) {
                    return step->get_value(i->name, i->type);
                }},
            v->data);
    };
    std::vector<shared_ptr<Value>> to_body = {outer_val, inner_val};
    for (const auto &arg : outer_loop.body.args) {
        to_body.push_back(through_step(arg));
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

    // What the step reads from outside itself -- the inner trip count, and
    // whichever bounds and strides are not constants -- make_instruction
    // threaded in as arguments of the step, behind its index. It had no
    // predecessor to hand them on to, since the step's one predecessor is the
    // loop being made here: so the loop's edge into it passes them, each the
    // header's value of the same name.
    std::vector<shared_ptr<Value>> into_step;
    for (size_t k = 1; k < step->args.size(); k++) {
        into_step.push_back(
            header->get_value(step->args[k].name, step->args[k].type));
    }

    header->terminator.data = Terminator::ParFor{
        collapsed,
        index_constant(itype, 0),
        total,
        index_constant(itype, 1),
        Terminator::Jump{step->name, std::move(into_step)},
        outer_loop.cont};

    // The inner loop's continuation only existed to end the outer loop's body,
    // which the body's own yield now does.
    const string dead = inner_loop.cont.name;
    for (auto &block : added) {
        f->blocks.push_back(std::move(block));
    }

    const Cfg live(*f, f->blocks.front()->name);
    std::vector<shared_ptr<Block>> kept;
    for (auto &block : f->blocks) {
        if (block->name == dead && !live.contains(dead)) {
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
