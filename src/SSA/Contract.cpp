#include "SSA/Contract.h"

#include "IR/Equality.h"
#include "IR/Expr.h"
#include "IR/Schedule.h"

#include "Error.h"

#include <algorithm>

#include <map>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using UseCounts = std::map<const Instruction *, size_t>;

void count(const std::shared_ptr<Value> &value, UseCounts &uses) {
    if (value == nullptr) {
        return;
    }
    if (const auto *instr =
            std::get_if<std::shared_ptr<Instruction>>(&value->data)) {
        ++uses[instr->get()];
    }
}

void count(const Terminator::Jump &jump, UseCounts &uses) {
    for (const auto &arg : jump.args) {
        count(arg, uses);
    }
}

// How many times each instruction's result is named, anywhere in the function.
//
// Terminators included, and that is not a detail: a jump argument is a use like
// any other, and a product counted only over instructions could look dead while
// a block argument still carries it.
UseCounts use_counts(const Function &f) {
    UseCounts uses;
    for (const auto &block : f.blocks) {
        for (const auto &instr : block->instrs) {
            for (const auto &operand : instr->operands) {
                count(operand, uses);
            }
        }
        std::visit(
            Overloaded{
                [](const std::monostate &) {},
                [&](const Terminator::Jump &j) { count(j, uses); },
                [&](const Terminator::Dispatch &d) {
                    count(d.cond, uses);
                    for (const auto &target : d.targets) {
                        count(target, uses);
                    }
                },
                [&](const Terminator::Return &r) { count(r.value, uses); },
                [&](const Terminator::ParFor &p) {
                    count(p.start, uses);
                    count(p.end, uses);
                    count(p.stride, uses);
                    count(p.body, uses);
                    count(p.cont, uses);
                },
                [](const Terminator::Yield &) {},
                [&](const Terminator::Call &c) {
                    count(c.call, uses);
                    count(c.cont, uses);
                },
            },
            block->terminator.data);
    }
    return uses;
}

bool is_float_valued(const Type &type) {
    return type.is_vector() ? type.element_of().is_float() : type.is_float();
}

// The product this add may absorb, or nothing.
//
// Three conditions, and each is there to keep the fusion free rather than to
// keep it legal. The product must be used only here, or fusing would leave the
// multiply behind for its other reader and compute `a * b` twice. It must be in
// the same block, or the fusion would drag the work to wherever the add is --
// which for a product hoisted out of a loop means dragging it back in. And the
// types must match exactly, so that a widened lane count or a mixed precision
// is left alone rather than quietly reinterpreted.
std::shared_ptr<Instruction> product_of(const std::shared_ptr<Value> &operand,
                                        const Block &block,
                                        const Type &type,
                                        const UseCounts &uses) {
    const auto *held = std::get_if<std::shared_ptr<Instruction>>(&operand->data);
    if (held == nullptr) {
        return nullptr;
    }
    const std::shared_ptr<Instruction> &instr = *held;
    if (instr->op != Instruction::Op::Mul || instr->operands.size() != 2) {
        return nullptr;
    }
    if (!equals(instr->type, type)) {
        return nullptr;
    }
    const auto found = uses.find(instr.get());
    if (found == uses.end() || found->second != 1) {
        return nullptr;
    }
    const auto in_block =
        std::find(block.instrs.cbegin(), block.instrs.cend(), instr);
    if (in_block == block.instrs.cend()) {
        return nullptr;
    }
    return instr;
}

} // namespace

void contract_fp(Function &f) {
    const UseCounts uses = use_counts(f);

    for (const auto &block : f.blocks) {
        // What the fusions consumed. Erased after the walk rather than during
        // it, because `product_of` searches this same list.
        std::vector<std::shared_ptr<Instruction>> absorbed;

        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Add ||
                instr->operands.size() != 2 || !is_float_valued(instr->type)) {
                continue;
            }
            // The first operand before the second, which is the arbitrary half
            // of this -- see the note in the header about gcc having no rule
            // here either.
            std::shared_ptr<Instruction> mul =
                product_of(instr->operands[0], *block, instr->type, uses);
            std::shared_ptr<Value> addend = instr->operands[1];
            if (mul == nullptr) {
                mul = product_of(instr->operands[1], *block, instr->type, uses);
                addend = instr->operands[0];
            }
            if (mul == nullptr) {
                continue;
            }

            // The add becomes the fma. Rewritten in place rather than replaced,
            // so that everything already naming its result goes on doing so.
            instr->op = Instruction::Op::Intrinsic;
            instr->intrinsic = ir::Intrinsic::fma;
            instr->operands = {mul->operands[0], mul->operands[1],
                               std::move(addend)};
            absorbed.push_back(std::move(mul));
        }

        for (const auto &dead : absorbed) {
            const auto at =
                std::find(block->instrs.begin(), block->instrs.end(), dead);
            internal_assert(at != block->instrs.end())
                << "A fused product left its block: " << dead->name;
            block->instrs.erase(at);
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
