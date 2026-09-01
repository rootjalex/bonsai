#include "SSA/DemoteAtomics.h"

#include "SSA/Contention.h"

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

bool is_accumulate(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
        return true;
    default:
        return false;
    }
}

} // namespace

void DemoteAtomics::run(Function &f) {
    const std::map<std::string, std::vector<ParallelLoop>> enclosing =
        parallel_loops_by_block(f);

    for (const auto &block : f.blocks) {
        const auto loops = enclosing.find(block->name);
        if (loops == enclosing.end()) {
            // Unreachable, so nothing runs it at all.
            continue;
        }
        for (const auto &instr : block->instrs) {
            if (!instr->atomic || !is_accumulate(instr->op) ||
                instr->operands.size() != 2) {
                continue;
            }
            // Only Disjoint permits dropping it. Unknown means the address
            // could not be shown either way, and an atomic that might be
            // needed has to stay -- being wrong here is a race, where being
            // cautious is only slow.
            if (contention_of(instr->operands[0], loops->second) ==
                Contention::Disjoint) {
                instr->atomic = false;
            }
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
