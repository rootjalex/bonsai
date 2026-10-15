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

void DemoteAtomics::run(Function &f, bool called_in_parallel,
                        const std::vector<Contention> *params) {
    const std::map<std::string, std::vector<ParallelLoop>> enclosing =
        parallel_loops_by_block(f);
    // For following a subscript that arrives as a block parameter back to
    // what computed it (the pixel of a pass's rest, threaded through the
    // blocks between the call and the film's add).
    Origins origins(f);

    // Whether a parameter of the function is handed storage of its own by
    // every call that runs at once (SSA/Contention.h, parameter_contention).
    const auto private_parameter = [&](const std::shared_ptr<Value> &ptr,
                                       const std::string &block) {
        if (params == nullptr) {
            return false;
        }
        const std::optional<size_t> k = parameter_of_address(ptr, origins, block);
        return k.has_value() && *k < params->size() &&
               (*params)[*k] == Contention::Disjoint;
    };

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
            // A function some bound loop calls runs once per concurrent
            // iteration, under loops no block of it shows. The addresses
            // two of those calls cannot share are storage this call made
            // for itself, and a parameter the whole program shows every
            // concurrent call is handed its own storage through -- the
            // sample's reducer, allocated in the loop's body and handed
            // down the path's calls; the record slot a drain's entry holds.
            // Any other arrived from a caller as a word two calls may well
            // share -- the run's reducer -- so that is Unknown, and the
            // atomic stays: a device atomic where the loop's own body would
            // get the block reduction (SSA/BlockAccumulates.h).
            if (called_in_parallel &&
                !allocated_by_function(instr->operands[0], origins, block->name) &&
                !private_parameter(instr->operands[0], block->name)) {
                continue;
            }
            // Only Disjoint permits dropping it. Unknown means the address
            // could not be shown either way, and an atomic that might be
            // needed has to stay -- being wrong here is a race, where being
            // cautious is only slow.
            if (contention_of(instr->operands[0], loops->second, &origins,
                              block->name) == Contention::Disjoint) {
                instr->atomic = false;
            }
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
