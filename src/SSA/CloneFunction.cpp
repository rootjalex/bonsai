#include "SSA/CloneFunction.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <map>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// Values are copied rather than shared, so that editing the clone cannot
// disturb the original. An instruction operand is remapped to the copy of the
// instruction it refers to; constants and arguments are self-contained and
// are copied by value.
shared_ptr<Value>
clone_value(const shared_ptr<Value> &value,
            const map<const Instruction *, shared_ptr<Instruction>> &instrs) {
    if (!value) {
        return nullptr;
    }
    if (const auto *instr = std::get_if<shared_ptr<Instruction>>(&value->data)) {
        const auto it = instrs.find(instr->get());
        internal_assert(it != instrs.end())
            << "Value refers to an instruction outside the function: "
            << (*instr)->name;
        return std::make_shared<Value>(it->second);
    }
    return std::make_shared<Value>(*value);
}

vector<shared_ptr<Value>>
clone_values(const vector<shared_ptr<Value>> &values,
             const map<const Instruction *, shared_ptr<Instruction>> &instrs) {
    vector<shared_ptr<Value>> cloned;
    cloned.reserve(values.size());
    for (const auto &value : values) {
        cloned.push_back(clone_value(value, instrs));
    }
    return cloned;
}

Terminator::Jump
clone_jump(const Terminator::Jump &jump,
           const map<const Instruction *, shared_ptr<Instruction>> &instrs) {
    return Terminator::Jump{jump.name, clone_values(jump.args, instrs)};
}

} // namespace

shared_ptr<Function> clone_function(const Function &func) {
    auto clone = std::make_shared<Function>();
    clone->ret_type = func.ret_type;
    clone->attributes = func.attributes;

    // First pass: the blocks and their instructions, so that an operand can
    // be remapped no matter which block defines it.
    map<const Instruction *, shared_ptr<Instruction>> instrs;
    map<string, shared_ptr<Block>> blocks;
    for (const auto &block : func.blocks) {
        auto copy = std::make_shared<Block>();
        copy->name = block->name;
        copy->args = block->args;
        copy->owner = clone;

        for (const auto &instr : block->instrs) {
            auto instr_copy = std::make_shared<Instruction>(
                instr->name, instr->type, instr->op,
                vector<shared_ptr<Value>>{}, copy);
            instrs[instr.get()] = instr_copy;
            copy->instrs.push_back(instr_copy);
            // The copy's name counter starts from zero, so it has to be told
            // which names are taken or it will hand out one of them again.
            clone->reserve_name(instr->name);
        }

        blocks[copy->name] = copy;
        clone->blocks.push_back(copy);
    }

    // Second pass: operands, terminators, lookups and predecessors, all of
    // which may refer to instructions defined anywhere in the function.
    for (const auto &block : func.blocks) {
        auto copy = blocks.at(block->name);

        for (size_t i = 0; i < block->instrs.size(); i++) {
            copy->instrs[i]->operands =
                clone_values(block->instrs[i]->operands, instrs);
        }

        for (const auto &[name, value] : block->lookups) {
            copy->lookups[name] = clone_value(value, instrs);
        }

        for (const auto &pred : block->preds) {
            const auto locked = pred.lock();
            internal_assert(locked) << "Predecessor of " << block->name
                                    << " died before it could be cloned";
            copy->preds.push_back(blocks.at(locked->name));
        }

        copy->terminator.data = std::visit(
            overloads{
                [&](const std::monostate &m) -> decltype(Terminator::data) {
                    return m;
                },
                [&](const Terminator::Jump &j) -> decltype(Terminator::data) {
                    return clone_jump(j, instrs);
                },
                [&](const Terminator::Dispatch &d)
                    -> decltype(Terminator::data) {
                    Terminator::Dispatch copy_d;
                    copy_d.cond = clone_value(d.cond, instrs);
                    for (const auto &target : d.targets) {
                        copy_d.targets.push_back(clone_jump(target, instrs));
                    }
                    return copy_d;
                },
                [&](const Terminator::Return &r) -> decltype(Terminator::data) {
                    return Terminator::Return{clone_value(r.value, instrs)};
                },
                [&](const Terminator::ParFor &p) -> decltype(Terminator::data) {
                    Terminator::ParFor copy_p;
                    copy_p.index = p.index;
                    copy_p.start = clone_value(p.start, instrs);
                    copy_p.end = clone_value(p.end, instrs);
                    copy_p.stride = clone_value(p.stride, instrs);
                    copy_p.body = clone_jump(p.body, instrs);
                    copy_p.cont = clone_jump(p.cont, instrs);
                    return copy_p;
                },
                [&](const Terminator::Yield &y) -> decltype(Terminator::data) {
                    return y;
                },
                [&](const Terminator::Call &c) -> decltype(Terminator::data) {
                    return Terminator::Call{clone_jump(c.call, instrs),
                                            clone_jump(c.cont, instrs), c.drop};
                },
            },
            block->terminator.data);
    }

    return clone;
}

string unify_returns(Function &func) {
    vector<shared_ptr<Block>> returning;
    for (const auto &block : func.blocks) {
        if (std::holds_alternative<Terminator::Return>(block->terminator.data)) {
            returning.push_back(block);
        }
    }

    if (returning.size() == 1) {
        return returning.front()->name;
    }
    internal_assert(!returning.empty())
        << "Function has no return to unify; it never comes back";

    auto exit = std::make_shared<Block>();
    exit->name = "!return";
    exit->owner = func.blocks.front()->owner;

    // The returned value becomes the exit block's argument, which is exactly
    // the phi a blend replaces once the region is linearized.
    const bool returns_value = std::get<Terminator::Return>(
                                   returning.front()->terminator.data)
                                   .value != nullptr;
    if (returns_value) {
        exit->args.push_back(Argument{func.ret_type, "!ret"});
    }

    for (const auto &block : returning) {
        const auto ret = std::get<Terminator::Return>(block->terminator.data);
        internal_assert((ret.value != nullptr) == returns_value)
            << "Function mixes returning a value with returning nothing, in "
            << block->name;

        Terminator::Jump jump{exit->name};
        if (returns_value) {
            jump.args.push_back(ret.value);
        }
        block->terminator.data = jump;
        exit->preds.push_back(block);
    }

    Terminator::Return ret;
    if (returns_value) {
        ret.value = std::make_shared<Value>(exit->args.front());
        exit->lookups["!ret"] = ret.value;
    }
    exit->terminator.data = ret;

    func.blocks.push_back(exit);
    return exit->name;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
