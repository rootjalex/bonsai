#include "SSA/SSA.h"

#include "IR/Equality.h"
#include "IR/Printer.h"

#include <iostream>

namespace bonsai {
namespace ir {
namespace ssa {

void Argument::dump(std::ostream &os) const { os << name << " : " << type; }

void Constant::dump(std::ostream &os) const {
    os << "(" << type << ")";
    std::visit([&](auto &&v) { os << v; }, data);
}

void Value::dump(std::ostream &os) const {
    std::visit(
        overloads{
            [&](const std::shared_ptr<Instruction> &i) { os << i->name; },
            [&](const Constant &c) { c.dump(os); },
            [&](const Argument &a) { os << a.name; },
        },
        data);
}

const Type &Value::get_type() const {
    const Type *ty = std::visit(
        overloads{
            [](const std::shared_ptr<Instruction> &i) -> const Type * {
                return &i->type;
            },
            [](const Constant &c) -> const Type * { return &c.type; },
            [](const Argument &a) -> const Type * { return &a.type; },
        },
        data);
    return *ty;
}

static const char *op_name(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Add:
        return "add";
    case Instruction::Op::Bc:
        return "bc";
    case Instruction::Op::Call:
        return "call";
    case Instruction::Op::Div:
        return "div";
    case Instruction::Op::Leq:
        return "leq";
    case Instruction::Op::LoadField:
        return "load_field";
    case Instruction::Op::Lt:
        return "lt";
    case Instruction::Op::MakeStruct:
        return "make_struct";
    case Instruction::Op::Max:
        return "max";
    case Instruction::Op::Min:
        return "min";
    case Instruction::Op::Mul:
        return "mul";
    case Instruction::Op::Set:
        return "set";
    case Instruction::Op::Sub:
        return "sub";
    }
    return "unknown";
}

void Instruction::dump(std::ostream &os) const {
    // TODO: print type?
    os << name << " = ";
    os << op_name(op) << "(";
    for (size_t i = 0, e = operands.size(); i < e; i++) {
        if (i) {
            os << ", ";
        }
        operands[i]->dump(os);
    }
    os << ")";
}

void Terminator::dump(std::ostream &os) const {
    std::visit(overloads{
                   [&](const std::monostate &m) { os << "<EMPTY>"; },
                   [&](const Jump &j) {
                       os << "jmp " << j.name << "(";
                       for (size_t i = 0; i < j.args.size(); ++i) {
                           if (i)
                               os << ", ";
                           j.args[i]->dump(os);
                       }
                       os << ")";
                   },
                   [&](const Dispatch &d) {
                       os << "dispatch ";
                       d.cond->dump(os);
                       os << " [";
                       for (size_t i = 0; i < d.targets.size(); ++i) {
                           if (i)
                               os << ", ";
                           os << d.targets[i].name << "(";
                           for (size_t j = 0; j < d.targets[i].args.size();
                                ++j) {
                               if (j)
                                   os << ", ";
                               d.targets[i].args[j]->dump(os);
                           }
                           os << ")";
                       }
                       os << "]";
                   },
                   [&](const Return &r) {
                       os << "ret";
                       if (r.value) {
                           os << " ";
                           r.value->dump(os);
                       }
                   },
               },
               data);
}

std::shared_ptr<Value> Block::get_value(const std::string &name,
                                        const Type &type) {
    auto it = lookups.find(name);
    if (it == lookups.end()) {
        // Needed from calling block.
        const auto func = owner.lock();
        internal_assert(func && !func->blocks.empty() &&
                        this != func->blocks.front().get())
            << "Var: " << name
            << " not in current block, and current block is entry!";
        Argument a = {type, name};
        args.push_back(a);
        auto value = std::make_shared<Value>(std::move(a));
        auto [_, inserted] = lookups.insert({name, value});
        internal_assert(inserted)
            << "Failed to insert necessary argument: " << name;
        // This cannot induce a cycle because the insert happens before
        // recursion! Do *NOT* reorder these.
        // TODO: should NOT recurse through `returns`!! But those shouldn't be
        // predecessors anyways...
        for (const auto &pred : preds) {
            auto p = pred.lock();
            internal_assert(p) << "Predecessor pointer died";
            // TODO: Terminators need to be updated as well!!
            internal_assert(!std::holds_alternative<Terminator::Return>(p->terminator.data))
                << "Block: " << p->name << " returns but is listed as a predecessor to: " << name;
            (void)p->get_value(name, type);
        }
        
        return value;
    }
    internal_assert(equals(type, it->second->get_type()))
        << "Expected type: " << type << " for var: " << name << " but got "
        << it->second->get_type() << " instead.";
    return it->second;
}

void Block::dump(std::ostream &os) const {
    os << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
        if (i)
            os << ", ";
        args[i].dump(os);
    }
    os << "):\n";
    for (auto &i : instrs) {
        os << "  ";
        i->dump(os);
        os << "\n";
    }
    os << "  ";
    terminator.dump(os);
    os << "\n";
}

void Function::dump(std::ostream &os) const {
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i == 0) {
            os << "func ";
        } else {
        }
        blocks[i]->dump(os);
        os << "\n";
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
