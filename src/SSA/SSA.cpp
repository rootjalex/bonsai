#include "SSA/SSA.h"

#include "IR/Equality.h"
#include "IR/Printer.h"

#include <iostream>

namespace bonsai {
namespace ir {
namespace ssa {

void Argument::dump(std::ostream &os) const { os << name << " : " << type; }

void Constant::dump(std::ostream &os) const {
    if (!type.is_func()) {
        // Func signatures are too annoying to read.
        os << "(" << type << ")";
    }
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

std::optional<Argument> Value::get_argument() const {
    return std::visit(
        overloads{
            [](const std::shared_ptr<Instruction> &i)
                -> std::optional<Argument> {
                internal_assert(i->type.defined());
                return Argument{i->type, i->name};
            },
            [](const Constant &c) -> std::optional<Argument> { return {}; },
            [](const Argument &a) -> std::optional<Argument> { return a; },
        },
        data);
}

static const char *op_name(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Abs:
        return "add";
    case Instruction::Op::Add:
        return "add";
    case Instruction::Op::Alloc:
        return "alloc";
    case Instruction::Op::Alloca:
        return "alloca";
    case Instruction::Op::Append:
        return "append";
    case Instruction::Op::Bc:
        return "bc";
    case Instruction::Op::Cast:
        return "cast";
    case Instruction::Op::Div:
        return "div";
    case Instruction::Op::Eps:
        return "eps";
    case Instruction::Op::Eq:
        return "eq";
    case Instruction::Op::ExtractIdx:
        return "extract_idx";
    case Instruction::Op::GEP:
        return "gep";
    case Instruction::Op::LAnd:
        return "land";
    case Instruction::Op::LOr:
        return "lor";
    case Instruction::Op::Leq:
        return "leq";
    case Instruction::Op::Load:
        return "load";
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
    case Instruction::Op::Mod:
        return "mod";
    case Instruction::Op::Mul:
        return "mul";
    case Instruction::Op::Reinterpret:
        return "reinterpret";
    case Instruction::Op::Set:
        return "set";
    case Instruction::Op::Store:
        return "set";
    case Instruction::Op::Sub:
        return "sub";
    }
    return "unknown";
}

void Instruction::dump(std::ostream &os) const {
    if (op == Instruction::Op::Store) {
        internal_assert(name.empty())
            << "Name must be empty for store: " << name;
        os << "store ";
        internal_assert(operands.size() == 2);
        operands[0]->dump(os);
        os << " ";
        operands[1]->dump(os);
        return;
    } else if (op == Instruction::Op::Append) {
        internal_assert(name.empty())
            << "Name must be empty for append: " << name;
        os << "append ";
        internal_assert(operands.size() == 2);
        operands[0]->dump(os);
        os << " ";
        operands[1]->dump(os);
        return;
    }
    // TODO: print type?
    if (!name.empty()) {
        os << name << " = ";
    }

    size_t start = 0;

    os << op_name(op);
    if (op == Instruction::Op::Alloc || op == Instruction::Op::Alloca ||
        op == Instruction::Op::Cast || op == Instruction::Op::Eps ||
        op == Instruction::Op::MakeStruct ||
        op == Instruction::Op::Reinterpret) {
        os << "<" << type << ">";
    }
    os << "(";

    for (size_t i = start, e = operands.size(); i < e; i++) {
        if (i > start) {
            os << ", ";
        }
        operands[i]->dump(os);
    }
    os << ")";
}

namespace {

void dump_target(std::ostream &os, const Terminator::Jump &j) {
    os << j.name << "(";
    for (size_t i = 0; i < j.args.size(); ++i) {
        if (i)
            os << ", ";
        j.args[i]->dump(os);
    }
    os << ")";
}

} // namespace

void Terminator::dump(std::ostream &os) const {
    std::visit(overloads{
                   [&](const std::monostate &m) { os << "<EMPTY>"; },
                   [&](const Jump &j) {
                       os << "jmp ";
                       dump_target(os, j);
                   },
                   [&](const Dispatch &d) {
                       os << "dispatch ";
                       d.cond->dump(os);
                       os << " [";
                       for (size_t i = 0; i < d.targets.size(); ++i) {
                           if (i)
                               os << ", ";
                           dump_target(os, d.targets[i]);
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
                   [&](const ParFor &p) {
                       os << "parfor " << p.index << " ";
                       p.start->dump(os);
                       os << ":";
                       p.end->dump(os);
                       os << ":";
                       p.stride->dump(os);
                       os << " ";
                       dump_target(os, p.body);
                       os << " ";
                       dump_target(os, p.cont);
                   },
                   [&](const Yield &y) { os << "yield"; },
                   [&](const Call &c) {
                       os << "call";
                       if (!c.drop) {
                           os << "c";
                       }
                       os << " ";
                       dump_target(os, c.call);
                       os << " ";
                       dump_target(os, c.cont);
                   },
               },
               data);
}

std::shared_ptr<Value> Block::get_value(const std::string &name,
                                        const Type &type) {
    auto it = lookups.find(name);

    if (it != lookups.end()) {
        internal_assert(equals(type, it->second->get_type()))
            << "Expected type: " << type << " for var: " << name << " but got "
            << it->second->get_type() << " instead.";
        return it->second;
    }

    // Needed from calling block.
    const auto func = owner.lock();
    if (!(func && !func->blocks.empty() &&
          this != func->blocks.front().get())) {
        if (func) {
            func->dump(std::cerr);
        }
    }
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
    for (const auto &pred : preds) {
        auto p = pred.lock();
        internal_assert(p) << "Predecessor pointer died";

        std::visit(overloads{
                       [&](const std::monostate &m) {
                           (void)m;
                           internal_error << "Block: " << p->name
                                          << " is a predecessor to: " << name
                                          << " but does not have a terminator.";
                       },
                       [&](Terminator::Jump &j) {
                           // Recursively adds to parent block.
                           j.args.push_back(p->get_value(name, type));
                       },
                       [&](Terminator::Dispatch &d) {
                           for (auto &target : d.targets) {
                               if (target.name == this->name) {
                                   // Recursively adds to parent block.
                                   target.args.push_back(
                                       p->get_value(name, type));
                                   // Can we early-return here?
                               }
                           }
                       },
                       [&](const Terminator::Return &r) {
                           (void)r;
                           internal_error
                               << "Block: " << p->name
                               << " returns but is listed as a predecessor to: "
                               << name;
                       },
                       [&](Terminator::ParFor &pf) {
                           if (pf.body.name == this->name) {
                               // Recursively adds to parent block.
                               pf.body.args.push_back(p->get_value(name, type));
                           } else if (pf.cont.name == this->name) {
                               // Recursively adds to parent block.
                               pf.cont.args.push_back(p->get_value(name, type));
                           } else {
                               p->dump(std::cerr);
                               internal_error << this->name
                                              << " lists predecessor ^ that "
                                                 "does not point to it.";
                           }
                       },
                       [&](const Terminator::Yield &y) { (void)y; },
                       [&](Terminator::Call &c) {
                           if (c.call.name == this->name) {
                               // Recursively adds to parent block.
                               c.call.args.push_back(p->get_value(name, type));
                           } else if (c.cont.name == this->name) {
                               // Recursively adds to parent block.
                               c.cont.args.push_back(p->get_value(name, type));
                           } else {
                               p->dump(std::cerr);
                               internal_error << this->name
                                              << " lists predecessor ^ that "
                                                 "does not point to it.";
                           }
                       }},

                   p->terminator.data);
    }
    return value;
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

void Block::dump() const { this->dump(std::cout); }

void Function::dump(std::ostream &os) const {
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i == 0) {
            os << "func ";
        }
        blocks[i]->dump(os);
        os << "\n";
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
