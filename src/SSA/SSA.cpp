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
        return "abs";
    case Instruction::Op::AccAdd:
        return "acc.add";
    case Instruction::Op::AccMul:
        return "acc.mul";
    case Instruction::Op::AccSub:
        return "acc.sub";
    case Instruction::Op::AccArgmin:
        return "acc.argmin";
    case Instruction::Op::AccArgmax:
        return "acc.argmax";
    case Instruction::Op::AccMin:
        return "accmin";
    case Instruction::Op::AccMax:
        return "acc.max";
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
        return "store";
    case Instruction::Op::Sub:
        return "sub";
    }
}

bool is_store_instr(const Instruction::Op &op) {
    switch (op) {
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::AccMin:
    case Instruction::Op::Store:
        return true;
    case Instruction::Op::AccMax:
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Alloc:
    case Instruction::Op::Alloca:
    case Instruction::Op::Append:
    case Instruction::Op::Bc:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::GEP:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Load:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Set:
    case Instruction::Op::Sub:
        return false;
    }
}

void Instruction::dump(std::ostream &os) const {
    if (is_store_instr(op)) {
        internal_assert(name.empty())
            << "Name must be empty for store/acc: " << name;
        os << op_name(op) << " ";
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

    internal_assert(!name.empty()) << op_name(op);
    os << name << " = ";

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
                       for (size_t i = 0; i < d.targets.size(); i++) {
                           if (i) {
                               os << ", ";
                           }
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

void Block::make_instruction(const std::string &name, Type type,
                             std::shared_ptr<Value> v) {
    // If v already refers to an Instruction, just rename it (copy
    // propagation)
    if (auto instr_ptr = std::get_if<std::shared_ptr<Instruction>>(&v->data)) {
        internal_assert(*instr_ptr) << "Null instruction value";

        auto &instr = *instr_ptr;

        // Update name
        instr->name = name;

        // Update lookup table
        // TODO: remove existing name??
        auto [it, inserted] = lookups.insert({name, v});
        if (!inserted) {
            it->second = v; // overwrite existing entry
        }

        return;
    }

    // Otherwise, create a new Set instruction
    std::vector<std::shared_ptr<Value>> vs = {std::move(v)};
    std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
        name, std::move(type), Instruction::Op::Set, std::move(vs),
        weak_from_this());
    instrs.push_back(instr);

    auto [_, inserted] = lookups.insert({name, std::make_shared<Value>(instr)});
    internal_assert(inserted) << name << "already exists in block!\n";
}

std::shared_ptr<Value>
Block::make_instruction(Type type, Instruction::Op op,
                        std::vector<std::shared_ptr<Value>> vs,
                        bool allow_rename) {
    // Re-thread any operands from other blocks (necessary due to call
    // continuations).
    for (auto &operand : vs) {
        std::visit(overloads{
                       [&](const std::shared_ptr<Instruction> &i) {
                           if (i->owner.lock().get() != this) {
                               operand = get_value(i->name, i->type);
                           }
                       },
                       [&](const Argument &a) {
                           auto it = lookups.find(a.name);
                           if (it != lookups.end()) {
                               // Already local — use the canonical local value.
                               operand = it->second;
                               return;
                           }
                           // Not local — thread it in.
                           operand = get_value(a.name, a.type);
                       },
                       [](const Constant &) {}, // constants need no threading
                   },
                   operand->data);
    }

    auto locked = owner.lock();
    internal_assert(locked)
        << "Function was deallocated during make_instruction";
    std::string name = locked->get_unique_name();
    std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
        name, std::move(type), op, std::move(vs), weak_from_this());
    instrs.push_back(instr);
    auto v = std::make_shared<Value>(std::move(instr));
    if (allow_rename) {
        lookups[name] = v;
    } else {
        auto [_, inserted] = lookups.insert({name, v});
        internal_assert(inserted) << name << "already exists in block!\n";
    }
    return v;
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

        std::visit(
            overloads{[&](const std::monostate &m) {
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
