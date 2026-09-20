#include "SSA/Simplify.h"

#include "IR/Equality.h"

#include "Error.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::vector;
using ValuePtr = shared_ptr<Value>;

const Instruction *def_of(const ValuePtr &v) {
    if (v == nullptr) {
        return nullptr;
    }
    const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
    return held == nullptr ? nullptr : held->get();
}

const Constant *constant_of(const ValuePtr &v) {
    return v == nullptr ? nullptr : std::get_if<Constant>(&v->data);
}

std::optional<bool> const_bool(const ValuePtr &v) {
    const Constant *c = constant_of(v);
    if (c == nullptr || !c->type.is<Bool_t>()) {
        return std::nullopt;
    }
    return std::visit(
        overloads{
            [](bool b) -> std::optional<bool> { return b; },
            [](int64_t i) -> std::optional<bool> { return i != 0; },
            [](uint64_t u) -> std::optional<bool> { return u != 0; },
            [](double d) -> std::optional<bool> { return d != 0.0; },
            [](const std::string &) -> std::optional<bool> {
                return std::nullopt;
            },
            [](const Undefined &) -> std::optional<bool> {
                return std::nullopt;
            },
        },
        c->data);
}

ValuePtr make_bool(bool b) {
    return std::make_shared<Value>(Constant{Bool_t::make(), b});
}

// Are `a` and `b` one value: the same instruction, the same argument, or
// equal constants?
bool same_value(const ValuePtr &a, const ValuePtr &b) {
    if (a == b) {
        return true;
    }
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &x) {
                const auto *y = std::get_if<shared_ptr<Instruction>>(&b->data);
                return y != nullptr && y->get() == x.get();
            },
            [&](const Argument &x) {
                const auto *y = std::get_if<Argument>(&b->data);
                return y != nullptr && y->name == x.name;
            },
            [&](const Constant &x) {
                const auto *y = std::get_if<Constant>(&b->data);
                return y != nullptr && equals(x.type, y->type) &&
                       x.data == y->data;
            },
        },
        a->data);
}

// Is `in` the operation `op` of `type` over `operands`?
bool same_expression(const Instruction *in, Instruction::Op op,
                     const Type &type, const vector<ValuePtr> &operands) {
    if (in == nullptr || in->op != op || !equals(in->type, type) ||
        in->operands.size() != operands.size()) {
        return false;
    }
    for (size_t i = 0; i < operands.size(); i++) {
        if (!same_value(in->operands[i], operands[i])) {
            return false;
        }
    }
    return true;
}

// A comparison of two constants, or nothing when they are not both constants
// of one kind.
std::optional<bool> compare_constants(Instruction::Op op, const ValuePtr &a,
                                      const ValuePtr &b) {
    const Constant *x = constant_of(a);
    const Constant *y = constant_of(b);
    if (x == nullptr || y == nullptr || x->data.index() != y->data.index() ||
        std::holds_alternative<Undefined>(x->data)) {
        return std::nullopt; // an undefined value compares as nothing
    }
    const auto compare = [&](const auto &p, const auto &q) -> bool {
        switch (op) {
        case Instruction::Op::Lt:
            return p < q;
        case Instruction::Op::Leq:
            return p <= q;
        case Instruction::Op::Eq:
            return p == q;
        case Instruction::Op::Ne:
            return p != q;
        default:
            internal_error << "Not a comparison: " << int(op);
            return false;
        }
    };
    return std::visit(
        overloads{
            [&](bool p) { return compare(p, std::get<bool>(y->data)); },
            [&](int64_t p) { return compare(p, std::get<int64_t>(y->data)); },
            [&](uint64_t p) {
                return compare(p, std::get<uint64_t>(y->data));
            },
            [&](double p) { return compare(p, std::get<double>(y->data)); },
            [&](const std::string &p) {
                return compare(p, std::get<std::string>(y->data));
            },
            [&](const Undefined &) { return false; }, // excluded above
        },
        x->data);
}

// Whether an instruction only computes a value, so that one nothing reads
// can go. Storage, effects and the fetch-and-add are kept; so is `rand`,
// which steps the generator's state whether or not its draw is read.
bool pure(const Instruction &in) {
    if (in.name.empty()) {
        return false;
    }
    switch (in.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Any:
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
    case Instruction::Op::Load:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Popcount:
    case Instruction::Op::Ramp:
    case Instruction::Op::Reduce:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Shuffle:
    case Instruction::Op::SizeOf:
    case Instruction::Op::Sub:
    case Instruction::Op::Vote:
    case Instruction::Op::Xor:
        return true;
    case Instruction::Op::Intrinsic:
        return in.intrinsic != ir::Intrinsic::rand;
    default:
        return false;
    }
}

using Replacements = std::map<const Instruction *, ValuePtr>;

struct Simplifier {
    explicit Simplifier(Function &func) : func(func) {}

    Function &func;
    // Where a rule's new instructions go: before instruction `at` of `block`.
    shared_ptr<Block> block;
    size_t at = 0;
    Replacements replaced;
    // The instructions the rules made, so that a merge of two equal
    // expressions keeps the one the program had.
    std::set<const Instruction *> made;

    // What `v` stands for once every replacement is followed.
    ValuePtr resolve(ValuePtr v) const {
        for (auto it = replaced.find(def_of(v)); it != replaced.end();
             it = replaced.find(def_of(v))) {
            v = it->second;
        }
        return v;
    }

    // An operation over `operands`, simplified: an existing value when a
    // rule gives one, and a new instruction before the current one otherwise.
    ValuePtr make(Instruction::Op op, Type type, vector<ValuePtr> operands) {
        if (ValuePtr v = rule(op, type, operands)) {
            return v;
        }
        auto instr = std::make_shared<Instruction>(
            func.get_unique_name(), std::move(type), op, std::move(operands),
            block);
        block->instrs.insert(block->instrs.begin() + at, instr);
        at++;
        made.insert(instr.get());
        auto v = std::make_shared<Value>(instr);
        block->lookups[instr->name] = v;
        return v;
    }

    // Of two operands that are the same expression, the one to keep: the
    // program's own over one a rule made.
    ValuePtr keep(const ValuePtr &a, const ValuePtr &b) const {
        return made.count(def_of(b)) > 0 ? a : b;
    }

    ValuePtr rule(Instruction::Op op, const Type &type,
                  const vector<ValuePtr> &ops) {
        switch (op) {
        case Instruction::Op::Not: {
            if (ops.size() != 1 || !type.is<Bool_t>()) {
                break;
            }
            if (std::optional<bool> c = const_bool(ops[0])) {
                return make_bool(!*c);
            }
            // !!x = x
            if (const Instruction *d = def_of(ops[0]);
                d != nullptr && d->op == Instruction::Op::Not &&
                d->operands.size() == 1) {
                return d->operands[0];
            }
            break;
        }
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr: {
            if (ops.size() != 2 || !type.is<Bool_t>()) {
                break;
            }
            const bool is_and = op == Instruction::Op::LAnd;
            const ValuePtr &a = ops[0], &b = ops[1];
            // x & x = x; x | x = x. Also when x is spelled twice.
            if (same_value(a, b)) {
                return a;
            }
            const Instruction *da = def_of(a);
            if (da != nullptr &&
                same_expression(def_of(b), da->op, da->type, da->operands)) {
                return keep(a, b);
            }
            for (const auto &[x, y] : {std::pair{a, b}, std::pair{b, a}}) {
                if (std::optional<bool> c = const_bool(x)) {
                    // true & y = y, false & y = false; and the dual.
                    return *c == is_and ? y : make_bool(!is_and);
                }
            }
            break;
        }
        case Instruction::Op::Select: {
            if (ops.size() != 3) {
                break;
            }
            if (std::optional<bool> c = const_bool(ops[0])) {
                return *c ? ops[1] : ops[2];
            }
            if (same_value(ops[1], ops[2])) {
                return ops[1];
            }
            break;
        }
        case Instruction::Op::Vote: {
            // Every lane holds the same constant, so that is the decision.
            // Anything else stays a vote until a gang is there to hold it
            // (see lower_votes in SSA/Vectorize.cpp) or the run it decides
            // is put on a stack (SSA/QueueRecursion.cpp).
            if (ops.size() == 1 && const_bool(ops[0]).has_value()) {
                return ops[0];
            }
            break;
        }
        case Instruction::Op::Lt:
        case Instruction::Op::Leq:
        case Instruction::Op::Eq:
        case Instruction::Op::Ne: {
            if (ops.size() != 2 || !type.is<Bool_t>()) {
                break;
            }
            if (std::optional<bool> c = compare_constants(op, ops[0], ops[1])) {
                return make_bool(*c);
            }
            if (same_value(ops[0], ops[1]) && !ops[0]->get_type().is_float()) {
                // x < x, x == x, and the rest, for anything without a NaN.
                return make_bool(op == Instruction::Op::Leq ||
                                 op == Instruction::Op::Eq);
            }
            if (op != Instruction::Op::Lt) {
                break;
            }
            // cast(a) < cast(b), a and b bools, both to one numeric type:
            // the only way a number made from a bool is below another is
            // false below true, so this is !a & b.
            const Instruction *da = def_of(ops[0]), *db = def_of(ops[1]);
            const auto bool_cast = [](const Instruction *d) {
                return d != nullptr && d->op == Instruction::Op::Cast &&
                       d->operands.size() == 1 && d->type.is_scalar() &&
                       (d->type.is_int() || d->type.is_uint() ||
                        d->type.is_float()) &&
                       d->operands[0]->get_type().is<Bool_t>();
            };
            if (bool_cast(da) && bool_cast(db) && equals(da->type, db->type)) {
                ValuePtr not_a = make(Instruction::Op::Not, Bool_t::make(),
                                      {da->operands[0]});
                return make(Instruction::Op::LAnd, Bool_t::make(),
                            {not_a, db->operands[0]});
            }
            break;
        }
        default:
            break;
        }
        return nullptr;
    }
};

void resolve_jump(Terminator::Jump &jump, const Simplifier &s) {
    for (auto &arg : jump.args) {
        arg = s.resolve(arg);
    }
}

// Points every use in the function at what its value became.
void resolve_uses(Function &func, const Simplifier &s) {
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            for (auto &operand : instr->operands) {
                operand = s.resolve(operand);
            }
        }
        std::visit(overloads{
                       [](std::monostate &) {},
                       [&](Terminator::Jump &j) { resolve_jump(j, s); },
                       [&](Terminator::Dispatch &d) {
                           d.cond = s.resolve(d.cond);
                           for (auto &target : d.targets) {
                               resolve_jump(target, s);
                           }
                       },
                       [&](Terminator::Return &r) {
                           if (r.value != nullptr) {
                               r.value = s.resolve(r.value);
                           }
                       },
                       [&](Terminator::ParFor &p) {
                           p.start = s.resolve(p.start);
                           p.end = s.resolve(p.end);
                           p.stride = s.resolve(p.stride);
                           resolve_jump(p.body, s);
                           resolve_jump(p.cont, s);
                       },
                       [](Terminator::Yield &) {},
                       [&](Terminator::Call &c) {
                           resolve_jump(c.call, s);
                           resolve_jump(c.cont, s);
                       },
                       [&](Terminator::MultiCall &c) {
                           resolve_jump(c.call, s);
                           resolve_jump(c.cont, s);
                           for (auto &vs : c.varying) {
                               for (auto &v : vs) {
                                   v = s.resolve(v);
                               }
                           }
                           for (auto &k : c.keys) {
                               k = s.resolve(k);
                           }
                       },
                   },
                   block->terminator.data);
    }
}

using UseCounts = std::map<const Instruction *, size_t>;

void count(const ValuePtr &value, UseCounts &uses) {
    if (const Instruction *d = def_of(value)) {
        ++uses[d];
    }
}

void count(const Terminator::Jump &jump, UseCounts &uses) {
    for (const auto &arg : jump.args) {
        count(arg, uses);
    }
}

// How many times each instruction's result is named anywhere in the
// function, terminators included.
UseCounts use_counts(const Function &func) {
    UseCounts uses;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            for (const auto &operand : instr->operands) {
                count(operand, uses);
            }
        }
        std::visit(overloads{
                       [](const std::monostate &) {},
                       [&](const Terminator::Jump &j) { count(j, uses); },
                       [&](const Terminator::Dispatch &d) {
                           count(d.cond, uses);
                           for (const auto &target : d.targets) {
                               count(target, uses);
                           }
                       },
                       [&](const Terminator::Return &r) {
                           count(r.value, uses);
                       },
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
                       [&](const Terminator::MultiCall &c) {
                           count(c.call, uses);
                           count(c.cont, uses);
                           for (const auto &vs : c.varying) {
                               for (const auto &v : vs) {
                                   count(v, uses);
                               }
                           }
                           for (const auto &k : c.keys) {
                               count(k, uses);
                           }
                       },
                   },
                   block->terminator.data);
    }
    return uses;
}

// Removes the instructions `dead` says to, and the lookups that named them.
void erase(Function &func, const std::set<const Instruction *> &dead) {
    for (const auto &block : func.blocks) {
        std::erase_if(block->instrs, [&](const shared_ptr<Instruction> &in) {
            return dead.count(in.get()) > 0;
        });
        std::erase_if(block->lookups, [&](const auto &entry) {
            return dead.count(def_of(entry.second)) > 0;
        });
    }
}

// Removes every pure instruction nothing reads, and then whatever that
// leaves unread, until nothing more goes.
void remove_dead(Function &func) {
    for (;;) {
        const UseCounts uses = use_counts(func);
        std::set<const Instruction *> dead;
        for (const auto &block : func.blocks) {
            for (const auto &instr : block->instrs) {
                if (pure(*instr) && uses.count(instr.get()) == 0) {
                    dead.insert(instr.get());
                }
            }
        }
        if (dead.empty()) {
            return;
        }
        erase(func, dead);
    }
}

} // namespace

void simplify(Function &func) {
    Simplifier s(func);
    for (const auto &block : func.blocks) {
        s.block = block;
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> instr = block->instrs[i];
            for (auto &operand : instr->operands) {
                operand = s.resolve(operand);
            }
            if (instr->name.empty()) {
                continue; // an effect, not a value
            }
            s.at = i;
            ValuePtr v = s.rule(instr->op, instr->type, instr->operands);
            // A rule may have put new instructions in front of this one.
            i = s.at;
            if (v == nullptr) {
                continue;
            }
            s.replaced[instr.get()] = v;
        }
    }
    if (s.replaced.empty()) {
        return;
    }

    resolve_uses(func, s);
    // A name looked up after this should find what took its place.
    for (const auto &block : func.blocks) {
        for (auto &[name, value] : block->lookups) {
            value = s.resolve(value);
        }
    }
    std::set<const Instruction *> gone;
    for (const auto &[instr, _] : s.replaced) {
        gone.insert(instr);
    }
    erase(func, gone);
    remove_dead(func);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
