#include "SSA/LoopArithmetic.h"

#include <variant>

namespace bonsai {
namespace ir {
namespace ssa {

using std::shared_ptr;
using std::string;

namespace {

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

} // namespace

bool varies_with(const shared_ptr<Value> &v, const string &index,
                 const std::set<string> &region) {
    std::set<const Instruction *> seen;
    return varies_with(v, index, region, seen);
}

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
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return int64_t(*u);
    }
    return std::nullopt;
}

shared_ptr<Value> index_constant(const Type &type, int64_t v) {
    return type.is_uint()
               ? std::make_shared<Value>(Constant{type, uint64_t(v)})
               : std::make_shared<Value>(Constant{type, v});
}

shared_ptr<Value> arith(Block &block, const Type &type, Instruction::Op op,
                        const shared_ptr<Value> &lhs,
                        const shared_ptr<Value> &rhs) {
    const auto a = as_int(lhs), b = as_int(rhs);
    if (a.has_value() && b.has_value()) {
        switch (op) {
        case Instruction::Op::Add:
            return index_constant(type, *a + *b);
        case Instruction::Op::Sub:
            return index_constant(type, *a - *b);
        case Instruction::Op::Mul:
            return index_constant(type, *a * *b);
        case Instruction::Op::Div:
            if (*b != 0) {
                return index_constant(type, *a / *b);
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

shared_ptr<Value> trip_count(Block &block, const Type &type,
                             const shared_ptr<Value> &begin,
                             const shared_ptr<Value> &end,
                             const shared_ptr<Value> &stride) {
    auto span = arith(block, type, Instruction::Op::Sub, end, begin);
    const auto by = as_int(stride);
    if (by.has_value() && *by == 1) {
        // The count is the span itself: `(span + 1 - 1) / 1` would be two
        // instructions that say nothing, and a name a type may then carry.
        return span;
    }
    auto bumped = arith(block, type, Instruction::Op::Add, span, stride);
    auto less_one = arith(block, type, Instruction::Op::Sub, bumped,
                          index_constant(type, 1));
    return arith(block, type, Instruction::Op::Div, less_one, stride);
}

shared_ptr<Value> normalized_index(Block &block, const Type &type,
                                   const shared_ptr<Value> &index,
                                   const shared_ptr<Value> &begin,
                                   const shared_ptr<Value> &stride) {
    auto from_zero = arith(block, type, Instruction::Op::Sub, index, begin);
    return arith(block, type, Instruction::Op::Div, from_zero, stride);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
