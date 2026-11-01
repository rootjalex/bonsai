#include "SSA/LaunchValues.h"

#include "Error.h"
#include "SSA/CodeGen_Stmt.h"
#include "SSA/Definitions.h"

#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::string;

// Whether an instruction's value is a function of its operands alone -- no
// read of memory, no effect, no draw of a random number -- so that the host
// may compute it again from the operands computed again. An element of a
// vector value is a register operation; an element of an array is a read of
// memory. An intrinsic is not admitted: the ones a launch value could want
// (a maximum, say) are operations of their own above, and the rest read,
// write or draw.
bool pure_instruction(const Instruction &in) {
    switch (in.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::Inf:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::SizeOf:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
        return true;
    case Instruction::Op::ExtractIdx:
        return !in.operands.empty() && in.operands[0]->get_type().is<Vector_t>();
    default:
        return false;
    }
}

Expr constant_expr(const Constant &c) {
    return std::visit(
        overloads{
            [](bool b) -> Expr { return BoolImm::make(b); },
            [&](int64_t i) -> Expr { return IntImm::make(c.type, i); },
            [&](uint64_t u) -> Expr { return UIntImm::make(c.type, u); },
            [&](double d) -> Expr { return FloatImm::make(c.type, d); },
            [](const string &) -> Expr { return Expr(); },
            [&](const Undefined &) -> Expr { return Undef::make(c.type); },
        },
        c.data);
}

struct Resolver {
    const Function &func;
    const Block &body_head;
    Definitions defs;
    string *why;
    // Where each capture of the body is defined -- the function's parameter
    // or the instruction the launching block passes for it -- so that a
    // value of the body that resolves to the same definition is that
    // capture, however many block arguments it went through on the way.
    std::vector<std::pair<Definition, const Argument *>> captures;
    // Whether the head's first argument is the loop's index, which no value
    // of the launch may depend on: a body head's is; a function's entry
    // block, whose arguments are all parameters, has none.
    const bool index_first;

    Resolver(const Function &func, const Block &body_head, string *why,
             bool index_first = true)
        : func(func), body_head(body_head),
          defs(func, /*lenient=*/true, /*bound=*/true), why(why),
          index_first(index_first) {
        for (size_t i = index_first ? 1 : 0; i < body_head.args.size(); i++) {
            const Argument &arg = body_head.args[i];
            captures.emplace_back(
                defs.of(body_head.name, std::make_shared<Value>(arg)), &arg);
        }
    }

    std::optional<Expr> fail(const string &reason) {
        if (why != nullptr) {
            *why = reason;
        }
        return std::nullopt;
    }

    std::optional<Expr> constant(const Constant &c) {
        Expr e = constant_expr(c);
        return e.defined() ? std::optional<Expr>(e) : fail("a string constant");
    }

    std::optional<Expr> of(const string &block, const shared_ptr<Value> &v) {
        if (const auto *c = std::get_if<Constant>(&v->data)) {
            return constant(*c);
        }
        const Definition d = defs.of(block, v);
        if (!d.value) {
            return fail("a value with no definition");
        }
        if (const auto *c = std::get_if<Constant>(&d.value->data)) {
            return constant(*c);
        }
        for (const auto &[definition, arg] : captures) {
            if (same_definition(definition, d)) {
                return Var::make(arg->type, arg->name);
            }
        }
        if (const auto *a = std::get_if<Argument>(&d.value->data)) {
            if (index_first && d.block == body_head.name &&
                !body_head.args.empty() && body_head.args.front().name == a->name) {
                return fail("the loop's own index `" + a->name +
                            "`, which differs from one iteration to the next");
            }
            return fail("`" + a->name + "`, a value merged or carried inside the body (in " +
                        d.block + ")");
        }
        const auto &in = std::get<shared_ptr<Instruction>>(d.value->data);
        if (!pure_instruction(*in)) {
            return fail("`" + in->name + "`, computed by " + string(op_name(in->op)) +
                        ", which reads memory, has an effect or draws a random number");
        }
        std::vector<Expr> operands;
        operands.reserve(in->operands.size());
        for (const auto &operand : in->operands) {
            const std::optional<Expr> o = of(d.block, operand);
            if (!o.has_value()) {
                return std::nullopt;
            }
            operands.push_back(*o);
        }
        return pure_instruction_expr(*in, std::move(operands));
    }

    // An expression a body's allocation is sized by (SSA/Storage.h,
    // as_expr): a constant, or a variable naming a value of the body -- an
    // instruction's, found in its block's table, or an argument reached by
    // name -- which the definitions resolve as `block` refers to it.
    std::optional<Expr> of(const string &block, const Expr &e) {
        if (e.is<IntImm>() || e.is<UIntImm>() || e.is<BoolImm>() || e.is<FloatImm>()) {
            return e;
        }
        const Var *var = e.as<Var>();
        if (var == nullptr) {
            return fail("an expression the body computes");
        }
        for (const auto &b : func.blocks) {
            if (const auto it = b->lookups.find(var->name); it != b->lookups.end()) {
                return of(block, it->second);
            }
        }
        return of(block, std::make_shared<Value>(Argument{var->type, var->name}));
    }
};

} // namespace

std::optional<Expr> launch_value(const Function &func, const Block &body_head,
                                 const string &block, const shared_ptr<Value> &v,
                                 string *why) {
    return Resolver(func, body_head, why).of(block, v);
}

std::optional<Expr> launch_value(const Function &func, const Block &body_head,
                                 const string &block, const Expr &e, string *why) {
    return Resolver(func, body_head, why).of(block, e);
}

std::optional<Expr> parameter_value(const Function &func, const Expr &e, string *why) {
    internal_assert(!func.blocks.empty());
    const Block &entry = *func.blocks.front();
    return Resolver(func, entry, why, /*index_first=*/false).of(entry.name, e);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
