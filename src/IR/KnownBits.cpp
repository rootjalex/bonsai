#include "IR/KnownBits.h"

#include "IR/Stmt.h"

#include <map>
#include <optional>
#include <string>

namespace bonsai {
namespace ir {

namespace {

// Every bit of a type: the mask of a value of `t`, or every bit of a 64-bit
// word for anything that is not a scalar integer narrower than that.
uint64_t full(const Type &t) {
    if (t.is_vector() || !(t.is_int() || t.is_uint() || t.is<Bool_t>())) {
        return ~uint64_t(0);
    }
    if (t.is<Bool_t>()) {
        return 1;
    }
    const uint32_t bits = t.bits();
    return bits >= 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
}

// One past the highest set bit of `mask`; 0 for 0.
uint32_t width(uint64_t mask) {
    uint32_t w = 0;
    while (mask != 0) {
        w++;
        mask >>= 1;
    }
    return w;
}

// Every bit up to and including `w - 1`.
uint64_t low(uint32_t w) {
    return w >= 64 ? ~uint64_t(0) : (uint64_t(1) << w) - 1;
}

// The variables in scope and the bits each may have.
using Env = std::map<std::string, uint64_t>;

struct Analysis {
    const FuncMap &funcs;
    // How deep through calls the analysis follows, against recursion.
    int depth = 0;
    static constexpr int kMaxDepth = 8;

    explicit Analysis(const FuncMap &funcs) : funcs(funcs) {}

    uint64_t bits(const Expr &e, const Env &env) {
        const uint64_t all = full(e.type());
        if (const auto *u = e.as<UIntImm>()) {
            return u->value & all;
        }
        if (const auto *i = e.as<IntImm>()) {
            return i->value < 0 ? all : uint64_t(i->value) & all;
        }
        if (const auto *b = e.as<BoolImm>()) {
            return b->value ? 1 : 0;
        }
        if (const auto *v = e.as<Var>()) {
            const auto found = env.find(v->name);
            return found != env.end() ? found->second & all : all;
        }
        if (const auto *c = e.as<Cast>()) {
            if (c->mode != Cast::Mode::Convert) {
                return all;
            }
            const Type &from = c->value.type();
            if (from.is_vector() || !(from.is_uint() || from.is_int() || from.is<Bool_t>())) {
                return all; // a float's integer part: anything
            }
            if (from.is_int() && !e.type().is_int()) {
                // A negative value cast to unsigned is every bit; a value
                // whose top bit the analysis has cleared is non-negative.
                const uint64_t m = bits(c->value, env);
                return (m >> (from.bits() - 1)) & 1 ? all : m & all;
            }
            if (from.is_int() && e.type().is_int() && e.type().bits() > from.bits()) {
                // Sign extension: a possibly negative value fills the top.
                const uint64_t m = bits(c->value, env);
                return (m >> (from.bits() - 1)) & 1 ? all : m & all;
            }
            return bits(c->value, env) & all;
        }
        if (const auto *s = e.as<Select>()) {
            return (bits(s->tvalue, env) | bits(s->fvalue, env)) & all;
        }
        if (const auto *b = e.as<BinOp>()) {
            const auto shift_amount = [&]() -> std::optional<uint32_t> {
                if (const auto *u = b->b.as<UIntImm>()) {
                    return u->value < 64 ? std::optional<uint32_t>(uint32_t(u->value))
                                         : std::nullopt;
                }
                if (const auto *i = b->b.as<IntImm>()) {
                    return i->value >= 0 && i->value < 64
                               ? std::optional<uint32_t>(uint32_t(i->value))
                               : std::nullopt;
                }
                return std::nullopt;
            };
            switch (b->op) {
            case BinOp::BwAnd:
                return bits(b->a, env) & bits(b->b, env) & all;
            case BinOp::BwOr:
            case BinOp::Xor:
                return (bits(b->a, env) | bits(b->b, env)) & all;
            case BinOp::Shl: {
                const auto k = shift_amount();
                return k.has_value() ? (bits(b->a, env) << *k) & all : all;
            }
            case BinOp::Shr: {
                if (e.type().is_int()) {
                    return all; // an arithmetic shift may sign-extend
                }
                const auto k = shift_amount();
                const uint64_t m = bits(b->a, env);
                return (k.has_value() ? m >> *k : m) & all;
            }
            case BinOp::Add: {
                // A carry reaches one bit past the wider operand.
                const uint32_t w = std::max(width(bits(b->a, env)), width(bits(b->b, env)));
                return low(w + 1) & all;
            }
            case BinOp::Mul: {
                const uint32_t w = width(bits(b->a, env)) + width(bits(b->b, env));
                return low(w) & all;
            }
            case BinOp::Div:
            case BinOp::Mod:
                // No larger than the dividend (unsigned); signed may negate.
                return e.type().is_int() ? all : low(width(bits(b->a, env))) & all;
            case BinOp::Eq:
            case BinOp::Neq:
            case BinOp::Lt:
            case BinOp::Le:
            case BinOp::LAnd:
            case BinOp::LOr:
                return 1 & all;
            case BinOp::Sub:
            default:
                return all;
            }
        }
        if (const auto *call = e.as<Call>()) {
            const Var *f = call->func.as<Var>();
            if (f == nullptr || depth >= kMaxDepth) {
                return all;
            }
            const auto found = funcs.find(f->name);
            if (found == funcs.end() || !found->second->body.defined() ||
                found->second->args.size() != call->args.size()) {
                return all;
            }
            Env inner;
            for (size_t i = 0; i < call->args.size(); i++) {
                inner[found->second->args[i].name] = bits(call->args[i], env);
            }
            depth++;
            const std::optional<uint64_t> result = body_bits(found->second->body, inner);
            depth--;
            return result.has_value() ? *result & all : all;
        }
        return all;
    }

    // The bits a function body's returned value may have: the body read as
    // straight-line statements with branches, the variables it binds
    // tracked in `env`. Nullopt when the body has a statement the analysis
    // does not read (a loop, a store through an index), or no return.
    std::optional<uint64_t> body_bits(const Stmt &s, Env &env) {
        returned = 0;
        gave_up = false;
        walk(s, env);
        if (gave_up) {
            return std::nullopt;
        }
        return returned;
    }

    uint64_t returned = 0;
    bool gave_up = false;

    // A plain variable write: `x = e` or `x : T = e`, with nothing indexed.
    static std::optional<std::string> plain(const WriteLoc &loc) {
        if (!loc.accesses.empty()) {
            return std::nullopt;
        }
        return loc.base;
    }

    void walk(const Stmt &s, Env &env) {
        if (gave_up || !s.defined()) {
            return;
        }
        if (const auto *seq = s.as<Sequence>()) {
            for (const Stmt &one : seq->stmts) {
                walk(one, env);
            }
            return;
        }
        if (const auto *let = s.as<LetStmt>()) {
            if (const auto name = plain(let->loc)) {
                env[*name] = bits(let->value, env);
                return;
            }
            gave_up = true;
            return;
        }
        if (const auto *alloc = s.as<Allocate>()) {
            if (const auto name = plain(alloc->loc)) {
                env[*name] = alloc->value.defined() ? bits(alloc->value, env)
                                                    : full(alloc->loc.type);
                return;
            }
            gave_up = true;
            return;
        }
        if (const auto *store = s.as<Store>()) {
            if (const auto name = plain(store->loc); name && !store->mask.defined()) {
                env[*name] = bits(store->value, env);
                return;
            }
            gave_up = true;
            return;
        }
        if (const auto *ret = s.as<Return>()) {
            if (ret->value.defined()) {
                returned |= bits(ret->value, env);
            }
            return;
        }
        if (const auto *branch = s.as<IfElse>()) {
            // Both arms from the same scope; afterwards a variable may hold
            // either arm's value.
            Env then_env = env, else_env = env;
            walk(branch->then_body, then_env);
            walk(branch->else_body, else_env);
            for (auto &[name, mask] : env) {
                const auto t = then_env.find(name);
                const auto o = else_env.find(name);
                mask = (t != then_env.end() ? t->second : mask) |
                       (o != else_env.end() ? o->second : mask);
            }
            return;
        }
        if (s.as<Print>() != nullptr || s.as<Label>() != nullptr) {
            return;
        }
        gave_up = true; // a loop, a call statement, an indexed store...
    }
};

} // namespace

uint64_t possible_bits(const Expr &e, const FuncMap &funcs) {
    Analysis analysis(funcs);
    const Env env;
    return analysis.bits(e, env);
}

uint32_t live_bits(const Expr &e, const FuncMap &funcs) {
    return std::max<uint32_t>(1, width(possible_bits(e, funcs)));
}

} // namespace ir
} // namespace bonsai
