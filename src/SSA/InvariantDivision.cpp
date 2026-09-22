#include "SSA/InvariantDivision.h"

#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "Error.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;
using ValuePtr = shared_ptr<Value>;

// Where a value comes from: an instruction, or a block whose argument it is
// with no single instruction behind it -- a function parameter, a call's
// result, a loop's index, or a join of different values.
struct Definition {
    string block;
    // The defining instruction, or null for an argument of `block`.
    shared_ptr<Instruction> instr;
    // The argument's name when `instr` is null.
    string arg;

    bool operator<(const Definition &o) const {
        if (block != o.block) {
            return block < o.block;
        }
        if (instr.get() != o.instr.get()) {
            return instr.get() < o.instr.get();
        }
        return arg < o.arg;
    }
};

// Follows an argument of `block` back through the jumps that pass it, block
// argument by block argument, until it reaches what the arguments all stand
// for. An argument fed the same instruction by every path is that
// instruction, wherever the path came in; one fed different things -- or a
// constant, or nothing, as a call's result or a parfor's index is -- is a
// definition of its own, at the block that receives it.
class Tracer {
  public:
    // `boundary`, when given, is a block the trace does not look behind: the
    // entry of a region being vectorized, whose arguments are the values it
    // receives from outside. A divisor defined beyond it is then "the
    // region's argument", and its multiplier is computed at the region's
    // top -- once per gang -- rather than threaded in from outside, which
    // the edge into a parfor's body does not carry.
    Tracer(const Function &func, const Cfg &cfg, const string *boundary)
        : entry(func.blocks.front()->name),
          boundary(boundary != nullptr ? *boundary : ""), cfg(cfg) {}

    optional<Definition> trace(const ValuePtr &v, const string &in_block) {
        if (const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data)) {
            const shared_ptr<Block> owner = (*held)->owner.lock();
            internal_assert(owner) << "instruction " << (*held)->name
                                   << " has no block";
            return Definition{owner->name, *held, ""};
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            set<std::pair<string, string>> visiting;
            const optional<Definition> def = resolve(in_block, a->name, visiting);
            // Reached only around a cycle, which cannot happen from the
            // outside; the argument is then its own definition here.
            return def.has_value() ? *def
                                   : Definition{in_block, nullptr, a->name};
        }
        return std::nullopt; // a constant
    }

  private:
    const string entry;
    const string boundary;
    const Cfg &cfg;

    // What the argument `name` of `block` stands for: the one definition
    // every path in passes, or the argument itself when the paths disagree,
    // pass a constant, or define it on the edge. Nothing, for a path that
    // has come back around to an argument it is already resolving -- the
    // value a loop hands itself on its back edge, which adds no definition.
    optional<Definition> resolve(const string &block, const string &name,
                                 set<std::pair<string, string>> &visiting) {
        if (!visiting.insert({block, name}).second) {
            return std::nullopt;
        }
        const Definition here{block, nullptr, name};
        const BlockId here_id = cfg.id(block);
        const shared_ptr<Block> &b = cfg.block(here_id);
        size_t k = 0;
        for (; k < b->args.size(); k++) {
            if (b->args[k].name == name) {
                break;
            }
        }
        if (k == b->args.size() || block == entry || block == boundary) {
            // A parameter of the function, or a name this cannot see
            // through.
            return here;
        }
        if (cfg.preds[here_id].empty()) {
            return here;
        }
        set<Definition> found;
        // Whether some path in came back around to an argument already being
        // resolved -- a loop handing a value to itself, through however many
        // blocks: a loop body with a call in it passes the value on to the
        // call's continuation, which passes it back to the header.
        bool cycled = false;
        for (BlockId p : cfg.preds[here_id]) {
            const string &pred = cfg.name(p);
            for (Terminator::Jump *jump : jumps_of(cfg[p])) {
                if (jump->name != block) {
                    continue;
                }
                // An edge that defines the block's first argument itself --
                // a call's result, a parfor's index -- passes one argument
                // fewer than the block takes.
                size_t at = k;
                if (jump->args.size() + 1 == b->args.size()) {
                    if (k == 0) {
                        return here;
                    }
                    at = k - 1;
                }
                if (at >= jump->args.size()) {
                    return here;
                }
                const ValuePtr &passed = jump->args[at];
                if (const auto *held =
                        std::get_if<shared_ptr<Instruction>>(&passed->data)) {
                    const shared_ptr<Block> owner = (*held)->owner.lock();
                    internal_assert(owner);
                    found.insert(Definition{owner->name, *held, ""});
                } else if (const auto *a = std::get_if<Argument>(&passed->data)) {
                    if (const optional<Definition> sub =
                            resolve(pred, a->name, visiting)) {
                        found.insert(*sub);
                    } else {
                        cycled = true;
                    }
                } else {
                    return here; // a constant on one path
                }
                if (found.size() > 1) {
                    return here;
                }
            }
        }
        if (found.size() == 1) {
            return *found.begin();
        }
        // Every path in came back around: this block adds no definition of
        // its own, and the answer is whatever the paths from outside the
        // cycle pass -- which the caller, further up the cycle, collects.
        if (found.empty() && cycled) {
            return std::nullopt;
        }
        return here;
    }
};

// Builds instructions into `block`, in front of `at`, keeping `at` pointing
// at the same instruction.
struct Emitter {
    Function &func;
    shared_ptr<Block> block;
    size_t at;

    ValuePtr constant(const Type &type, uint64_t k) const {
        if (type.is_uint()) {
            return std::make_shared<Value>(Constant{type, uint64_t(k)});
        }
        return std::make_shared<Value>(Constant{type, int64_t(k)});
    }

    ValuePtr emit(Instruction::Op op, const Type &type, vector<ValuePtr> operands,
                  ir::Intrinsic::OpType intrinsic = ir::Intrinsic::abs) {
        auto instr = std::make_shared<Instruction>(
            func.get_unique_name(), type, op, std::move(operands), block);
        instr->intrinsic = intrinsic;
        block->instrs.insert(block->instrs.begin() + long(at), instr);
        at++;
        auto value = std::make_shared<Value>(instr);
        const auto [_, inserted] = block->lookups.insert({instr->name, value});
        internal_assert(inserted) << instr->name << " already in " << block->name;
        return value;
    }

    ValuePtr intrinsic(ir::Intrinsic::OpType which, const Type &type,
                       vector<ValuePtr> operands) {
        return emit(Instruction::Op::Intrinsic, type, std::move(operands), which);
    }
};

// What a division by one divisor needs computed once: the multiplier and the
// shifts, by the names they are defined under, to be threaded to wherever a
// division sits.
struct Multiplier {
    Type type;
    string m;
    // Unsigned: the two shifts of the round-up method; signed: the one
    // shift, and whether the divisor is negative.
    string sh1, sh2;
    string negative;
};

optional<uint64_t> upper_bound(const ValuePtr &v, unsigned depth);

// What a divisor's magnitude is known to be at most: what upper_bound
// proves, or for a type of 32 bits or fewer, what the type says.
optional<uint64_t> divisor_bound(const ValuePtr &d, const Type &type) {
    if (const optional<uint64_t> b = upper_bound(d, 0)) {
        return b;
    }
    if (type.bits() <= 32) {
        return type.is_uint() ? (uint64_t(1) << type.bits()) - 1
                              : uint64_t(1) << (type.bits() - 1);
    }
    return std::nullopt;
}

// Does the two-step multiplier (bounded_multiplier) apply to a divisor of
// `type` known to be at most `bound`?
bool two_step_fits(const Type &type, const optional<uint64_t> &bound) {
    const uint64_t bits = type.bits();
    if (!bound || bits > 64 || bits < 8 || bits % 2 != 0) {
        return false;
    }
    const uint64_t half = bits / 2;
    return *bound < (uint64_t(1) << 53) / ((uint64_t(1) << half) + 1);
}

// The multiplier `numerator_high * 2^N / ad + 1` (mod 2^N) by two exact
// double divisions, when `ad` is known small enough for them to be exact;
// or nothing, and the intrinsic does it with a division twice the width.
//
// `numerator_high` is the multiple of 2^N being divided -- 2^l - d for the
// unsigned round-up method, 2^(l-1) for the signed one -- and is below
// `ad`. Long division in base 2^(N/2): with H = N/2, the quotient is
// q1 2^H + q2 where q1 = floor(numerator_high 2^H / ad), r1 its remainder,
// and q2 = floor(r1 2^H / ad); q1 < 2^H since numerator_high < ad, and
// q2 < 2^H since r1 < ad, so the two assemble without a carry, and the
// whole is below 2^N as Granlund & Montgomery's multiplier is. Each step
// is a correctly rounded double quotient truncated toward zero, which is
// the integer quotient whenever numerator plus divisor is below 2^53 (see
// CodeGen_LLVM::vector_int_division for the argument): step one needs
// numerator_high 2^H + ad < 2^53 and step two r1 2^H + ad < 2^53, and both
// hold when ad (2^H + 1) < 2^53 -- below 2^21 for 64-bit words. The two
// numerators are below ad 2^H and have to fit the word they are computed
// in: for a 64-bit word that is ad < 2^32, which the bound above implies;
// a word of 32 bits or fewer is computed in 64 bits, where any divisor of
// its width fits. A vector of divisors then costs two vector double
// divisions where the wide integer division costs one scalar divide, or a
// library call, per lane. Why a divisor that small is common: apps/pbrt
// divides a 64-bit Halton index by a prime from a u16 table.
optional<ValuePtr> bounded_multiplier(Emitter &e, const ValuePtr &ad,
                                      const ValuePtr &numerator_high,
                                      const Type &type,
                                      const optional<uint64_t> &bound) {
    const uint64_t bits = type.bits();
    if (!two_step_fits(type, bound)) {
        return std::nullopt;
    }
    const uint64_t half = bits / 2;
    // The word the two numerators are computed in: 64 bits, so that
    // ad 2^H fits it -- for a 64-bit word the bound (below 2^21) already says
    // so; for a narrower word every divisor of its width does.
    const bool widen_it = bits < 64;
    const Type work = widen_it ? UInt_t::make(64) : type;
    const auto up = [&](const ValuePtr &v) {
        return widen_it ? e.emit(Instruction::Op::Cast, work, {v}) : v;
    };
    const Type f64 = Float_t::make_f64();
    ValuePtr d = up(ad);
    ValuePtr shift = e.constant(work, half);
    ValuePtr fd = e.emit(Instruction::Op::Cast, f64, {d});
    ValuePtr n1 = e.emit(Instruction::Op::Shl, work, {up(numerator_high), shift});
    ValuePtr q1 = e.emit(Instruction::Op::Cast, work,
                         {e.emit(Instruction::Op::Div, f64,
                                 {e.emit(Instruction::Op::Cast, f64, {n1}), fd})});
    ValuePtr r1 = e.emit(Instruction::Op::Sub, work,
                         {n1, e.emit(Instruction::Op::Mul, work, {q1, d})});
    ValuePtr n2 = e.emit(Instruction::Op::Shl, work, {r1, shift});
    ValuePtr q2 = e.emit(Instruction::Op::Cast, work,
                         {e.emit(Instruction::Op::Div, f64,
                                 {e.emit(Instruction::Op::Cast, f64, {n2}), fd})});
    ValuePtr high = e.emit(Instruction::Op::Shl, work, {q1, shift});
    ValuePtr sum = e.emit(Instruction::Op::Add, work, {high, q2});
    ValuePtr m = e.emit(Instruction::Op::Add, work, {sum, e.constant(work, 1)});
    return widen_it ? e.emit(Instruction::Op::Cast, type, {m}) : m;
}

// Emits the multiplier of `d` into `e`, which is positioned right after the
// divisor's definition. `d` is the divisor as an operand valid there.
Multiplier emit_multiplier(Emitter &e, const ValuePtr &d, const Type &type) {
    const uint64_t bits = type.bits();
    Multiplier out;
    out.type = type;
    auto name_of = [](const ValuePtr &v) {
        return std::get<shared_ptr<Instruction>>(v->data)->name;
    };
    // The multiplier itself is the intrinsic: on a scalar, one hardware
    // division twice the width (x86's `div` takes a 128-bit dividend, and
    // libgcc's __udivti3 is that one instruction when the high word is below
    // the divisor, as it is here). A gang's vector of them is another
    // matter -- see expand_bounded_multipliers, which the vectorizer runs.
    ValuePtr m = e.intrinsic(ir::Intrinsic::div_multiplier, type, {d});
    out.m = name_of(m);
    if (type.is_uint()) {
        // l = ceil(log2 d) = N - clz(d - 1); sh1 = min(l, 1), sh2 = max(l, 1)
        // - 1 (Granlund & Montgomery, figure 4.2). With sh1 = 0 for d = 1
        // the quotient below comes out as n itself, so d = 1 needs no case
        // of its own.
        ValuePtr dm1 = e.emit(Instruction::Op::Sub, type, {d, e.constant(type, 1)});
        ValuePtr clz = e.intrinsic(ir::Intrinsic::clz, type, {dm1});
        ValuePtr l = e.emit(Instruction::Op::Sub, type, {e.constant(type, bits), clz});
        ValuePtr sh1 = e.emit(Instruction::Op::Min, type, {l, e.constant(type, 1)});
        ValuePtr l1 = e.emit(Instruction::Op::Max, type, {l, e.constant(type, 1)});
        ValuePtr sh2 = e.emit(Instruction::Op::Sub, type, {l1, e.constant(type, 1)});
        out.sh1 = name_of(sh1);
        out.sh2 = name_of(sh2);
        return out;
    }
    // Signed: l = max(ceil(log2 |d|), 1), the shift is l - 1, and the sign of
    // d is applied to the quotient at the end (figure 5.2).
    ValuePtr zero = e.constant(type, 0);
    ValuePtr negative = e.emit(Instruction::Op::Lt, Bool_t::make(), {d, zero});
    ValuePtr negated = e.emit(Instruction::Op::Sub, type, {zero, d});
    ValuePtr ad = e.emit(Instruction::Op::Select, type, {negative, negated, d});
    ValuePtr adm1 = e.emit(Instruction::Op::Sub, type, {ad, e.constant(type, 1)});
    ValuePtr clz = e.intrinsic(ir::Intrinsic::clz, type, {adm1});
    ValuePtr l = e.emit(Instruction::Op::Sub, type, {e.constant(type, bits), clz});
    ValuePtr l1 = e.emit(Instruction::Op::Max, type, {l, e.constant(type, 1)});
    ValuePtr sh = e.emit(Instruction::Op::Sub, type, {l1, e.constant(type, 1)});
    out.sh1 = name_of(sh);
    out.negative = name_of(negative);
    return out;
}

// The two-step form of a div_multiplier `instr` whose divisor is bounded,
// emitted in front of it in `block` at `index`; `instr` becomes the result.
// Returns how many instructions were added in front. See
// expand_bounded_multipliers for when this is wanted.
size_t expand_multiplier(Function &func, const shared_ptr<Block> &block,
                         size_t index, const optional<uint64_t> &bound) {
    const shared_ptr<Instruction> instr = block->instrs[index];
    const Type type = instr->type;
    const uint64_t bits = type.bits();
    const ValuePtr d = instr->operands[0];
    Emitter e{func, block, index};
    ValuePtr zero = e.constant(type, 0);
    ValuePtr one = e.constant(type, 1);
    optional<ValuePtr> m;
    if (type.is_uint()) {
        // m' = floor(2^N (2^l - d) / d) + 1. A zero divisor divides by one
        // instead: its multiplier is never used, but this is computed where
        // the divisor is defined, for lanes that never divide too. 2^l as
        // an N-bit number is zero when l = N, and a shift by N is not.
        ValuePtr ds = e.emit(Instruction::Op::Select, type,
                             {e.emit(Instruction::Op::Eq, Bool_t::make(), {d, zero}),
                              one, d});
        ValuePtr dm1 = e.emit(Instruction::Op::Sub, type, {ds, one});
        ValuePtr clz = e.intrinsic(ir::Intrinsic::clz, type, {dm1});
        ValuePtr l = e.emit(Instruction::Op::Sub, type, {e.constant(type, bits), clz});
        ValuePtr two_l = e.emit(
            Instruction::Op::Select, type,
            {e.emit(Instruction::Op::Eq, Bool_t::make(), {l, e.constant(type, bits)}),
             zero, e.emit(Instruction::Op::Shl, type, {one, l})});
        ValuePtr a = e.emit(Instruction::Op::Sub, type, {two_l, ds});
        m = bounded_multiplier(e, ds, a, type, bound);
    } else {
        // m = floor(2^(N + l - 1) / |d|) + 1, taken modulo 2^N: the multiple
        // of 2^N divided is 2^(l - 1), l = max(ceil(log2 |d|), 1). The bound
        // is on |d|: upper_bound bounds only a value it knows non-negative,
        // and a narrow type bounds its magnitude.
        ValuePtr negative = e.emit(Instruction::Op::Lt, Bool_t::make(), {d, zero});
        ValuePtr ad = e.emit(Instruction::Op::Select, type,
                             {negative, e.emit(Instruction::Op::Sub, type, {zero, d}), d});
        ValuePtr ads = e.emit(Instruction::Op::Select, type,
                              {e.emit(Instruction::Op::Eq, Bool_t::make(), {ad, zero}),
                               one, ad});
        ValuePtr adm1 = e.emit(Instruction::Op::Sub, type, {ads, one});
        ValuePtr clz = e.intrinsic(ir::Intrinsic::clz, type, {adm1});
        ValuePtr l = e.emit(Instruction::Op::Sub, type, {e.constant(type, bits), clz});
        ValuePtr l1 = e.emit(Instruction::Op::Max, type, {l, one});
        ValuePtr sh = e.emit(Instruction::Op::Sub, type, {l1, one});
        ValuePtr half_pow = e.emit(Instruction::Op::Shl, type, {one, sh});
        m = bounded_multiplier(e, ads, half_pow, type, bound);
    }
    internal_assert(m.has_value())
        << "the two-step multiplier was asked for a divisor it does not fit: "
        << instr->name;
    // The instruction keeps its name, as the result plus nothing, which the
    // simplifier folds away.
    instr->op = Instruction::Op::Add;
    instr->operands = {*m, zero};
    return e.at - index;
}

// Rewrites the division `instr`, at `index` in `block`, to use `mult`. The
// instruction keeps its name and type, so nothing that reads it changes.
void rewrite_division(Function &func, const shared_ptr<Block> &block,
                      size_t index, const Multiplier &mult) {
    const shared_ptr<Instruction> instr = block->instrs[index];
    const Type type = instr->type;
    const bool is_mod = instr->op == Instruction::Op::Mod;
    const ValuePtr n = instr->operands[0];
    const ValuePtr d = instr->operands[1];

    ValuePtr m = block->get_value(mult.m, type);
    Emitter e{func, block, index};
    ValuePtr q;
    if (type.is_uint()) {
        // q = (t + ((n - t) >> sh1)) >> sh2 with t = mulhi(m, n).
        ValuePtr sh1 = block->get_value(mult.sh1, type);
        ValuePtr sh2 = block->get_value(mult.sh2, type);
        ValuePtr t = e.intrinsic(ir::Intrinsic::mulhi, type, {m, n});
        ValuePtr diff = e.emit(Instruction::Op::Sub, type, {n, t});
        ValuePtr half = e.emit(Instruction::Op::Shr, type, {diff, sh1});
        ValuePtr sum = e.emit(Instruction::Op::Add, type, {t, half});
        if (is_mod) {
            q = e.emit(Instruction::Op::Shr, type, {sum, sh2});
        } else {
            instr->op = Instruction::Op::Shr;
            instr->operands = {sum, sh2};
            q = std::make_shared<Value>(instr);
        }
    } else {
        // q = (mulhi(m, n) + n) >> sh, plus one when negative, negated for a
        // negative divisor. The arithmetic shift rounds toward minus
        // infinity, and a negative quotient is one short of the truncation
        // C and this IR define.
        ValuePtr sh = block->get_value(mult.sh1, type);
        ValuePtr negative = block->get_value(mult.negative, Bool_t::make());
        ValuePtr zero = e.constant(type, 0);
        ValuePtr t = e.intrinsic(ir::Intrinsic::mulhi, type, {m, n});
        ValuePtr sum = e.emit(Instruction::Op::Add, type, {t, n});
        ValuePtr q1 = e.emit(Instruction::Op::Shr, type, {sum, sh});
        ValuePtr below = e.emit(Instruction::Op::Lt, Bool_t::make(), {q1, zero});
        ValuePtr q1p = e.emit(Instruction::Op::Add, type, {q1, e.constant(type, 1)});
        ValuePtr q2 = e.emit(Instruction::Op::Select, type, {below, q1p, q1});
        ValuePtr nq2 = e.emit(Instruction::Op::Sub, type, {zero, q2});
        if (is_mod) {
            q = e.emit(Instruction::Op::Select, type, {negative, nq2, q2});
        } else {
            instr->op = Instruction::Op::Select;
            instr->operands = {negative, nq2, q2};
            q = std::make_shared<Value>(instr);
        }
    }
    if (is_mod) {
        // n - q * d.
        ValuePtr qd = e.emit(Instruction::Op::Mul, type, {q, d});
        instr->op = Instruction::Op::Sub;
        instr->operands = {n, qd};
    }
}

// --- Constant divisors ----------------------------------------------------------
//
// A constant divisor's multiplier is computed here, at compile time, by the
// algorithms of Hacker's Delight chapter 10 -- `magicu` (figure 10-2) for an
// unsigned divisor and `magic` (figure 10-1) for a signed one, written for a
// word of N bits -- and the division becomes the shortest sequence that
// works for that constant, which is how Halide lowers one (lower_int_uint_div
// in CodeGen_Internal.cpp, over the tables of IntegerDivisionTable.h): a
// shift alone for a power of two; a multiply-high and a shift when the
// multiplier fits the word; and the round-up form, a multiply-high, a
// halved difference and a shift, when it needs one bit more. Halide's tables
// hold the same numbers, generated once; computing them costs nothing at
// compile time and reaches 64-bit divisors, which the tables stop short of.
//
// Signed division truncates toward zero here, as C's does and as ir::BinOp
// defines it, so the signed sequence is Hacker's Delight's (section 10-4)
// and not Halide's, whose division rounds toward minus infinity.

// The multiplier and shift for one constant: the multiplier as the N-bit
// pattern it is, and for an unsigned divisor whether the round-up form is
// needed (Hacker's Delight's "add indicator").
struct Magic {
    uint64_t multiplier;
    unsigned shift;
    bool add;
};

// Hacker's Delight figure 10-2, `magicu`, for N bits: `d` in [2, 2^N - 1] and
// not a power of two. All arithmetic modulo 2^N, as the figure's is modulo
// 2^32.
Magic magic_unsigned(uint64_t d, unsigned bits) {
    const uint64_t mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    const auto m = [mask](uint64_t x) { return x & mask; };
    const uint64_t half = uint64_t(1) << (bits - 1); // 2^(N-1)
    Magic out{0, 0, false};
    // nc = 2^N - 1 - (2^N mod d): the largest multiple of d below 2^N,
    // less one.
    const uint64_t nc = m(mask - (m(0 - d) % d));
    unsigned p = bits - 1;
    uint64_t q1 = half / nc;          // 2^p / nc
    uint64_t r1 = half - q1 * nc;     // 2^p mod nc
    uint64_t q2 = (half - 1) / d;     // (2^p - 1) / d
    uint64_t r2 = (half - 1) - q2 * d; // (2^p - 1) mod d
    uint64_t delta = 0;
    do {
        p++;
        if (r1 >= nc - r1) {
            q1 = m(2 * q1 + 1);
            r1 = m(2 * r1 - nc);
        } else {
            q1 = m(2 * q1);
            r1 = m(2 * r1);
        }
        if (r2 + 1 >= d - r2) {
            if (q2 >= half - 1) {
                out.add = true;
            }
            q2 = m(2 * q2 + 1);
            r2 = m(2 * r2 + 1 - d);
        } else {
            if (q2 >= half) {
                out.add = true;
            }
            q2 = m(2 * q2);
            r2 = m(2 * r2 + 1);
        }
        delta = d - 1 - r2;
    } while (p < 2 * bits && (q1 < delta || (q1 == delta && r1 == 0)));
    out.multiplier = m(q2 + 1);
    out.shift = p - bits;
    return out;
}

// Hacker's Delight figure 10-1, `magic`, for N bits: `d` with 2 <= |d|, the
// most negative value included. The multiplier comes back as an N-bit
// pattern to be read as signed.
Magic magic_signed(uint64_t pattern, unsigned bits) {
    const uint64_t mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    const auto m = [mask](uint64_t x) { return x & mask; };
    const uint64_t half = uint64_t(1) << (bits - 1);
    const uint64_t ud = m(pattern);
    const bool negative = (ud & half) != 0;
    const uint64_t ad = negative ? m(0 - ud) : ud; // |d|, 2^(N-1) for the minimum
    const uint64_t t = half + (ud >> (bits - 1));
    const uint64_t anc = t - 1 - t % ad;
    unsigned p = bits - 1;
    uint64_t q1 = half / anc;
    uint64_t r1 = half - q1 * anc;
    uint64_t q2 = half / ad;
    uint64_t r2 = half - q2 * ad;
    uint64_t delta = 0;
    do {
        p++;
        q1 = m(2 * q1);
        r1 = m(2 * r1);
        if (r1 >= anc) {
            q1 = m(q1 + 1);
            r1 = m(r1 - anc);
        }
        q2 = m(2 * q2);
        r2 = m(2 * r2);
        if (r2 >= ad) {
            q2 = m(q2 + 1);
            r2 = m(r2 - ad);
        }
        delta = ad - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));
    uint64_t multiplier = m(q2 + 1);
    if (negative) {
        multiplier = m(0 - multiplier);
    }
    return Magic{multiplier, p - bits, false};
}

// The N-bit pattern a constant divisor holds, or nothing for a divisor that
// is not a constant.
optional<uint64_t> constant_divisor(const ValuePtr &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return uint64_t(*i);
    }
    return std::nullopt;
}

// A constant of `type` with the N-bit `pattern`, sign-extended for a signed
// type so that the constant reads as the two's complement value it is.
ValuePtr pattern_constant(const Type &type, uint64_t pattern) {
    const unsigned bits = unsigned(type.bits());
    const uint64_t mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    pattern &= mask;
    if (type.is_uint()) {
        return std::make_shared<Value>(Constant{type, pattern});
    }
    const uint64_t sign = uint64_t(1) << (bits - 1);
    const int64_t value =
        (pattern & sign) != 0 ? int64_t(pattern | ~mask) : int64_t(pattern);
    return std::make_shared<Value>(Constant{type, value});
}

// What `v` is known to be at most, when it is known to be non-negative and
// bounded by what it is built from: constants, selects among them, sums and
// products of them, masks, shifts, casts from narrower unsigned types, and
// remainders by constants. Nothing for anything else -- an argument, a
// load -- which is where the search stops. What Halide's simplifier knows
// through bounds inference, in the little of it a division needs: a
// dividend that stays below twice its divisor is divided by one comparison.
optional<uint64_t> upper_bound(const ValuePtr &v, unsigned depth = 0) {
    if (depth > 8) {
        return std::nullopt;
    }
    if (const auto *c = std::get_if<Constant>(&v->data)) {
        if (const auto *u = std::get_if<uint64_t>(&c->data)) {
            return *u;
        }
        if (const auto *i = std::get_if<int64_t>(&c->data)) {
            return *i >= 0 ? optional<uint64_t>(uint64_t(*i)) : std::nullopt;
        }
        if (const auto *b = std::get_if<bool>(&c->data)) {
            return *b ? 1 : 0;
        }
        return std::nullopt;
    }
    const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
    if (held == nullptr) {
        return std::nullopt;
    }
    const Instruction &in = **held;
    const auto bound = [&](size_t k) -> optional<uint64_t> {
        return k < in.operands.size() ? upper_bound(in.operands[k], depth + 1)
                                      : std::nullopt;
    };
    const auto both = [&]() -> optional<std::pair<uint64_t, uint64_t>> {
        const auto a = bound(0);
        const auto b = bound(1);
        if (a && b) {
            return std::make_pair(*a, *b);
        }
        return std::nullopt;
    };
    // Are these the same value: the same instruction, argument or constant?
    const auto same = [](const ValuePtr &a, const ValuePtr &b) {
        if (const auto *ia = std::get_if<shared_ptr<Instruction>>(&a->data)) {
            const auto *ib = std::get_if<shared_ptr<Instruction>>(&b->data);
            return ib != nullptr && ia->get() == ib->get();
        }
        if (const auto *aa = std::get_if<Argument>(&a->data)) {
            const auto *ab = std::get_if<Argument>(&b->data);
            return ab != nullptr && aa->name == ab->name;
        }
        return false;
    };
    switch (in.op) {
    case Instruction::Op::Select: {
        // `select(x < c, x, x - c)` is x reduced by c once -- what a
        // remainder becomes below (see rewrite_constant_division), and so
        // the shape the next remainder in a chain of them meets: at most
        // c - 1 where x stayed below c, and x's bound less c where it did
        // not.
        if (const auto *cond =
                std::get_if<shared_ptr<Instruction>>(&in.operands[0]->data);
            cond != nullptr && (*cond)->op == Instruction::Op::Lt &&
            same((*cond)->operands[0], in.operands[1])) {
            const auto *sub =
                std::get_if<shared_ptr<Instruction>>(&in.operands[2]->data);
            const auto c = constant_divisor((*cond)->operands[1]);
            if (sub != nullptr && (*sub)->op == Instruction::Op::Sub &&
                same((*sub)->operands[0], in.operands[1]) && c && *c > 0 &&
                (in.type.is_uint() || int64_t(*c) > 0)) {
                const auto d = constant_divisor((*sub)->operands[1]);
                const auto x = bound(1);
                if (d && *d == *c && x) {
                    return std::max(std::min(*c - 1, *x),
                                    *x >= *c ? *x - *c : 0);
                }
            }
        }
        const auto a = bound(1);
        const auto b = bound(2);
        return a && b ? optional<uint64_t>(std::max(*a, *b)) : std::nullopt;
    }
    case Instruction::Op::Reduce: {
        // Which lane holds the extremum is a lane index.
        if (in.reduce == ir::VectorReduce::Idxmax ||
            in.reduce == ir::VectorReduce::Idxmin) {
            const Type &of = in.operands[0]->get_type();
            return of.is_vector() ? optional<uint64_t>(of.lanes() - 1)
                                  : std::nullopt;
        }
        return std::nullopt;
    }
    case Instruction::Op::Bc:
        // Every lane holds the one value.
        return bound(0);
    case Instruction::Op::Ramp: {
        // base + stride * (lanes - 1) in the last lane.
        const auto base = bound(0);
        const auto stride = bound(1);
        if (base && stride && in.type.is_vector()) {
            uint64_t span = 0, top = 0;
            if (!__builtin_mul_overflow(*stride, uint64_t(in.type.lanes() - 1),
                                        &span) &&
                !__builtin_add_overflow(*base, span, &top)) {
                return top;
            }
        }
        return std::nullopt;
    }
    case Instruction::Op::Add: {
        const auto ab = both();
        if (ab && ab->first <= ~uint64_t(0) - ab->second) {
            return ab->first + ab->second;
        }
        return std::nullopt;
    }
    case Instruction::Op::Mul: {
        const auto ab = both();
        uint64_t product = 0;
        if (ab && !__builtin_mul_overflow(ab->first, ab->second, &product)) {
            return product;
        }
        return std::nullopt;
    }
    case Instruction::Op::BwAnd: {
        // Below either operand that is a non-negative constant, whatever the
        // other holds: the constant's sign bit is clear, so the result's is.
        for (size_t k = 0; k < 2; k++) {
            if (std::holds_alternative<Constant>(in.operands[k]->data)) {
                if (const auto c = bound(k)) {
                    const auto other = bound(1 - k);
                    return other ? std::min(*c, *other) : *c;
                }
            }
        }
        return std::nullopt;
    }
    case Instruction::Op::Shr: {
        const auto a = bound(0);
        const auto k = bound(1);
        if (a && k && std::holds_alternative<Constant>(in.operands[1]->data)) {
            return *k >= 64 ? 0 : *a >> *k;
        }
        return std::nullopt;
    }
    case Instruction::Op::Min: {
        const auto a = bound(0);
        const auto b = bound(1);
        if (a && b) {
            return std::min(*a, *b);
        }
        // Of an unsigned pair, below whichever is known.
        if (in.type.is_uint()) {
            return a ? a : b;
        }
        return std::nullopt;
    }
    case Instruction::Op::Max: {
        const auto ab = both();
        return ab ? optional<uint64_t>(std::max(ab->first, ab->second))
                  : std::nullopt;
    }
    case Instruction::Op::Cast: {
        const Type from = in.operands[0]->get_type();
        if (from.is_bool()) {
            return 1;
        }
        if (from.is_uint() && from.bits() < 64 && from.bits() < in.type.bits()) {
            return (uint64_t(1) << from.bits()) - 1;
        }
        if (from.is<Int_t, UInt_t>() && from.bits() <= in.type.bits()) {
            return bound(0); // widening keeps the value
        }
        return std::nullopt;
    }
    case Instruction::Op::Mod: {
        const auto d = constant_divisor(in.operands[1]);
        if (d && *d != 0 && (in.type.is_uint() || int64_t(*d) > 0) &&
            (in.type.is_uint() || bound(0))) {
            return *d - 1;
        }
        return std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

// Rewrites the division or remainder `instr`, at `index` in `block`, by the
// non-zero constant `pattern`. Returns how many instructions were added in
// front of it.
size_t rewrite_constant_division(Function &func, const shared_ptr<Block> &block,
                                 size_t index, uint64_t pattern) {
    const shared_ptr<Instruction> instr = block->instrs[index];
    const Type type = instr->type;
    const unsigned bits = unsigned(type.bits());
    const uint64_t mask = bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
    const uint64_t d = pattern & mask;
    internal_assert(d != 0) << "division of " << instr->name << " by zero";
    const bool is_mod = instr->op == Instruction::Op::Mod;
    const ValuePtr n = instr->operands[0];
    Emitter e{func, block, index};
    const Type utype = UInt_t::make(bits);

    // The result of the division as an instruction of its own when the
    // remainder is wanted, or as `instr` rewritten in place: `finish` puts
    // the last operation into `instr` when it is the answer.
    ValuePtr q;
    const auto finish = [&](Instruction::Op op, vector<ValuePtr> operands) {
        if (is_mod) {
            q = e.emit(op, type, std::move(operands));
        } else {
            instr->op = op;
            instr->operands = std::move(operands);
            q = std::make_shared<Value>(instr);
        }
    };

    // n itself, as `n + 0`, which the simplifier folds: a plain copy (Set) is
    // reserved for the program's own names. The remainder is `n & 0`.
    const auto copy_of_n = [&]() { finish(Instruction::Op::Add, {n, e.constant(type, 0)}); };
    const auto zero_remainder = [&]() {
        instr->op = Instruction::Op::BwAnd;
        instr->operands = {n, e.constant(type, 0)};
    };

    // A dividend known to stay below twice a positive divisor -- pbrt's
    // `(axis + 1) % 3` with the axis one of three -- divides by one
    // comparison: the quotient is whether it reached the divisor, the
    // remainder is the dividend less the divisor when it did.
    const bool positive = type.is_uint() || (d & (uint64_t(1) << (bits - 1))) == 0;
    if (positive) {
        if (const optional<uint64_t> bound = upper_bound(n);
            bound && *bound < 2 * d) {
            if (*bound < d) {
                if (is_mod) {
                    instr->op = Instruction::Op::Add;
                    instr->operands = {n, e.constant(type, 0)};
                } else {
                    zero_remainder(); // n & 0: the quotient is zero
                }
                return e.at - index;
            }
            ValuePtr divisor = pattern_constant(type, d);
            ValuePtr below = e.emit(Instruction::Op::Lt, Bool_t::make(), {n, divisor});
            if (is_mod) {
                ValuePtr less = e.emit(Instruction::Op::Sub, type, {n, divisor});
                instr->op = Instruction::Op::Select;
                instr->operands = {below, n, less};
            } else {
                instr->op = Instruction::Op::Select;
                instr->operands = {below, e.constant(type, 0), e.constant(type, 1)};
            }
            return e.at - index;
        }
    }

    if (type.is_uint()) {
        if (d == 1) {
            if (is_mod) {
                zero_remainder();
            } else {
                copy_of_n();
            }
            return e.at - index;
        }
        if ((d & (d - 1)) == 0) {
            // A power of two: the shift, and the low bits for the remainder.
            unsigned k = 0;
            while ((uint64_t(1) << k) != d) {
                k++;
            }
            if (is_mod) {
                instr->op = Instruction::Op::BwAnd;
                instr->operands = {n, e.constant(type, d - 1)};
                return e.at - index;
            }
            finish(Instruction::Op::Shr, {n, e.constant(type, k)});
            return e.at - index;
        }
        const Magic magic = magic_unsigned(d, bits);
        ValuePtr m = pattern_constant(type, magic.multiplier);
        ValuePtr t = e.intrinsic(ir::Intrinsic::mulhi, type, {n, m});
        if (!magic.add) {
            // q = mulhi(n, m) >> s: the multiplier fits the word.
            finish(Instruction::Op::Shr, {t, e.constant(type, magic.shift)});
        } else {
            // q = (((n - t) >> 1) + t) >> (s - 1): the multiplier's top bit
            // is the one past the word, put back by adding n's half.
            ValuePtr diff = e.emit(Instruction::Op::Sub, type, {n, t});
            ValuePtr half = e.emit(Instruction::Op::Shr, type, {diff, e.constant(type, 1)});
            ValuePtr sum = e.emit(Instruction::Op::Add, type, {half, t});
            finish(Instruction::Op::Shr, {sum, e.constant(type, magic.shift - 1)});
        }
    } else {
        const uint64_t sign = uint64_t(1) << (bits - 1);
        const bool negative = (d & sign) != 0;
        const uint64_t ad = negative ? ((0 - d) & mask) : d;
        ValuePtr zero = e.constant(type, 0);
        if (ad == 1) {
            // n, or its negation for -1; the remainder is zero.
            if (is_mod) {
                zero_remainder();
            } else if (negative) {
                finish(Instruction::Op::Sub, {zero, n});
            } else {
                copy_of_n();
            }
            return e.at - index;
        }
        if ((ad & (ad - 1)) == 0) {
            // |d| = 2^k: an arithmetic shift rounds toward minus infinity,
            // so a negative n first gets 2^k - 1 added -- its sign, spread
            // by an arithmetic shift, then cut to k bits by a logical one --
            // which makes the shift truncate toward zero (Hacker's Delight
            // 10-1). Negated for a negative divisor.
            unsigned k = 0;
            while ((uint64_t(1) << k) != ad) {
                k++;
            }
            ValuePtr spread = e.emit(Instruction::Op::Shr, type, {n, e.constant(type, bits - 1)});
            ValuePtr as_unsigned = e.emit(Instruction::Op::Reinterpret, utype, {spread});
            ValuePtr low = e.emit(Instruction::Op::Shr, utype,
                                  {as_unsigned, e.constant(utype, bits - k)});
            ValuePtr bias = e.emit(Instruction::Op::Reinterpret, type, {low});
            ValuePtr biased = e.emit(Instruction::Op::Add, type, {n, bias});
            if (negative) {
                ValuePtr shifted = e.emit(Instruction::Op::Shr, type, {biased, e.constant(type, k)});
                finish(Instruction::Op::Sub, {zero, shifted});
            } else {
                finish(Instruction::Op::Shr, {biased, e.constant(type, k)});
            }
        } else {
            // Hacker's Delight 10-4: q0 = mulhi(n, m), plus n when d > 0 and
            // m reads negative, minus n when d < 0 and m reads positive; q0
            // shifted arithmetically, then one added when negative, since
            // the shift rounded toward minus infinity.
            const Magic magic = magic_signed(d, bits);
            const bool m_negative = (magic.multiplier & sign) != 0;
            ValuePtr m = pattern_constant(type, magic.multiplier);
            ValuePtr t = e.intrinsic(ir::Intrinsic::mulhi, type, {n, m});
            if (!negative && m_negative) {
                t = e.emit(Instruction::Op::Add, type, {t, n});
            } else if (negative && !m_negative) {
                t = e.emit(Instruction::Op::Sub, type, {t, n});
            }
            ValuePtr shifted =
                magic.shift == 0
                    ? t
                    : e.emit(Instruction::Op::Shr, type, {t, e.constant(type, magic.shift)});
            ValuePtr sign_of = e.emit(Instruction::Op::Shr, type, {shifted, e.constant(type, bits - 1)});
            ValuePtr as_unsigned = e.emit(Instruction::Op::Reinterpret, utype, {sign_of});
            ValuePtr one_if_negative_u = e.emit(Instruction::Op::Shr, utype,
                                                {as_unsigned, e.constant(utype, bits - 1)});
            ValuePtr one_if_negative =
                e.emit(Instruction::Op::Reinterpret, type, {one_if_negative_u});
            finish(Instruction::Op::Add, {shifted, one_if_negative});
        }
    }
    if (is_mod) {
        // n - q * d.
        ValuePtr qd = e.emit(Instruction::Op::Mul, type, {q, pattern_constant(type, d)});
        instr->op = Instruction::Op::Sub;
        instr->operands = {n, qd};
    }
    return e.at - index;
}

// Every division or remainder by a constant in `func`, rewritten. Returns
// how many.
size_t divide_by_constants(Function &func) {
    size_t rewritten = 0;
    for (const shared_ptr<Block> &block : func.blocks) {
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> &instr = block->instrs[i];
            if ((instr->op != Instruction::Op::Div &&
                 instr->op != Instruction::Op::Mod) ||
                !instr->type.is<Int_t, UInt_t>() || instr->operands.size() != 2 ||
                !equals(instr->operands[1]->get_type(), instr->type)) {
                continue;
            }
            const optional<uint64_t> d = constant_divisor(instr->operands[1]);
            const unsigned bits = unsigned(instr->type.bits());
            const uint64_t mask =
                bits == 64 ? ~uint64_t(0) : (uint64_t(1) << bits) - 1;
            if (!d.has_value() || (*d & mask) == 0) {
                continue; // not a constant, or a zero left to trap
            }
            i += rewrite_constant_division(func, block, i, *d);
            rewritten++;
        }
    }
    return rewritten;
}

} // namespace

size_t divide_bounded_by_floats(Function &func, const Divergence &divergence) {
    size_t rewritten = 0;
    for (const shared_ptr<Block> &block : func.blocks) {
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> instr = block->instrs[i];
            if ((instr->op != Instruction::Op::Div &&
                 instr->op != Instruction::Op::Mod) ||
                !instr->type.is<Int_t, UInt_t>() || instr->operands.size() != 2 ||
                !equals(instr->operands[1]->get_type(), instr->type) ||
                !divergence.instrs.count(instr.get())) {
                continue;
            }
            const optional<uint64_t> a = upper_bound(instr->operands[0]);
            const optional<uint64_t> b = upper_bound(instr->operands[1]);
            uint64_t reach = 0;
            if (!a || !b || __builtin_add_overflow(*a, *b, &reach)) {
                continue;
            }
            // The float whose mantissa the operands fit under, 24 bits for a
            // float and 53 for a double; a pair too wide for a float is only
            // worth a double when the lanes are 64 bits, since narrower lanes
            // always fit a double and the code generator takes those.
            Type ft;
            if (reach < (uint64_t(1) << 24)) {
                ft = Float_t::make_f32();
            } else if (instr->type.bits() == 64 && reach < (uint64_t(1) << 53)) {
                ft = Float_t::make_f64();
            } else {
                continue;
            }
            const Type type = instr->type;
            const bool is_mod = instr->op == Instruction::Op::Mod;
            const ValuePtr n = instr->operands[0];
            const ValuePtr d = instr->operands[1];
            Emitter e{func, block, i};
            ValuePtr fn = e.emit(Instruction::Op::Cast, ft, {n});
            ValuePtr fd = e.emit(Instruction::Op::Cast, ft, {d});
            ValuePtr fq = e.emit(Instruction::Op::Div, ft, {fn, fd});
            if (is_mod) {
                // A cast from float to integer truncates toward zero, and the
                // remainder is n - qd.
                ValuePtr q = e.emit(Instruction::Op::Cast, type, {fq});
                ValuePtr qd = e.emit(Instruction::Op::Mul, type, {q, d});
                instr->op = Instruction::Op::Sub;
                instr->operands = {n, qd};
            } else {
                instr->op = Instruction::Op::Cast;
                instr->operands = {fq};
            }
            i += e.at - i;
            rewritten++;
        }
    }
    return rewritten;
}

namespace {

// The rewrite behind both entry points. A division is worth a multiplier
// when its divisor is invariant in one of two senses: across a loop the
// division is in (its definition lies outside), or -- with `divergence` in
// hand, inside a gang -- across the lanes, the divisor uniform while the
// dividend varies, so that the one multiplier serves sixteen divisions the
// machine would otherwise do one lane at a time.
size_t divide_by_definitions(Function &func, const Divergence *divergence,
                             const string *region_entry) {
    if (func.blocks.empty()) {
        return 0;
    }
    const size_t constants = divide_by_constants(func);
    const Cfg cfg(func);
    const DomTree dom = compute_dominator_tree(cfg);
    const LoopForest loops = compute_loop_forest(cfg, dom);
    if (loops.empty() && divergence == nullptr) {
        return constants;
    }
    Tracer tracer(func, cfg, region_entry);

    // The divisions worth the multiplier. A scalar integer, not a constant.
    struct Site {
        string block;
        shared_ptr<Instruction> instr;
    };
    map<Definition, vector<Site>> worth;
    map<Definition, vector<Site>> others;
    for (BlockId b : cfg.rpo) {
        const shared_ptr<Block> &block = cfg.block(b);
        const string &name = block->name;
        const Loop *loop = loops.innermost(b);
        for (const shared_ptr<Instruction> &instr : block->instrs) {
            if ((instr->op != Instruction::Op::Div &&
                 instr->op != Instruction::Op::Mod) ||
                !instr->type.is<Int_t, UInt_t>() || instr->operands.size() != 2 ||
                !equals(instr->operands[1]->get_type(), instr->type)) {
                continue;
            }
            const optional<Definition> def =
                tracer.trace(instr->operands[1], name);
            if (!def.has_value()) {
                continue; // a constant divisor: zero, the one kind left
            }
            const bool loop_invariant =
                loop != nullptr && !loop->blocks.contains(cfg.id(def->block));
            const bool lane_invariant =
                divergence != nullptr &&
                !divergence->is_varying(name, *instr->operands[1]) &&
                divergence->is_varying(name, *instr->operands[0]);
            (loop_invariant || lane_invariant ? worth : others)[*def].push_back(
                Site{name, instr});
        }
    }
    if (worth.empty()) {
        return constants;
    }
    // Threading the multiplier to a division's block (Block::get_value) walks
    // the predecessor lists, which have to be current. Only now, so that a
    // function with nothing to rewrite is left exactly as it was.
    refresh_preds(func);

    size_t rewritten = constants;
    for (auto &[def, sites] : worth) {
        // The multiplier goes right after the divisor's definition: after
        // the instruction, or at the top of the block whose argument it is.
        const shared_ptr<Block> &at = cfg.block(cfg.id(def.block));
        size_t position = 0;
        ValuePtr divisor;
        if (def.instr) {
            for (; position < at->instrs.size(); position++) {
                if (at->instrs[position].get() == def.instr.get()) {
                    break;
                }
            }
            internal_assert(position < at->instrs.size())
                << def.instr->name << " is not in " << def.block;
            position++;
            divisor = std::make_shared<Value>(def.instr);
        } else {
            const auto found = at->lookups.find(def.arg);
            internal_assert(found != at->lookups.end())
                << def.arg << " is not an argument of " << def.block;
            divisor = found->second;
        }
        Emitter e{func, at, position};
        const Multiplier mult = emit_multiplier(e, divisor, divisor->get_type());

        // Every division by this divisor, the ones in loops and the rest.
        vector<Site> all = sites;
        if (const auto rest = others.find(def); rest != others.end()) {
            all.insert(all.end(), rest->second.begin(), rest->second.end());
        }
        for (const Site &site : all) {
            const shared_ptr<Block> &block = cfg.block(cfg.id(site.block));
            size_t index = 0;
            for (; index < block->instrs.size(); index++) {
                if (block->instrs[index].get() == site.instr.get()) {
                    break;
                }
            }
            internal_assert(index < block->instrs.size())
                << site.instr->name << " is not in " << site.block;
            rewrite_division(func, block, index, mult);
            rewritten++;
        }
    }
    return rewritten;
}

} // namespace

size_t divide_by_invariants(Function &func) {
    return divide_by_definitions(func, nullptr, nullptr);
}

size_t divide_by_uniform_divisors(Function &func, const Divergence &divergence,
                                  const string &region_entry) {
    return divide_by_definitions(func, &divergence, &region_entry);
}

size_t expand_bounded_multipliers(Function &func, const Divergence &divergence) {
    size_t expanded = 0;
    for (const shared_ptr<Block> &block : func.blocks) {
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> &instr = block->instrs[i];
            if (instr->op != Instruction::Op::Intrinsic ||
                instr->intrinsic != ir::Intrinsic::div_multiplier ||
                instr->operands.size() != 1 || !instr->type.is<Int_t, UInt_t>() ||
                !divergence.instrs.count(instr.get())) {
                continue; // not a multiplier, or one the gang shares
            }
            const optional<uint64_t> bound =
                divisor_bound(instr->operands[0], instr->type);
            if (!two_step_fits(instr->type, bound)) {
                continue;
            }
            i += expand_multiplier(func, block, i, bound);
            expanded++;
        }
    }
    return expanded;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
