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
    Tracer(const Function &func, const Cfg &cfg)
        : entry(func.blocks.front()->name), cfg(cfg) {}

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
        if (k == b->args.size() || block == entry) {
            // A parameter of the function, or a name this cannot see
            // through.
            return here;
        }
        if (cfg.preds[here_id].empty()) {
            return here;
        }
        set<Definition> found;
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
                    }
                } else {
                    return here; // a constant on one path
                }
                if (found.size() > 1) {
                    return here;
                }
            }
        }
        return found.size() == 1 ? *found.begin() : here;
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

// Emits the multiplier of `d` into `e`, which is positioned right after the
// divisor's definition. `d` is the divisor as an operand valid there.
Multiplier emit_multiplier(Emitter &e, const ValuePtr &d, const Type &type) {
    const uint64_t bits = type.bits();
    Multiplier out;
    out.type = type;
    auto name_of = [](const ValuePtr &v) {
        return std::get<shared_ptr<Instruction>>(v->data)->name;
    };

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

} // namespace

size_t divide_by_invariants(Function &func) {
    if (func.blocks.empty()) {
        return 0;
    }
    const Cfg cfg(func);
    const DomTree dom = compute_dominator_tree(cfg);
    const LoopForest loops = compute_loop_forest(cfg, dom);
    if (loops.empty()) {
        return 0;
    }
    Tracer tracer(func, cfg);

    // The divisions worth the multiplier: one whose divisor is defined
    // outside a loop the division is in. A scalar integer, not a constant.
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
                continue; // a constant divisor, which the backend handles
            }
            const bool invariant =
                loop != nullptr && !loop->blocks.contains(cfg.id(def->block));
            (invariant ? worth : others)[*def].push_back(Site{name, instr});
        }
    }
    if (worth.empty()) {
        return 0;
    }
    // Threading the multiplier to a division's block (Block::get_value) walks
    // the predecessor lists, which have to be current. Only now, so that a
    // function with nothing to rewrite is left exactly as it was.
    refresh_preds(func);

    size_t rewritten = 0;
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

} // namespace ssa
} // namespace ir
} // namespace bonsai
