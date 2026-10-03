#include "SSA/Contract.h"

#include "SSA/Analysis.h"

#include "IR/Equality.h"
#include "IR/Expr.h"
#include "IR/Schedule.h"

#include "Error.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using UseCounts = std::map<const Instruction *, size_t>;

void count(const std::shared_ptr<Value> &value, UseCounts &uses) {
    if (value == nullptr) {
        return;
    }
    if (const auto *instr =
            std::get_if<std::shared_ptr<Instruction>>(&value->data)) {
        ++uses[instr->get()];
    }
}

void count(const Terminator::Jump &jump, UseCounts &uses) {
    for (const auto &arg : jump.args) {
        count(arg, uses);
    }
}

// How many times each instruction's result is named, anywhere in the function.
//
// Terminators included, and that is not a detail: a jump argument is a use like
// any other, and a product counted only over instructions could look dead while
// a block argument still carries it.
UseCounts use_counts(const Function &f) {
    UseCounts uses;
    for (const auto &block : f.blocks) {
        for (const auto &instr : block->instrs) {
            for (const auto &operand : instr->operands) {
                count(operand, uses);
            }
        }
        std::visit(
            Overloaded{
                [](const std::monostate &) {},
                [&](const Terminator::Jump &j) { count(j, uses); },
                [&](const Terminator::Dispatch &d) {
                    count(d.cond, uses);
                    for (const auto &target : d.targets) {
                        count(target, uses);
                    }
                },
                [&](const Terminator::Return &r) { count(r.value, uses); },
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
                    // Each varying value is a use in its own right: a product
                    // read by two of the calls is read twice, and fusing it
                    // into one of them would leave the other recomputing it.
                    for (const auto &vs : c.varying) {
                        for (const auto &v : vs) {
                            count(v, uses);
                        }
                    }
                    for (const auto &k : c.keys) {
                        count(k, uses);
                    }
                    for (const auto &k : c.conds) {
                        count(k, uses);
                    }
                },
            },
            block->terminator.data);
    }
    return uses;
}

bool is_float_valued(const Type &type) {
    return type.is_vector() ? type.element_of().is_float() : type.is_float();
}

// A zero, as the SSA builder writes a negation's minuend (SSA/Convert.cpp,
// `UnOp::Neg` is `0 - x`): the constant itself, or a broadcast of it when the
// value has lanes. Either sign of zero, since `-0.0 - x` is a negation too.
bool is_zero(const std::shared_ptr<Value> &value) {
    const Value *at = value.get();
    if (const auto *held =
            std::get_if<std::shared_ptr<Instruction>>(&at->data)) {
        if ((*held)->op != Instruction::Op::Bc || (*held)->operands.empty()) {
            return false;
        }
        at = (*held)->operands[0].get();
    }
    const auto *constant = std::get_if<Constant>(&at->data);
    if (constant == nullptr) {
        return false;
    }
    const auto *d = std::get_if<double>(&constant->data);
    return d != nullptr && *d == 0.0;
}

// A negation, `0 - x`: what the SSA builder makes of a unary minus.
bool is_negation(const Instruction &instr) {
    return instr.op == Instruction::Op::Sub && instr.operands.size() == 2 &&
           is_zero(instr.operands[0]);
}

// Where a consumer may absorb a product from.
//
// Its own block, always. Another block when that block dominates the
// consumer's -- so the product's operands are in scope there -- and the two
// are in the same loops: the same natural loops, and inside the same parfor
// bodies, which the graph has as regions under a ParFor edge rather than as
// cycles. Same loops, or the fusion would drag the product to wherever the
// consumer is, which for a product hoisted out of a loop means dragging it
// back in.
//
// Across blocks at all because the SSA builder places an expression where
// its operands are ready rather than where the program wrote it. pbrt's
// `max(0, Gaussian(p.x) - expX) * max(0, Gaussian(p.y) - expY)` has each
// Gaussian's product and its subtraction side by side in gcc's GIMPLE, and
// gcc, which fuses within a basic block only, fuses both; here the first
// product lands after the first FastExp's branches and its subtraction after
// the second's, two blocks apart.
class Placement {
  public:
    explicit Placement(const Function &f)
        : cfg(f), dom(compute_dominator_tree(cfg)),
          loops(compute_loop_forest(cfg, dom)), bodies(cfg.size()) {
        for (const BlockId b : cfg.rpo) {
            const auto *parfor =
                std::get_if<Terminator::ParFor>(&cfg[b].terminator.data);
            if (parfor == nullptr) {
                continue;
            }
            for (const BlockId inside :
                 reachable_from(cfg, cfg.id(parfor->body.name))) {
                bodies[inside].push_back(b);
            }
        }
    }

    bool may_absorb(const Block &from, const Block &into) const {
        if (&from == &into) {
            return true;
        }
        const BlockId a = cfg.find(from), b = cfg.find(into);
        if (a == NO_BLOCK || b == NO_BLOCK || !dom.dominates(a, b)) {
            return false;
        }
        return loops.innermost(a) == loops.innermost(b) &&
               bodies[a] == bodies[b];
    }

    // Whether a value defined in `def` is in scope in `use`: `def` dominates
    // it.
    bool dominates(const Block &def, const Block &use) const {
        if (&def == &use) {
            return true;
        }
        const BlockId a = cfg.find(def), b = cfg.find(use);
        return a != NO_BLOCK && b != NO_BLOCK && dom.dominates(a, b);
    }

  private:
    Cfg cfg;
    DomTree dom;
    LoopForest loops;
    // Per block, the ParFor blocks whose bodies contain it, in id order.
    std::vector<std::vector<BlockId>> bodies;
};

// An instruction a consumer in `block` may absorb: used only there, and
// defined where `Placement` allows.
//
// Used only there, or fusing would leave the instruction behind for its
// other reader and do its work twice. Like the placement rule, a condition
// that keeps the fusion free rather than legal.
bool absorbable(const std::shared_ptr<Instruction> &instr, const Block &block,
                const Placement &placement, const UseCounts &uses) {
    const auto found = uses.find(instr.get());
    if (found == uses.end() || found->second != 1) {
        return false;
    }
    const std::shared_ptr<Block> owner = instr->owner.lock();
    return owner != nullptr && placement.may_absorb(*owner, block);
}

// A product an add or subtract may absorb: the multiply, and the negation it
// arrived through, if it did -- `-(a*b) + c` fuses as `fma(-a, b, c)`, which
// is gcc's NEGATE_EXPR case.
struct Product {
    std::shared_ptr<Instruction> mul;
    std::shared_ptr<Instruction> negation;
};

// Why an operand was not absorbed, for BONSAI_EXPLAIN_CONTRACT: set it and
// every float add or subtract that stays one says, per operand, which
// condition refused.
bool explaining() {
    static const bool on = std::getenv("BONSAI_EXPLAIN_CONTRACT") != nullptr;
    return on;
}

// The product this operand is, or nothing.
//
// The types must match exactly, so that a widened lane count or a mixed
// precision is left alone rather than quietly reinterpreted.
std::optional<Product> product_of(const std::shared_ptr<Value> &operand,
                                  const Block &block, const Type &type,
                                  const Placement &placement,
                                  const UseCounts &uses, std::string *why) {
    const auto *held = std::get_if<std::shared_ptr<Instruction>>(&operand->data);
    if (held == nullptr) {
        if (why) {
            *why = "not an instruction's value";
        }
        return std::nullopt;
    }
    const std::shared_ptr<Instruction> &instr = *held;
    if (!equals(instr->type, type)) {
        if (why) {
            *why = instr->name + " has another type";
        }
        return std::nullopt;
    }
    if (!absorbable(instr, block, placement, uses)) {
        if (why) {
            const auto found = uses.find(instr.get());
            const std::shared_ptr<Block> owner = instr->owner.lock();
            *why = instr->name + " is used " +
                   std::to_string(found == uses.end() ? 0 : found->second) +
                   " times, in block " + (owner ? owner->name : "<none>");
        }
        return std::nullopt;
    }
    if (instr->op == Instruction::Op::Mul && instr->operands.size() == 2) {
        return Product{instr, nullptr};
    }
    if (is_negation(*instr)) {
        const auto *inner =
            std::get_if<std::shared_ptr<Instruction>>(&instr->operands[1]->data);
        if (inner != nullptr && (*inner)->op == Instruction::Op::Mul &&
            (*inner)->operands.size() == 2 && equals((*inner)->type, type) &&
            absorbable(*inner, block, placement, uses)) {
            return Product{*inner, instr};
        }
    }
    if (why) {
        *why = instr->name + " is not a product";
    }
    return std::nullopt;
}

// An operand of the product, as the fma in `block` reads it: the value
// itself. The product's block dominates the consumer's (`Placement`), so
// everything the product read is in scope at the consumer as it stands -- a
// constant, an instruction of a dominating block, or an argument of one --
// the way a rewrite's values are referenced across the blocks it left (see
// SSA/Vectorize.cpp).
//
// Not re-resolved by name through Block::get_value: that reads the block's
// current binding of the name, which for a loop's variable is its last
// assignment in the block -- `total = total + o * noise(p); o = o * omega;`
// would fuse to `fma(o', noise, total)` with `o'` the product below it, a
// value not yet defined where it is read. The instruction's operand holds
// the value `o` was at the multiply, and that is what the fma takes.
std::shared_ptr<Value> in_block(const std::shared_ptr<Value> &value,
                                const Block &block,
                                const Placement &placement) {
    if (const auto *held =
            std::get_if<std::shared_ptr<Instruction>>(&value->data)) {
        const std::shared_ptr<Block> owner = (*held)->owner.lock();
        internal_assert(owner != nullptr &&
                        (owner.get() == &block ||
                         placement.dominates(*owner, block)))
            << "A fused product's operand " << (*held)->name
            << " is not in scope in block " << block.name;
    }
    return value;
}

// One add or subtract and the product it absorbs, with which of the fma's
// operands are negated to say the same thing: `c - a*b` is `fma(-a, b, c)`,
// `a*b - c` is `fma(a, b, -c)`.
struct Fusion {
    std::shared_ptr<Instruction> consumer;
    Product product;
    std::shared_ptr<Value> addend;
    bool negate_product;
    bool negate_addend;
};

} // namespace

void contract_fp(Function &f) {
    if (f.blocks.empty()) {
        return;
    }
    const UseCounts uses = use_counts(f);
    const Placement placement(f);

    for (const auto &block : f.blocks) {
        // Decided over the block as it is, applied after: a fusion adds a
        // negation before its consumer and removes what it absorbed, and
        // `product_of` searches this same list.
        std::vector<Fusion> fusions;

        for (const auto &instr : block->instrs) {
            const bool minus = instr->op == Instruction::Op::Sub;
            if ((instr->op != Instruction::Op::Add && !minus) ||
                instr->operands.size() != 2 || !is_float_valued(instr->type)) {
                continue;
            }
            // A negation is not a subtraction with a product on its right:
            // gcc fuses `-(a*b)` into the add that consumes it, or not at all,
            // and so does this, from the consumer's side.
            if (minus && is_negation(*instr)) {
                continue;
            }
            // The first operand before the second, which is the arbitrary half
            // of this -- see the note in the header about gcc having no rule
            // here either.
            size_t at = 0;
            std::string why[2];
            std::optional<Product> product =
                product_of(instr->operands[0], *block, instr->type, placement,
                           uses, explaining() ? &why[0] : nullptr);
            if (!product) {
                at = 1;
                product =
                    product_of(instr->operands[1], *block, instr->type,
                               placement, uses, explaining() ? &why[1] : nullptr);
            }
            if (!product) {
                if (explaining()) {
                    std::cerr << "contract: " << f.blocks.front()->name << " "
                              << block->name
                              << " " << instr->name << " stays "
                              << (minus ? "a subtract" : "an add") << ": "
                              << why[0] << "; " << why[1] << "\n";
                }
                continue;
            }
            // `-(a*b)` arrived negated; a subtraction negates its right-hand
            // side, which is the product when the product is on the right and
            // the addend when it is on the left.
            bool negate_product = product->negation != nullptr;
            bool negate_addend = false;
            if (minus && at == 1) {
                negate_product = !negate_product;
            } else if (minus) {
                negate_addend = true;
            }
            fusions.push_back({instr, *product, instr->operands[1 - at],
                               negate_product, negate_addend});
        }

        // A value's negation, placed just before `consumer`: `-0.0 - x`, which
        // is -x for every x, both zeroes included, and what LLVM spells
        // `fneg`. Made through `make_instruction` so that a value from
        // another block is threaded in as any operand is, then moved from
        // the end of the block, where that appends, to where it is read.
        const auto negate = [&](const std::shared_ptr<Value> &x,
                                const Type &type,
                                const std::shared_ptr<Instruction> &consumer) {
            const auto before = [&](const std::shared_ptr<Value> &made) {
                const auto &instr =
                    std::get<std::shared_ptr<Instruction>>(made->data);
                internal_assert(block->instrs.back() == instr)
                    << "make_instruction did not append " << instr->name;
                block->instrs.pop_back();
                const auto pos = std::find(block->instrs.begin(),
                                           block->instrs.end(), consumer);
                internal_assert(pos != block->instrs.end())
                    << "A fusing add left its block: " << consumer->name;
                block->instrs.insert(pos, instr);
            };
            const Type elem = type.is_vector() ? type.element_of() : type;
            std::shared_ptr<Value> zero =
                std::make_shared<Value>(Constant{elem, -0.0});
            if (type.is_vector()) {
                auto lanes = std::make_shared<Value>(
                    Constant{UInt_t::make(32), uint64_t(type.lanes())});
                zero = block->make_instruction(type, Instruction::Op::Bc,
                                               {std::move(zero), lanes});
                before(zero);
            }
            std::shared_ptr<Value> negated = block->make_instruction(
                type, Instruction::Op::Sub, {std::move(zero), x});
            before(negated);
            return negated;
        };

        // Whether a value is the same on every pass of the loop around
        // `block`, as far as can be told here: a constant, an argument, a
        // value of another block, or a broadcast of one -- the gang's copy of
        // a uniform value, which the vectorizer puts beside its use and LLVM
        // hoists. Where a sign goes matters for such a value: LLVM pushes a
        // negation into a multiply's right operand (InstCombine's
        // hoistFNegAboveFMulFDiv) and hoists the negation of an invariant one
        // out of the loop, where it holds a register for the loop's whole
        // run -- three of them, for a ray's direction in the triangle test --
        // while a negation of a value computed in the loop folds into the
        // fused multiply-add's own sign (`vfmsub`, `vfnmadd`) and costs
        // nothing. Either operand may carry the sign: a negation is exact,
        // and so is swapping a multiply's operands.
        // A value of this block counts as invariant when it is pure
        // arithmetic over invariant values: the broadcast of one component
        // of the ray's direction is a broadcast of an extract of a value
        // from above, all three placed here by the vectorizer and all three
        // hoisted by LLVM.
        // A block's arguments are the values threaded in from above under
        // their own names and the loop's own variables, its index among
        // them: only an argument named as one of the function's parameters
        // is known to be the same throughout.
        const auto is_parameter = [&](const std::string &name) {
            const std::vector<Argument> &params = f.blocks.front()->args;
            return std::any_of(params.begin(), params.end(),
                               [&](const Argument &p) { return p.name == name; });
        };
        std::map<const Instruction *, bool> invariant;
        std::function<bool(const std::shared_ptr<Value> &)> looks_invariant =
            [&](const std::shared_ptr<Value> &v) -> bool {
            if (const auto *arg = std::get_if<Argument>(&v->data)) {
                return is_parameter(arg->name);
            }
            const auto *held =
                std::get_if<std::shared_ptr<Instruction>>(&v->data);
            if (held == nullptr) {
                return true; // a constant
            }
            const Instruction &in = **held;
            if (in.owner.lock().get() != block.get()) {
                return true;
            }
            if (const auto known = invariant.find(&in);
                known != invariant.end()) {
                return known->second;
            }
            bool pure;
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
            case Instruction::Op::ExtractIdx:
            case Instruction::Op::Inf:
            case Instruction::Op::LAnd:
            case Instruction::Op::Leq:
            case Instruction::Op::LoadField:
            case Instruction::Op::LOr:
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
            case Instruction::Op::Shuffle:
            case Instruction::Op::Sub:
            case Instruction::Op::Xor:
                pure = true;
                break;
            case Instruction::Op::Intrinsic:
                pure = !ir::Intrinsic::has_effects(in.intrinsic);
                break;
            default:
                pure = false;
            }
            bool result = pure;
            // Marked first, against a cycle, which the SSA form has none of.
            invariant[&in] = false;
            for (size_t i = 0; result && i < in.operands.size(); i++) {
                result = looks_invariant(in.operands[i]);
            }
            invariant[&in] = result;
            return result;
        };

        for (Fusion &fusion : fusions) {
            const std::shared_ptr<Instruction> &instr = fusion.consumer;
            const std::shared_ptr<Instruction> &mul = fusion.product.mul;
            // The product's operands, here: from the block the product was
            // in, when that is another block.
            std::shared_ptr<Value> a =
                in_block(mul->operands[0], *block, placement);
            std::shared_ptr<Value> b =
                in_block(mul->operands[1], *block, placement);
            if (fusion.negate_product) {
                // The sign on the operand computed in the loop, where it folds
                // into the instruction (see looks_invariant).
                if (looks_invariant(a) && !looks_invariant(b)) {
                    b = negate(b, instr->type, instr);
                } else {
                    a = negate(a, instr->type, instr);
                }
            }
            std::shared_ptr<Value> addend = fusion.addend;
            if (fusion.negate_addend) {
                // A negated product, `a*b - c*d`: LLVM will push the sign
                // into the product's right operand, so the one computed in
                // the loop goes there (see looks_invariant).
                if (const auto *held = std::get_if<std::shared_ptr<Instruction>>(
                        &addend->data);
                    held != nullptr && (*held)->op == Instruction::Op::Mul &&
                    (*held)->operands.size() == 2 &&
                    (*held)->owner.lock().get() == block.get() &&
                    !looks_invariant((*held)->operands[0]) &&
                    looks_invariant((*held)->operands[1])) {
                    std::swap((*held)->operands[0], (*held)->operands[1]);
                }
                addend = negate(addend, instr->type, instr);
            }
            // The add becomes the fma. Rewritten in place rather than replaced,
            // so that everything already naming its result goes on doing so.
            instr->op = Instruction::Op::Intrinsic;
            instr->intrinsic = ir::Intrinsic::fma;
            instr->operands = {std::move(a), std::move(b), std::move(addend)};

            // What was absorbed goes, from whichever block held it.
            for (const std::shared_ptr<Instruction> &dead :
                 {mul, fusion.product.negation}) {
                if (dead == nullptr) {
                    continue;
                }
                const std::shared_ptr<Block> home = dead->owner.lock();
                internal_assert(home != nullptr)
                    << "A fused product has no block: " << dead->name;
                const auto at =
                    std::find(home->instrs.begin(), home->instrs.end(), dead);
                internal_assert(at != home->instrs.end())
                    << "A fused product left its block: " << dead->name;
                home->instrs.erase(at);
            }
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
