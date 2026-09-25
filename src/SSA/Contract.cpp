#include "SSA/Contract.h"

#include "SSA/Analysis.h"

#include "IR/Equality.h"
#include "IR/Expr.h"
#include "IR/Schedule.h"

#include "Error.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
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

// `value` as an operand of an instruction in `block`: a value from another
// block reaches it as a block argument, threaded in as the builder threads
// every cross-block operand (Block::get_value); a constant or a value of
// this block is itself.
std::shared_ptr<Value> in_block(const std::shared_ptr<Value> &value,
                                Block &block) {
    if (const auto *held =
            std::get_if<std::shared_ptr<Instruction>>(&value->data)) {
        if ((*held)->owner.lock().get() != &block) {
            return block.get_value((*held)->name, (*held)->type);
        }
        return value;
    }
    if (const auto *arg = std::get_if<Argument>(&value->data)) {
        return block.get_value(arg->name, arg->type);
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

        for (Fusion &fusion : fusions) {
            const std::shared_ptr<Instruction> &instr = fusion.consumer;
            const std::shared_ptr<Instruction> &mul = fusion.product.mul;
            // The product's operands, here: from the block the product was
            // in, when that is another block.
            std::shared_ptr<Value> a = in_block(mul->operands[0], *block);
            std::shared_ptr<Value> b = in_block(mul->operands[1], *block);
            if (fusion.negate_product) {
                a = negate(a, instr->type, instr);
            }
            std::shared_ptr<Value> addend = fusion.addend;
            if (fusion.negate_addend) {
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
