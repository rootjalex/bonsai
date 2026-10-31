#include "SSA/HoistAllocations.h"

#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "SSA/SSA.h"
#include "SSA/Storage.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

// A parfor of the function: the block that holds it and the blocks of its
// body, which is where anything nested inside it lives.
struct Nest {
    BlockId header;
    const Terminator::ParFor *loop;
    BlockSet body;
};

// The innermost parfor whose body holds `b`, or null.
const Nest *innermost_parfor(const vector<Nest> &nests, BlockId b) {
    const Nest *found = nullptr;
    for (const Nest &n : nests) {
        if (!n.body.contains(b)) {
            continue;
        }
        if (found == nullptr || found->body.contains(n.header)) {
            found = &n;
        }
    }
    return found;
}

// Where the value named `name` in a type is defined, as the block `home`
// refers to it: through `home`'s own arguments when it takes the value, and
// otherwise the one instruction or parameter of that name -- a name a type
// carries from a block above, which the code generators find the same way.
Definition definition_of(Definitions &defs, const Function &func,
                         const shared_ptr<Block> &home, const string &name) {
    if (const auto it = home->lookups.find(name); it != home->lookups.end()) {
        return defs.of(home->name, it->second);
    }
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->name == name) {
                return Definition{std::make_shared<Value>(instr), block->name};
            }
        }
    }
    // A block's argument -- a parameter, a merge, a copy threaded from
    // either -- settled to what it is threaded from.
    for (const auto &block : func.blocks) {
        for (const Argument &arg : block->args) {
            if (arg.name == name) {
                return defs.of(block->name, std::make_shared<Value>(arg));
            }
        }
    }
    internal_error << "hoist_invariant_allocations: " << home->name
                   << " allocates storage sized by " << name
                   << ", which nothing in " << func.blocks.front()->name
                   << " defines";
    return {};
}

// An instruction with no effect but its value: arithmetic, comparison,
// selection, conversion. What may move with an allocation whose size it
// computes (see hoist_invariant_allocations). Anything that reads or
// writes memory, calls, accumulates, pushes, claims a slot or takes an
// address is not on the list and stays where it is.
bool pure(const Instruction &i) {
    using Op = Instruction::Op;
    switch (i.op) {
    case Op::Abs: case Op::Add: case Op::BwAnd: case Op::BwOr: case Op::Cast:
    case Op::Div: case Op::Eq: case Op::LAnd: case Op::LOr: case Op::Leq:
    case Op::Lt: case Op::Max: case Op::Min: case Op::Mod: case Op::Mul:
    case Op::Ne: case Op::Not: case Op::Popcount: case Op::Reinterpret:
    case Op::Select: case Op::Shl: case Op::Shr: case Op::SizeOf: case Op::Sub:
    case Op::Xor:
        return true;
    default:
        return false;
    }
}

const string &name_of(const Value &v) {
    return std::visit(
        overloads{[](const Argument &a) -> const string & { return a.name; },
                  [](const shared_ptr<Instruction> &i) -> const string & {
                      return i->name;
                  },
                  [](const Constant &) -> const string & {
                      static const string none;
                      return none;
                  }},
        v.data);
}

// Whether every read of the slot `slot` -- through its address, in this
// function or a callee handed it -- is preceded in its own iteration by a
// store of the whole slot: a store whose address is the slot itself, in a
// block that dominates the use, or earlier in the same block. A slot whose
// address is stored somewhere as a value is not proven: the address may be
// read back after an iteration has ended.
bool written_before_read(const Cfg &cfg, const DomTree &dom,
                         const Instruction *slot) {
    using Site = std::pair<BlockId, size_t>; // block, position among its instrs
    vector<Site> stores, uses;
    bool escapes = false;
    const auto is_slot = [&](const shared_ptr<Value> &v) {
        const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
        return held != nullptr && held->get() == slot;
    };
    for (BlockId b : cfg.rpo) {
        Block &block = cfg[b];
        for (size_t at = 0; at < block.instrs.size(); at++) {
            const auto &instr = block.instrs[at];
            for (size_t k = 0; k < instr->operands.size(); k++) {
                if (!is_slot(instr->operands[k])) {
                    continue;
                }
                if (instr->op == Instruction::Op::Store && k == 0) {
                    stores.emplace_back(b, at);
                } else if (instr->op == Instruction::Op::Store && k == 1) {
                    escapes = true;
                } else {
                    uses.emplace_back(b, at);
                }
            }
        }
        for_each_value(block.terminator, [&](shared_ptr<Value> &v) {
            if (is_slot(v)) {
                uses.emplace_back(b, block.instrs.size());
            }
        });
    }
    if (escapes || stores.empty()) {
        return false;
    }
    for (const Site &use : uses) {
        bool covered = false;
        for (const Site &store : stores) {
            if (store.first == use.first ? store.second < use.second
                                         : dom.dominates(store.first, use.first)) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            return false;
        }
    }
    return true;
}

} // namespace

size_t hoist_invariant_allocations(Function &func) {
    if (func.blocks.empty()) {
        return 0;
    }
    const Cfg cfg(func);
    const DomTree dom = compute_dominator_tree(cfg);
    const LoopForest loops = compute_loop_forest(cfg, dom);
    vector<Nest> nests;
    for (BlockId b : cfg.rpo) {
        if (const auto *p =
                std::get_if<Terminator::ParFor>(&cfg[b].terminator.data)) {
            nests.push_back(Nest{b, p, reachable_from(cfg, cfg.id(p->body.name))});
        }
    }
    Definitions defs(func, /*lenient=*/true);

    // Every move is decided against the graph as it is, then made: a move
    // changes no edge, so the analyses above stay right, but it does change
    // the instruction lists being walked.
    struct Move {
        shared_ptr<Instruction> instr;
        shared_ptr<Block> from, to;
        map<string, Definition> sizes;
        // Pure instructions the sizes depend on, computed inside the loops
        // the allocation leaves, moved to `to` ahead of it in dependency
        // order (see `movable` below).
        vector<shared_ptr<Instruction>> companions;
    };
    vector<Move> moves;
    // Where an earlier move takes a companion: a later allocation that needs
    // the same instruction finds it there.
    map<const Instruction *, BlockId> moved_to;
    const auto has = [](const vector<shared_ptr<Instruction>> &list,
                        const shared_ptr<Instruction> &i) {
        return std::find(list.begin(), list.end(), i) != list.end();
    };
    // Whether `v`, as the block named `block` refers to it, is available at
    // `next` -- a constant, or defined in a block that dominates it -- or
    // can be made available by moving pure arithmetic there: an instruction
    // with no effect but its value, every operand of which is available or
    // movable in turn (a loop index, a value carried round the loop, a
    // load, a call stop it). Loop-invariant code motion (Allen and Cocke,
    // "A Catalogue of Optimizing Transformations", 1971; Muchnick, Advanced
    // Compiler Design and Implementation, ch. 13; LLVM's LICM pass) for the
    // one thing a hoisted allocation needs, its size: the deferral sizes a
    // queue owned by a loop whose producers are a nest by the product of
    // their trip counts, written at the head of the owner's iteration, and
    // without this the product, pure arithmetic on values settled before
    // every loop, kept the allocation in the loop (gpu-optix-pass.bonsai,
    // 10-30). The movable instructions go onto `out` in dependency order,
    // each once.
    std::function<bool(const shared_ptr<Value> &, const string &, BlockId,
                       vector<shared_ptr<Instruction>> &)>
        movable = [&](const shared_ptr<Value> &v, const string &block, BlockId next,
                      vector<shared_ptr<Instruction>> &out) -> bool {
            const Definition d = defs.of(block, v);
            if (d.block.empty()) {
                return true; // a constant
            }
            const BlockId def = cfg.find(d.block);
            if (def != NO_BLOCK && dom.dominates(def, next)) {
                return true;
            }
            const auto *held = std::get_if<shared_ptr<Instruction>>(&d.value->data);
            if (held == nullptr || !pure(**held)) {
                return false;
            }
            if (const auto earlier = moved_to.find(held->get()); earlier != moved_to.end()) {
                return dom.dominates(earlier->second, next);
            }
            if (has(out, *held)) {
                return true;
            }
            vector<shared_ptr<Instruction>> mine;
            for (const auto &operand : (*held)->operands) {
                if (!movable(operand, d.block, next, mine)) {
                    return false;
                }
            }
            for (const auto &i : mine) {
                if (!has(out, i)) {
                    out.push_back(i);
                }
            }
            out.push_back(*held);
            return true;
        };
    for (BlockId b : cfg.rpo) {
        const shared_ptr<Block> home = cfg.block(b);
        for (const auto &instr : home->instrs) {
            if (instr->op != Instruction::Op::Alloca &&
                instr->op != Instruction::Op::Alloc) {
                continue;
            }
            // Nothing reads what the storage held when an iteration began:
            // said by the rewrite that made it, or shown for a slot by a
            // store of the whole slot ahead of every use.
            if (!instr->scratch) {
                if (instr->type.is_reference() ||
                    !written_before_read(cfg, dom, instr.get())) {
                    continue;
                }
            }
            map<string, Definition> sizes;
            for (const string &name : names_in_type(instr->type)) {
                sizes.emplace(name, definition_of(defs, func, home, name));
            }
            // Outward one loop at a time, to the last block the sizes are
            // available in, stopping at a loop whose iterations are threads.
            BlockId at = b;
            BlockId target = NO_BLOCK;
            vector<shared_ptr<Instruction>> companions;
            while (true) {
                const Loop *natural = loops.innermost(at);
                const Nest *parfor = innermost_parfor(nests, at);
                if (natural == nullptr && parfor == nullptr) {
                    break;
                }
                BlockId next = NO_BLOCK;
                // The inner of the two constructs around `at`: a parfor whose
                // header lies inside the natural loop is inside it.
                if (parfor != nullptr &&
                    (natural == nullptr || natural->blocks.contains(parfor->header))) {
                    if (parfor->loop->binding.has_value()) {
                        break;
                    }
                    next = parfor->header;
                } else {
                    next = dom.idom[natural->header];
                    if (next == NO_BLOCK || next == natural->header) {
                        break;
                    }
                    // A parfor body is not a natural loop -- its yields have
                    // no successors -- so a loop headed by the body's first
                    // block has its dominator outside the parfor. Leaving
                    // the loop is then leaving the parfor too, and goes by
                    // the parfor's rule.
                    if (parfor != nullptr && !parfor->body.contains(next)) {
                        if (parfor->loop->binding.has_value()) {
                            break;
                        }
                        next = parfor->header;
                    }
                }
                // Every size available at `next`, or made so by moving the
                // pure arithmetic that computes it (`movable`).
                bool available = true;
                vector<shared_ptr<Instruction>> needed;
                for (const auto &[name, d] : sizes) {
                    if (d.block.empty()) {
                        continue; // a constant
                    }
                    if (!movable(d.value, d.block, next, needed)) {
                        available = false;
                        break;
                    }
                }
                if (!available) {
                    break;
                }
                target = next;
                companions = std::move(needed);
                at = next;
            }
            if (target == NO_BLOCK) {
                continue;
            }
            for (const auto &c : companions) {
                moved_to.emplace(c.get(), target);
            }
            moves.push_back(Move{instr, home, cfg.block(target), std::move(sizes),
                                 std::move(companions)});
        }
    }

    // Take `i` out of the block that holds it and put it at the end of `to`,
    // its name following it.
    const auto relocate = [](const shared_ptr<Instruction> &i, const shared_ptr<Block> &from,
                             const shared_ptr<Block> &to) {
        auto &instrs = from->instrs;
        for (auto it = instrs.begin(); it != instrs.end(); ++it) {
            if (*it == i) {
                instrs.erase(it);
                break;
            }
        }
        if (const auto look = from->lookups.find(i->name); look != from->lookups.end()) {
            const auto *held = std::get_if<shared_ptr<Instruction>>(&look->second->data);
            if (held != nullptr && held->get() == i.get()) {
                from->lookups.erase(look);
            }
        }
        i->owner = to;
        to->instrs.push_back(i);
        to->lookups[i->name] = std::make_shared<Value>(i);
    };
    for (Move &m : moves) {
        // The companions first, in dependency order, each operand that was
        // a block's argument replaced by the value it is threaded from, as
        // the target block refers to it.
        for (const auto &c : m.companions) {
            const shared_ptr<Block> from = c->owner.lock();
            internal_assert(from != nullptr)
                << "hoist_invariant_allocations: " << c->name << " has no block";
            for (auto &operand : c->operands) {
                if (std::holds_alternative<Argument>(operand->data)) {
                    operand = defs.of(from->name, operand).value;
                }
            }
            relocate(c, from, m.to);
        }
        auto &from = m.from->instrs;
        for (auto it = from.begin(); it != from.end(); ++it) {
            if (*it == m.instr) {
                from.erase(it);
                break;
            }
        }
        if (const auto look = m.from->lookups.find(m.instr->name);
            look != m.from->lookups.end()) {
            const auto *held = std::get_if<shared_ptr<Instruction>>(&look->second->data);
            if (held != nullptr && held->get() == m.instr.get()) {
                m.from->lookups.erase(look);
            }
        }
        // The sizes by the names their definitions go by, which the target
        // block sees: a size the old block took as an argument under one name
        // may be an instruction of another.
        map<string, Expr> renames;
        for (const auto &[name, d] : m.sizes) {
            if (d.block.empty()) {
                continue;
            }
            const string &as = name_of(*d.value);
            if (!as.empty() && as != name) {
                renames.emplace(name, Var::make(d.value->get_type(), as));
            }
        }
        if (!renames.empty()) {
            m.instr->type = rename_in_type(m.instr->type, renames);
        }
        m.instr->owner = m.to;
        m.to->instrs.push_back(m.instr);
        m.to->lookups[m.instr->name] = std::make_shared<Value>(m.instr);
    }
    return moves.size();
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
