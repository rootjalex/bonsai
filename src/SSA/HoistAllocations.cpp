#include "SSA/HoistAllocations.h"

#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "SSA/SSA.h"
#include "SSA/Storage.h"

#include "Error.h"
#include "Utils.h"

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
    };
    vector<Move> moves;
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
                bool available = true;
                for (const auto &[name, d] : sizes) {
                    if (d.block.empty()) {
                        continue; // a constant
                    }
                    const BlockId def = cfg.find(d.block);
                    if (def == NO_BLOCK || !dom.dominates(def, next)) {
                        available = false;
                        break;
                    }
                }
                if (!available) {
                    break;
                }
                target = next;
                at = next;
            }
            if (target == NO_BLOCK) {
                continue;
            }
            moves.push_back(Move{instr, home, cfg.block(target), std::move(sizes)});
        }
    }

    for (Move &m : moves) {
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
