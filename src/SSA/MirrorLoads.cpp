#include "SSA/MirrorLoads.h"

#include "Error.h"
#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "SSA/Simplify.h"
#include "Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;
using ValuePtr = shared_ptr<Value>;

const Instruction *def_of(const ValuePtr &v) {
    const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
    return held == nullptr ? nullptr : held->get();
}

// The name a value is known by: an instruction's, an argument's (a value
// carried into the blocks below keeps the instruction's name as their
// argument), none for a constant.
string name_of(const ValuePtr &v) {
    if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data)) {
        return (*in)->name;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return a->name;
    }
    return "";
}

bool writes_cell(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Store:
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
        return true;
    default:
        return false;
    }
}

// A pure computation of its operands and nothing more, which a clone of
// it at another place computes alike: what a chain from the load is made
// of, and what an operand of the chain may be a computation of.
bool recomputable(const Instruction &in) {
    if (in.name.empty() || in.storage.has_value()) {
        return false;
    }
    switch (in.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eq:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::LoadField:
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
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
    case Instruction::Op::Inf:
    case Instruction::Op::Eps:
        return true;
    case Instruction::Op::Intrinsic:
        switch (in.intrinsic) {
        case ir::Intrinsic::abs:
        case ir::Intrinsic::min:
        case ir::Intrinsic::max:
        case ir::Intrinsic::fma:
        case ir::Intrinsic::sqrt:
        case ir::Intrinsic::sqr:
        case ir::Intrinsic::rcp_approx:
            return true;
        default:
            return false;
        }
    default:
        return false;
    }
}

struct Mirroring {
    Function &func;
    const set<string> params; // the entry block's arguments, by name
    // How many places read each instruction's value.
    map<const Instruction *, size_t> uses;

    Mirroring(Function &f) : func(f), params(parameter_names(f)) {
        // Readers are operands and what terminators read or pass; a block's
        // table of names is not one.
        const auto count = [&](ValuePtr &v) {
            if (const Instruction *d = def_of(v)) {
                uses[d]++;
            }
        };
        for (const shared_ptr<Block> &block : func.blocks) {
            for (const shared_ptr<Instruction> &in : block->instrs) {
                for (ValuePtr &o : in->operands) {
                    if (o != nullptr) {
                        count(o);
                    }
                }
            }
            for_each_value(block->terminator, count);
        }
    }

    static set<string> parameter_names(const Function &f) {
        set<string> names;
        for (const Argument &a : f.blocks.front()->args) {
            names.insert(a.name);
        }
        return names;
    }

    // Whether `v` can be computed anywhere in the function: a constant, a
    // parameter, or a pure computation of such, to a bounded depth.
    bool available(const ValuePtr &v, int depth = 0) const {
        if (std::holds_alternative<Constant>(v->data)) {
            return true;
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            return params.count(a->name) > 0;
        }
        const Instruction *d = def_of(v);
        if (d == nullptr || depth > 8 || !recomputable(*d)) {
            return false;
        }
        for (const ValuePtr &o : d->operands) {
            if (!available(o, depth + 1)) {
                return false;
            }
        }
        return true;
    }

    // Whether a value names the cell: the allocation itself, or an argument
    // carrying it by its name (a parameter's, or the allocation's).
    static bool names(const ValuePtr &v, const string &cell) {
        return name_of(v) == cell;
    }

    // A clone of `v` at `at` (position `pos` of block `at`, advanced past
    // what is inserted), the instructions of its computation cloned too,
    // with `replaced` standing for the values given; constants and the
    // parameters as they are.
    ValuePtr clone_at(const shared_ptr<Block> &at, size_t &pos, const ValuePtr &v,
                      map<const Instruction *, ValuePtr> &replaced) {
        const Instruction *d = def_of(v);
        if (d == nullptr) {
            return v;
        }
        if (const auto it = replaced.find(d); it != replaced.end()) {
            return it->second;
        }
        vector<ValuePtr> operands;
        for (const ValuePtr &o : d->operands) {
            operands.push_back(clone_at(at, pos, o, replaced));
        }
        auto copy = std::make_shared<Instruction>(func.get_unique_name(), d->type,
                                                  d->op, std::move(operands), at);
        copy->intrinsic = d->intrinsic;
        at->instrs.insert(at->instrs.begin() + pos, copy);
        pos++;
        auto value = std::make_shared<Value>(copy);
        at->lookups[copy->name] = value;
        replaced[d] = value;
        return value;
    }

    // The cells: every allocation of the entry block and every pointer
    // parameter whose name is used only as the pointer of a load, a store
    // or an accumulate, or passed between blocks by name.
    struct Cell {
        string name;
        ValuePtr pointer; // how to refer to it anywhere: by its name
        Type held;
    };
    vector<Cell> cells() const {
        vector<Cell> found;
        const shared_ptr<Block> &entry = func.blocks.front();
        const auto consider = [&](const string &name, const ValuePtr &pointer,
                                  const Type &type) {
            if (!type.is<Ptr_t>() || type.is_reference()) {
                return;
            }
            bool only_memory = true;
            for (const shared_ptr<Block> &block : func.blocks) {
                for (const shared_ptr<Instruction> &in : block->instrs) {
                    for (size_t k = 0; k < in->operands.size(); k++) {
                        if (!names(in->operands[k], name)) {
                            continue;
                        }
                        const bool as_pointer =
                            k == 0 && (in->op == Instruction::Op::Load ||
                                       writes_cell(in->op));
                        if (!as_pointer) {
                            only_memory = false;
                        }
                    }
                }
                // A call's argument is the address going somewhere else.
                if (const auto *callee = block->terminator.callee()) {
                    for (const ValuePtr &a : callee->args) {
                        if (names(a, name)) {
                            only_memory = false;
                        }
                    }
                }
                if (const auto *multi =
                        std::get_if<Terminator::MultiCall>(&block->terminator.data)) {
                    for (const auto &row : multi->varying) {
                        for (const ValuePtr &a : row) {
                            if (names(a, name)) {
                                only_memory = false;
                            }
                        }
                    }
                }
            }
            if (only_memory) {
                found.push_back({name, pointer, type.element_of()});
            }
        };
        for (const Argument &a : entry->args) {
            consider(a.name, std::make_shared<Value>(a), a.type);
        }
        for (const shared_ptr<Instruction> &in : entry->instrs) {
            if (in->op == Instruction::Op::Alloca && !in->name.empty()) {
                consider(in->name, std::make_shared<Value>(in), in->type);
            }
        }
        return found;
    }

    // What the chain costs at every iteration: the instructions a backend
    // emits for it. A field read, a reinterpretation and a negation are
    // nothing (a field of a value in registers is a register, bits are bits,
    // and a `!` folds into the compare or branch that reads it); every other
    // operation is one. A chain worth a mirror costs at least two: a mirror
    // costs its update at every write of the cell, and a chain of one would
    // at best move one instruction from the reads to the writes.
    static size_t cost_of(const vector<shared_ptr<Instruction>> &chain) {
        size_t cost = 0;
        for (const shared_ptr<Instruction> &in : chain) {
            switch (in->op) {
            case Instruction::Op::LoadField:
            case Instruction::Op::Reinterpret:
            case Instruction::Op::Not:
                break;
            default:
                cost++;
                break;
            }
        }
        return cost;
    }

    // From a load of a cell, the chain of single readers that is a pure
    // computation with every other operand available: the last of them, or
    // the load itself when there is none.
    shared_ptr<Instruction> chain_end(const shared_ptr<Block> &block,
                                      const shared_ptr<Instruction> &load,
                                      vector<shared_ptr<Instruction>> &chain) const {
        shared_ptr<Instruction> cur = load;
        for (;;) {
            if (uses.count(cur.get()) == 0 || uses.at(cur.get()) != 1) {
                return cur;
            }
            shared_ptr<Instruction> next;
            for (const shared_ptr<Instruction> &in : block->instrs) {
                for (const ValuePtr &o : in->operands) {
                    if (def_of(o) == cur.get()) {
                        next = in;
                    }
                }
            }
            if (!next || !recomputable(*next)) {
                return cur;
            }
            for (const ValuePtr &o : next->operands) {
                if (def_of(o) != cur.get() && !available(o)) {
                    return cur;
                }
            }
            chain.push_back(next);
            cur = next;
        }
    }

    size_t run() {
        Cfg cfg(func);
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest loops = compute_loop_forest(cfg, dom);
        if (loops.empty()) {
            return 0;
        }
        size_t mirrored = 0;
        const shared_ptr<Block> &entry = func.blocks.front();
        for (const Cell &cell : cells()) {
            // The writes of the cell, each followed by the mirror's update.
            vector<std::pair<shared_ptr<Block>, size_t>> writes;
            for (const shared_ptr<Block> &block : func.blocks) {
                for (size_t i = 0; i < block->instrs.size(); i++) {
                    const shared_ptr<Instruction> &in = block->instrs[i];
                    if (writes_cell(in->op) && !in->operands.empty() &&
                        names(in->operands[0], cell.name)) {
                        writes.emplace_back(block, i);
                    }
                }
            }
            // The loads inside loops, each with its chain.
            struct Candidate {
                shared_ptr<Block> block;
                shared_ptr<Instruction> load;
                vector<shared_ptr<Instruction>> chain;
            };
            vector<Candidate> candidates;
            for (const shared_ptr<Block> &block : func.blocks) {
                const BlockId b = cfg.find(block->name);
                if (b == NO_BLOCK || loops.innermost(b) == nullptr) {
                    continue;
                }
                for (const shared_ptr<Instruction> &in : block->instrs) {
                    if (in->op != Instruction::Op::Load || in->operands.size() != 1 ||
                        !names(in->operands[0], cell.name)) {
                        continue;
                    }
                    Candidate c{block, in, {}};
                    chain_end(block, in, c.chain);
                    if (cost_of(c.chain) >= 2) {
                        candidates.push_back(std::move(c));
                    }
                }
            }
            for (const Candidate &c : candidates) {
                const shared_ptr<Instruction> last = c.chain.back();
                // The mirror, allocated at the entry.
                auto mirror = std::make_shared<Instruction>(
                    func.get_unique_name() + "!mirror", Ptr_t::make(last->type),
                    Instruction::Op::Alloca, vector<ValuePtr>{}, entry);
                size_t first_instr = 0;
                while (first_instr < entry->instrs.size() &&
                       entry->instrs[first_instr]->op == Instruction::Op::Alloca) {
                    first_instr++;
                }
                entry->instrs.insert(entry->instrs.begin() + first_instr, mirror);
                auto mirror_value = std::make_shared<Value>(mirror);
                entry->lookups[mirror->name] = mirror_value;

                // The chain recomputed from a fresh load and stored into
                // the mirror, at the entry (after the allocations and the
                // cell's own first store if it is there) and after every
                // write of the cell.
                const auto update = [&](const shared_ptr<Block> &at, size_t pos) {
                    auto load = std::make_shared<Instruction>(
                        func.get_unique_name(), c.load->type, Instruction::Op::Load,
                        vector<ValuePtr>{cell.pointer}, at);
                    at->instrs.insert(at->instrs.begin() + pos, load);
                    pos++;
                    auto loaded = std::make_shared<Value>(load);
                    at->lookups[load->name] = loaded;
                    map<const Instruction *, ValuePtr> replaced;
                    replaced[c.load.get()] = loaded;
                    ValuePtr value = clone_at(at, pos, std::make_shared<Value>(last),
                                              replaced);
                    auto store = std::make_shared<Instruction>(
                        Instruction::Op::Store, vector<ValuePtr>{mirror_value, value},
                        at);
                    at->instrs.insert(at->instrs.begin() + pos, store);
                    return pos + 1;
                };
                // After every write, from the last position in a block
                // backwards so that the earlier positions stay right; a
                // write in the entry block sits one further on now, past the
                // mirror's allocation.
                vector<std::pair<shared_ptr<Block>, size_t>> ordered = writes;
                std::sort(ordered.begin(), ordered.end(),
                          [](const auto &a, const auto &b) { return a.second > b.second; });
                bool written_in_entry = false;
                for (const auto &[block, index] : ordered) {
                    const bool in_entry = block.get() == entry.get();
                    written_in_entry = written_in_entry || in_entry;
                    const size_t at = in_entry && index >= first_instr ? index + 1 : index;
                    update(block, at + 1);
                }
                // And at the entry, for what the cell holds before any write
                // of this function's -- a pointer parameter's contents as the
                // caller left them -- right after the allocations. A cell
                // this function writes in its entry block is covered by that
                // write's update.
                if (!written_in_entry) {
                    size_t pos = 0;
                    while (pos < entry->instrs.size() &&
                           entry->instrs[pos]->op == Instruction::Op::Alloca) {
                        pos++;
                    }
                    update(entry, pos);
                }

                // The value in the loop is the mirror's.
                size_t where = 0;
                for (size_t i = 0; i < c.block->instrs.size(); i++) {
                    if (c.block->instrs[i].get() == last.get()) {
                        where = i;
                    }
                }
                auto read = std::make_shared<Instruction>(
                    func.get_unique_name(), last->type, Instruction::Op::Load,
                    vector<ValuePtr>{mirror_value}, c.block);
                c.block->instrs.insert(c.block->instrs.begin() + where, read);
                auto read_value = std::make_shared<Value>(read);
                c.block->lookups[read->name] = read_value;
                replace_uses(func, last.get(), read_value);
                mirrored++;
            }
        }
        if (mirrored > 0) {
            remove_dead(func);
        }
        return mirrored;
    }
};

} // namespace

size_t mirror_derived_loads(Function &func) {
    if (func.blocks.empty()) {
        return 0;
    }
    Mirroring m(func);
    return m.run();
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
