#include "SSA/DestinationPassing.h"

#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "SSA/SSA.h"

#include "Error.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
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
using InstrPtr = shared_ptr<Instruction>;

const Instruction *instruction_of(const ValuePtr &v) {
    const auto *held = std::get_if<InstrPtr>(&v->data);
    return held == nullptr ? nullptr : held->get();
}

// A definition's identity, for sets: an instruction by its pointer, a block
// parameter by its block and name, a constant by its spelling.
string key_of(const Definition &d) {
    if (const Instruction *in = instruction_of(d.value)) {
        return "i:" + in->name;
    }
    if (const optional<Argument> a = d.value->get_argument()) {
        return "a:" + d.block + ":" + a->name;
    }
    std::ostringstream os;
    d.value->dump(os);
    return "c:" + os.str();
}

optional<uint64_t> constant_index(const ValuePtr &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return *i < 0 ? std::nullopt : optional<uint64_t>(uint64_t(*i));
    }
    return std::nullopt;
}

bool is_acc(Instruction::Op op) {
    switch (op) {
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

// Whether operand `k` of an accumulate is the place of one of its alongside
// pairs (ir::Accumulate::alongside): the pairs close the operand list, place
// first.
bool is_alongside_place(const Instruction &in, size_t k) {
    return is_acc(in.op) && k >= in.alongside_begin() &&
           (k - in.alongside_begin()) % 2 == 0;
}

// Reads of memory: a Load through a pointer, an element taken out of an
// array (not a lane out of a vector), and the place an accumulate reads and
// writes.
bool reads_memory_at(const Instruction &in, size_t k) {
    switch (in.op) {
    case Instruction::Op::Load:
        return k == 0;
    case Instruction::Op::ExtractIdx:
        return k == 0 && !in.operands[0]->get_type().is_vector();
    default:
        return is_acc(in.op) && k == 0;
    }
}

// Where the values a jump passes land: the index of the parameter of
// `target` that argument `k` of the jump `j` becomes (see passed_to).
size_t param_index(const Block &from, const Terminator::Jump &j, size_t k) {
    return std::visit(
        overloads{
            [&](const Terminator::ParFor &p) -> size_t {
                return &j == &p.body ? k + 1 : k;
            },
            [&](const Terminator::Call &c) -> size_t {
                return (&j == &c.cont && !c.drop) ? k + 1 : k;
            },
            [&](const auto &) -> size_t { return k; },
        },
        from.terminator.data);
}

// The jumps out of `block` within this function, with the callee's jump of a
// Call or MultiCall left out (it goes to another function).
vector<const Terminator::Jump *> edges_of(const Block &block) {
    vector<const Terminator::Jump *> out;
    std::visit(overloads{
                   [&](const std::monostate &) {},
                   [&](const Terminator::Jump &j) { out.push_back(&j); },
                   [&](const Terminator::Dispatch &d) {
                       for (const auto &t : d.targets) {
                           out.push_back(&t);
                       }
                   },
                   [&](const Terminator::Return &) {},
                   [&](const Terminator::ParFor &p) {
                       out.push_back(&p.body);
                       out.push_back(&p.cont);
                   },
                   [&](const Terminator::Yield &) {},
                   [&](const Terminator::Call &c) { out.push_back(&c.cont); },
                   [&](const Terminator::MultiCall &c) {
                       out.push_back(&c.cont);
                   },
               },
               block.terminator.data);
    return out;
}

// The values a terminator reads for itself, not to pass on: a dispatch's
// condition, a return's value, a loop's bounds, a run's keys and conditions.
vector<ValuePtr> read_by(const Block &block) {
    vector<ValuePtr> out;
    std::visit(overloads{
                   [&](const Terminator::Dispatch &d) { out.push_back(d.cond); },
                   [&](const Terminator::Return &r) {
                       if (r.value) {
                           out.push_back(r.value);
                       }
                   },
                   [&](const Terminator::ParFor &p) {
                       out.insert(out.end(), {p.start, p.end, p.stride});
                   },
                   [&](const Terminator::MultiCall &c) {
                       out.insert(out.end(), c.keys.begin(), c.keys.end());
                       out.insert(out.end(), c.conds.begin(), c.conds.end());
                       for (const auto &row : c.varying) {
                           out.insert(out.end(), row.begin(), row.end());
                       }
                   },
                   [&](const auto &) {},
               },
               block.terminator.data);
    return out;
}

bool ends_region(const Block &block) {
    return std::holds_alternative<Terminator::Yield>(block.terminator.data) ||
           std::holds_alternative<Terminator::Return>(block.terminator.data);
}

// Where the value `v`, as `block` refers to it, is defined. Definitions
// settles a block argument to what is threaded into it; a name a block
// refers to from above without taking it as an argument -- a parameter of
// the function, an instruction of a dominating block -- comes back (in the
// lenient mode) as an argument of the referring block, and is settled here
// to the one definition of that name, as the code generators find it.
Definition settle(Definitions &defs, const BlockMap &bmap, const Function &func,
                  const string &block, const ValuePtr &v) {
    const Definition d = defs.of(block, v);
    const optional<Argument> a = d.value->get_argument();
    if (!a.has_value() || d.block.empty()) {
        return d;
    }
    const auto home = bmap.find(d.block);
    if (home != bmap.end()) {
        for (const Argument &declared : home->second->args) {
            if (declared.name == a->name) {
                return d; // a real argument of its block
            }
        }
    }
    for (const auto &b : func.blocks) {
        for (const auto &in : b->instrs) {
            if (in->name == a->name) {
                return Definition{std::make_shared<Value>(in), b->name};
            }
        }
    }
    for (const auto &b : func.blocks) {
        for (const Argument &declared : b->args) {
            if (declared.name == a->name) {
                return defs.of(b->name, std::make_shared<Value>(declared));
            }
        }
    }
    return d;
}

// The analysis of one function, shared by the checks below.
struct Context {
    Function &func;
    const FuncMap &fmap;
    Definitions defs;
    BlockMap bmap;
    std::unordered_map<string, const Instruction *> instrs; // for roots_of

    Context(Function &f, const FuncMap &m)
        : func(f), fmap(m), defs(f, /*lenient=*/true) {
        bmap = make_block_map(f);
        for (const auto &block : f.blocks) {
            for (const auto &in : block->instrs) {
                if (!in->name.empty()) {
                    instrs[in->name] = in.get();
                }
            }
        }
    }

    Definition of(const string &block, const ValuePtr &v) {
        return settle(defs, bmap, func, block, v);
    }

    bool is(const Block &block, const ValuePtr &v, const Definition &d) {
        return same_definition(of(block.name, v), d);
    }
};

// BONSAI_EXPLAIN_DESTINATIONS=1 prints every refusal: which function, which
// slot, and the condition that failed.
bool explaining() {
    static const bool on = std::getenv("BONSAI_EXPLAIN_DESTINATIONS") != nullptr;
    return on;
}

bool refuse(const string &where, const string &why) {
    if (explaining()) {
        std::cerr << "destination passing: " << where << ": " << why << "\n";
    }
    return false;
}

// A callee's parameter `k` is only written through: every use of it is the
// address of a Store, the place of an alongside pair, or an argument of a
// call whose callee uses it so. Followed through the threading of the value
// along edges under its own name, and into the merges it flows into (a
// loop header's parameter, carried round unchanged), each of which has to
// be only written through as well; any read of it is a refusal.
bool only_written(const FuncMap &fmap, const string &callee, size_t k,
                  set<std::pair<string, size_t>> &visiting,
                  const string &where) {
    if (!visiting.insert({callee, k}).second) {
        return true; // a recursion we are already inside of
    }
    const auto it = fmap.find(callee);
    if (it == fmap.end() || it->second->blocks.empty()) {
        return refuse(where, "handed to " + callee + ", which is not here");
    }
    Function &f = *it->second;
    const shared_ptr<Block> &entry = f.blocks.front();
    if (k >= entry->args.size()) {
        return refuse(where, "handed to " + callee + " past its parameters");
    }
    const BlockMap bmap = make_block_map(f);
    Definitions defs(f, /*lenient=*/true);
    vector<Definition> pending = {settle(
        defs, bmap, f, entry->name, std::make_shared<Value>(entry->args[k]))};
    set<string> checked;
    const auto is_one = [&](const Block &b, const ValuePtr &v,
                            const Definition &d) {
        return same_definition(settle(defs, bmap, f, b.name, v), d);
    };
    while (!pending.empty()) {
        const Definition param = pending.back();
        pending.pop_back();
        if (!checked.insert(key_of(param)).second) {
            continue;
        }
        for (const auto &block : f.blocks) {
            for (const auto &in : block->instrs) {
                for (size_t i = 0; i < in->operands.size(); i++) {
                    if (!is_one(*block, in->operands[i], param)) {
                        continue;
                    }
                    const bool store_address =
                        in->op == Instruction::Op::Store && i == 0;
                    if (!store_address && !is_alongside_place(*in, i)) {
                        return refuse(where, callee + " reads its parameter " +
                                                 std::to_string(k) + " (" +
                                                 op_name(in->op) + ")");
                    }
                }
            }
            for (const ValuePtr &v : read_by(*block)) {
                if (is_one(*block, v, param)) {
                    return refuse(where, callee + " reads its parameter " +
                                             std::to_string(k) +
                                             " in a terminator");
                }
            }
            for (const Terminator::Jump *j : edges_of(*block)) {
                const auto target = bmap.find(j->name);
                for (size_t i = 0; i < j->args.size(); i++) {
                    if (!is_one(*block, j->args[i], param)) {
                        continue;
                    }
                    if (target == bmap.end()) {
                        return refuse(where, callee + " jumps to " + j->name +
                                                 ", which is not a block");
                    }
                    const size_t p = param_index(*block, *j, i);
                    if (p >= target->second->args.size()) {
                        return refuse(where, callee + " passes its parameter " +
                                                 std::to_string(k) +
                                                 " past " + j->name +
                                                 "'s parameters");
                    }
                    const Definition landed = settle(
                        defs, bmap, f, target->second->name,
                        std::make_shared<Value>(target->second->args[p]));
                    if (!same_definition(landed, param)) {
                        pending.push_back(landed); // a merge: check it too
                    }
                }
            }
            if (const Terminator::Jump *c = block->terminator.callee()) {
                for (size_t i = 0; i < c->args.size(); i++) {
                    if (is_one(*block, c->args[i], param) &&
                        !only_written(fmap, c->name, i, visiting, where)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

// Are two addresses one place? The same definition, or the same chain of
// GEP and FieldPtr over the same definitions.
bool same_address(Context &cx, const Block &ba, const ValuePtr &a,
                  const Block &bb, const ValuePtr &b) {
    const Definition da = cx.of(ba.name, a);
    const Definition db = cx.of(bb.name, b);
    if (same_definition(da, db)) {
        return true;
    }
    const Instruction *ia = instruction_of(da.value);
    const Instruction *ib = instruction_of(db.value);
    if (ia == nullptr || ib == nullptr || ia->op != ib->op ||
        (ia->op != Instruction::Op::GEP && ia->op != Instruction::Op::FieldPtr) ||
        ia->operands.size() != ib->operands.size()) {
        return false;
    }
    const Block &ha = *cx.bmap.at(da.block);
    const Block &hb = *cx.bmap.at(db.block);
    for (size_t i = 0; i < ia->operands.size(); i++) {
        if (!same_address(cx, ha, ia->operands[i], hb, ib->operands[i])) {
            return false;
        }
    }
    return true;
}

// A store of the copied value: the sink the destination is read off.
struct Sink {
    shared_ptr<Block> block;
    InstrPtr store;
};

// Follows the loaded value forward, through the structs it is packed into
// and the fields taken back out, through the block arguments it is passed
// as, to the stores of it. `path` is where the value sits inside the value
// being followed: the field indices down to it. Returns false where it
// reaches anything else.
bool follow(Context &cx, const Definition &def, vector<unsigned> path,
            set<string> &seen, vector<Sink> &sinks) {
    string k = key_of(def);
    for (unsigned p : path) {
        k += "." + std::to_string(p);
    }
    if (!seen.insert(k).second) {
        return true;
    }
    for (const auto &block : cx.func.blocks) {
        for (const auto &in : block->instrs) {
            for (size_t i = 0; i < in->operands.size(); i++) {
                if (!cx.is(*block, in->operands[i], def)) {
                    continue;
                }
                switch (in->op) {
                case Instruction::Op::MakeStruct: {
                    vector<unsigned> inner = {unsigned(i)};
                    inner.insert(inner.end(), path.begin(), path.end());
                    if (!follow(cx, Definition{std::make_shared<Value>(in),
                                               block->name},
                                std::move(inner), seen, sinks)) {
                        return false;
                    }
                    break;
                }
                case Instruction::Op::LoadField: {
                    if (i != 0 || path.empty()) {
                        return false; // a field of the record itself
                    }
                    const optional<uint64_t> field = constant_index(in->operands[1]);
                    if (!field.has_value()) {
                        return false;
                    }
                    if (*field != path.front()) {
                        break; // another field: not the record
                    }
                    if (!follow(cx, Definition{std::make_shared<Value>(in),
                                               block->name},
                                vector<unsigned>(path.begin() + 1, path.end()),
                                seen, sinks)) {
                        return false;
                    }
                    break;
                }
                case Instruction::Op::Store:
                    if (i != 1 || !path.empty() || in->operands.size() != 2 ||
                        in->compact) {
                        return false;
                    }
                    sinks.push_back({block, in});
                    break;
                default:
                    return false;
                }
            }
        }
        for (const ValuePtr &v : read_by(*block)) {
            if (cx.is(*block, v, def)) {
                return false;
            }
        }
        if (const Terminator::Jump *c = block->terminator.callee()) {
            for (const ValuePtr &v : c->args) {
                if (cx.is(*block, v, def)) {
                    return false; // handed to a callee
                }
            }
        }
        for (const Terminator::Jump *j : edges_of(*block)) {
            const auto target = cx.bmap.find(j->name);
            for (size_t i = 0; i < j->args.size(); i++) {
                if (!cx.is(*block, j->args[i], def)) {
                    continue;
                }
                if (target == cx.bmap.end()) {
                    return false;
                }
                const size_t p = param_index(*block, *j, i);
                if (p >= target->second->args.size()) {
                    return false;
                }
                const Definition landed = cx.of(
                    target->second->name,
                    std::make_shared<Value>(target->second->args[p]));
                if (same_definition(landed, def)) {
                    continue; // threaded on under its own name: seen above
                }
                if (!follow(cx, landed, path, seen, sinks)) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Builds the address `v` (as `from` names it) in `into`, before position
// `at`: a chain of GEP and FieldPtr is rebuilt over its operands, and a
// value is fetched by name where `into` can see it. Dry (no `into`) to ask
// whether it can be.
struct Materialize {
    Context &cx;
    const DomTree &dom;
    const Cfg &cfg;
    shared_ptr<Block> into; // null: dry run
    size_t at;
    size_t slot_index; // where the slot is in `into`, for ordering

    bool available(const Definition &d) const {
        if (d.block.empty()) {
            return true; // a constant
        }
        const BlockId home = cfg.find(d.block);
        const BlockId here = cfg.find(into ? into->name : at_block);
        if (home == NO_BLOCK || here == NO_BLOCK) {
            return false;
        }
        if (home != here) {
            return dom.dominates(home, here);
        }
        // The same block: a parameter is defined at its start; an
        // instruction must come before the slot.
        const Instruction *in = instruction_of(d.value);
        if (in == nullptr) {
            return true;
        }
        const Block &b = *cx.bmap.at(d.block);
        for (size_t i = 0; i < b.instrs.size(); i++) {
            if (b.instrs[i].get() == in) {
                return i < slot_index;
            }
        }
        return false;
    }

    string at_block; // for a dry run

    optional<ValuePtr> build(const Block &from, const ValuePtr &v) {
        const Definition d = cx.of(from.name, v);
        if (d.block.empty()) {
            return d.value; // a constant
        }
        const Instruction *in = instruction_of(d.value);
        if (in != nullptr && (in->op == Instruction::Op::GEP ||
                              in->op == Instruction::Op::FieldPtr)) {
            const Block &home = *cx.bmap.at(d.block);
            vector<ValuePtr> operands;
            for (const ValuePtr &o : in->operands) {
                optional<ValuePtr> built = build(home, o);
                if (!built.has_value()) {
                    return std::nullopt;
                }
                operands.push_back(*built);
            }
            if (!into) {
                return d.value; // dry: any value stands in
            }
            auto made = std::make_shared<Instruction>(
                cx.func.get_unique_name(), in->type, in->op,
                std::move(operands), into);
            into->instrs.insert(into->instrs.begin() + long(at), made);
            at++;
            slot_index++;
            auto value = std::make_shared<Value>(made);
            into->lookups.insert({made->name, value});
            return value;
        }
        if (!available(d)) {
            return std::nullopt;
        }
        if (!into) {
            return d.value;
        }
        if (in != nullptr) {
            return into->get_value(in->name, in->type);
        }
        const optional<Argument> a = d.value->get_argument();
        internal_assert(a.has_value());
        return into->get_value(a->name, a->type);
    }
};

struct Candidate {
    shared_ptr<Block> block; // the slot's
    InstrPtr slot;
    Definition def;
    vector<Sink> sinks;
    set<string> writers; // blocks that write the slot, directly or by a call
};

// The one rewrite, or nothing. Every refusal is a `return false`.
bool pass_one(Context &cx, const Cfg &cfg, const DomTree &dom, Candidate &c) {
    Function &func = cx.func;
    const string where = func.blocks.front()->name + ": " + c.slot->name;
    // Uses of the slot: writes, calls that only write, and the one load.
    optional<std::pair<shared_ptr<Block>, InstrPtr>> load;
    for (const auto &block : func.blocks) {
        for (const auto &in : block->instrs) {
            for (size_t i = 0; i < in->operands.size(); i++) {
                if (!cx.is(*block, in->operands[i], c.def)) {
                    continue;
                }
                if (in->op == Instruction::Op::Load && i == 0) {
                    if (load.has_value()) {
                        return refuse(where, "loaded twice");
                    }
                    load = {block, in};
                } else if ((in->op == Instruction::Op::Store && i == 0) ||
                           is_alongside_place(*in, i)) {
                    c.writers.insert(block->name);
                } else {
                    return refuse(where, string("used by ") + op_name(in->op));
                }
            }
        }
        for (const ValuePtr &v : read_by(*block)) {
            if (cx.is(*block, v, c.def)) {
                return refuse(where, "read by a terminator");
            }
        }
        if (const Terminator::Jump *call = block->terminator.callee()) {
            const auto *multi =
                std::get_if<Terminator::MultiCall>(&block->terminator.data);
            if (multi != nullptr) {
                for (const auto &row : multi->varying) {
                    for (const ValuePtr &v : row) {
                        if (cx.is(*block, v, c.def)) {
                            return refuse(where, "varies over a run of calls");
                        }
                    }
                }
            }
            for (size_t i = 0; i < call->args.size(); i++) {
                if (!cx.is(*block, call->args[i], c.def)) {
                    continue;
                }
                set<std::pair<string, size_t>> visiting;
                if (!only_written(cx.fmap, call->name, i, visiting, where)) {
                    return false;
                }
                c.writers.insert(block->name);
            }
        }
        for (const Terminator::Jump *j : edges_of(*block)) {
            const auto target = cx.bmap.find(j->name);
            for (size_t i = 0; i < j->args.size(); i++) {
                if (!cx.is(*block, j->args[i], c.def)) {
                    continue;
                }
                if (target == cx.bmap.end()) {
                    return refuse(where, "passed to " + j->name +
                                             ", which is not a block");
                }
                const size_t p = param_index(*block, *j, i);
                if (p >= target->second->args.size()) {
                    return refuse(where, "passed past " + j->name +
                                             "'s parameters");
                }
                const Definition landed = cx.of(
                    target->second->name,
                    std::make_shared<Value>(target->second->args[p]));
                if (!same_definition(landed, c.def)) {
                    return refuse(where, "merged with another pointer at " +
                                             j->name);
                }
            }
        }
    }
    if (!load.has_value()) {
        return refuse(where, "never loaded");
    }
    if (c.writers.empty()) {
        return refuse(where, "never written");
    }

    // The loaded value's sinks: stores of it, whole, at one address.
    set<string> seen;
    if (!follow(cx, Definition{std::make_shared<Value>(load->second), load->first->name},
                {}, seen, c.sinks)) {
        return refuse(where, "the loaded value is read, not just stored");
    }
    if (c.sinks.empty()) {
        return refuse(where, "the loaded value is stored nowhere");
    }
    const Sink &first = c.sinks.front();
    const ValuePtr dest = first.store->operands[0];
    for (const Sink &s : c.sinks) {
        if (!same_address(cx, *first.block, dest, *s.block, s.store->operands[0])) {
            return refuse(where, "stored at two addresses");
        }
    }
    if (!dest->get_type().same_as(c.slot->type)) {
        return refuse(where, "the destination's type differs from the slot's");
    }

    // The destination's memory is otherwise untouched.
    const Roots droots = roots_of(dest, cx.instrs);
    if (droots.unknown || droots.names.empty()) {
        return refuse(where, "the destination's root cannot be named");
    }
    KernelWrites region;
    region.prefixes = droots.names;
    set<string> dest_stores; // blocks storing to D
    for (const auto &block : func.blocks) {
        for (const auto &in : block->instrs) {
            for (size_t i = 0; i < in->operands.size(); i++) {
                const ValuePtr &o = in->operands[i];
                if (!o->get_type().is<Ptr_t>() && !o->get_type().is_reference()) {
                    continue; // not an address: a value
                }
                const Roots r = roots_of(o, cx.instrs);
                if (r.unknown || !region.writes(r.names)) {
                    continue;
                }
                // Rooted at D. A store to D itself is the program's own;
                // a read, a store elsewhere in D's root, or any other use
                // of the address (an escape) is a refusal.
                if (in->op == Instruction::Op::Store && i == 0 &&
                    same_address(cx, *first.block, dest, *block, o)) {
                    dest_stores.insert(block->name);
                    continue;
                }
                if (in->op == Instruction::Op::GEP ||
                    in->op == Instruction::Op::FieldPtr ||
                    in->op == Instruction::Op::Cast ||
                    in->op == Instruction::Op::Reinterpret) {
                    continue; // an address being formed: judged at its use
                }
                return refuse(where, string("the destination's memory is ") +
                                         (reads_memory_at(*in, i) ? "read by "
                                                                  : "used by ") +
                                         op_name(in->op) + " in " + block->name);
            }
        }
        for (const ValuePtr &v : read_by(*block)) {
            const Roots r = roots_of(v, cx.instrs);
            if (!r.unknown && region.writes(r.names)) {
                return refuse(where, "the destination's address is read by " +
                                         block->name + "'s terminator");
            }
        }
        if (const Terminator::Jump *call = block->terminator.callee()) {
            for (const ValuePtr &v : call->args) {
                const Roots r = roots_of(v, cx.instrs);
                if (!r.unknown && region.writes(r.names)) {
                    return refuse(where, "the destination's address is handed to " +
                                             call->name);
                }
            }
        }
    }
    if (dest_stores.empty()) {
        return refuse(where, "no store to the destination was found");
    }

    // Every path from the slot to the end of its region stores D, and no
    // write of the slot follows a store to D.
    const BlockId start = cfg.find(c.block->name);
    if (start == NO_BLOCK) {
        return refuse(where, "the slot's block is unreachable");
    }
    if (dest_stores.contains(c.block->name)) {
        return refuse(where, "the destination is stored in the slot's own block");
    }
    {
        BlockSet visited(cfg.size());
        vector<BlockId> work = {start};
        visited.insert(start);
        while (!work.empty()) {
            const BlockId b = work.back();
            work.pop_back();
            if (dest_stores.contains(cfg.name(b))) {
                continue; // the store is made here
            }
            if (ends_region(cfg[b])) {
                return refuse(where, "a path to " + cfg.name(b) +
                                         " stores nothing to the destination");
            }
            for (BlockId s : cfg.succs[b]) {
                if (visited.insert(s)) {
                    work.push_back(s);
                }
            }
        }
    }
    for (const string &d : dest_stores) {
        const BlockId db = cfg.find(d);
        if (db == NO_BLOCK) {
            return refuse(where, "a store to the destination is unreachable");
        }
        const BlockSet after = reachable_from(cfg, db);
        for (const string &w : c.writers) {
            const BlockId wb = cfg.find(w);
            // A write in the same block as a store of D (where the order
            // would decide; refused rather than looked at) or in any block
            // after it.
            if (wb == NO_BLOCK || wb == db || after.contains(wb)) {
                return refuse(where, "the slot is written in " + w +
                                         ", after the destination is stored in " +
                                         d);
            }
        }
    }

    // The destination's address is computable where the slot is made.
    size_t slot_at = 0;
    for (; slot_at < c.block->instrs.size(); slot_at++) {
        if (c.block->instrs[slot_at] == c.slot) {
            break;
        }
    }
    internal_assert(slot_at < c.block->instrs.size());
    {
        Materialize dry{cx, dom, cfg, nullptr, slot_at, slot_at};
        dry.at_block = c.block->name;
        if (!dry.build(*first.block, dest).has_value()) {
            return refuse(where, "the destination's address is not computable "
                                 "where the slot is made");
        }
    }

    // The rewrite.
    Materialize make{cx, dom, cfg, c.block, slot_at, slot_at};
    optional<ValuePtr> address = make.build(*first.block, dest);
    internal_assert(address.has_value());
    replace_uses(func, c.slot.get(), *address);
    for (const Sink &s : c.sinks) {
        std::erase_if(s.block->instrs,
                      [&](const InstrPtr &in) { return in == s.store; });
    }
    if (explaining()) {
        std::cerr << "destination passing: " << where << ": written into "
                  << first.block->name << "'s destination\n";
    }
    return true;
}

} // namespace

size_t pass_destinations(Function &func, const FuncMap &fmap) {
    size_t passed = 0;
    // One slot at a time: the rewrite changes the function, and the
    // analyses are rebuilt for the next. A slot refused once is not asked
    // about again.
    set<const Instruction *> tried;
    for (;;) {
        Context cx(func, fmap);
        const Cfg cfg(func);
        const DomTree dom = compute_dominator_tree(cfg);
        bool any = false;
        for (const auto &block : func.blocks) {
            for (const auto &in : block->instrs) {
                if (in->op != Instruction::Op::Alloca ||
                    !in->type.is<Ptr_t>() || tried.contains(in.get())) {
                    continue;
                }
                tried.insert(in.get());
                Candidate c{block, in,
                            Definition{std::make_shared<Value>(in), block->name},
                            {},
                            {}};
                if (pass_one(cx, cfg, dom, c)) {
                    passed++;
                    any = true;
                    break;
                }
            }
            if (any) {
                break;
            }
        }
        if (!any) {
            return passed;
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
