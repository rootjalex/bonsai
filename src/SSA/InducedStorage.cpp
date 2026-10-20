#include "SSA/InducedStorage.h"

#include "Error.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/Defer.h"
#include "SSA/Reach.h"
#include "SSA/SSA.h"
#include "SSA/Simplify.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;
using InducedStorage = Function::InducedStorage;
using Op = Instruction::Op;

namespace {

// The name a value goes by: an instruction's, or a block argument's, which
// threads a definition onwards under the definition's own name.
const string *name_of(const shared_ptr<Value> &v) {
    if (!v) {
        return nullptr;
    }
    if (const auto *i = std::get_if<shared_ptr<Instruction>>(&v->data)) {
        return &(*i)->name;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return &a->name;
    }
    return nullptr;
}

bool refers(const shared_ptr<Value> &v, const string &name) {
    const string *n = name_of(v);
    return n != nullptr && *n == name;
}

optional<uint64_t> constant_index(const shared_ptr<Value> &v) {
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

// An address into a storage has one of two shapes: a pointer, or a vector
// of pointers, one per lane of a gang -- the slot addresses of sixteen
// entries a gang drain runs at once, ispc's varying pointer, which the
// vectorizer made of the drain's slot address and of every chain function's
// parameter that receives it. The split follows the shape: a leaf's address
// is a vector of pointers where the slot's was, and what moves through it
// is a gang-wide value per leaf (widen, IR/Type.h), as take_apart() and the
// queue's own gang reads already have it.
uint32_t lanes_of_address(const Type &type) {
    if (const auto *v = type.as<Vector_t>(); v != nullptr && v->etype.is<Ptr_t>()) {
        return v->lanes;
    }
    return 1;
}

// What an address of either shape points at, or null for no address
// (ir::pointee_of, IR/Type.h, is the pointer's alone).
const Type *address_pointee(const Type &type) {
    if (const auto *p = type.as<Ptr_t>()) {
        return &p->etype;
    }
    if (const auto *v = type.as<Vector_t>(); v != nullptr && v->etype.is<Ptr_t>()) {
        return &v->etype.as<Ptr_t>()->etype;
    }
    return nullptr;
}

// An address of `pointee` of the shape of `like`: a pointer, or a vector of
// as many pointers.
Type address_like(const Type &like, const Type &pointee) {
    if (const auto *v = like.as<Vector_t>(); v != nullptr && v->etype.is<Ptr_t>()) {
        return Vector_t::make(Ptr_t::make(pointee, v->etype.as<Ptr_t>()->readonly),
                              v->lanes);
    }
    const auto *p = like.as<Ptr_t>();
    return Ptr_t::make(pointee, p != nullptr && p->readonly);
}

// A value of `type` as a gang of `lanes` holds it: itself for one lane.
Type held_by(const Type &type, uint32_t lanes) {
    return lanes > 1 ? widen(type, lanes) : type;
}

// Every jump out of `block`, paired with the index its first argument binds
// to in the target's argument list (1 for a parfor body, whose first
// argument is the index, and for a call continuation handed the returned
// value first).
vector<std::pair<Terminator::Jump *, size_t>> jumps(Block &block) {
    vector<std::pair<Terminator::Jump *, size_t>> result;
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) { result.push_back({&j, 0}); },
                   [&](Terminator::Dispatch &d) {
                       for (auto &target : d.targets) {
                           result.push_back({&target, 0});
                       }
                   },
                   [&](Terminator::Return &) {},
                   [&](Terminator::ParFor &p) {
                       result.push_back({&p.body, 1});
                       result.push_back({&p.cont, 0});
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       result.push_back({&c.call, 0});
                       result.push_back({&c.cont, c.drop ? size_t(0) : 1});
                   },
                   [&](Terminator::Spawn &s) {
                       result.push_back({&s.call, 0});
                       result.push_back({&s.cont, 0});
                   },
                   [&](Terminator::Join &j) { result.push_back({&j.next, 0}); },
                   [&](Terminator::MultiCall &c) {
                       result.push_back({&c.call, 0});
                       result.push_back({&c.cont, c.drop ? size_t(0) : 1});
                   },
               },
               block.terminator.data);
    return result;
}

// What the split keeps about one storage over the whole program, for the
// unread rule at the end: which leaves anything loads, and everything made
// on each leaf -- the derived addresses, the stores and accumulates through
// them, the parameters chain functions were given for it -- so that an
// unread leaf's can all go.
struct Tracking {
    string handle;
    string owner;
    vector<Leaf> leaves;
    vector<bool> read;
    vector<vector<shared_ptr<Instruction>>> emitted;
    vector<vector<std::pair<string, string>>> params; // (function, parameter)
    vector<shared_ptr<Value>> arrays;                 // the leaf arrays, in the owner
};

// A value that stands for part of an induced storage once it is split: the
// storage itself (`array`), whose parts are the leaf arrays and whose uses
// are indexed; or the address of one slot's value of `type`, down `path`
// from the element, whose parts are the leaves' addresses. The parts are by
// leaf index over the whole storage, null outside [first, last) and for a
// pad.
struct Split {
    shared_ptr<Tracking> track;
    Type type;
    bool array = false;
    size_t first = 0, last = 0;
    vector<unsigned> path;
    vector<shared_ptr<Value>> parts;

    bool one_leaf_exactly() const {
        return first + 1 == last && track->leaves[first].kind == Leaf::Kind::Stored &&
               track->leaves[first].path == path;
    }
};

struct FunctionState {
    map<string, Split> splits; // by the name of the value they replace
    vector<string> pending;    // names whose uses are still to rewrite
    set<string> done;          // names rewritten: their threading is dead
    set<string> dead_names;    // allocations and parameters removed outright
};

struct Site {
    string func;
    string block;
};

class Splitter {
  public:
    explicit Splitter(FuncMap &funcs) : funcs(funcs) {
        for (const auto &[fname, f] : funcs) {
            for (const auto &block : f->blocks) {
                if (const Terminator::Jump *callee = block->terminator.callee()) {
                    callers[callee->name].push_back(Site{fname, block->name});
                }
            }
        }
    }

    void run() {
        for (auto &[fname, f] : funcs) {
            for (size_t s = 0; s < f->induced.size(); s++) {
                if (!f->induced[s].split) {
                    split_storage(fname, s);
                }
            }
        }
        drain();
        finish();
    }

  private:
    FuncMap &funcs;
    map<string, vector<Site>> callers;
    map<string, FunctionState> states;
    map<string, map<string, Split>> param_splits; // by function, by parameter
    vector<shared_ptr<Tracking>> tracks;
    vector<string> work; // functions with pending names
    // Two storages whose leaves [first, last) flow into one parameter: a read
    // through it is a read of either (split_parameter, finish).
    struct Shared {
        shared_ptr<Tracking> a, b;
        size_t first, last;
    };
    vector<Shared> shared;

    Function &function(const string &name) {
        const auto it = funcs.find(name);
        internal_assert(it != funcs.end()) << "no function " << name;
        return *it->second;
    }

    shared_ptr<Block> block_of(Function &f, const string &name) {
        for (const auto &b : f.blocks) {
            if (b->name == name) {
                return b;
            }
        }
        internal_error << "no block " << name;
        return nullptr;
    }

    void enqueue(const string &fname, const string &vname, Split split) {
        FunctionState &st = states[fname];
        st.splits[vname] = std::move(split);
        st.pending.push_back(vname);
        work.push_back(fname);
    }

    // A name no value of `f` has.
    string fresh(Function &f, string name) {
        const auto taken = [&](const string &n) {
            for (const auto &b : f.blocks) {
                if (b->lookups.contains(n)) {
                    return true;
                }
            }
            return false;
        };
        string candidate = name;
        for (unsigned k = 2; taken(candidate); k++) {
            candidate = name + "!" + std::to_string(k);
        }
        return candidate;
    }

    // The tail of a leaf's name past the stored thing's: `_n`, `_lambda_pdf`.
    static string tail_of(const Tracking &t, const string &local, size_t k) {
        const string &n = t.leaves[k].name;
        if (n.size() >= local.size() && n.compare(0, local.size(), local) == 0) {
            return n.substr(local.size());
        }
        return "_" + n;
    }

    // A named instruction inserted at `at` of `block`, `at` left after it.
    shared_ptr<Value> emit_at(Function &f, const shared_ptr<Block> &block, size_t &at,
                              const Type &type, Op op, vector<shared_ptr<Value>> ops) {
        auto in = std::make_shared<Instruction>(f.get_unique_name(), type, op,
                                                std::move(ops), block);
        block->instrs.insert(block->instrs.begin() + long(at), in);
        at++;
        auto v = std::make_shared<Value>(in);
        block->lookups[in->name] = v;
        return v;
    }

    shared_ptr<Instruction> effect_at(const shared_ptr<Block> &block, size_t &at, Op op,
                                      vector<shared_ptr<Value>> ops) {
        auto in = std::make_shared<Instruction>(op, std::move(ops), block);
        block->instrs.insert(block->instrs.begin() + long(at), in);
        at++;
        return in;
    }

    static void erase_at(const shared_ptr<Block> &block, size_t i) {
        const shared_ptr<Instruction> &in = block->instrs[i];
        if (!in->name.empty()) {
            block->lookups.erase(in->name);
        }
        block->instrs.erase(block->instrs.begin() + long(i));
    }

    static void erase_instruction(const shared_ptr<Instruction> &in) {
        const shared_ptr<Block> block = in->owner.lock();
        if (!block) {
            return;
        }
        for (size_t i = 0; i < block->instrs.size(); i++) {
            if (block->instrs[i] == in) {
                erase_at(block, i);
                return;
            }
        }
    }

    // An allocation like `like` -- as scratch, in the memory the schedule
    // placed it -- of `type`, inserted right after it.
    shared_ptr<Value> allocate_beside(Function &f, const Instruction &like, const Type &type,
                                      const string &name) {
        const shared_ptr<Block> block = like.owner.lock();
        internal_assert(block) << name << ": the storage it is made beside outlived its block";
        size_t at = block->instrs.size();
        for (size_t i = 0; i < block->instrs.size(); i++) {
            if (block->instrs[i].get() == &like) {
                at = i + 1;
            }
        }
        auto in = std::make_shared<Instruction>(
            fresh(f, name), type.is_reference() ? type : Ptr_t::make(type), like.op,
            vector<shared_ptr<Value>>{}, block);
        in->scratch = like.scratch;
        in->storage = like.storage;
        block->instrs.insert(block->instrs.begin() + long(at), in);
        auto v = std::make_shared<Value>(in);
        block->lookups[in->name] = v;
        return v;
    }

    // Whether `type` comes apart into leaves (leaves_of): numbers and bools,
    // short vectors of them, structs of those. An array or an address inside
    // has no leaf to be stored by, and such storage stays as it is.
    static bool splittable(const Type &type) {
        if (const auto *s = type.as<Struct_t>()) {
            return std::all_of(s->fields.begin(), s->fields.end(),
                               [](const TypedVar &f) { return splittable(f.type); });
        }
        if (const auto *v = type.as<Vector_t>()) {
            return splittable(v->etype);
        }
        return type.is_scalar();
    }

    // The storage `s` of `fname`: its leaf arrays made beside it, its uses
    // queued for the rewrite.
    void split_storage(const string &fname, size_t s) {
        Function &f = function(fname);
        InducedStorage &st = f.induced[s];
        internal_assert(st.arrays.size() == 1) << st.handle << ": registered with "
                                                << st.arrays.size() << " arrays, not one";
        const shared_ptr<Value> orig = st.arrays[0];
        const auto *oi = std::get_if<shared_ptr<Instruction>>(&orig->data);
        internal_assert(oi != nullptr) << st.handle << ": its storage is not an allocation";
        const auto *array_t = (*oi)->type.as<Array_t>();
        internal_assert(array_t != nullptr && equals(array_t->etype, st.element))
            << st.handle << ": its storage is " << (*oi)->type << ", not an array of "
            << st.element;
        if (!splittable(st.element)) {
            // An array inside, or an address: no leaves to store it by.
            st.split = true;
            st.leaf_names = {(*oi)->name};
            return;
        }
        auto track = std::make_shared<Tracking>();
        track->handle = st.handle;
        track->owner = fname;
        leaves_of(st.local, st.element, st.adts, /*in_pad=*/false, st.vector_leaves, {},
                  track->leaves);
        const size_t n = track->leaves.size();
        internal_assert(n > 0) << st.handle << ": " << st.element << " has no leaves";
        if (n == 1 && track->leaves[0].kind == Leaf::Kind::Stored &&
            equals(track->leaves[0].type, st.element)) {
            // One leaf, the element itself: the array is already the layout.
            st.split = true;
            st.leaf_names = {(*oi)->name};
            return;
        }
        track->read.assign(n, false);
        track->emitted.resize(n);
        track->params.resize(n);
        track->arrays.assign(n, nullptr);
        Split split;
        split.track = track;
        split.type = st.element;
        split.array = true;
        split.first = 0;
        split.last = n;
        split.parts.assign(n, nullptr);
        for (size_t k = 0; k < n; k++) {
            const Leaf &leaf = track->leaves[k];
            if (leaf.kind != Leaf::Kind::Stored) {
                continue;
            }
            track->arrays[k] = allocate_beside(
                f, **oi, Array_t::make(leaf.type, array_t->size, array_t->readonly),
                st.prefix + leaf.name + st.suffix);
            split.parts[k] = track->arrays[k];
        }
        tracks.push_back(track);
        st.split = true;
        st.arrays.clear();
        st.leaf_names.clear();
        for (size_t k = 0; k < n; k++) {
            if (track->arrays[k]) {
                st.arrays.push_back(track->arrays[k]);
                st.leaf_names.push_back(*name_of(track->arrays[k]));
            }
        }
        states[fname].dead_names.insert((*oi)->name);
        enqueue(fname, (*oi)->name, std::move(split));
    }

    void drain() {
        while (!work.empty()) {
            const string fname = work.back();
            work.pop_back();
            FunctionState &st = states[fname];
            while (!st.pending.empty()) {
                const string vname = st.pending.back();
                st.pending.pop_back();
                rewrite_uses(fname, vname);
            }
        }
    }

    // Every use `fname` makes of the value named `vname`, rewritten to its
    // parts.
    void rewrite_uses(const string &fname, const string &vname) {
        Function &f = function(fname);
        const Split split = states[fname].splits.at(vname);
        refresh_preds(f);
        const vector<shared_ptr<Block>> blocks = f.blocks;
        for (const shared_ptr<Block> &block : blocks) {
            // Each rewrite leaves the instruction no longer referring to the
            // name (replaced, or its operand retargeted), so a scan from the
            // top after each one ends.
            for (bool again = true; again;) {
                again = false;
                for (size_t i = 0; i < block->instrs.size(); i++) {
                    const shared_ptr<Instruction> in = block->instrs[i];
                    const bool uses = std::any_of(in->operands.begin(), in->operands.end(),
                                                  [&](const shared_ptr<Value> &v) {
                                                      return refers(v, vname);
                                                  });
                    if (!uses) {
                        continue;
                    }
                    rewrite_instruction(f, fname, block, i, in, split, vname);
                    again = true;
                    break;
                }
            }
            rewrite_terminator(f, fname, block, split, vname);
        }
        states[fname].done.insert(vname);
    }

    static string describe(const string &fname, const Block &block, const Instruction &in) {
        return fname + ", block " + block.name + ": " + op_name(in.op) +
               (in.name.empty() ? string() : " " + in.name);
    }

    // The leaves of `split` that lie under `path`: a contiguous run.
    std::pair<size_t, size_t> range_under(const Split &split, const vector<unsigned> &path) {
        size_t a = split.last, b = split.first;
        for (size_t k = split.first; k < split.last; k++) {
            if (covers(path, split.track->leaves[k].path)) {
                a = std::min(a, k);
                b = std::max(b, k + 1);
            }
        }
        return {a, b};
    }

    // `in` at `i` of `block` uses the value of `split` named `vname`; hands
    // back the index to go on from.
    size_t rewrite_instruction(Function &f, const string &fname, const shared_ptr<Block> &block,
                               size_t i, const shared_ptr<Instruction> &in, const Split &split,
                               const string &vname) {
        Tracking &track = *split.track;
        const vector<Leaf> &leaves = track.leaves;
        FunctionState &st = states[fname];
        const auto part_at = [&](size_t k) { return reach(block, split.parts[k]); };
        switch (in->op) {
        case Op::GEP: {
            internal_assert(refers(in->operands[0], vname) && in->operands.size() == 2)
                << track.handle << ": " << describe(fname, *block, *in)
                << " uses the address as an index";
            if (split.array) {
                // The address of one slot, per leaf.
                Split element = split;
                element.array = false;
                element.parts.assign(leaves.size(), nullptr);
                size_t at = i;
                const bool slot = f.record_slots.contains(in.get());
                for (size_t k = split.first; k < split.last; k++) {
                    if (!split.parts[k]) {
                        continue;
                    }
                    // Of the shape of the slot's address: a pointer, or a
                    // gang drain's vector of pointers (lanes_of_address).
                    element.parts[k] = emit_at(f, block, at,
                                               address_like(in->type, leaves[k].type), Op::GEP,
                                               {part_at(k), in->operands[1]});
                    const auto &made = std::get<shared_ptr<Instruction>>(element.parts[k]->data);
                    track.emitted[k].push_back(made);
                    if (slot) {
                        f.record_slots.insert(made.get());
                    }
                }
                if (slot) {
                    f.record_slots.erase(in.get());
                }
                enqueue(fname, in->name, std::move(element));
                erase_at(block, at);
                return at;
            }
            // Into a vector kept whole: the component's address off the leaf's.
            internal_assert(split.one_leaf_exactly())
                << track.handle << ": " << describe(fname, *block, *in)
                << " indexes into a " << split.type << " that is stored as "
                << (split.last - split.first) << " arrays";
            in->operands[0] = part_at(split.first);
            track.read[split.first] = true; // what goes through it is not followed
            return i + 1;
        }
        case Op::FieldPtr: {
            internal_assert(refers(in->operands[0], vname) && in->operands.size() == 2)
                << track.handle << ": " << describe(fname, *block, *in);
            const optional<uint64_t> k = constant_index(in->operands[1]);
            const auto *s = split.type.as<Struct_t>();
            internal_assert(k.has_value() && s != nullptr && *k < s->fields.size())
                << track.handle << ": " << describe(fname, *block, *in)
                << " takes a field of a " << split.type;
            Split field = split;
            field.array = false;
            field.type = s->fields[*k].type;
            field.path.push_back(unsigned(*k));
            std::tie(field.first, field.last) = range_under(split, field.path);
            internal_assert(field.first < field.last)
                << track.handle << ": " << describe(fname, *block, *in) << " names field "
                << s->fields[*k].name << ", which has no leaf";
            enqueue(fname, in->name, std::move(field));
            erase_at(block, i);
            return i;
        }
        case Op::Load:
        case Op::ExtractIdx: {
            const bool indexed = in->op == Op::ExtractIdx;
            internal_assert(refers(in->operands[0], vname))
                << track.handle << ": " << describe(fname, *block, *in);
            internal_assert(indexed == split.array)
                << track.handle << ": " << describe(fname, *block, *in)
                << (split.array ? " loads the whole storage" : " indexes a slot's address");
            const vector<shared_ptr<Value>> extra(in->operands.begin() + 1, in->operands.end());
            // A gang's read: through a vector of pointers, or of the storage
            // itself at a vector of indices, one per lane.
            uint32_t lanes = lanes_of_address(in->operands[0]->get_type());
            if (indexed && !extra.empty()) {
                if (const auto *v = extra[0]->get_type().as<Vector_t>()) {
                    lanes = v->lanes;
                }
            }
            materialize(f, fname, in, split, in->op, extra, lanes);
            return i;
        }
        case Op::Store: {
            internal_assert(refers(in->operands[0], vname) && !split.array &&
                            in->operands.size() >= 2)
                << track.handle << ": " << describe(fname, *block, *in)
                << (split.array ? " stores the whole storage" : " stores the address itself");
            for (size_t j = 1; j < in->operands.size(); j++) {
                internal_assert(!refers(in->operands[j], vname))
                    << track.handle << ": " << describe(fname, *block, *in)
                    << " stores the address itself";
            }
            vector<shared_ptr<Value>> extra(in->operands.begin() + 2, in->operands.end());
            size_t at = i;
            size_t l = split.first;
            const Emit emit = [&](const Type &t, Op op, vector<shared_ptr<Value>> ops) {
                return emit_at(f, block, at, t, op, std::move(ops));
            };
            vector<shared_ptr<Value>> parts;
            // A gang's store through a vector of pointers writes a gang-wide
            // value per leaf (take_apart knows the widened shapes).
            take_apart(in->operands[1], leaves_from(leaves, split.last), l,
                       lanes_of_address(in->operands[0]->get_type()), emit, parts);
            internal_assert(l == split.last && parts.size() == split.last - split.first)
                << track.handle << ": " << describe(fname, *block, *in) << " stores a "
                << split.type << " of " << parts.size() << " scalars where the storage has "
                << (split.last - split.first);
            for (size_t k = split.first; k < split.last; k++) {
                const shared_ptr<Value> &part = parts[k - split.first];
                if (!part || leaves[k].kind != Leaf::Kind::Stored) {
                    continue;
                }
                vector<shared_ptr<Value>> ops{part_at(k), part};
                ops.insert(ops.end(), extra.begin(), extra.end());
                auto made = effect_at(block, at, Op::Store, std::move(ops));
                made->atomic = in->atomic;
                made->compact = in->compact;
                made->slack = in->slack;
                track.emitted[k].push_back(made);
            }
            erase_at(block, at);
            return at;
        }
        case Op::AccAdd:
        case Op::AccSub:
        case Op::AccMul:
        case Op::AccMin:
        case Op::AccMax:
        case Op::AccArgmin:
        case Op::AccArgmax: {
            internal_assert(refers(in->operands[0], vname) && !split.array)
                << track.handle << ": " << describe(fname, *block, *in);
            if (split.one_leaf_exactly() &&
                equals(leaves[split.first].type, split.type)) {
                in->operands[0] = part_at(split.first);
                track.read[split.first] = true;
                track.emitted[split.first].push_back(in);
                return i + 1;
            }
            internal_assert((in->op == Op::AccAdd || in->op == Op::AccSub) && in->alongside == 0)
                << track.handle << ": " << describe(fname, *block, *in)
                << " accumulates into a " << split.type << " stored as "
                << (split.last - split.first) << " arrays, which only an add or a "
                << "subtract does leaf by leaf";
            vector<shared_ptr<Value>> extra(in->operands.begin() + 2, in->operands.end());
            size_t at = i;
            size_t l = split.first;
            const Emit emit = [&](const Type &t, Op op, vector<shared_ptr<Value>> ops) {
                return emit_at(f, block, at, t, op, std::move(ops));
            };
            vector<shared_ptr<Value>> parts;
            take_apart(in->operands[1], leaves_from(leaves, split.last), l,
                       lanes_of_address(in->operands[0]->get_type()), emit, parts);
            internal_assert(l == split.last) << track.handle;
            for (size_t k = split.first; k < split.last; k++) {
                const shared_ptr<Value> &part = parts[k - split.first];
                if (!part || leaves[k].kind != Leaf::Kind::Stored) {
                    continue;
                }
                vector<shared_ptr<Value>> ops{part_at(k), part};
                ops.insert(ops.end(), extra.begin(), extra.end());
                auto made = effect_at(block, at, in->op, std::move(ops));
                made->atomic = in->atomic;
                track.read[k] = true;
                track.emitted[k].push_back(made);
            }
            erase_at(block, at);
            return at;
        }
        default:
            internal_error << track.handle << ": its address is used by "
                           << describe(fname, *block, *in)
                           << ", which the struct-of-arrays rewrite cannot follow";
        }
        return i + 1;
    }

    // The value `in` reads out of the storage -- a slot's value of
    // `split.type`, or with `op` ExtractIdx an element of the storage itself
    // -- from its leaves. A struct the program only picks fields of
    // (`rec.hits`, a load of the whole and a LoadField) is not loaded whole:
    // each pick gets the leaves of its field alone, so that a leaf nothing
    // reads is loaded nowhere and the unread rule can drop it. Ends with
    // `in` gone: replaced by the value, or removed when nothing read it.
    // `lanes` is the gang's where the read is a gang's -- through a vector
    // of pointers, or at a vector of indices -- and the value read is then
    // the widened one, leaf by gang-wide leaf; one otherwise.
    void materialize(Function &f, const string &fname, const shared_ptr<Instruction> &in,
                     const Split &split, Op op, const vector<shared_ptr<Value>> &extra,
                     uint32_t lanes) {
        Tracking &track = *split.track;
        const vector<Leaf> &leaves = track.leaves;
        vector<shared_ptr<Instruction>> picks;
        bool only_picks = !split.array && split.type.is<Struct_t>();
        bool any_use = false;
        for (const auto &b : f.blocks) {
            for (const auto &instr : b->instrs) {
                const bool uses = std::any_of(instr->operands.begin(), instr->operands.end(),
                                              [&](const shared_ptr<Value> &v) {
                                                  return refers(v, in->name);
                                              });
                if (!uses) {
                    continue;
                }
                any_use = true;
                if (instr->op == Op::LoadField && instr->operands.size() == 2 &&
                    refers(instr->operands[0], in->name) &&
                    !refers(instr->operands[1], in->name) &&
                    constant_index(instr->operands[1]).has_value()) {
                    picks.push_back(instr);
                } else {
                    only_picks = false;
                }
            }
            if (used_by_terminator(*b, in->name)) {
                any_use = true;
                only_picks = false;
            }
        }
        if (!any_use) {
            erase_instruction(in);
            return;
        }
        if (only_picks) {
            const auto *s = split.type.as<Struct_t>();
            for (const shared_ptr<Instruction> &pick : picks) {
                const uint64_t k = *constant_index(pick->operands[1]);
                internal_assert(k < s->fields.size())
                    << track.handle << ": " << fname << " picks field " << k << " of a "
                    << split.type;
                Split field = split;
                field.type = s->fields[k].type;
                field.path.push_back(unsigned(k));
                std::tie(field.first, field.last) = range_under(split, field.path);
                internal_assert(field.first < field.last)
                    << track.handle << ": " << fname << " picks field " << s->fields[k].name
                    << ", which has no leaf";
                materialize(f, fname, pick, field, Op::Load, {}, lanes);
            }
            erase_instruction(in);
            return;
        }
        const shared_ptr<Block> block = in->owner.lock();
        internal_assert(block) << track.handle << ": a read outlived its block";
        size_t at = block->instrs.size();
        for (size_t j = 0; j < block->instrs.size(); j++) {
            if (block->instrs[j] == in) {
                at = j;
            }
        }
        internal_assert(at < block->instrs.size()) << track.handle;
        size_t l = split.first;
        const Emit emit = [&](const Type &t, Op o, vector<shared_ptr<Value>> ops) {
            return emit_at(f, block, at, t, o, std::move(ops));
        };
        // A gang reads a leaf as one gang-wide vector, and a value as the
        // struct of those (widen): the next leaf is kept whole where the
        // type asked for is the widened leaf.
        const auto stored_whole = [&](const Type &t) {
            return l < split.last &&
                   (keeps_whole(leaves[l], t) ||
                    (lanes > 1 && equals(held_by(leaves[l].type, lanes), t)));
        };
        const auto next_leaf = [&](const Type &t) -> shared_ptr<Value> {
            internal_assert(l < split.last)
                << track.handle << ": " << describe(fname, *block, *in) << " reads a "
                << split.type << " of more scalars than the storage has";
            const size_t k = l++;
            const Leaf &leaf = leaves[k];
            if (leaf.kind == Leaf::Kind::Pad) {
                return undef_value(t); // padding: read by nothing
            }
            internal_assert(leaf.kind == Leaf::Kind::Stored) << track.handle;
            track.read[k] = true;
            vector<shared_ptr<Value>> ops{reach(block, split.parts[k])};
            ops.insert(ops.end(), extra.begin(), extra.end());
            const Type held = held_by(leaf.type, lanes);
            auto v = emit(held, op, std::move(ops));
            if (lanes == 1 && packs(leaf, t)) {
                return emit(t, Op::Cast, {v});
            }
            internal_assert(equals(held, t))
                << track.handle << ": leaf " << leaf.name << " is a " << held
                << " where the program reads a " << t;
            return v;
        };
        shared_ptr<Value> value =
            rebuild(held_by(split.type, lanes), emit, next_leaf, stored_whole);
        internal_assert(l == split.last)
            << track.handle << ": " << describe(fname, *block, *in) << " reads a "
            << split.type << " of fewer scalars than the storage has";
        replace_uses(f, in.get(), value);
    }

    static bool used_by_terminator(const Block &block, const string &name) {
        bool found = false;
        std::visit(overloads{
                       [](const std::monostate &) {},
                       [&](const Terminator::Jump &j) {
                           for (const auto &v : j.args) {
                               found = found || refers(v, name);
                           }
                       },
                       [&](const Terminator::Dispatch &d) {
                           found = refers(d.cond, name);
                           for (const auto &t : d.targets) {
                               for (const auto &v : t.args) {
                                   found = found || refers(v, name);
                               }
                           }
                       },
                       [&](const Terminator::Return &r) { found = refers(r.value, name); },
                       [&](const Terminator::ParFor &p) {
                           found = refers(p.start, name) || refers(p.end, name) ||
                                   refers(p.stride, name);
                           for (const auto &v : p.body.args) {
                               found = found || refers(v, name);
                           }
                           for (const auto &v : p.cont.args) {
                               found = found || refers(v, name);
                           }
                       },
                       [&](const Terminator::Yield &) {},
                       [&](const Terminator::Call &c) {
                           for (const auto &v : c.call.args) {
                               found = found || refers(v, name);
                           }
                           for (const auto &v : c.cont.args) {
                               found = found || refers(v, name);
                           }
                       },
                       [&](const Terminator::Spawn &s) {
                           for (const auto &v : s.call.args) {
                               found = found || refers(v, name);
                           }
                           for (const auto &v : s.cont.args) {
                               found = found || refers(v, name);
                           }
                       },
                       [&](const Terminator::Join &j) {
                           for (const auto &v : j.next.args) {
                               found = found || refers(v, name);
                           }
                       },
                       [&](const Terminator::MultiCall &c) {
                           for (const auto &v : c.call.args) {
                               found = found || refers(v, name);
                           }
                           for (const auto &v : c.cont.args) {
                               found = found || refers(v, name);
                           }
                           for (const auto &vs : c.varying) {
                               for (const auto &v : vs) {
                                   found = found || refers(v, name);
                               }
                           }
                       },
                   },
                   block.terminator.data);
        return found;
    }

    // `leaves` cut off at `last`, so that take_apart() stops where this
    // split's run of leaves ends.
    static vector<Leaf> leaves_from(const vector<Leaf> &leaves, size_t last) {
        return vector<Leaf>(leaves.begin(), leaves.begin() + long(last));
    }

    void rewrite_terminator(Function &f, const string &fname, const shared_ptr<Block> &block,
                            const Split &split, const string &vname) {
        Terminator &t = block->terminator;
        if (auto *r = std::get_if<Terminator::Return>(&t.data)) {
            internal_assert(!refers(r->value, vname))
                << split.track->handle << ": " << fname << " returns its address";
            return;
        }
        if (auto *d = std::get_if<Terminator::Dispatch>(&t.data)) {
            internal_assert(!refers(d->cond, vname))
                << split.track->handle << ": " << fname << " dispatches on its address";
        }
        if (auto *mc = std::get_if<Terminator::MultiCall>(&t.data)) {
            for (const auto &vs : mc->varying) {
                for (const auto &v : vs) {
                    internal_assert(!refers(v, vname))
                        << split.track->handle << ": " << fname
                        << " hands its address as a varying argument of a run of calls";
                }
            }
        }
        if (Terminator::Jump *call = t.callee()) {
            // Handed to a callee: the parameter becomes one per leaf, at
            // every call of the callee.
            for (bool again = true; again;) {
                again = false;
                for (size_t j = 0; j < call->args.size(); j++) {
                    if (refers(call->args[j], vname)) {
                        split_parameter(call->name, j, split);
                        again = true;
                        break;
                    }
                }
            }
        }
        // Threaded to another block under another name: that name stands
        // for the same parts.
        for (auto &[jump, first_arg] : jumps(*block)) {
            if (jump == t.callee()) {
                continue;
            }
            for (size_t k = 0; k < jump->args.size(); k++) {
                if (!refers(jump->args[k], vname)) {
                    continue;
                }
                const shared_ptr<Block> target = block_of(f, jump->name);
                const size_t index = k + first_arg;
                internal_assert(index < target->args.size())
                    << split.track->handle << ": " << fname << ", " << block->name
                    << " hands " << target->name << " more arguments than it takes";
                const string &tname = target->args[index].name;
                FunctionState &st = states[fname];
                if (tname != vname && !st.splits.contains(tname)) {
                    enqueue(fname, tname, split);
                }
            }
        }
    }

    // The value `arg` hands a callee at `site`, as parts: a value already
    // split, a local of the caller's (split here), or a parameter of the
    // caller's (split as the callee's is).
    const Split &parts_for(const Site &site, const shared_ptr<Value> &arg, const Split &like) {
        FunctionState &st = states[site.func];
        const string *n = name_of(arg);
        internal_assert(n != nullptr)
            << like.track->handle << ": " << site.func << ", block " << site.block
            << " hands a callee a constant where an address is expected";
        if (const auto it = st.splits.find(*n); it != st.splits.end()) {
            return it->second;
        }
        if (const auto ps = param_splits.find(site.func); ps != param_splits.end()) {
            if (const auto it = ps->second.find(*n); it != ps->second.end()) {
                return it->second;
            }
        }
        Function &f = function(site.func);
        if (const auto *a = std::get_if<Argument>(&arg->data)) {
            // A parameter of the caller: an entry argument, threaded here.
            const Block &entry = *f.blocks.front();
            for (size_t j = 0; j < entry.args.size(); j++) {
                if (entry.args[j].name == a->name) {
                    return split_parameter(site.func, j, like);
                }
            }
        }
        if (const auto *i = std::get_if<shared_ptr<Instruction>>(&arg->data)) {
            const Instruction &in = **i;
            if (in.op == Op::Alloca || in.op == Op::Alloc) {
                return split_local(site.func, *i, like);
            }
            if (in.op == Op::GEP) {
                // A slot of a storage not yet split: split it first, which
                // rewrites this address too.
                const string *base = name_of(in.operands[0]);
                for (size_t s = 0; base != nullptr && s < f.induced.size(); s++) {
                    const InducedStorage &is = f.induced[s];
                    if (!is.split && is.arrays.size() == 1 && refers(is.arrays[0], *base)) {
                        split_storage(site.func, s);
                        rewrite_uses(site.func, *base);
                        if (const auto it = st.splits.find(*n); it != st.splits.end()) {
                            return it->second;
                        }
                    }
                }
            }
        }
        internal_error << like.track->handle << ": " << site.func << ", block " << site.block
                       << " hands a callee the address " << *n << ", which is not a slot "
                       << "of induced storage, a local of its own, or a parameter of its "
                       << "own, so the struct-of-arrays rewrite cannot follow it";
        return like;
    }

    // A `mut` local of `fname` of the split's type, as one local per leaf.
    const Split &split_local(const string &fname, const shared_ptr<Instruction> &local,
                             const Split &like) {
        Function &f = function(fname);
        const auto *ptr = local->type.as<Ptr_t>();
        internal_assert(ptr != nullptr && equals(ptr->etype, like.type) && !like.array)
            << like.track->handle << ": " << fname << " hands a callee its local "
            << local->name << " of type " << local->type << " where a " << like.type
            << " slot is expected";
        Split split = like;
        split.parts.assign(like.track->leaves.size(), nullptr);
        for (size_t k = like.first; k < like.last; k++) {
            if (!like.parts[k]) {
                continue;
            }
            const Leaf &leaf = like.track->leaves[k];
            split.parts[k] = allocate_beside(f, *local, leaf.type,
                                             local->name + tail_of(*like.track, leaf_local(like), k));
        }
        states[fname].dead_names.insert(local->name);
        enqueue(fname, local->name, split);
        return states[fname].splits.at(local->name);
    }

    // The stored thing's name the leaves of `split` begin with.
    static string leaf_local(const Split &split) {
        const Tracking &t = *split.track;
        const string &first = t.leaves[split.first].name;
        // Every leaf's name begins with the local's; the shortest common
        // prefix ending before an underscore is it.
        size_t cut = first.size();
        for (size_t k = split.first; k < split.last; k++) {
            const string &n = t.leaves[k].name;
            size_t c = 0;
            while (c < cut && c < n.size() && n[c] == first[c]) {
                c++;
            }
            cut = c;
        }
        const size_t us = first.rfind('_', cut);
        return us == string::npos ? first.substr(0, cut) : first.substr(0, us);
    }

    // Parameter `j` of `callee`, handed the address of a `split`, as one
    // parameter per leaf, at every call of the callee.
    const Split &split_parameter(const string &callee, size_t j, const Split &like) {
        Function &g = function(callee);
        Block &entry = *g.blocks.front();
        internal_assert(j < entry.args.size())
            << like.track->handle << ": " << callee << " is handed an address as argument "
            << j << " but takes " << entry.args.size();
        const Argument old = entry.args[j];
        if (const auto ps = param_splits.find(callee); ps != param_splits.end()) {
            if (const auto it = ps->second.find(old.name); it != ps->second.end()) {
                // Another storage hands the same parameter (two records of
                // one type, two arms' records): what the callee reads through
                // it, it reads of both, so the two share their reads (finish).
                if (it->second.track != like.track) {
                    internal_assert(it->second.first == like.first &&
                                    it->second.last == like.last)
                        << like.track->handle << " and " << it->second.track->handle
                        << " hand " << callee << " different parts of a storage as "
                        << old.name;
                    shared.push_back({it->second.track, like.track, like.first, like.last});
                }
                return it->second;
            }
        }
        // The parameter is the slot's address, in the shape the vectorizer
        // left it: a pointer, or in a gang's variant a vector of pointers,
        // one slot per lane (lanes_of_address); each leaf's is the same.
        const Type *pointee = address_pointee(old.type);
        const auto *arr = old.type.as<Array_t>();
        internal_assert((pointee != nullptr && !like.array && equals(*pointee, like.type)) ||
                        (arr != nullptr && like.array && equals(arr->etype, like.type)))
            << like.track->handle << ": " << callee << " takes " << old.name << " as a "
            << old.type << " where it is handed the address of a " << like.type;
        Split split = like;
        split.parts.assign(like.track->leaves.size(), nullptr);
        const string local = leaf_local(like);
        vector<Argument> made;
        for (size_t k = like.first; k < like.last; k++) {
            if (!like.parts[k]) {
                continue;
            }
            const Leaf &leaf = like.track->leaves[k];
            Argument a = old;
            a.name = fresh(g, old.name + tail_of(*like.track, local, k));
            a.type = like.array ? Array_t::make(leaf.type, arr->size, arr->readonly)
                                : address_like(old.type, leaf.type);
            made.push_back(a);
            like.track->params[k].push_back({callee, a.name});
        }
        // In place of the old parameter, so that a call's arguments and the
        // parameters stay in step whatever order the splits run in.
        entry.args.erase(entry.args.begin() + long(j));
        entry.lookups.erase(old.name);
        size_t at = j;
        size_t m = 0;
        for (size_t k = like.first; k < like.last; k++) {
            if (!like.parts[k]) {
                continue;
            }
            entry.args.insert(entry.args.begin() + long(at), made[m]);
            auto v = std::make_shared<Value>(made[m]);
            entry.lookups[made[m].name] = v;
            split.parts[k] = v;
            at++;
            m++;
        }
        param_splits[callee][old.name] = split;
        states[callee].dead_names.insert(old.name);
        enqueue(callee, old.name, split);
        // Every call: the argument replaced by its parts, found by the value
        // handed (another split may have moved its position meanwhile).
        for (const Site &site : callers[callee]) {
            Function &f = function(site.func);
            const shared_ptr<Block> block = block_of(f, site.block);
            Terminator::Jump *call = block->terminator.callee();
            internal_assert(call != nullptr && call->name == callee) << callee;
            internal_assert(j < call->args.size())
                << like.track->handle << ": " << site.func << ", block " << site.block
                << " calls " << callee << " with fewer arguments than it takes";
            const shared_ptr<Value> handed = call->args[j];
            const Split &parts = parts_for(site, handed, like);
            internal_assert(parts.first == like.first && parts.last == like.last)
                << like.track->handle << ": " << site.func << ", block " << site.block
                << " hands " << callee << " a different part of the storage than its "
                << "other callers do";
            refresh_preds(f);
            size_t pos = call->args.size();
            for (size_t q = 0; q < call->args.size(); q++) {
                if (call->args[q] == handed) {
                    pos = q;
                }
            }
            internal_assert(pos < call->args.size()) << callee;
            call->args.erase(call->args.begin() + long(pos));
            for (size_t k = like.first; k < like.last; k++) {
                if (!like.parts[k]) {
                    continue;
                }
                internal_assert(parts.parts[k] != nullptr)
                    << like.track->handle << ": " << site.func << " hands " << callee
                    << " a storage without leaf " << like.track->leaves[k].name;
                call->args.insert(call->args.begin() + long(pos), reach(block, parts.parts[k]));
                pos++;
            }
        }
        return param_splits[callee][old.name];
    }

    // Removes the block arguments of `f` that thread `name`, and the
    // operands of the jumps that supply them; the entry's parameter is the
    // callers' affair (split_parameter, drop_parameter).
    void erase_threading(Function &f, const string &name) {
        refresh_preds(f);
        for (size_t b = 1; b < f.blocks.size(); b++) {
            const shared_ptr<Block> &block = f.blocks[b];
            for (size_t j = block->args.size(); j-- > 0;) {
                if (block->args[j].name != name) {
                    continue;
                }
                block->args.erase(block->args.begin() + long(j));
                for (const auto &weak : block->preds) {
                    const shared_ptr<Block> pred = weak.lock();
                    if (!pred) {
                        continue;
                    }
                    for (auto &[jump, first_arg] : jumps(*pred)) {
                        if (jump->name != block->name || j < first_arg) {
                            continue;
                        }
                        const size_t k = j - first_arg;
                        internal_assert(k < jump->args.size())
                            << "the jump from " << pred->name << " to " << block->name
                            << " is missing an argument for " << name;
                        jump->args.erase(jump->args.begin() + long(k));
                    }
                }
            }
            block->lookups.erase(name);
        }
    }

    // Removes parameter `name` of `callee` and the argument every call
    // hands for it.
    void drop_parameter(const string &callee, const string &name) {
        Function &g = function(callee);
        Block &entry = *g.blocks.front();
        size_t j = entry.args.size();
        for (size_t q = 0; q < entry.args.size(); q++) {
            if (entry.args[q].name == name) {
                j = q;
            }
        }
        if (j == entry.args.size()) {
            return;
        }
        entry.args.erase(entry.args.begin() + long(j));
        entry.lookups.erase(name);
        for (const Site &site : callers[callee]) {
            Function &f = function(site.func);
            const shared_ptr<Block> block = block_of(f, site.block);
            Terminator::Jump *call = block->terminator.callee();
            internal_assert(call != nullptr && j < call->args.size()) << callee;
            call->args.erase(call->args.begin() + long(j));
        }
        states[callee].dead_names.insert(name);
    }

    // Does anything in `f` hold the value named `name`, other than a block
    // argument threading it?
    static bool used(const Function &f, const string &name) {
        for (const auto &block : f.blocks) {
            for (const auto &in : block->instrs) {
                for (const auto &v : in->operands) {
                    if (refers(v, name)) {
                        return true;
                    }
                }
            }
            bool found = false;
            std::visit(overloads{
                           [](const std::monostate &) {},
                           [&](const Terminator::Jump &) {},
                           [&](const Terminator::Dispatch &d) { found = refers(d.cond, name); },
                           [&](const Terminator::Return &r) { found = refers(r.value, name); },
                           [&](const Terminator::ParFor &p) {
                               found = refers(p.start, name) || refers(p.end, name) ||
                                       refers(p.stride, name);
                           },
                           [&](const Terminator::Yield &) {},
                           [&](const Terminator::Call &c) {
                               for (const auto &v : c.call.args) {
                                   found = found || refers(v, name);
                               }
                           },
                           [&](const Terminator::Spawn &s) {
                               for (const auto &v : s.call.args) {
                                   found = found || refers(v, name);
                               }
                           },
                           [&](const Terminator::Join &) {},
                           [&](const Terminator::MultiCall &c) {
                               for (const auto &v : c.call.args) {
                                   found = found || refers(v, name);
                               }
                           },
                       },
                       block->terminator.data);
            if (found) {
                return true;
            }
        }
        return false;
    }

    void finish() {
        // A leaf read through a parameter two storages hand is read of both.
        for (bool again = true; again;) {
            again = false;
            for (const Shared &s : shared) {
                for (size_t k = s.first; k < s.last; k++) {
                    const bool read = s.a->read[k] || s.b->read[k];
                    again = again || read != s.a->read[k] || read != s.b->read[k];
                    s.a->read[k] = read;
                    s.b->read[k] = read;
                }
            }
        }
        // The unread rule: a leaf nothing loads is not stored and has no
        // array; the parameters made for it go too.
        for (const shared_ptr<Tracking> &track : tracks) {
            Function &owner = function(track->owner);
            InducedStorage *storage = nullptr;
            for (InducedStorage &s : owner.induced) {
                if (s.handle == track->handle) {
                    storage = &s;
                }
            }
            for (size_t k = 0; k < track->leaves.size(); k++) {
                if (!track->arrays[k] || track->read[k]) {
                    continue;
                }
                for (const shared_ptr<Instruction> &in : track->emitted[k]) {
                    erase_instruction(in);
                }
                for (const auto &[g, pname] : track->params[k]) {
                    drop_parameter(g, pname);
                }
                const string &aname = *name_of(track->arrays[k]);
                erase_instruction(std::get<shared_ptr<Instruction>>(track->arrays[k]->data));
                states[track->owner].dead_names.insert(aname);
                if (storage != nullptr) {
                    for (size_t a = 0; a < storage->arrays.size(); a++) {
                        if (storage->arrays[a] == track->arrays[k]) {
                            storage->arrays.erase(storage->arrays.begin() + long(a));
                            storage->leaf_names.erase(storage->leaf_names.begin() + long(a));
                            break;
                        }
                    }
                }
            }
        }
        // The storage that was split is gone; so is every name the rewrite
        // replaced, and the threading that carried them.
        for (auto &[fname, st] : states) {
            Function &f = function(fname);
            for (const string &name : st.dead_names) {
                for (const auto &block : f.blocks) {
                    for (size_t i = 0; i < block->instrs.size();) {
                        const auto &in = block->instrs[i];
                        if (in->name == name &&
                            (in->op == Op::Alloca || in->op == Op::Alloc)) {
                            internal_assert(!used(f, name))
                                << name << " is still used in " << fname
                                << " after its storage was split";
                            erase_at(block, i);
                            continue;
                        }
                        i++;
                    }
                }
            }
            set<string> names = st.done;
            names.insert(st.dead_names.begin(), st.dead_names.end());
            for (const string &name : names) {
                if (!used(f, name)) {
                    erase_threading(f, name);
                }
            }
            // What built a value only to be taken apart -- the struct an
            // initializer made, read part by part by the stores -- is dead now.
            remove_dead(f);
        }
    }
};

} // namespace

void register_induced(Function &func, InducedStorage storage) {
    for (const InducedStorage &s : func.induced) {
        internal_assert(s.handle != storage.handle)
            << func.blocks.front()->name << " already has induced storage named "
            << storage.handle;
    }
    func.induced.push_back(std::move(storage));
}

void dump_induced(std::ostream &os, const Function &func) {
    for (const InducedStorage &s : func.induced) {
        os << "; storage " << s.handle << " : " << s.element << " as";
        if (s.arrays.empty()) {
            os << " nothing (unread)";
        }
        for (size_t k = 0; k < s.arrays.size(); k++) {
            const string *n = name_of(s.arrays[k]);
            os << (k == 0 ? " " : ", ") << (n != nullptr ? *n : string("?"));
        }
        os << "\n";
    }
}

void split_induced_storage(FuncMap &funcs) {
    Splitter(funcs).run();
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
