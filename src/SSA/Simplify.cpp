#include "SSA/Simplify.h"

#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "SSA/ConstantIntervals.h"
#include "SSA/Definitions.h"
#include "SSA/Storage.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::vector;
using ValuePtr = shared_ptr<Value>;

const Instruction *def_of(const ValuePtr &v) {
    if (v == nullptr) {
        return nullptr;
    }
    const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
    return held == nullptr ? nullptr : held->get();
}

const Constant *constant_of(const ValuePtr &v) {
    return v == nullptr ? nullptr : std::get_if<Constant>(&v->data);
}

std::optional<bool> const_bool(const ValuePtr &v) {
    const Constant *c = constant_of(v);
    if (c == nullptr || !c->type.is<Bool_t>()) {
        return std::nullopt;
    }
    return std::visit(
        overloads{
            [](bool b) -> std::optional<bool> { return b; },
            [](int64_t i) -> std::optional<bool> { return i != 0; },
            [](uint64_t u) -> std::optional<bool> { return u != 0; },
            [](double d) -> std::optional<bool> { return d != 0.0; },
            [](const std::string &) -> std::optional<bool> {
                return std::nullopt;
            },
            [](const Undefined &) -> std::optional<bool> {
                return std::nullopt;
            },
        },
        c->data);
}

ValuePtr make_bool(bool b) {
    return std::make_shared<Value>(Constant{Bool_t::make(), b});
}

// Are `a` and `b` one value: the same instruction, the same argument, or
// equal constants?
bool same_value(const ValuePtr &a, const ValuePtr &b) {
    if (a == b) {
        return true;
    }
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &x) {
                const auto *y = std::get_if<shared_ptr<Instruction>>(&b->data);
                return y != nullptr && y->get() == x.get();
            },
            [&](const Argument &x) {
                const auto *y = std::get_if<Argument>(&b->data);
                return y != nullptr && y->name == x.name;
            },
            [&](const Constant &x) {
                const auto *y = std::get_if<Constant>(&b->data);
                return y != nullptr && equals(x.type, y->type) &&
                       x.data == y->data;
            },
        },
        a->data);
}

// Is `in` the operation `op` of `type` over `operands`?
bool same_expression(const Instruction *in, Instruction::Op op,
                     const Type &type, const vector<ValuePtr> &operands) {
    if (in == nullptr || in->op != op || !equals(in->type, type) ||
        in->operands.size() != operands.size()) {
        return false;
    }
    for (size_t i = 0; i < operands.size(); i++) {
        if (!same_value(in->operands[i], operands[i])) {
            return false;
        }
    }
    return true;
}

// A comparison of two constants, or nothing when they are not both constants
// of one kind.
std::optional<bool> compare_constants(Instruction::Op op, const ValuePtr &a,
                                      const ValuePtr &b) {
    const Constant *x = constant_of(a);
    const Constant *y = constant_of(b);
    if (x == nullptr || y == nullptr || x->data.index() != y->data.index() ||
        std::holds_alternative<Undefined>(x->data)) {
        return std::nullopt; // an undefined value compares as nothing
    }
    const auto compare = [&](const auto &p, const auto &q) -> bool {
        switch (op) {
        case Instruction::Op::Lt:
            return p < q;
        case Instruction::Op::Leq:
            return p <= q;
        case Instruction::Op::Eq:
            return p == q;
        case Instruction::Op::Ne:
            return p != q;
        default:
            internal_error << "Not a comparison: " << int(op);
            return false;
        }
    };
    return std::visit(
        overloads{
            [&](bool p) { return compare(p, std::get<bool>(y->data)); },
            [&](int64_t p) { return compare(p, std::get<int64_t>(y->data)); },
            [&](uint64_t p) {
                return compare(p, std::get<uint64_t>(y->data));
            },
            [&](double p) { return compare(p, std::get<double>(y->data)); },
            [&](const std::string &p) {
                return compare(p, std::get<std::string>(y->data));
            },
            [&](const Undefined &) { return false; }, // excluded above
        },
        x->data);
}

} // namespace

bool pure(const Instruction &in) {
    if (in.name.empty()) {
        return false;
    }
    switch (in.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Any:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::GEP:
    case Instruction::Op::Inf:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Load:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Popcount:
    case Instruction::Op::Ramp:
    case Instruction::Op::Reduce:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Shuffle:
    case Instruction::Op::SizeOf:
    case Instruction::Op::Sub:
    case Instruction::Op::Vote:
    case Instruction::Op::Xor:
        return true;
    case Instruction::Op::Intrinsic:
        return !ir::Intrinsic::has_effects(in.intrinsic);
    default:
        return false;
    }
}

bool reads_memory(const Instruction &in) {
    switch (in.op) {
    case Instruction::Op::Load:
        return true;
    case Instruction::Op::ExtractIdx: {
        if (in.operands.empty()) {
            return false;
        }
        const Type &container = in.operands[0]->get_type();
        return container.is_reference() || is_dynamic_array_struct_type(container);
    }
    default:
        return false;
    }
}

namespace {

using Replacements = std::map<const Instruction *, ValuePtr>;

// How many times each instruction's value is read (defined below).
using UseCounts = std::map<const Instruction *, size_t>;
UseCounts use_counts(const Function &func);

struct Simplifier {
    explicit Simplifier(Function &func, const ConstantIntervals *intervals)
        : func(func), intervals(intervals) {}

    Function &func;
    // What every value is known to lie between (SSA/ConstantIntervals.h),
    // when the caller computed it, for the rules that are exact only under
    // such a fact; null when none was computed, and those rules stand down.
    const ConstantIntervals *intervals = nullptr;

    // What `v` lies between, as the current block reads it; everything
    // without the analysis.
    ConstantInterval bounds(const ValuePtr &v) const {
        if (intervals == nullptr || block == nullptr || v == nullptr) {
            return ConstantInterval::everything();
        }
        return intervals->of(func, *block, v);
    }
    // A non-negative number: its bits as a signed integer lie in
    // [0, 0x7f800000] and order as the float does.
    bool nonnegative_number(const ValuePtr &v) const {
        const ConstantInterval i = bounds(v);
        return i.number && i.nonnegative;
    }
    // Its sign bit clear in every value, a NaN's included: its bits as a
    // signed integer are non-negative.
    bool sign_clear(const ValuePtr &v) const { return bounds(v).sign_clear; }

    // Common subexpressions: an operation over the same operands computed
    // again where the first is in scope -- its block dominating -- is the
    // first. LLVM's EarlyCSE over the dominator tree: a table per block,
    // consulted along with its dominators'. The blocks are walked as the
    // function stores them, so a block visited before a dominator of it
    // gains nothing from that dominator's table and loses nothing. A pure
    // value computation that reads no memory is reusable; a load is not
    // (what is stored between two of them is another question), nor an
    // instruction with storage riding on it, nor a Set, which names.
    std::optional<Cfg> cfg;
    std::optional<DomTree> dom;
    vector<std::unordered_map<std::string, ValuePtr>> tables;
    BlockId at_block = NO_BLOCK;

    void enter(const shared_ptr<Block> &b) {
        block = b;
        generation = 0;
        if (!cfg.has_value()) {
            cfg.emplace(func);
            dom = compute_dominator_tree(*cfg);
            tables.assign(cfg->size(), {});
        }
        at_block = cfg->find(b->name);
    }

    // An instruction that is its operation, type and operands and nothing
    // more (rebuildable), or one whose one extra -- an intrinsic's name, a
    // reduction's kind, a shuffle's order -- the key carries (extra_of);
    // and the pure reads and address computations, which their operands
    // describe. Not a sizeof, whose measured type rides on it alone, nor an
    // allocation, nor a Set, which names.
    static bool reusable(const Instruction &in) {
        if (in.name.empty() || in.storage.has_value() || !pure(in)) {
            return false;
        }
        if (rebuildable(in)) {
            return true;
        }
        switch (in.op) {
        case Instruction::Op::Intrinsic:
        case Instruction::Op::Reduce:
        case Instruction::Op::Shuffle:
        case Instruction::Op::Div:
        case Instruction::Op::Mod:
        case Instruction::Op::Ramp:
        case Instruction::Op::Popcount:
        case Instruction::Op::Any:
        case Instruction::Op::GEP:
        case Instruction::Op::FieldPtr:
        case Instruction::Op::Load:
        case Instruction::Op::ExtractIdx:
        case Instruction::Op::Inf:
        case Instruction::Op::Eps:
            return true;
        default:
            return false;
        }
    }

    // A read of memory -- a load, a lane of a stored array -- is the same
    // value as an earlier read of the same place only while nothing has
    // been written in between. Within a block that is counted: the
    // generation steps at every store, accumulate, atomic and effect, a
    // read is remembered under the generation it was made in, and only its
    // own block's reads are consulted (the generation says nothing across
    // blocks). EarlyCSE's generation, over one block. The children's test
    // reads the node's row once for its own slab test and again for the
    // `distmin` inlined beside it, through the same load of the current
    // node, with nothing stored between.
    size_t generation = 0;
    // A read of memory that a store might change: not one of read-only
    // storage (an extern's layout, whose types say so, see SSA/Storage.h),
    // which is the same value wherever it is read and merges like any pure
    // computation.
    static bool reads_mutable_memory(const Instruction &in) {
        if (!reads_memory(in) || in.operands.empty()) {
            return false;
        }
        return !in.operands[0]->get_type().is_readonly();
    }
    static bool writes_memory(const Instruction &in) {
        if (in.name.empty()) {
            return true; // a store, an accumulate, a prefetch, a print: an effect
        }
        switch (in.op) {
        case Instruction::Op::AtomicAdd:
            return true;
        default:
            return false;
        }
    }

    // The text that says what an instruction computes: its operation, its
    // type, what rides on it, and its operands as resolved -- an
    // instruction by its name, which is one per function; an argument by
    // its name, which a value keeps as it is carried between blocks; a
    // constant as written with its type.
    std::string key_of(Instruction::Op op, const Type &type,
                       const vector<ValuePtr> &ops, const std::string &extra) const {
        std::ostringstream os;
        os << int(op) << '|' << type << '|' << extra << '|';
        for (const ValuePtr &o : ops) {
            if (o == nullptr) {
                os << "null,";
                continue;
            }
            resolve(o)->dump(os);
            os << ',';
        }
        return os.str();
    }
    std::string extra_of(const Instruction &in) const {
        std::ostringstream os;
        switch (in.op) {
        case Instruction::Op::Intrinsic:
            os << "i" << int(in.intrinsic);
            break;
        case Instruction::Op::Reduce:
            os << "r" << int(in.reduce);
            break;
        case Instruction::Op::Shuffle:
            os << "s";
            for (int k : in.shuffle) {
                os << k << '.';
            }
            break;
        default:
            break;
        }
        if (reads_mutable_memory(in)) {
            os << "g" << generation;
        }
        return os.str();
    }
    // Up the dominators, but not past a call or a parfor: a block reached
    // through a call's continuation or a loop's body does not reuse the
    // values of the block that made the call or ran the loop, nor of
    // anything above it. A deferral (SSA/Defer.h) cuts the graph exactly
    // there -- the continuation of a deferred call runs from a queue's
    // drain with the values its entry captured, and a value of the calling
    // block or of the enclosing body is not among them, where one the
    // continuation computes for itself from a captured value is -- so a
    // merge across that edge would hand the drain a value it cannot have.
    // (LLVM merges across these later, where no such cut remains.)
    ValuePtr find_equal(const std::string &key, bool this_block_only = false) {
        if (at_block == NO_BLOCK || !dom.has_value()) {
            return nullptr;
        }
        for (BlockId b = at_block; b != NO_BLOCK;) {
            if (const auto it = tables[b].find(key); it != tables[b].end()) {
                return resolve(it->second);
            }
            if (this_block_only || b == dom->root || b >= dom->idom.size()) {
                break;
            }
            const BlockId above = dom->idom[b];
            if (above == NO_BLOCK) {
                break;
            }
            const Terminator &t = cfg->block(above)->terminator;
            if (std::holds_alternative<Terminator::Call>(t.data) ||
                std::holds_alternative<Terminator::MultiCall>(t.data) ||
                std::holds_alternative<Terminator::ParFor>(t.data)) {
                break;
            }
            b = above;
        }
        return nullptr;
    }
    ValuePtr find_equal(const Instruction &in) {
        if (!reusable(in)) {
            return nullptr;
        }
        return find_equal(key_of(in.op, in.type, in.operands, extra_of(in)),
                          reads_mutable_memory(in));
    }
    void remember(const shared_ptr<Instruction> &in, const ValuePtr &v) {
        if (at_block == NO_BLOCK || !reusable(*in)) {
            return;
        }
        tables[at_block][key_of(in->op, in->type, in->operands, extra_of(*in))] = v;
        // A value another block may now refer to by name has to be found by
        // it there (Block::get_value walks the predecessors' lookups); one
        // only ever read in its own block need not have been.
        block->lookups.emplace(in->name, v);
    }
    // The use counts as the pass began, for a rule that asks whether a value
    // is read anywhere else. Built on first use; a rule's rewrites only take
    // uses away from what they replace.
    std::optional<UseCounts> uses;
    size_t uses_of(const Instruction *d) {
        if (!uses.has_value()) {
            uses = use_counts(func);
        }
        const auto it = uses->find(d);
        return it == uses->end() ? 0 : it->second;
    }
    // Where a rule's new instructions go: before instruction `at` of `block`.
    shared_ptr<Block> block;
    size_t at = 0;
    Replacements replaced;
    // The instructions the rules made, so that a merge of two equal
    // expressions keeps the one the program had.
    std::set<const Instruction *> made;
    // Where a value a block refers to by name is defined, for a rule that
    // needs the defining instruction of an argument (SSA/Definitions.h).
    // Built on first use: the walk changes operands, not the jumps and
    // arguments the answers rest on.
    std::optional<Definitions> definitions;

    // The instruction `v` is defined by, followed through the block
    // arguments it may have arrived as; null for a constant, a parameter, or
    // a merge of different values.
    const Instruction *defined_by(const ValuePtr &v) {
        if (const Instruction *d = def_of(v)) {
            return d;
        }
        if (v == nullptr || !std::holds_alternative<Argument>(v->data) || !block) {
            return nullptr;
        }
        if (!definitions.has_value()) {
            definitions.emplace(func, /*lenient=*/true);
        }
        return def_of(definitions->of(block->name, v).value);
    }

    // Field `index` of `v` when `v` is a merge of structs every one of which
    // was built with the same constant in that field: the tag of a variant
    // that every incoming arm built as that variant -- a BSDF and its
    // regularized copy, say -- which no one instruction defines. Followed
    // back through the block arguments to the structs, an edge carried round
    // a loop unchanged being skipped; null when any edge brings something
    // else.
    ValuePtr constant_field_of_merge(const ValuePtr &v, uint64_t index) {
        const auto *arg = v ? std::get_if<Argument>(&v->data) : nullptr;
        if (arg == nullptr || !block) {
            return nullptr;
        }
        const BlockMap bmap = make_block_map(func);
        ValuePtr found;
        std::set<std::pair<std::string, std::string>> visiting;
        const std::function<bool(const shared_ptr<Block> &, const Argument &)> walk =
            [&](const shared_ptr<Block> &at, const Argument &a) -> bool {
            if (!visiting.insert({at->name, a.name}).second) {
                return true; // round a loop unchanged: what enters decides
            }
            if (at.get() == func.blocks.front().get()) {
                return false; // a parameter: anything
            }
            size_t k = at->args.size();
            for (size_t i = 0; i < at->args.size(); i++) {
                if (at->args[i].name == a.name) {
                    k = i;
                }
            }
            if (k == at->args.size() || at->preds.empty()) {
                return false;
            }
            for (const auto &weak : at->preds) {
                const shared_ptr<Block> pred = weak.lock();
                if (!pred) {
                    return false;
                }
                const ValuePtr passed = passed_to(*pred, *at, k);
                if (!passed) {
                    return false; // the edge defines it
                }
                if (const Instruction *d = def_of(passed)) {
                    if (d->op != Instruction::Op::MakeStruct ||
                        index >= d->operands.size() ||
                        constant_of(d->operands[index]) == nullptr) {
                        return false;
                    }
                    if (found && !same_value(*found, *d->operands[index])) {
                        return false;
                    }
                    found = d->operands[index];
                    continue;
                }
                const auto *pa = std::get_if<Argument>(&passed->data);
                if (pa == nullptr || !walk(pred, *pa)) {
                    return false;
                }
            }
            return true;
        };
        const auto it = bmap.find(block->name);
        if (it == bmap.end() || !walk(it->second, *arg)) {
            return nullptr;
        }
        return found;
    }

    // What `v` stands for once every replacement is followed.
    ValuePtr resolve(ValuePtr v) const {
        for (auto it = replaced.find(def_of(v)); it != replaced.end();
             it = replaced.find(def_of(v))) {
            v = it->second;
        }
        return v;
    }

    // An operation over `operands`, simplified: an existing value when a
    // rule gives one, and a new instruction before the current one otherwise.
    ValuePtr make(Instruction::Op op, Type type, vector<ValuePtr> operands) {
        if (ValuePtr v = rule(op, type, operands)) {
            return v;
        }
        // An instruction like it already in scope is it (see find_equal).
        if (ValuePtr same = find_equal(key_of(op, type, operands, ""))) {
            return same;
        }
        return insert(std::make_shared<Instruction>(
            func.get_unique_name(), std::move(type), op, std::move(operands),
            block));
    }

    // An intrinsic over `operands`, as a new instruction before the current
    // one.
    ValuePtr make_intrinsic(ir::Intrinsic::OpType which, Type type,
                            vector<ValuePtr> operands) {
        auto instr = std::make_shared<Instruction>(
            func.get_unique_name(), std::move(type), Instruction::Op::Intrinsic,
            std::move(operands), block);
        instr->intrinsic = which;
        if (ValuePtr same = find_equal(*instr)) {
            return same;
        }
        return insert(std::move(instr));
    }

    ValuePtr insert(shared_ptr<Instruction> instr) {
        block->instrs.insert(block->instrs.begin() + at, instr);
        at++;
        made.insert(instr.get());
        auto v = std::make_shared<Value>(instr);
        block->lookups[instr->name] = v;
        remember(instr, v);
        return v;
    }

    // The address of the struct value `s` in read-only storage, when that is
    // where it was read from -- an element of an extern's layout, a field of
    // one -- and null otherwise. The address is formed here, before the
    // current instruction, from the operands of the read.
    ValuePtr stored_struct_address(const ValuePtr &s) {
        const Instruction *d = def_of(s);
        if (d == nullptr || !d->type.is<Struct_t>()) {
            return nullptr;
        }
        switch (d->op) {
        case Instruction::Op::Load:
            if (d->operands.size() == 1 &&
                d->operands[0]->get_type().is<Ptr_t>() &&
                d->operands[0]->get_type().is_readonly()) {
                return d->operands[0];
            }
            return nullptr;
        case Instruction::Op::ExtractIdx:
            // One element of a read-only array, at one index.
            if (d->operands.size() == 2 &&
                d->operands[0]->get_type().is<Array_t>() &&
                d->operands[0]->get_type().is_readonly() &&
                d->operands[1]->get_type().is_scalar()) {
                return make(Instruction::Op::GEP,
                            Ptr_t::make(d->type, /*readonly=*/true),
                            {d->operands[0], d->operands[1]});
            }
            return nullptr;
        case Instruction::Op::LoadField:
            if (d->operands.size() == 2) {
                return stored_field_address(d);
            }
            return nullptr;
        default:
            return nullptr;
        }
    }

    // The address of the field a LoadField reads, when the struct it reads
    // it from is in read-only storage; null otherwise.
    ValuePtr stored_field_address(const Instruction *load_field) {
        ValuePtr owner = stored_struct_address(load_field->operands[0]);
        if (owner == nullptr) {
            return nullptr;
        }
        const Struct_t *s = owner->get_type().as<Ptr_t>()->etype.as<Struct_t>();
        const Constant *c = constant_of(load_field->operands[1]);
        if (s == nullptr || c == nullptr ||
            !std::holds_alternative<uint64_t>(c->data) ||
            std::get<uint64_t>(c->data) >= s->fields.size()) {
            return nullptr;
        }
        // Typed as the field is stored -- a layout's `[[packed]] f32x8` --
        // rather than as the load reads it.
        const Type &stored = s->fields[std::get<uint64_t>(c->data)].type;
        return make(Instruction::Op::FieldPtr,
                    Ptr_t::make(stored, /*readonly=*/true),
                    {owner, load_field->operands[1]});
    }

    // The address of the vector `v` in read-only storage, when it was loaded
    // whole from there; null otherwise. Looks through the cast that reads a
    // layout's packed vector as the ordinary one -- the same lanes at the
    // same places in memory.
    ValuePtr stored_vector_address(const ValuePtr &v) {
        const Instruction *d = def_of(v);
        if (d == nullptr || !d->type.is<Vector_t>()) {
            return nullptr;
        }
        switch (d->op) {
        case Instruction::Op::Load:
            if (d->operands.size() == 1 &&
                d->operands[0]->get_type().is<Ptr_t>() &&
                d->operands[0]->get_type().is_readonly()) {
                return d->operands[0];
            }
            return nullptr;
        case Instruction::Op::Cast: {
            if (d->operands.size() != 1) {
                return nullptr;
            }
            const Type &from = d->operands[0]->get_type();
            if (!from.is<Vector_t>() || from.lanes() != d->type.lanes() ||
                !equals(from.element_of(), d->type.element_of())) {
                return nullptr;
            }
            return stored_vector_address(d->operands[0]);
        }
        case Instruction::Op::ExtractIdx:
            // One element of a read-only array of vectors, at one index.
            if (d->operands.size() == 2 &&
                d->operands[0]->get_type().is<Array_t>() &&
                d->operands[0]->get_type().is_readonly() &&
                d->operands[1]->get_type().is_scalar()) {
                return make(Instruction::Op::GEP,
                            Ptr_t::make(d->type, /*readonly=*/true),
                            {d->operands[0], d->operands[1]});
            }
            return nullptr;
        case Instruction::Op::LoadField:
            if (d->operands.size() == 2) {
                return stored_field_address(d);
            }
            return nullptr;
        default:
            return nullptr;
        }
    }

    // The lane a mask with one bit on names, as an index of `index_type`:
    // the mask read as an integer and its trailing zeros counted (`kmov`,
    // `tzcnt`; Embree's `bsf` on its movemask). Only where the mask's lanes
    // are the bits of an integer type; a shorter mask has no integer to
    // count, and stays with the compress.
    ValuePtr lowest_lane(const ValuePtr &mask, const Type &index_type) {
        const Type &mt = mask->get_type();
        if (!mt.is<Vector_t>() || !mt.element_of().is<Bool_t>()) {
            return nullptr;
        }
        const uint32_t n = mt.lanes();
        if (n != 8 && n != 16 && n != 32 && n != 64) {
            return nullptr;
        }
        const Type bits_t = UInt_t::make(n);
        ValuePtr bits = make(Instruction::Op::Reinterpret, bits_t, {mask});
        ValuePtr zeros = make_intrinsic(ir::Intrinsic::ctz, bits_t, {bits});
        if (equals(bits_t, index_type)) {
            return zeros;
        }
        return make(Instruction::Op::Cast, index_type, {zeros});
    }

    // Of two operands that are the same expression, the one to keep: the
    // program's own over one a rule made.
    ValuePtr keep(const ValuePtr &a, const ValuePtr &b) const {
        return made.count(def_of(b)) > 0 ? a : b;
    }

    // Is `v` the integer constant zero, of any integer type?
    static bool is_zero(const ValuePtr &v) {
        const Constant *c = constant_of(v);
        if (c == nullptr || !(c->type.is_int() || c->type.is_uint())) {
            return false;
        }
        return std::visit(overloads{
                              [](int64_t i) { return i == 0; },
                              [](uint64_t u) { return u == 0; },
                              [](const auto &) { return false; },
                          },
                          c->data);
    }

    // The scalar a type's lanes hold, or the type itself when it has none.
    static Type scalar_of(const Type &t) {
        return t.is_vector() ? t.element_of() : t;
    }

    // Is `v` one bool in every lane -- a bool constant, or a broadcast of
    // one -- and which?
    static std::optional<bool> uniform_bool(const ValuePtr &v) {
        if (std::optional<bool> c = const_bool(v)) {
            return c;
        }
        const Instruction *d = def_of(v);
        if (d != nullptr && d->op == Instruction::Op::Bc &&
            !d->operands.empty()) {
            return const_bool(d->operands[0]);
        }
        return std::nullopt;
    }

    // The bool `b` as a value of type `t`: the constant, or its broadcast
    // over the lanes of a mask type.
    ValuePtr uniform_bool_constant(const Type &t, bool b) {
        if (!t.is_vector()) {
            return make_bool(b);
        }
        ValuePtr width = std::make_shared<Value>(
            Constant{UInt_t::make(32), uint64_t(t.lanes())});
        return make(Instruction::Op::Bc, t, {make_bool(b), std::move(width)});
    }

    // An instruction that is its operation, type and operands and nothing
    // more -- no intrinsic kind, reduction, shuffle or storage riding on it
    // -- so that one like it over other operands is what `make` makes.
    static bool rebuildable(const Instruction &in) {
        switch (in.op) {
        case Instruction::Op::Abs:
        case Instruction::Op::Add:
        case Instruction::Op::Bc:
        case Instruction::Op::BwAnd:
        case Instruction::Op::BwOr:
        case Instruction::Op::Cast:
        case Instruction::Op::Eq:
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr:
        case Instruction::Op::Leq:
        case Instruction::Op::LoadField:
        case Instruction::Op::Lt:
        case Instruction::Op::MakeStruct:
        case Instruction::Op::Max:
        case Instruction::Op::Min:
        case Instruction::Op::Mul:
        case Instruction::Op::Ne:
        case Instruction::Op::Not:
        case Instruction::Op::Reinterpret:
        case Instruction::Op::Select:
        case Instruction::Op::Shl:
        case Instruction::Op::Shr:
        case Instruction::Op::Sub:
        case Instruction::Op::Xor:
            return !in.storage.has_value();
        default:
            return false;
        }
    }

    // Lane `i` of the result depends on lane `i` of each vector operand and
    // on the scalar operands alone, no lane read from another: what a fact
    // about one lane of a mask may be carried through. A struct of vectors
    // keeps its lanes aligned field by field, so building one and reading
    // its fields are lanewise too; a mask reinterpreted as a word packs its
    // lanes and is not, nor is a reduction, a lane read at an index, a
    // shuffle or a compress.
    static bool lanewise(const Instruction &in) {
        switch (in.op) {
        case Instruction::Op::Abs:
        case Instruction::Op::Add:
        case Instruction::Op::Bc:
        case Instruction::Op::BwAnd:
        case Instruction::Op::BwOr:
        case Instruction::Op::Eq:
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr:
        case Instruction::Op::Leq:
        case Instruction::Op::LoadField:
        case Instruction::Op::Lt:
        case Instruction::Op::MakeStruct:
        case Instruction::Op::Max:
        case Instruction::Op::Min:
        case Instruction::Op::Mul:
        case Instruction::Op::Ne:
        case Instruction::Op::Not:
        case Instruction::Op::Select:
        case Instruction::Op::Shl:
        case Instruction::Op::Shr:
        case Instruction::Op::Sub:
        case Instruction::Op::Xor:
            return true;
        case Instruction::Op::Cast:
        case Instruction::Op::Reinterpret: {
            // Between vectors of one lane count (f32x8 as u32x8), or between
            // scalars; a mask as a word, or a word as a mask, moves lanes.
            if (in.operands.size() != 1) {
                return false;
            }
            const Type &from = in.operands[0]->get_type();
            if (in.type.is_vector() && from.is_vector()) {
                return in.type.lanes() == from.lanes();
            }
            return !in.type.is_vector() && !from.is_vector();
        }
        default:
            return false;
        }
    }

    // Set while `assume` rebuilds: the rules that call it stand down inside
    // it, since the walk substitutes everywhere it reaches already, and a
    // rule under a rule under a rule is what makes a rewrite exponential.
    size_t assuming = 0;

    // `v` with every occurrence of `fact` in it taken as `truth` -- the
    // occurrences reached through pure value computations alone, and for a
    // mask through lanewise ones, so that a lane's fact decides that lane's
    // result and no other's -- or `v` itself when it reaches none. What
    // `a & b` knows inside `b`: Halide's simplifier learns `a` true there
    // (Simplify_And.cpp, learn_true), and so does `select(c, t, f)` inside
    // `t`, and `c` false inside `f`. Linear in the part of `v` it walks,
    // which is bounded.
    ValuePtr assume(const ValuePtr &v, const ValuePtr &fact, bool truth) {
        const Type fact_t = fact->get_type();
        const bool vector_fact = fact_t.is_vector();
        std::map<const Instruction *, ValuePtr> memo;
        size_t budget = 256;
        ValuePtr constant;
        assuming++;
        const std::function<ValuePtr(const ValuePtr &)> walk =
            [&](const ValuePtr &x) -> ValuePtr {
            if (same_value(x, fact)) {
                if (!constant) {
                    constant = uniform_bool_constant(fact_t, truth);
                }
                return constant;
            }
            const Instruction *d = def_of(x);
            if (d == nullptr) {
                return x; // a parameter, a block's argument, a constant
            }
            if (const auto it = memo.find(d); it != memo.end()) {
                return it->second;
            }
            if (budget == 0 || !rebuildable(*d) || !pure(*d) ||
                reads_memory(*d) || (vector_fact && !lanewise(*d))) {
                return x;
            }
            budget--;
            vector<ValuePtr> operands;
            bool changed = false;
            for (const ValuePtr &o : d->operands) {
                ValuePtr n = walk(o);
                // By value, not by handle: one instruction is held through
                // many handles, and the memo hands back the first.
                changed = changed || !same_value(n, o);
                operands.push_back(std::move(n));
            }
            ValuePtr result =
                changed ? make(d->op, d->type, std::move(operands)) : x;
            memo[d] = result;
            return result;
        };
        ValuePtr result = walk(v);
        assuming--;
        return result;
    }

    // Two comparisons of one kind joined by `&` or `|` that share a side
    // are one comparison against a min or a max of the other sides:
    //
    //     (a <= b) & (a <= c)  =  a <= min(b, c)      (b <= a) & (c <= a)  =  max(b, c) <= a
    //     (a <= b) | (a <= c)  =  a <= max(b, c)      (b <= a) | (c <= a)  =  min(b, c) <= a
    //
    // and the same with `<`. Over integers, whose order is total, exactly;
    // one compare and a min where there were two compares and an and. Not
    // over floats: min and max here are std::min and std::max (CodeGen_LLVM.
    // cpp), which pass a NaN in their second argument over, so `a <= min(b,
    // NaN)` is `a <= b` where `(a <= b) & (a <= NaN)` is false. The rule
    // would be exact there given that `c` is not a NaN, which nothing here
    // can say (SSA/ConstantIntervals.h bounds a value where it is a number).
    //
    // A strict comparison beside a non-strict one over integers is made
    // non-strict first, by one step on the side that is not shared --
    // `a < c` is `a <= c - 1`, `c < a` is `c + 1 <= a` -- where the
    // interval analysis says the step cannot wrap (the stepped side is not
    // the type's least, or greatest). So `(a <= b) & (a < c)` is `a <=
    // min(b, c - 1)`: the slab test's `tNear <= tFar` beside the prune's
    // `tNear < best` on the bits, whose least is zero.
    ValuePtr compare_pair(bool is_and, const Type &type, const ValuePtr &x,
                          const ValuePtr &y) {
        const Instruction *dx = def_of(x), *dy = def_of(y);
        if (dx == nullptr || dy == nullptr ||
            (dx->op != Instruction::Op::Lt && dx->op != Instruction::Op::Leq) ||
            (dy->op != Instruction::Op::Lt && dy->op != Instruction::Op::Leq) ||
            dx->operands.size() != 2 || dy->operands.size() != 2 ||
            !equals(dx->type, type) || !equals(dy->type, type)) {
            return nullptr;
        }
        const Type operand_t = dx->operands[0]->get_type();
        if (!equals(operand_t, dy->operands[0]->get_type()) ||
            !scalar_of(operand_t).is_int_or_uint()) {
            return nullptr;
        }
        // The two comparisons' sides, the strict one stepped where the two
        // differ in kind.
        Instruction::Op kind = dx->op;
        ValuePtr x0 = dx->operands[0], x1 = dx->operands[1];
        ValuePtr y0 = dy->operands[0], y1 = dy->operands[1];
        if (dx->op != dy->op) {
            // Which of the four pairings shares a side says which side of
            // the strict comparison is stepped.
            const bool strict_is_x = dx->op == Instruction::Op::Lt;
            ValuePtr &s0 = strict_is_x ? x0 : y0, &s1 = strict_is_x ? x1 : y1;
            const ValuePtr &o0 = strict_is_x ? y0 : x0, &o1 = strict_is_x ? y1 : x1;
            const Type scalar = scalar_of(operand_t);
            const ConstantInterval range = ConstantInterval::bounds_of_type(scalar);
            const auto one = [&] {
                return make_integer_constant(operand_t, 1);
            };
            if (same_value(s0, o0)) {
                // `s0 < s1` with s0 shared: `s0 <= s1 - 1`, s1 not the least.
                const ConstantInterval i = bounds(s1);
                if (!(i.number && i.min_defined && range.min_defined &&
                      i.min > range.min)) {
                    return nullptr;
                }
                s1 = make(Instruction::Op::Sub, operand_t, {s1, one()});
            } else if (same_value(s1, o1)) {
                // `s0 < s1` with s1 shared: `s0 + 1 <= s1`, s0 not the greatest.
                const ConstantInterval i = bounds(s0);
                if (!(i.number && i.max_defined && range.max_defined &&
                      i.max < range.max)) {
                    return nullptr;
                }
                s0 = make(Instruction::Op::Add, operand_t, {s0, one()});
            } else {
                return nullptr;
            }
            kind = Instruction::Op::Leq;
        }
        if (same_value(x0, y0)) {
            ValuePtr joined =
                make(is_and ? Instruction::Op::Min : Instruction::Op::Max,
                     operand_t, {x1, y1});
            return make(kind, type, {x0, std::move(joined)});
        }
        if (same_value(x1, y1)) {
            ValuePtr joined =
                make(is_and ? Instruction::Op::Max : Instruction::Op::Min,
                     operand_t, {x0, y0});
            return make(kind, type, {std::move(joined), x1});
        }
        return nullptr;
    }

    // The zero of type `t`: false, 0, 0.0, broadcast over a vector's lanes;
    // an empty build for a struct; null for anything else.
    ValuePtr zero_of(const Type &t) {
        if (t.is<Struct_t>()) {
            return make(Instruction::Op::MakeStruct, t, {});
        }
        const Type scalar = scalar_of(t);
        ValuePtr c;
        if (scalar.is<Bool_t>()) {
            c = make_bool(false);
        } else if (scalar.is_uint()) {
            c = std::make_shared<Value>(Constant{scalar, uint64_t(0)});
        } else if (scalar.is_int()) {
            c = std::make_shared<Value>(Constant{scalar, int64_t(0)});
        } else if (scalar.is_float()) {
            c = std::make_shared<Value>(Constant{scalar, 0.0});
        } else {
            return nullptr;
        }
        if (!t.is_vector()) {
            return c;
        }
        ValuePtr width = std::make_shared<Value>(
            Constant{UInt_t::make(32), uint64_t(t.lanes())});
        return make(Instruction::Op::Bc, t, {std::move(c), std::move(width)});
    }

    // The integer constant `n` of type `t`: a scalar's, or a scalar's
    // broadcast over a vector's lanes.
    ValuePtr make_integer_constant(const Type &t, int64_t n) {
        const Type scalar = scalar_of(t);
        ValuePtr c = scalar.is_uint()
                         ? std::make_shared<Value>(Constant{scalar, uint64_t(n)})
                         : std::make_shared<Value>(Constant{scalar, n});
        if (!t.is_vector()) {
            return c;
        }
        ValuePtr width = std::make_shared<Value>(
            Constant{UInt_t::make(32), uint64_t(t.lanes())});
        return make(Instruction::Op::Bc, t, {std::move(c), std::move(width)});
    }

    // An operation over broadcasts alone is the broadcast of the operation
    // over what they broadcast: `min(bc(s), bc(t))` is `bc(min(s, t))`, one
    // scalar operation where there were a vector one and two broadcasts.
    // Exact in every lane, since every lane held the same operands -- std::
    // min's treatment of a NaN included.
    ValuePtr group_broadcasts(Instruction::Op op, const Type &type,
                              const vector<ValuePtr> &ops) {
        if (!type.is_vector() || ops.empty()) {
            return nullptr;
        }
        switch (op) {
        case Instruction::Op::Abs:
        case Instruction::Op::Add:
        case Instruction::Op::BwAnd:
        case Instruction::Op::BwOr:
        case Instruction::Op::Eq:
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr:
        case Instruction::Op::Leq:
        case Instruction::Op::Lt:
        case Instruction::Op::Max:
        case Instruction::Op::Min:
        case Instruction::Op::Mul:
        case Instruction::Op::Ne:
        case Instruction::Op::Not:
        case Instruction::Op::Sub:
        case Instruction::Op::Xor:
            break;
        case Instruction::Op::Reinterpret:
            // Between vectors of one lane count alone: a mask read as a word
            // packs its lanes, and is not lanewise.
            if (ops.size() != 1 || !ops[0]->get_type().is_vector() ||
                ops[0]->get_type().lanes() != type.lanes()) {
                return nullptr;
            }
            break;
        default:
            return nullptr;
        }
        vector<ValuePtr> scalars;
        ValuePtr width;
        for (const ValuePtr &o : ops) {
            const Instruction *d = def_of(o);
            if (d == nullptr || d->op != Instruction::Op::Bc ||
                d->operands.size() != 2 || !d->type.is_vector() ||
                d->type.lanes() != type.lanes()) {
                return nullptr;
            }
            scalars.push_back(d->operands[0]);
            if (!width) {
                width = d->operands[1];
            }
        }
        ValuePtr scalar = make(op, type.element_of(), std::move(scalars));
        return make(Instruction::Op::Bc, type, {std::move(scalar), width});
    }

    // Over integers min and max are associative and commutative, so a
    // broadcast joining a chain of them that already holds one joins that
    // one: `min(min(x, bc(s)), bc(t))` is `min(x, min(bc(s), bc(t)))`, which
    // the rule above makes `min(x, bc(min(s, t)))` -- the min of the two
    // scalars once, where the vector min was in every lane, and hoisted by
    // the backend where `t` changes less often than the chain is evaluated:
    // a traversal's running best joining a node's slab test, as Embree's
    // `tray.tfar` carries it. Not over floats: std::min is neither
    // associative nor commutative where a NaN is (the second argument's is
    // passed over, the first's comes back), so moving an operand past
    // another changes which NaN a lane keeps. The chain's first broadcast,
    // to a bounded depth; each application leaves one broadcast fewer.
    ValuePtr join_broadcast(Instruction::Op op, const Type &type,
                            const vector<ValuePtr> &ops) {
        if ((op != Instruction::Op::Min && op != Instruction::Op::Max) ||
            ops.size() != 2 || !type.is_vector() ||
            !type.element_of().is_int_or_uint()) {
            return nullptr;
        }
        for (const auto &[chain, joining] :
             {std::pair{ops[0], ops[1]}, std::pair{ops[1], ops[0]}}) {
            const Instruction *b = def_of(joining);
            const Instruction *c = def_of(chain);
            if (b == nullptr || b->op != Instruction::Op::Bc || c == nullptr ||
                c->op != op || !equals(c->type, type)) {
                continue;
            }
            if (ValuePtr joined = join_into(chain, joining, op, type, 8)) {
                return joined;
            }
        }
        return nullptr;
    }

    // `chain` with its first broadcast leaf made the op of itself and
    // `joining`; null when it has none within `depth`.
    ValuePtr join_into(const ValuePtr &chain, const ValuePtr &joining,
                       Instruction::Op op, const Type &type, size_t depth) {
        const Instruction *d = def_of(chain);
        if (d == nullptr) {
            return nullptr;
        }
        if (d->op == Instruction::Op::Bc) {
            return make(op, type, {chain, joining});
        }
        if (depth == 0 || d->op != op || !equals(d->type, type) ||
            d->operands.size() != 2) {
            return nullptr;
        }
        for (size_t i = 0; i < 2; i++) {
            if (ValuePtr n = join_into(d->operands[i], joining, op, type,
                                       depth - 1)) {
                vector<ValuePtr> operands = d->operands;
                operands[i] = std::move(n);
                return make(op, type, std::move(operands));
            }
        }
        return nullptr;
    }

    ValuePtr rule(Instruction::Op op, const Type &type,
                  const vector<ValuePtr> &ops) {
        if (ValuePtr grouped = group_broadcasts(op, type, ops)) {
            return grouped;
        }
        switch (op) {
        case Instruction::Op::Add: {
            if (ops.size() != 2 || !(type.is_int() || type.is_uint())) {
                break;
            }
            for (const auto &[x, y] : {std::pair{ops[0], ops[1]},
                                       std::pair{ops[1], ops[0]}}) {
                // x + 0 = x
                if (is_zero(y)) {
                    return x;
                }
                // bc(a) + ramp(b, s) = ramp(a + b, s): a gang's index into
                // an array -- the loop's index, one per lane, plus the same
                // offset in every lane -- is still `lanes` consecutive
                // elements, and read or written as one vector access rather
                // than a gather or a scatter (see as_dense_ramp in
                // CodeGen/CodeGen_LLVM.cpp). The vectorizer makes this shape
                // whenever a split loop's inner index is vectorized: the
                // outer index broadcast, added to the ramp of lanes.
                const Instruction *dx = def_of(x), *dy = def_of(y);
                if (dx != nullptr && dy != nullptr &&
                    dx->op == Instruction::Op::Bc &&
                    dy->op == Instruction::Op::Ramp &&
                    dx->operands.size() == 2 && dy->operands.size() == 2 &&
                    equals(dx->operands[0]->get_type(),
                           dy->operands[0]->get_type())) {
                    ValuePtr base = make(Instruction::Op::Add,
                                         dy->operands[0]->get_type(),
                                         {dx->operands[0], dy->operands[0]});
                    return make(Instruction::Op::Ramp, type,
                                {base, dy->operands[1]});
                }
            }
            break;
        }
        case Instruction::Op::Not: {
            // Bools, and masks, lane by lane.
            const bool bools = type.is<Bool_t>() ||
                               (type.is_vector() && type.element_of().is<Bool_t>());
            if (ops.size() != 1 || !bools) {
                break;
            }
            if (std::optional<bool> c = uniform_bool(ops[0])) {
                return uniform_bool_constant(type, !*c);
            }
            // !!x = x
            if (const Instruction *d = def_of(ops[0]);
                d != nullptr && d->op == Instruction::Op::Not &&
                d->operands.size() == 1) {
                return d->operands[0];
            }
            break;
        }
        case Instruction::Op::LAnd:
        case Instruction::Op::LOr: {
            // Bools, and masks: a vectorized `&&` is one of these on
            // vectors of bools, and every rule below holds lane by lane.
            const bool bools = type.is<Bool_t>() ||
                               (type.is_vector() && type.element_of().is<Bool_t>());
            if (ops.size() != 2 || !bools) {
                break;
            }
            const bool is_and = op == Instruction::Op::LAnd;
            const ValuePtr &a = ops[0], &b = ops[1];
            // x & x = x; x | x = x. Also when x is spelled twice.
            if (same_value(a, b)) {
                return a;
            }
            const Instruction *da = def_of(a);
            if (da != nullptr &&
                same_expression(def_of(b), da->op, da->type, da->operands)) {
                return keep(a, b);
            }
            for (const auto &[x, y] : {std::pair{a, b}, std::pair{b, a}}) {
                if (std::optional<bool> c = uniform_bool(x)) {
                    // true & y = y, false & y = false; and the dual. On
                    // masks too, where the constant is a broadcast.
                    return *c == is_and ? y : uniform_bool_constant(type, !is_and);
                }
            }
            // In `a & b`, `a` holds wherever `b` matters and `b` wherever
            // `a` does; in `a | b` each fails wherever the other matters. An
            // occurrence of one inside the other is decided: `hit & (select(
            // hit, near, inf) <= best)` is `hit & (near <= best)`, the blend
            // gone from under the compare.
            if (assuming == 0) {
                for (const auto &[x, y] : {std::pair{a, b}, std::pair{b, a}}) {
                    ValuePtr decided = assume(y, x, is_and);
                    if (!same_value(decided, y)) {
                        return make(op, type, {x, std::move(decided)});
                    }
                }
            }
            if (ValuePtr joined = compare_pair(is_and, type, a, b)) {
                return joined;
            }
            break;
        }
        case Instruction::Op::Select: {
            if (ops.size() != 3) {
                break;
            }
            if (std::optional<bool> c = uniform_bool(ops[0])) {
                return *c ? ops[1] : ops[2];
            }
            if (same_value(ops[1], ops[2])) {
                return ops[1];
            }
            // `select(c, true, false)` is `c`, and `select(c, false, true)`
            // is `!c`: the linearizer's negation of a mask, and an option's
            // `set` once its merge has been read field by field (the
            // LoadField rule).
            if (equals(ops[0]->get_type(), type)) {
                const std::optional<bool> t = uniform_bool(ops[1]);
                const std::optional<bool> f = uniform_bool(ops[2]);
                if (t == true && f == false) {
                    return ops[0];
                }
                if (t == false && f == true) {
                    return make(Instruction::Op::Not, type, {ops[0]});
                }
            }
            // Inside the arm taken when the condition holds, it holds; inside
            // the other, it does not (see assume).
            if (assuming == 0) {
                ValuePtr then = assume(ops[1], ops[0], true);
                ValuePtr otherwise = assume(ops[2], ops[0], false);
                if (!same_value(then, ops[1]) || !same_value(otherwise, ops[2])) {
                    return make(op, type,
                                {ops[0], std::move(then), std::move(otherwise)});
                }
            }
            // `select(c, t, false)` is `c & t` and `select(c, true, f)` is
            // `c | f` -- the shape a vectorized `&&` of masks takes -- so
            // the comparisons they join are joined as under `&` and `|`.
            if (equals(ops[0]->get_type(), type)) {
                if (uniform_bool(ops[2]) == false) {
                    if (ValuePtr joined = compare_pair(true, type, ops[0], ops[1])) {
                        return joined;
                    }
                }
                if (uniform_bool(ops[1]) == true) {
                    if (ValuePtr joined = compare_pair(false, type, ops[0], ops[2])) {
                        return joined;
                    }
                }
            }
            break;
        }
        case Instruction::Op::Min:
        case Instruction::Op::Max: {
            if (ValuePtr joined = join_broadcast(op, type, ops)) {
                return joined;
            }
            break;
        }
        case Instruction::Op::MakeStruct: {
            // A struct built from every field of one value of its own type,
            // in order, is that value -- `make_struct<T>(load_field(v, 0),
            // ..., load_field(v, n-1))` is a copy of `v` -- and a vector
            // built from every lane of one vector the same. A traversal's
            // hit is taken apart and put back together this way at each
            // level of the option it travels in; folded, it is handed on as
            // it was rather than rebuilt a field at a time.
            if (ops.empty()) {
                break;
            }
            const Instruction::Op part = type.is_vector()
                                             ? Instruction::Op::ExtractIdx
                                             : Instruction::Op::LoadField;
            size_t count = 0;
            if (type.is_vector()) {
                count = type.lanes();
            } else if (const Struct_t *s = type.as<Struct_t>()) {
                count = s->fields.size();
            } else {
                break;
            }
            if (count != ops.size()) {
                break;
            }
            const Instruction *first = defined_by(ops[0]);
            if (first == nullptr || first->op != part ||
                first->operands.size() != 2) {
                break;
            }
            const ValuePtr &whole = first->operands[0];
            if (!equals(whole->get_type(), type)) {
                break;
            }
            const auto position = [](const ValuePtr &v) -> std::optional<uint64_t> {
                const Constant *c = constant_of(v);
                if (c == nullptr) {
                    return std::nullopt;
                }
                return std::visit(
                    overloads{
                        [](int64_t i) -> std::optional<uint64_t> {
                            return i < 0 ? std::nullopt
                                         : std::optional<uint64_t>(i);
                        },
                        [](uint64_t u) -> std::optional<uint64_t> { return u; },
                        [](const auto &) -> std::optional<uint64_t> {
                            return std::nullopt;
                        },
                    },
                    c->data);
            };
            bool copy = true;
            for (size_t i = 0; copy && i < ops.size(); i++) {
                const Instruction *d = defined_by(ops[i]);
                copy = d != nullptr && d->op == part && d->operands.size() == 2 &&
                       same_value(d->operands[0], whole) &&
                       position(d->operands[1]) == std::optional<uint64_t>(i);
            }
            if (copy) {
                return whole;
            }
            break;
        }
        case Instruction::Op::LoadField: {
            // A field of a struct just made is what it was made from. A
            // variant given its tag as a constant (a specialized copy, SSA/
            // Specialize.h) is made this way, and this is what lets the
            // match on it fold -- in the block that made it, and in the
            // blocks below that were handed it as an argument, where the
            // struct is found through the argument. The field's value is
            // defined where the struct was, which every block handed the
            // struct is reached through.
            if (ops.size() != 2) {
                break;
            }
            const Instruction *d = defined_by(ops[0]);
            const Constant *c = constant_of(ops[1]);
            if (c == nullptr) {
                break;
            }
            const std::optional<uint64_t> index = std::visit(
                overloads{
                    [](int64_t i) -> std::optional<uint64_t> {
                        return i < 0 ? std::nullopt : std::optional<uint64_t>(i);
                    },
                    [](uint64_t u) -> std::optional<uint64_t> { return u; },
                    [](const auto &) -> std::optional<uint64_t> {
                        return std::nullopt;
                    },
                },
                c->data);
            if (!index.has_value()) {
                break;
            }
            if (d == nullptr) {
                // Not one struct but a merge of several: the field folds
                // when every one was built with the same constant there.
                ValuePtr same = constant_field_of_merge(ops[0], *index);
                if (same && equals(same->get_type(), type)) {
                    return same;
                }
                break;
            }
            // A field of a select between two structs is the select between
            // the field of each -- the option a masked test merges, `select(
            // miss, none, some)`, read as `select(miss, none.set, some.set)`
            // -- where an arm is a struct just made or a merge itself, so
            // that the field folds on at least that side; the select's
            // condition has to fit the field (one bool for any field, a mask
            // for a field with its lanes).
            if (d->op == Instruction::Op::Select && d->operands.size() == 3) {
                const Type &cond_t = d->operands[0]->get_type();
                // A struct field under a mask is itself a struct of vectors,
                // as the whole was (its own fields are read in turn).
                const bool fits = !cond_t.is_vector() || type.is<Struct_t>() ||
                                  (type.is_vector() && type.lanes() == cond_t.lanes());
                const auto folds = [&](const ValuePtr &arm) {
                    const Instruction *a = defined_by(arm);
                    return a != nullptr && (a->op == Instruction::Op::MakeStruct ||
                                            a->op == Instruction::Op::Select);
                };
                if (fits && (folds(d->operands[1]) || folds(d->operands[2]))) {
                    ValuePtr then = make(Instruction::Op::LoadField, type,
                                         {d->operands[1], ops[1]});
                    ValuePtr otherwise = make(Instruction::Op::LoadField, type,
                                              {d->operands[2], ops[1]});
                    return make(Instruction::Op::Select, type,
                                {d->operands[0], std::move(then), std::move(otherwise)});
                }
                break;
            }
            // A struct read through a view of another shape with the same
            // parts in the same order -- the gang's `vector[vec3f, 8]`
            // built from three component vectors and read as the struct of
            // three vectors, `reinterpret<f32x3$v8>(build<f32x3x8>(x, y,
            // z))` -- hands out the part built there, where its type is the
            // field's.
            if (d->op == Instruction::Op::Reinterpret && d->operands.size() == 1) {
                const Instruction *built = defined_by(d->operands[0]);
                if (built != nullptr && built->op == Instruction::Op::MakeStruct &&
                    *index < built->operands.size() &&
                    equals(built->operands[*index]->get_type(), type)) {
                    const Struct_t *viewed = d->type.as<Struct_t>();
                    if (viewed != nullptr &&
                        viewed->fields.size() == built->operands.size()) {
                        return built->operands[*index];
                    }
                }
                break;
            }
            if (d->op != Instruction::Op::MakeStruct) {
                break;
            }
            if (d->operands.empty()) {
                // Built with no operands at all -- an option's empty variant
                // -- a struct is all zeros (the backends' null value), so a
                // field of it is its type's zero: false, 0, 0.0, or an empty
                // build of a struct field.
                return zero_of(type);
            }
            if (*index >= d->operands.size() ||
                !equals(d->operands[*index]->get_type(), type)) {
                break;
            }
            return d->operands[*index];
        }
        case Instruction::Op::Vote: {
            // Every lane holds the same constant, so that is the decision.
            // Anything else stays a vote until a gang is there to hold it
            // (see lower_votes in SSA/Vectorize.cpp) or the run it decides
            // is put on a stack (SSA/QueueRecursion.cpp).
            if (ops.size() == 1 && const_bool(ops[0]).has_value()) {
                return ops[0];
            }
            break;
        }
        case Instruction::Op::Reinterpret: {
            // Bits read as one type and back as the first are the bits
            // they were: `reinterpret<i32>(reinterpret<f32>(x))` is `x`.
            if (ops.size() != 1) {
                break;
            }
            const Instruction *d = defined_by(ops[0]);
            if (d != nullptr && d->op == Instruction::Op::Reinterpret &&
                d->operands.size() == 1 &&
                equals(d->operands[0]->get_type(), type)) {
                return d->operands[0];
            }
            break;
        }
        case Instruction::Op::Lt:
        case Instruction::Op::Leq:
        case Instruction::Op::Eq:
        case Instruction::Op::Ne: {
            if (ops.size() != 2) {
                break;
            }
            // A `<` or `<=` of two 32-bit floats is the same comparison of
            // their bits as signed integers where the right side is a
            // non-negative number -- its bits in [0, 0x7f800000], ordered as
            // the float is, +inf's the greatest -- and the left side's sign
            // bit is clear in every value: a non-negative number's bits
            // order as it does, and a NaN's lie past infinity's, where the
            // comparison is false of them as the float one is. Exact, by
            // the interval analysis's flags (SSA/ConstantIntervals.h), with
            // nothing read into a NaN or a negative zero: a NaN on the right
            // would compare true on the bits where the float compare is
            // false, so none is allowed there. Embree's `asInt(tNear) <=
            // asInt(tFar)`, here reached by rule from `tNear < best` once
            // the best is known a non-negative number; on the bits the
            // comparison joins the slab test's (compare_pair) where a float
            // compare could not. Scalars and masks alike.
            if ((op == Instruction::Op::Lt || op == Instruction::Op::Leq) &&
                scalar_of(ops[0]->get_type()).is_float() &&
                scalar_of(ops[0]->get_type()).bits() == 32 &&
                equals(ops[0]->get_type(), ops[1]->get_type()) &&
                sign_clear(ops[0]) && nonnegative_number(ops[1])) {
                const Type &ft = ops[0]->get_type();
                const Type it = ft.is_vector()
                                    ? Vector_t::make(Int_t::make(32), ft.lanes())
                                    : Int_t::make(32);
                ValuePtr a = make(Instruction::Op::Reinterpret, it, {ops[0]});
                ValuePtr b = make(Instruction::Op::Reinterpret, it, {ops[1]});
                return make(op, type, {std::move(a), std::move(b)});
            }
            if (!type.is<Bool_t>()) {
                break;
            }
            if (std::optional<bool> c = compare_constants(op, ops[0], ops[1])) {
                return make_bool(*c);
            }
            // popcount(m) != 0 is any(m), and popcount(m) == 0 is !any(m):
            // one `kortest` of the mask where the count compared is a mask
            // move, a `popcnt` and a `test`. What a traversal asks of a
            // node's hits before it sorts them (SSA/QueueRecursion.cpp).
            if (op == Instruction::Op::Ne || op == Instruction::Op::Eq) {
                for (const auto &[counted, zero] :
                     {std::pair{ops[0], ops[1]}, std::pair{ops[1], ops[0]}}) {
                    const Instruction *count = defined_by(counted);
                    const Constant *c = constant_of(zero);
                    if (count == nullptr ||
                        count->op != Instruction::Op::Popcount ||
                        count->operands.size() != 1 || c == nullptr ||
                        !count->operands[0]->get_type().is_vector()) {
                        continue;
                    }
                    const bool is_zero = std::visit(
                        overloads{[](uint64_t u) { return u == 0; },
                                  [](int64_t i) { return i == 0; },
                                  [](const auto &) { return false; }},
                        c->data);
                    if (!is_zero) {
                        continue;
                    }
                    ValuePtr any = make(Instruction::Op::Any, Bool_t::make(),
                                        {count->operands[0]});
                    if (op == Instruction::Op::Ne) {
                        return any;
                    }
                    return make(Instruction::Op::Not, Bool_t::make(), {any});
                }
            }
            if (same_value(ops[0], ops[1]) && !ops[0]->get_type().is_float()) {
                // x < x, x == x, and the rest, for anything without a NaN.
                return make_bool(op == Instruction::Op::Leq ||
                                 op == Instruction::Op::Eq);
            }
            if (op != Instruction::Op::Lt) {
                break;
            }
            // cast(a) < cast(b), a and b bools, both to one numeric type:
            // the only way a number made from a bool is below another is
            // false below true, so this is !a & b.
            const Instruction *da = def_of(ops[0]), *db = def_of(ops[1]);
            const auto bool_cast = [](const Instruction *d) {
                return d != nullptr && d->op == Instruction::Op::Cast &&
                       d->operands.size() == 1 && d->type.is_scalar() &&
                       (d->type.is_int() || d->type.is_uint() ||
                        d->type.is_float()) &&
                       d->operands[0]->get_type().is<Bool_t>();
            };
            if (bool_cast(da) && bool_cast(db) && equals(da->type, db->type)) {
                ValuePtr not_a = make(Instruction::Op::Not, Bool_t::make(),
                                      {da->operands[0]});
                return make(Instruction::Op::LAnd, Bool_t::make(),
                            {not_a, db->operands[0]});
            }
            break;
        }
        case Instruction::Op::ExtractIdx: {
            if (ops.size() != 2 || !ops[0]->get_type().is<Vector_t>() ||
                !ops[1]->get_type().is_scalar()) {
                break;
            }
            // A lane of a vector that lives in read-only storage, at a lane
            // only known when it runs, is read from the storage: a layout
            // row's `children[i]`, with `i` the low bits of a sorted key,
            // becomes a scalar load at the lane's address -- the child a
            // traversal descends into read off the node, as Embree's
            // `node->child(i)` is. The vector form kept the children in a
            // register, and since no instruction reads a register's lane at
            // a run-time index, spilled it to read one (`vmovdqa64 %zmm,
            // (%rsp)`; `mov (%rsp,%rbx,8)`): a store the load waits on, on
            // the chain to the next node's address. A lane known at compile
            // time is one instruction off the register, which the vector's
            // other uses hold anyway, and is left alone. Sound because
            // nothing writes the storage (Type::is_readonly), so the read
            // may be made where the lane is wanted rather than where the
            // vector was loaded; the vector's other uses keep it. A vector
            // of scalars only: a vector of vectors is stored one component
            // at a time (see the Vector_t visitor in CodeGen/CodeGen_LLVM.
            // cpp), so its lane is not at one address.
            if (constant_of(ops[1]) == nullptr &&
                ops[0]->get_type().element_of().is_scalar()) {
                if (ValuePtr whole = stored_vector_address(ops[0])) {
                    const Type stored = whole->get_type().as<Ptr_t>()->etype;
                    const Type element = stored.element_of();
                    const Type view_t = Array_t::make(
                        element,
                        UIntImm::make(UInt_t::make(32), stored.lanes()),
                        /*readonly=*/true);
                    ValuePtr view =
                        make(Instruction::Op::Reinterpret, view_t, {whole});
                    ValuePtr at =
                        make(Instruction::Op::GEP,
                             Ptr_t::make(element, /*readonly=*/true),
                             {view, ops[1]});
                    return make(Instruction::Op::Load, type, {at});
                }
            }
            // Lane 0 of a compress that leaves the lanes past the packed
            // ones unspecified, and that nothing else reads, is the lane
            // the mask's lowest set bit names: `compress(v, m)[0]` is
            // `v[ctz(m)]` -- the one child a traversal's one-hit arm
            // descends into, Embree's `bsf` and `node->child(r)` where the
            // compress read the vector through `vpcompressq` and a `vmovq`.
            // With no bit on, both are unspecified (`v[lanes]` is out of
            // range as the compress's lane 0 is nothing), so no condition on
            // the mask is needed; a compress with a fill is left alone, its
            // lane 0 being the fill then. The lane is a run-time one, so the
            // rule above reads it off the storage where the vector lives
            // there. A compress read at other lanes too -- the sorted arms'
            // packed keys, read at 0, 1 and 2 -- is made anyway, and its
            // lane 0 is then one instruction off the register, where the
            // lane named by a count would go through the stack.
            if (const Instruction *d = def_of(ops[0]);
                d != nullptr && d->op == Instruction::Op::Intrinsic &&
                d->intrinsic == ir::Intrinsic::compress &&
                d->operands.size() == 2 && is_zero(ops[1]) &&
                uses_of(d) == 1) {
                if (ValuePtr lane =
                        lowest_lane(d->operands[1], ops[1]->get_type())) {
                    return make(Instruction::Op::ExtractIdx, type,
                                {d->operands[0], lane});
                }
            }
            break;
        }
        default:
            break;
        }
        return nullptr;
    }
};

void resolve_jump(Terminator::Jump &jump, const Simplifier &s) {
    for (auto &arg : jump.args) {
        arg = s.resolve(arg);
    }
}

// Points every use in the function at what its value became.
void resolve_uses(Function &func, const Simplifier &s) {
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            for (auto &operand : instr->operands) {
                operand = s.resolve(operand);
            }
        }
        std::visit(overloads{
                       [](std::monostate &) {},
                       [&](Terminator::Jump &j) { resolve_jump(j, s); },
                       [&](Terminator::Dispatch &d) {
                           d.cond = s.resolve(d.cond);
                           for (auto &target : d.targets) {
                               resolve_jump(target, s);
                           }
                       },
                       [&](Terminator::Return &r) {
                           if (r.value != nullptr) {
                               r.value = s.resolve(r.value);
                           }
                       },
                       [&](Terminator::ParFor &p) {
                           p.start = s.resolve(p.start);
                           p.end = s.resolve(p.end);
                           p.stride = s.resolve(p.stride);
                           resolve_jump(p.body, s);
                           resolve_jump(p.cont, s);
                       },
                       [](Terminator::Yield &) {},
                       [&](Terminator::Call &c) {
                           resolve_jump(c.call, s);
                           resolve_jump(c.cont, s);
                       },
                       [&](Terminator::MultiCall &c) {
                           resolve_jump(c.call, s);
                           resolve_jump(c.cont, s);
                           for (auto &vs : c.varying) {
                               for (auto &v : vs) {
                                   v = s.resolve(v);
                               }
                           }
                           for (auto &k : c.keys) {
                               k = s.resolve(k);
                           }
                           for (auto &k : c.conds) {
                               k = s.resolve(k);
                           }
                       },
                   },
                   block->terminator.data);
    }
}

using UseCounts = std::map<const Instruction *, size_t>;

void count(const ValuePtr &value, UseCounts &uses) {
    if (const Instruction *d = def_of(value)) {
        ++uses[d];
    }
}

void count(const Terminator::Jump &jump, UseCounts &uses) {
    for (const auto &arg : jump.args) {
        count(arg, uses);
    }
}

// How many times each instruction's result is named anywhere in the
// function, terminators included.
UseCounts use_counts(const Function &func) {
    UseCounts uses;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            for (const auto &operand : instr->operands) {
                count(operand, uses);
            }
        }
        std::visit(overloads{
                       [](const std::monostate &) {},
                       [&](const Terminator::Jump &j) { count(j, uses); },
                       [&](const Terminator::Dispatch &d) {
                           count(d.cond, uses);
                           for (const auto &target : d.targets) {
                               count(target, uses);
                           }
                       },
                       [&](const Terminator::Return &r) {
                           count(r.value, uses);
                       },
                       [&](const Terminator::ParFor &p) {
                           count(p.start, uses);
                           count(p.end, uses);
                           count(p.stride, uses);
                           count(p.body, uses);
                           count(p.cont, uses);
                           // A queue drain launched over its capacity uses
                           // the queue's size, storage and slot, and (once a
                           // GPU bind has built it) the address of its count
                           // (Terminator::ParFor, Bind.cpp); a drain not
                           // GPU-bound has them cleared and these are null
                           // (SSA/Convert.cpp).
                           if (p.capacity) {
                               count(p.capacity, uses);
                           }
                           if (p.queue_base) {
                               count(p.queue_base, uses);
                           }
                           if (p.queue_slot) {
                               count(p.queue_slot, uses);
                           }
                           if (p.count_address) {
                               count(p.count_address, uses);
                           }
                       },
                       [](const Terminator::Yield &) {},
                       [&](const Terminator::Call &c) {
                           count(c.call, uses);
                           count(c.cont, uses);
                       },
                       [&](const Terminator::MultiCall &c) {
                           count(c.call, uses);
                           count(c.cont, uses);
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

// Removes the instructions `dead` says to, and the lookups that named them.
void erase(Function &func, const std::set<const Instruction *> &dead) {
    for (const auto &block : func.blocks) {
        std::erase_if(block->instrs, [&](const shared_ptr<Instruction> &in) {
            return dead.count(in.get()) > 0;
        });
        std::erase_if(block->lookups, [&](const auto &entry) {
            return dead.count(def_of(entry.second)) > 0;
        });
    }
}

// Removes every pure instruction nothing reads, and then whatever that
// leaves unread, until nothing more goes.
} // namespace

void remove_dead(Function &func) {
    for (;;) {
        const UseCounts uses = use_counts(func);
        std::set<const Instruction *> dead;
        for (const auto &block : func.blocks) {
            for (const auto &instr : block->instrs) {
                // A value the schedule reads is read (Function::schedule_uses).
                if (pure(*instr) && uses.count(instr.get()) == 0 &&
                    !func.schedule_uses.contains(instr->name)) {
                    dead.insert(instr.get());
                }
            }
        }
        if (dead.empty()) {
            return;
        }
        erase(func, dead);
    }
}

namespace {

// The target a dispatch on the constant `v` takes, of `targets` of them:
// target 0 on false or zero, target k on k; nothing when `v` is not a
// constant or names no target.
std::optional<size_t> dispatch_choice(const ValuePtr &v, size_t targets) {
    const Constant *c = constant_of(v);
    if (c == nullptr) {
        return std::nullopt;
    }
    const std::optional<uint64_t> which = std::visit(
        overloads{
            [](bool b) -> std::optional<uint64_t> { return b ? 1 : 0; },
            [](int64_t i) -> std::optional<uint64_t> {
                return i < 0 ? std::nullopt : std::optional<uint64_t>(i);
            },
            [](uint64_t u) -> std::optional<uint64_t> { return u; },
            [](const auto &) -> std::optional<uint64_t> {
                return std::nullopt;
            },
        },
        c->data);
    if (!which.has_value() || *which >= targets) {
        return std::nullopt;
    }
    return size_t(*which);
}

// A dispatch on a constant is a jump to the target it names: what a
// specialized copy's match on its variant becomes once the tag is read off
// the struct that fixed it (the LoadField rule). The arms nothing reaches
// any more are for the caller to remove. Returns whether any folded.
bool fold_constant_dispatches(Function &func) {
    bool any = false;
    for (const auto &block : func.blocks) {
        auto *d = std::get_if<Terminator::Dispatch>(&block->terminator.data);
        if (d == nullptr) {
            continue;
        }
        const std::optional<size_t> which =
            dispatch_choice(d->cond, d->targets.size());
        if (!which.has_value()) {
            continue;
        }
        Terminator::Jump taken = d->targets[*which];
        block->terminator.data = std::move(taken);
        any = true;
    }
    return any;
}

// The value `v` with the arguments named in `with` replaced by what the map
// says; `v` itself when it is not one of them.
ValuePtr substitute_argument(const ValuePtr &v,
                             const std::map<std::string, ValuePtr> &with) {
    const auto *arg = v ? std::get_if<Argument>(&v->data) : nullptr;
    if (arg == nullptr) {
        return v;
    }
    const auto it = with.find(arg->name);
    return it == with.end() ? v : it->second;
}

// An edge that decides a dispatch is threaded past it: a jump into a block
// that has no instructions and dispatches on one of its own arguments,
// passing a constant there, goes straight to the target the constant picks,
// and that target takes over the arguments the block was carrying. The
// loop a traversal becomes (SSA/QueueRecursion.cpp) has a header of exactly
// this shape -- `!visit(node, live, count): dispatch live [exit, body]` --
// into which every edge but the one that ran out of stack passes `true`,
// so the header's test of `live` ran once per node and once per leaf tile
// for a value every edge knew; after this the edges that know go straight
// to the body, the one that does not goes straight to the exit, and the
// header, with no edge left, goes. (Mueller and Whalley's "Avoiding
// Conditional Branches by Code Replication", PLDI 1995, is the idea, done
// here without the replication: the block replicated has nothing in it but
// the branch.) LLVM's own jump threading will not do this one, because the
// block is a loop header and threading through a header can make a loop
// irreducible (ThreadAcrossLoopHeaders is off by default). Here it cannot:
// a target is entered from the block alone, so an edge that goes to the
// target instead makes the target the header, with one entry still.
//
// The block's arguments move to the targets. Each target takes a copy of
// every argument that is read in a block it dominates, and those reads are
// pointed at the copy; a read in a block no one target dominates -- one
// reached from two of them -- would need a merge nothing here can write,
// and a block with such a read is left alone, as is a block whose targets
// have another way in. A target's own jump arguments, computed from the
// block's arguments, are rewritten on each threaded edge in terms of what
// that edge passes. Edges that pass no constant keep the block, and where
// they all pass one constant its dispatch becomes the jump that constant
// picks. One block per pass over the function, the graph rebuilt between,
// since each threading changes what the next one would read; returns
// whether any was threaded.
bool thread_argument_dispatches(Function &func) {
    bool any = false;
    for (;;) {
        const Cfg cfg(func);
        const DomTree dom = compute_dominator_tree(cfg);
        bool threaded = false;
        for (const BlockId b : cfg.rpo) {
            const shared_ptr<Block> &block = cfg.block(b);
            if (block.get() == func.blocks.front().get() ||
                !block->instrs.empty()) {
                continue; // the entry's arguments are the parameters
            }
            auto *d = std::get_if<Terminator::Dispatch>(&block->terminator.data);
            if (d == nullptr || d->targets.empty()) {
                continue;
            }
            const auto *cond = std::get_if<Argument>(&d->cond->data);
            if (cond == nullptr) {
                continue;
            }
            size_t which_arg = block->args.size();
            for (size_t i = 0; i < block->args.size(); i++) {
                if (block->args[i].name == cond->name) {
                    which_arg = i;
                }
            }
            if (which_arg == block->args.size()) {
                continue;
            }

            // The targets: distinct blocks, not this one, entered from this
            // one alone.
            vector<shared_ptr<Block>> targets;
            bool fit = true;
            for (const Terminator::Jump &t : d->targets) {
                const BlockId tb = cfg.find(t.name);
                if (tb == NO_BLOCK || tb == b || !dom.contains(tb)) {
                    fit = false;
                    break;
                }
                for (const BlockId p : cfg.preds[tb]) {
                    fit = fit && p == b;
                }
                for (const auto &seen : targets) {
                    fit = fit && seen.get() != cfg.block(tb).get();
                }
                targets.push_back(cfg.block(tb));
            }
            if (!fit) {
                continue;
            }

            // The edges in, and which of them pass a constant where the
            // dispatch reads.
            struct Edge {
                Terminator::Jump *jump;
                size_t which; // the target the constant picks
            };
            vector<Edge> constant_edges;
            size_t edges = cfg.preds[b].size();
            for (const BlockId p : cfg.preds[b]) {
                for (Terminator::Jump *jump : jumps_of(*cfg.block(p))) {
                    if (jump->name != block->name ||
                        jump->args.size() != block->args.size()) {
                        continue;
                    }
                    const std::optional<size_t> which =
                        dispatch_choice(jump->args[which_arg], targets.size());
                    if (which.has_value()) {
                        constant_edges.push_back(Edge{jump, *which});
                    }
                }
            }
            // A predecessor with two edges in appears twice in `preds` and
            // yields its jumps twice above; one entry per edge either way.
            std::sort(constant_edges.begin(), constant_edges.end(),
                      [](const Edge &x, const Edge &y) {
                          return x.jump < y.jump;
                      });
            constant_edges.erase(
                std::unique(constant_edges.begin(), constant_edges.end(),
                            [](const Edge &x, const Edge &y) {
                                return x.jump == y.jump;
                            }),
                constant_edges.end());
            if (constant_edges.empty()) {
                continue;
            }
            const bool dies = constant_edges.size() == edges;

            // Where the block's arguments are read, outside it: each read
            // must sit under exactly one target.
            std::set<std::string> names;
            for (const Argument &a : block->args) {
                names.insert(a.name);
            }
            // needed[m] = the arguments target m's subtree reads.
            vector<std::set<size_t>> needed(targets.size());
            for (const BlockId x : cfg.rpo) {
                if (x == b) {
                    continue;
                }
                std::set<std::string> read;
                for_each_value(*cfg.block(x), [&](ValuePtr &v) {
                    if (const auto *a = std::get_if<Argument>(&v->data);
                        a != nullptr && names.contains(a->name)) {
                        read.insert(a->name);
                    }
                });
                if (read.empty()) {
                    continue;
                }
                size_t under = targets.size();
                for (size_t m = 0; m < targets.size(); m++) {
                    if (dom.dominates(cfg.id(*targets[m]), x)) {
                        under = m;
                    }
                }
                if (under == targets.size()) {
                    fit = false;
                    break;
                }
                for (size_t j = 0; j < block->args.size(); j++) {
                    if (read.contains(block->args[j].name)) {
                        needed[under].insert(j);
                    }
                }
            }
            if (!fit) {
                continue;
            }

            // Each target takes the arguments its subtree reads, under the
            // block's own names where the block is about to go and fresh
            // ones where it stays, and its subtree reads the copies.
            const vector<Terminator::Jump> original = d->targets;
            for (size_t m = 0; m < targets.size(); m++) {
                std::map<std::string, ValuePtr> copies;
                for (const size_t j : needed[m]) {
                    Argument copy = block->args[j];
                    if (!dies) {
                        copy.name = func.get_unique_name();
                    }
                    copies[block->args[j].name] = targets[m]->add_argument(copy);
                    d->targets[m].args.push_back(
                        std::make_shared<Value>(block->args[j]));
                }
                if (copies.empty()) {
                    continue;
                }
                const BlockId tb = cfg.id(*targets[m]);
                for (const BlockId x : cfg.rpo) {
                    if (dom.dominates(tb, x)) {
                        for_each_value(*cfg.block(x), [&](ValuePtr &v) {
                            v = substitute_argument(v, copies);
                        });
                    }
                }
            }
            // The edges that know where they go, sent there, passing what
            // the target's jump would have computed from the block's
            // arguments and the copies the target now takes.
            for (const Edge &e : constant_edges) {
                std::map<std::string, ValuePtr> passed;
                for (size_t j = 0; j < block->args.size(); j++) {
                    passed[block->args[j].name] = e.jump->args[j];
                }
                vector<ValuePtr> args;
                for (const ValuePtr &v : original[e.which].args) {
                    args.push_back(substitute_argument(v, passed));
                }
                for (const size_t j : needed[e.which]) {
                    args.push_back(e.jump->args[j]);
                }
                e.jump->name = targets[e.which]->name;
                e.jump->args = std::move(args);
            }
            // The edges left: where they all pass one constant, the dispatch
            // is that target's jump.
            if (!dies) {
                std::optional<size_t> agreed;
                bool agree = true;
                refresh_preds(func);
                for (const auto &weak : block->preds) {
                    const shared_ptr<Block> pred = weak.lock();
                    const ValuePtr passed =
                        pred ? passed_to(*pred, *block, which_arg) : nullptr;
                    const std::optional<size_t> which =
                        dispatch_choice(passed, targets.size());
                    if (!which.has_value() ||
                        (agreed.has_value() && *agreed != *which)) {
                        agree = false;
                        break;
                    }
                    agreed = which;
                }
                if (agree && agreed.has_value()) {
                    Terminator::Jump taken = d->targets[*agreed];
                    block->terminator.data = std::move(taken);
                }
            }
            remove_unreachable_blocks(func);
            threaded = true;
            any = true;
            break;
        }
        if (!threaded) {
            return any;
        }
    }
}

} // namespace

namespace {

// One pass of the rules over every instruction of `func`, in order, and the
// removal of what they replaced; whether anything was replaced.
bool simplify_once(Function &func, const ConstantIntervals *intervals) {
    Simplifier s(func, intervals);
    for (const auto &block : func.blocks) {
        s.enter(block);
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> instr = block->instrs[i];
            for (auto &operand : instr->operands) {
                operand = s.resolve(operand);
            }
            if (Simplifier::writes_memory(*instr)) {
                s.generation++;
            }
            if (instr->name.empty()) {
                continue; // an effect, not a value
            }
            s.at = i;
            // An instruction like this one already in scope is it (see
            // find_equal); otherwise this one is remembered, before the
            // rules, so that what a rule makes of a later one can find it.
            if (ValuePtr same = s.find_equal(*instr)) {
                s.replaced[instr.get()] = same;
                continue;
            }
            {
                const auto looked_up = block->lookups.find(instr->name);
                s.remember(instr, looked_up != block->lookups.end()
                                      ? looked_up->second
                                      : std::make_shared<Value>(instr));
            }
            // A shuffle of one vector in its own order is that vector: the
            // concatenation of one source that a widened read of lanes
            // makes (ir::VectorShuffle::make_concat), and what stood between
            // a lane taken out of it and the storage the vector was read
            // from (the ExtractIdx rule) -- a prefetch's address at a lane
            // was a spill of the whole vector and a load from the slot where
            // the child itself was one load off the row. Decided here, where
            // the instruction's lane order is at hand.
            ValuePtr v;
            if (instr->op == Instruction::Op::Shuffle &&
                instr->operands.size() == 1 && instr->type.is_vector() &&
                equals(instr->operands[0]->get_type(), instr->type) &&
                instr->shuffle.size() == instr->type.lanes()) {
                bool identity = true;
                for (size_t k = 0; identity && k < instr->shuffle.size(); k++) {
                    identity = instr->shuffle[k] == int(k);
                }
                if (identity) {
                    v = instr->operands[0];
                }
            }
            if (v == nullptr) {
                v = s.rule(instr->op, instr->type, instr->operands);
            }
            // A rule may have put new instructions in front of this one.
            i = s.at;
            if (v == nullptr) {
                continue;
            }
            // The program's name for the value stays on it: a `let`'s name
            // is how a schedule points at a value (`hits.specialize(kind)`),
            // and what takes the instruction's place is that value.
            if (const auto *vi = std::get_if<shared_ptr<Instruction>>(&v->data);
                vi != nullptr && !instr->name.starts_with("@") &&
                (*vi)->name.starts_with("@")) {
                if (const auto owner = (*vi)->owner.lock()) {
                    owner->lookups.erase((*vi)->name);
                    owner->lookups[instr->name] = v;
                }
                (*vi)->name = instr->name;
            }
            s.replaced[instr.get()] = v;
        }
    }
    if (s.replaced.empty()) {
        return false;
    }
    resolve_uses(func, s);
    // A name looked up after this should find what took its place.
    for (const auto &block : func.blocks) {
        for (auto &[name, value] : block->lookups) {
            value = s.resolve(value);
        }
    }
    // And a name a type carries -- the size of an array made by a
    // rewrite (see SSA/Storage.h) -- names what took its place too, or
    // the type would name a value that no longer exists.
    std::map<std::string, ir::Expr> renames;
    for (const auto &[instr, v] : s.replaced) {
        const ValuePtr resolved = s.resolve(v);
        const Instruction *now = def_of(resolved);
        if (now != nullptr && now->name == instr->name) {
            continue; // the replacement took the name
        }
        renames.emplace(instr->name, as_expr(resolved));
    }
    if (!renames.empty()) {
        rename_in_types(func, renames);
    }
    std::set<const Instruction *> gone;
    for (const auto &[instr, _] : s.replaced) {
        gone.insert(instr);
    }
    erase(func, gone);
    remove_dead(func);
    return true;
}

} // namespace

void simplify(Function &func, const ConstantIntervals *intervals) {
    // To a fixed point, within a bound: a rule's result is simplified as
    // it is made, but an instruction visited before one of its operands
    // was replaced -- `!x` made of a `!y` that a later merge made `y`'s
    // double negation -- is not looked at again within the pass, and what
    // the next pass finds in it may carry further (the prune's compare
    // reaches the slab test's through three such steps).
    for (int round = 0; round < 8 && simplify_once(func, intervals); round++) {
    }
    if (fold_constant_dispatches(func)) {
        remove_unreachable_blocks(func);
        remove_dead(func);
    }
    if (thread_argument_dispatches(func)) {
        remove_dead(func);
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
