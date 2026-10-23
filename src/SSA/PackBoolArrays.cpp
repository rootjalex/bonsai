#include "SSA/PackBoolArrays.h"

#include "Error.h"
#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "Utils.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::set;
using std::shared_ptr;
using std::string;
using std::vector;
using ValuePtr = shared_ptr<Value>;

const Instruction *def_of(const ValuePtr &v) {
    const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
    return held == nullptr ? nullptr : held->get();
}

std::optional<uint64_t> constant_of(const ValuePtr &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data);
        i != nullptr && *i >= 0) {
        return uint64_t(*i);
    }
    return std::nullopt;
}

// Does `v` refer to the allocation `alloca`: the instruction itself, or a
// block argument threading it onward under its name (a definition and the
// arguments that carry it share a name, SSA/PromoteAllocas.cpp).
bool refers_to(const ValuePtr &v, const Instruction *alloca) {
    if (const auto *in = std::get_if<shared_ptr<Instruction>>(&v->data)) {
        return in->get() == alloca;
    }
    if (const auto *a = std::get_if<Argument>(&v->data)) {
        return a->name == alloca->name;
    }
    return false;
}

// Is `idx` the ramp 0, 1, ..., n - 1: the index of the whole array.
bool is_full_ramp(const ValuePtr &idx, uint32_t n) {
    const Instruction *d = def_of(idx);
    if (d == nullptr || d->op != Instruction::Op::Ramp ||
        d->operands.size() != 2 || !d->type.is<Vector_t>() ||
        d->type.lanes() != n) {
        return false;
    }
    return constant_of(d->operands[0]) == uint64_t(0) &&
           constant_of(d->operands[1]) == uint64_t(1);
}

// The blocks of the body of the parfor that ends `header`: what its body
// jump reaches. The body ends at a Yield, and the continuation is not a
// successor of the body (SSA/Analysis.h, successors).
set<string> body_of(const BlockMap &blocks, const Block &header) {
    const auto *loop =
        std::get_if<Terminator::ParFor>(&header.terminator.data);
    internal_assert(loop != nullptr);
    set<string> region;
    vector<string> work{loop->body.name};
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (!region.insert(name).second) {
            continue;
        }
        for (const string &next : successors(*blocks.at(name))) {
            work.push_back(next);
        }
    }
    return region;
}

// Writes instructions into a block before a position, moving the position
// past each.
struct Emitter {
    Function &func;
    const shared_ptr<Block> &block;
    size_t pos;

    ValuePtr make(Type type, Instruction::Op op, vector<ValuePtr> operands) {
        auto instr = std::make_shared<Instruction>(
            func.get_unique_name(), std::move(type), op, std::move(operands),
            block);
        block->instrs.insert(block->instrs.begin() + pos, instr);
        pos++;
        return std::make_shared<Value>(std::move(instr));
    }

    void store(ValuePtr ptr, ValuePtr v) {
        auto instr = std::make_shared<Instruction>(
            Instruction::Op::Store,
            vector<ValuePtr>{std::move(ptr), std::move(v)}, block);
        block->instrs.insert(block->instrs.begin() + pos, instr);
        pos++;
    }

    static ValuePtr constant(const Type &type, uint64_t k) {
        return std::make_shared<Value>(Constant{type, k});
    }

    // `idx` as a value of the word's type, which a shift by it needs.
    ValuePtr as_word(const ValuePtr &idx, const Type &word) {
        return equals(idx->get_type(), word)
                   ? idx
                   : make(word, Instruction::Op::Cast, {idx});
    }

    // Bit `idx` of the word behind `ref`, as a boolean.
    ValuePtr bit(const ValuePtr &ref, const ValuePtr &idx, const Type &word) {
        ValuePtr held = make(word, Instruction::Op::Load, {ref});
        ValuePtr shifted =
            make(word, Instruction::Op::Shr, {held, as_word(idx, word)});
        ValuePtr low = make(word, Instruction::Op::BwAnd,
                            {shifted, constant(word, 1)});
        return make(Bool_t::make(), Instruction::Op::Ne,
                    {low, constant(word, 0)});
    }

    // The low `n` bits of the word behind `ref`, as the vector of n
    // booleans the array read whole was.
    ValuePtr lanes(const ValuePtr &ref, uint32_t n, const Type &word) {
        ValuePtr held = make(word, Instruction::Op::Load, {ref});
        if (n != word.bits()) {
            held = make(UInt_t::make(n), Instruction::Op::Cast, {held});
        }
        return make(Vector_t::make(Bool_t::make(), n),
                    Instruction::Op::Reinterpret, {held});
    }

    // The vector of n booleans `v` as the word: its bits, widened.
    ValuePtr word_of(const ValuePtr &v, uint32_t n, const Type &word) {
        ValuePtr bits =
            make(UInt_t::make(n), Instruction::Op::Reinterpret, {v});
        if (n == word.bits()) {
            return bits;
        }
        return make(word, Instruction::Op::Cast, {bits});
    }
};

size_t position_of(const Block &block, const Instruction *instr) {
    for (size_t k = 0; k < block.instrs.size(); k++) {
        if (block.instrs[k].get() == instr) {
            return k;
        }
    }
    internal_error << "An instruction is not in its block: " << instr->name;
    return 0;
}

struct Candidate {
    shared_ptr<Block> home;
    shared_ptr<Instruction> alloca;
    uint32_t n;
};

// A use of the array at an index: a GEP (then written or read through) or
// an ExtractIdx (a read).
struct Access {
    shared_ptr<Block> block;
    shared_ptr<Instruction> instr;
};

// A store through a GEP of the array.
struct Write {
    shared_ptr<Block> block;
    shared_ptr<Instruction> store;
    shared_ptr<Instruction> gep;
};

// A load through a GEP of the array.
struct Read {
    shared_ptr<Block> block;
    shared_ptr<Instruction> load;
    shared_ptr<Instruction> gep;
};

// A store of the whole array: its initializer, `flags : mut array[bool, N]
// = {...}`, a Store of the array built element by element (MakeStruct) into
// the allocation itself.
struct Init {
    shared_ptr<Block> block;
    shared_ptr<Instruction> store;
};

// The elements of a whole-array value, where it is built from them.
std::optional<vector<ValuePtr>> elements_of(const ValuePtr &v, uint32_t n) {
    const Instruction *d = def_of(v);
    if (d == nullptr || d->op != Instruction::Op::MakeStruct ||
        d->operands.size() != n) {
        return std::nullopt;
    }
    for (const ValuePtr &e : d->operands) {
        if (!e->get_type().is<Bool_t>()) {
            return std::nullopt;
        }
    }
    return d->operands;
}

// Whether `instr` takes `alloca` as its first operand and something else
// as its second: the shape of a GEP or an ExtractIdx at an index.
bool indexes(const Instruction &instr, const Instruction *alloca) {
    return instr.operands.size() == 2 &&
           refers_to(instr.operands[0], alloca) &&
           !refers_to(instr.operands[1], alloca);
}

// The element type a store or a read of the array at `idx` has to have:
// the boolean, or the vector of all n for the whole array's ramp. Null for
// an index of another shape.
std::optional<Type> element_at(const ValuePtr &idx, uint32_t n) {
    if (is_full_ramp(idx, n)) {
        return Vector_t::make(Bool_t::make(), n);
    }
    if (idx->get_type().is<Vector_t>()) {
        return std::nullopt;
    }
    return Bool_t::make();
}

// Packs one array, or leaves it when a use is not one this rewrite knows.
bool pack(Function &func, const BlockMap &blocks,
          const vector<set<string>> &bound_bodies, const Candidate &c) {
    const Instruction *alloca = c.alloca.get();
    vector<Access> geps, extracts;
    vector<Init> inits;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr.get() == alloca) {
                continue;
            }
            const bool uses = std::any_of(
                instr->operands.begin(), instr->operands.end(),
                [&](const ValuePtr &v) { return refers_to(v, alloca); });
            if (!uses) {
                continue;
            }
            if (instr->op == Instruction::Op::Store &&
                instr->operands.size() == 2 &&
                refers_to(instr->operands[0], alloca) &&
                elements_of(instr->operands[1], c.n).has_value()) {
                inits.push_back({block, instr});
                continue;
            }
            if (!indexes(*instr, alloca)) {
                return false;
            }
            const std::optional<Type> element =
                element_at(instr->operands[1], c.n);
            if (!element.has_value()) {
                return false;
            }
            if (instr->op == Instruction::Op::GEP) {
                geps.push_back({block, instr});
            } else if (instr->op == Instruction::Op::ExtractIdx &&
                       equals(instr->type, *element)) {
                extracts.push_back({block, instr});
            } else {
                return false;
            }
        }
        // A terminator may pass the array along a jump to a block argument
        // of its name -- the builder's threading -- and nothing else.
        size_t passed = 0;
        for_each_value(block->terminator, [&](ValuePtr &v) {
            if (refers_to(v, alloca)) {
                passed++;
            }
        });
        size_t threaded = 0;
        for (Terminator::Jump *jump : jumps_of(*block)) {
            const auto target = blocks.find(jump->name);
            if (target == blocks.end()) {
                continue;
            }
            const vector<Argument> &args = target->second->args;
            const bool takes =
                std::any_of(args.begin(), args.end(), [&](const Argument &a) {
                    return a.name == alloca->name;
                });
            if (!takes) {
                continue;
            }
            for (const ValuePtr &v : jump->args) {
                if (refers_to(v, alloca)) {
                    threaded++;
                }
            }
        }
        if (passed != threaded) {
            return false;
        }
    }

    // Every GEP is written or read through, and goes nowhere else.
    vector<Write> writes;
    vector<Read> reads;
    for (const Access &g : geps) {
        const Instruction *gep = g.instr.get();
        const std::optional<Type> element = element_at(gep->operands[1], c.n);
        for (const auto &block : func.blocks) {
            for (const auto &instr : block->instrs) {
                const bool uses = std::any_of(
                    instr->operands.begin(), instr->operands.end(),
                    [&](const ValuePtr &v) { return def_of(v) == gep; });
                if (!uses) {
                    continue;
                }
                if (instr->op == Instruction::Op::Store &&
                    instr->operands.size() == 2 &&
                    def_of(instr->operands[0]) == gep &&
                    def_of(instr->operands[1]) != gep &&
                    equals(instr->operands[1]->get_type(), *element)) {
                    writes.push_back({block, instr, g.instr});
                } else if (instr->op == Instruction::Op::Load &&
                           instr->operands.size() == 1 &&
                           equals(instr->type, *element)) {
                    reads.push_back({block, instr, g.instr});
                } else {
                    return false;
                }
            }
            bool escapes = false;
            for_each_value(block->terminator, [&](ValuePtr &v) {
                if (def_of(v) == gep) {
                    escapes = true;
                }
            });
            for (const Argument &a : block->args) {
                if (a.name == gep->name) {
                    escapes = true;
                }
            }
            if (escapes) {
                return false;
            }
        }
    }

    // Written inside a parallel loop the array is allocated outside of, the
    // elements' stores are independent where the word's would race.
    const auto in_bound_loop_alone = [&](const Block &block) {
        for (const set<string> &body : bound_bodies) {
            if (body.contains(block.name) && !body.contains(c.home->name)) {
                return true;
            }
        }
        return false;
    };
    for (const Write &w : writes) {
        if (in_bound_loop_alone(*w.block)) {
            return false;
        }
    }

    // The array is a word from here on: the allocation, and the arguments
    // that thread it, are retyped in place, so that every reference to it
    // by name stays one.
    const Type word = UInt_t::make(c.n <= 32 ? 32 : 64);
    const Type word_ptr = Ptr_t::make(word);
    c.alloca->type = word_ptr;
    for (const auto &block : func.blocks) {
        for (Argument &a : block->args) {
            if (a.name == alloca->name) {
                a.type = word_ptr;
            }
        }
    }
    for_each_value(func, [&](ValuePtr &v) {
        if (const auto *a = std::get_if<Argument>(&v->data);
            a != nullptr && a->name == alloca->name) {
            Argument retyped = *a;
            retyped.type = word_ptr;
            v->data = std::move(retyped);
        }
    });
    {
        // Cleared where it is made: an element never written reads as
        // false, where the array's read as anything.
        Emitter e{func, c.home, position_of(*c.home, alloca) + 1};
        e.store(std::make_shared<Value>(c.alloca), Emitter::constant(word, 0));
    }
    for (const Init &init : inits) {
        const ValuePtr ref = init.store->operands[0];
        const vector<ValuePtr> elements =
            *elements_of(init.store->operands[1], c.n);
        Emitter e{func, init.block, position_of(*init.block, init.store.get())};
        ValuePtr fresh = Emitter::constant(word, 0);
        for (uint32_t k = 0; k < c.n; k++) {
            ValuePtr bit = e.make(word, Instruction::Op::Cast, {elements[k]});
            if (k != 0) {
                bit = e.make(word, Instruction::Op::Shl,
                             {bit, Emitter::constant(word, k)});
            }
            fresh = e.make(word, Instruction::Op::BwOr, {fresh, bit});
        }
        e.store(ref, fresh);
        init.block->instrs.erase(init.block->instrs.begin() + e.pos);
    }
    for (const Write &w : writes) {
        const ValuePtr ref = w.gep->operands[0];
        const ValuePtr idx = w.gep->operands[1];
        const ValuePtr v = w.store->operands[1];
        Emitter e{func, w.block, position_of(*w.block, w.store.get())};
        ValuePtr fresh;
        if (is_full_ramp(idx, c.n)) {
            fresh = e.word_of(v, c.n, word);
        } else {
            ValuePtr held = e.make(word, Instruction::Op::Load, {ref});
            ValuePtr i = e.as_word(idx, word);
            ValuePtr bit = e.make(word, Instruction::Op::Shl,
                                  {Emitter::constant(word, 1), i});
            ValuePtr cleared = e.make(
                word, Instruction::Op::BwAnd,
                {held, e.make(word, Instruction::Op::Not, {bit})});
            ValuePtr set = e.make(
                word, Instruction::Op::Shl,
                {e.make(word, Instruction::Op::Cast, {v}), i});
            fresh = e.make(word, Instruction::Op::BwOr, {cleared, set});
        }
        e.store(ref, fresh);
        // The store replaced, now right after what replaced it.
        w.block->instrs.erase(w.block->instrs.begin() + e.pos);
    }
    for (const Read &r : reads) {
        const ValuePtr ref = r.gep->operands[0];
        const ValuePtr idx = r.gep->operands[1];
        Emitter e{func, r.block, position_of(*r.block, r.load.get())};
        ValuePtr value = is_full_ramp(idx, c.n) ? e.lanes(ref, c.n, word)
                                                : e.bit(ref, idx, word);
        replace_uses(func, r.load.get(), value);
    }
    for (const Access &x : extracts) {
        const ValuePtr ref = x.instr->operands[0];
        const ValuePtr idx = x.instr->operands[1];
        Emitter e{func, x.block, position_of(*x.block, x.instr.get())};
        ValuePtr value = is_full_ramp(idx, c.n) ? e.lanes(ref, c.n, word)
                                                : e.bit(ref, idx, word);
        replace_uses(func, x.instr.get(), value);
    }
    for (const Access &g : geps) {
        internal_assert(!has_uses(func, g.instr.get()))
            << "A GEP of a packed array is still read: " << g.instr->name;
        g.block->instrs.erase(g.block->instrs.begin() +
                              position_of(*g.block, g.instr.get()));
    }
    return true;
}

} // namespace

size_t pack_bool_arrays(Function &func) {
    const BlockMap blocks = make_block_map(func);
    vector<set<string>> bound_bodies;
    for (const auto &block : func.blocks) {
        const auto *loop =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (loop != nullptr && loop->binding.has_value()) {
            bound_bodies.push_back(body_of(blocks, *block));
        }
    }
    vector<Candidate> candidates;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Alloca) {
                continue;
            }
            // The scalar boolean alone: Type::is_bool() is true of a vector
            // of them too, and an array of a gang's vectors -- the packet
            // traversal's per-lane tests -- is not this shape.
            const Array_t *array = instr->type.as<Array_t>();
            if (array == nullptr || !array->etype.is<Bool_t>()) {
                continue;
            }
            const std::optional<uint64_t> n =
                get_constant_value<uint64_t>(array->size);
            if (!n.has_value() || *n == 0 || *n > 64) {
                continue;
            }
            candidates.push_back({block, instr, uint32_t(*n)});
        }
    }
    size_t packed = 0;
    for (const Candidate &c : candidates) {
        if (pack(func, blocks, bound_bodies, c)) {
            packed++;
        }
    }
    return packed;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
