#include "SSA/Contention.h"
#include "SSA/Analysis.h"

#include "Error.h"

#include <set>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

const Instruction *as_instruction(const std::shared_ptr<Value> &v) {
    if (!v) {
        return nullptr;
    }
    if (const auto *i = std::get_if<std::shared_ptr<Instruction>>(&v->data)) {
        return i->get();
    }
    return nullptr;
}

// Whether two Values stand for the same thing.
//
// Instructions are shared, so pointer identity settles those. A block
// parameter is a name in a block, and the loop index reaching an accumulate
// has been threaded through however many blocks lie between, arriving as an
// argument of the same name each time -- so names are what identify those.
bool same_value(const std::shared_ptr<Value> &a,
                const std::shared_ptr<Value> &b) {
    if (!a || !b) {
        return false;
    }
    if (a == b) {
        return true;
    }
    const auto *ia = std::get_if<std::shared_ptr<Instruction>>(&a->data);
    const auto *ib = std::get_if<std::shared_ptr<Instruction>>(&b->data);
    if (ia && ib) {
        return ia->get() == ib->get();
    }
    const auto *aa = std::get_if<Argument>(&a->data);
    const auto *ab = std::get_if<Argument>(&b->data);
    if (aa && ab) {
        return aa->name == ab->name;
    }
    return false;
}

// Whether `v` is computed from `target` at all.
//
// This is the cheap half of the question, and the half SSA answers directly:
// an index that never mentions the parallel loop's index is the same address
// in every iteration, which settles contention in the other direction. Walking
// def chains backwards is a graph traversal here where in a dataflow IR it
// would be a fixed point.
// The value `v` stands for as `block` refers to it, followed back through the
// block parameters that pass it along (Origins), and the block that defines
// it there. Without origins a parameter is itself.
struct Resolved {
    std::shared_ptr<Value> value;
    std::string block;
};

Resolved resolve(const std::shared_ptr<Value> &v, const std::string &block,
                 Origins *origins) {
    if (origins == nullptr || !v || block.empty() ||
        !std::holds_alternative<Argument>(v->data)) {
        return {v, block};
    }
    const Definition d = origins->defs.of(block, v);
    return {d.value ? d.value : v, d.value ? d.block : block};
}

std::string block_of(const Instruction *instr, const std::string &fallback) {
    if (const auto owner = instr->owner.lock()) {
        return owner->name;
    }
    return fallback;
}

bool depends_on(const std::shared_ptr<Value> &v,
                const std::shared_ptr<Value> &target,
                std::set<const void *> &seen, const std::string &block = "",
                Origins *origins = nullptr) {
    if (!v) {
        return false;
    }
    if (same_value(v, target)) {
        return true;
    }
    const Instruction *instr = as_instruction(v);
    if (instr == nullptr) {
        // An argument that is not the target, or a constant. A block
        // parameter may carry the index in from a predecessor: followed back
        // to what it was threaded from when the caller gave the means
        // (Origins); left as independent otherwise, which is the unsafe
        // direction, so callers without origins only use this to *keep* an
        // atomic, never to drop one.
        const Resolved r = resolve(v, block, origins);
        if (r.value != v) {
            if (same_value(r.value, target)) {
                return true;
            }
            if (as_instruction(r.value) != nullptr) {
                return depends_on(r.value, target, seen, r.block, origins);
            }
        }
        return false;
    }
    if (!seen.insert(instr).second) {
        return false;
    }
    const std::string here = block_of(instr, block);
    for (const auto &operand : instr->operands) {
        if (depends_on(operand, target, seen, here, origins)) {
            return true;
        }
    }
    return false;
}

// Whether `v`, as `block` refers to it, is independent of `target`: a
// constant; an instruction over independent operands; a parameter that
// resolves to one of those, to another loop's index, or to a parameter of
// the function. A parameter that resolves to a merge of different values is
// not shown independent, which is the direction that keeps the atomic.
bool free_of(const std::shared_ptr<Value> &v, const std::shared_ptr<Value> &target,
             const std::string &block, Origins *origins,
             std::set<const void *> &seen) {
    if (!v || same_value(v, target)) {
        return false;
    }
    if (std::holds_alternative<Constant>(v->data)) {
        return true;
    }
    if (const Instruction *instr = as_instruction(v)) {
        if (!seen.insert(instr).second) {
            return true;
        }
        const std::string here = block_of(instr, block);
        for (const auto &operand : instr->operands) {
            if (!free_of(operand, target, here, origins, seen)) {
                return false;
            }
        }
        return true;
    }
    // A block parameter.
    if (origins == nullptr) {
        return false;
    }
    const Resolved r = resolve(v, block, origins);
    if (r.value != v && as_instruction(r.value) != nullptr) {
        return free_of(r.value, target, r.block, origins, seen);
    }
    if (std::holds_alternative<Constant>(r.value->data)) {
        return true;
    }
    const auto *a = std::get_if<Argument>(&r.value->data);
    if (a == nullptr || same_value(r.value, target)) {
        return false;
    }
    if (origins->loop_indices.contains(a->name)) {
        return true; // another loop's index
    }
    return r.block == origins->func.blocks.front()->name; // a parameter
}

// A non-zero integer constant, if `v` is one.
bool nonzero_constant(const std::shared_ptr<Value> &v) {
    if (!v) {
        return false;
    }
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return false;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return *i != 0;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u != 0;
    }
    return false;
}

// Whether `v` is an injective function of `target` over the integers: the
// index itself; a sum with, or difference by, something the index does not
// reach (`p_band + p_in`, `i - start`); a product or left shift by a
// non-zero constant (`i * 4`, `i << 2`); a cast of one to a type at least as
// wide. Distinct iterations of the loop then give distinct values, which is
// what makes the elements they subscript distinct. Wrapping is not
// considered: an index that overflows its type addresses nothing sensible.
bool injective_in(const std::shared_ptr<Value> &v,
                  const std::shared_ptr<Value> &target, const std::string &block,
                  Origins *origins) {
    if (same_value(v, target)) {
        return true;
    }
    const Instruction *instr = as_instruction(v);
    if (instr == nullptr) {
        // A block parameter: what it was threaded from, if that can be found.
        const Resolved r = resolve(v, block, origins);
        if (r.value != v && as_instruction(r.value) != nullptr) {
            return injective_in(r.value, target, r.block, origins);
        }
        return false;
    }
    if (instr->operands.empty()) {
        return false;
    }
    const std::string here = block_of(instr, block);
    const auto independent = [&](const std::shared_ptr<Value> &x) {
        std::set<const void *> seen;
        return free_of(x, target, here, origins, seen);
    };
    const auto injective = [&](const std::shared_ptr<Value> &x) {
        return injective_in(x, target, here, origins);
    };
    const auto &ops = instr->operands;
    switch (instr->op) {
    case Instruction::Op::Add:
        return ops.size() == 2 && ((injective(ops[0]) && independent(ops[1])) ||
                                   (independent(ops[0]) && injective(ops[1])));
    case Instruction::Op::Sub:
        return ops.size() == 2 && injective(ops[0]) && independent(ops[1]);
    case Instruction::Op::Mul:
        return ops.size() == 2 && ((injective(ops[0]) && nonzero_constant(ops[1])) ||
                                   (nonzero_constant(ops[0]) && injective(ops[1])));
    case Instruction::Op::Shl:
        return ops.size() == 2 && injective(ops[0]) &&
               std::get_if<Constant>(&ops[1]->data) != nullptr;
    case Instruction::Op::Cast: {
        const Type from = ops[0]->get_type();
        const Type to = instr->type;
        return ops.size() == 1 && from.is_int() && to.is_int() &&
               to.bytes() >= from.bytes() && injective(ops[0]);
    }
    default:
        return false;
    }
}

// The index operands of the GEP chain that produced `ptr`, outermost first,
// along with the base the chain started from.
struct Address {
    std::shared_ptr<Value> base;
    std::vector<std::shared_ptr<Value>> indices;
};

Address address_of(const std::shared_ptr<Value> &ptr) {
    Address address;
    std::shared_ptr<Value> cursor = ptr;
    while (true) {
        const Instruction *instr = as_instruction(cursor);
        if (instr == nullptr || instr->op != Instruction::Op::GEP ||
            instr->operands.size() != 2) {
            address.base = cursor;
            break;
        }
        address.indices.push_back(instr->operands[1]);
        cursor = instr->operands[0];
    }
    // Collected innermost first while walking back up the chain.
    std::reverse(address.indices.begin(), address.indices.end());
    return address;
}

// Whether `base`, followed back through the parameters that pass it along
// (Origins), is an allocation made by a block of `loop`'s body: storage each
// iteration of that loop makes for itself.
bool allocated_inside(const std::shared_ptr<Value> &base,
                      const ParallelLoop &loop, const std::string &block,
                      Origins *origins) {
    if (loop.body.empty()) {
        return false;
    }
    const Resolved r = resolve(base, block, origins);
    const Instruction *instr = as_instruction(r.value);
    if (instr == nullptr || (instr->op != Instruction::Op::Alloca &&
                             instr->op != Instruction::Op::Alloc)) {
        return false;
    }
    const auto owner = instr->owner.lock();
    if (!owner) {
        return false;
    }
    const Cfg body(origins->func, loop.body);
    return body.find(owner->name) != NO_BLOCK;
}

} // namespace

std::map<std::string, std::vector<ParallelLoop>>
parallel_loops_by_block(const Function &f) {
    std::map<std::string, std::vector<ParallelLoop>> enclosing;
    if (f.blocks.empty()) {
        return enclosing;
    }

    std::map<std::string, std::shared_ptr<Block>> by_name;
    for (const auto &block : f.blocks) {
        by_name[block->name] = block;
    }

    // The nest is walked rather than inferred: a ParFor terminator names the
    // block its body starts at and the block that follows the loop, so
    // descending into the first with the loop pushed and into the second
    // without it gives every block the nest it sits in. This is the structure
    // the source had, still visible because the SSA kept it.
    std::vector<std::pair<std::string, std::vector<ParallelLoop>>> worklist;
    worklist.emplace_back(f.blocks.front()->name,
                          std::vector<ParallelLoop>{});

    while (!worklist.empty()) {
        auto [name, stack] = std::move(worklist.back());
        worklist.pop_back();

        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            continue;
        }
        // A block reached twice keeps the nest it was first given. The nest is
        // a property of where a block sits in the source's loop structure, and
        // that does not depend on which edge arrived.
        if (!enclosing.emplace(name, stack).second) {
            continue;
        }

        const Block &block = *it->second;
        std::visit(
            overloads{
                [](const std::monostate &) {},
                [&](const Terminator::Jump &jump) {
                    worklist.emplace_back(jump.name, stack);
                },
                [&](const Terminator::Dispatch &dispatch) {
                    for (const auto &target : dispatch.targets) {
                        worklist.emplace_back(target.name, stack);
                    }
                },
                [&](const Terminator::ParFor &parfor) {
                    std::vector<ParallelLoop> inner = stack;
                    // Only a bound loop actually runs two iterations at once.
                    // An unbound parfor permits any order, which includes one
                    // after another, and nothing can collide with itself.
                    if (parfor.binding.has_value()) {
                        ParallelLoop loop;
                        loop.index = parfor.index;
                        loop.start = parfor.start;
                        loop.end = parfor.end;
                        loop.stride = parfor.stride;
                        loop.body = parfor.body.name;
                        // The body block takes the varying index as its first
                        // parameter; that Value is what an address is compared
                        // against.
                        const auto body = by_name.find(parfor.body.name);
                        if (body != by_name.end() &&
                            !body->second->args.empty()) {
                            loop.index_value = std::make_shared<Value>(
                                body->second->args.front());
                        }
                        inner.push_back(std::move(loop));
                    }
                    worklist.emplace_back(parfor.body.name, std::move(inner));
                    worklist.emplace_back(parfor.cont.name, stack);
                },
                [](const Terminator::Yield &) {
                    // Ends the body of a loop; the block after it was already
                    // reached through that loop's `cont`.
                },
                [&](const Terminator::Call &call) {
                    worklist.emplace_back(call.cont.name, stack);
                },
                [&](const Terminator::MultiCall &call) {
                    worklist.emplace_back(call.cont.name, stack);
                },
                [](const Terminator::Return &) {},
            },
            block.terminator.data);
    }
    return enclosing;
}

Origins::Origins(const Function &f) : func(f), defs(f, /*lenient=*/true) {
    std::map<std::string, std::shared_ptr<Block>> by_name;
    for (const auto &block : f.blocks) {
        by_name[block->name] = block;
    }
    for (const auto &block : f.blocks) {
        const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (p == nullptr) {
            continue;
        }
        const auto body = by_name.find(p->body.name);
        if (body != by_name.end() && !body->second->args.empty()) {
            loop_indices.insert(body->second->args.front().name);
        }
    }
}

Contention contention_of(const std::shared_ptr<Value> &ptr,
                         const std::vector<ParallelLoop> &enclosing,
                         Origins *origins, const std::string &block) {
    if (enclosing.empty()) {
        // Nothing runs concurrently, so nothing collides.
        return Contention::Disjoint;
    }

    const Address address = address_of(ptr);

    bool all_disjoint = true;
    bool any_shared = false;
    for (const ParallelLoop &loop : enclosing) {
        if (!loop.index_value) {
            // The loop's index could not be found, so nothing can be said
            // about an address in terms of it.
            all_disjoint = false;
            continue;
        }

        // Tier zero, the iteration's own storage: a base that is an
        // allocation made inside this loop's body is a fresh slot per
        // iteration -- a `mut` local of the body, the sums a point's walks
        // add to -- which no other iteration of this loop can reach,
        // whatever the subscripts. Without origins the function is not at
        // hand to ask where the allocation sits, and the base is treated as
        // any other, which keeps the atomic.
        if (origins != nullptr &&
            allocated_inside(address.base, loop, block, origins)) {
            continue;
        }

        // Tier one, syntactic: the index *is* one of the subscripts, so
        // distinct iterations land on distinct elements.
        bool is_subscript = false;
        for (const auto &index : address.indices) {
            if (same_value(index, loop.index_value)) {
                is_subscript = true;
                break;
            }
        }
        if (is_subscript) {
            continue;
        }

        // Tier two, provenance: no part of the address mentions the index, so
        // every iteration of this loop computes the same address. That is the
        // definite answer in the other direction, and worth distinguishing --
        // it is the case where an accumulate without `atomic` is a race rather
        // than merely unproven.
        std::set<const void *> seen;
        bool mentions =
            depends_on(address.base, loop.index_value, seen, block, origins);
        for (const auto &index : address.indices) {
            if (mentions) {
                break;
            }
            seen.clear();
            mentions = depends_on(index, loop.index_value, seen, block, origins);
        }
        if (!mentions) {
            // Every iteration of this loop lands on the same address. Note
            // that this clears all_disjoint as well as recording the sharing:
            // one loop's iterations colliding is enough, whatever the others
            // do.
            all_disjoint = false;
            any_shared = true;
            continue;
        }

        // Tier three, affine: a subscript that is the index moved by
        // something the index does not reach -- `p_band + p_in`, the pixel
        // of a band's continuation; `i * 4 + c` -- takes distinct values on
        // distinct iterations, so the elements are distinct too (see
        // injective_in). What else is derived from the index by arithmetic
        // -- `out[i / 2]`, `out[permutation[i]]` -- stays Unknown, which is
        // the answer that costs speed rather than correctness.
        bool affine = false;
        for (const auto &index : address.indices) {
            if (injective_in(index, loop.index_value, block, origins)) {
                affine = true;
                break;
            }
        }
        if (affine) {
            continue;
        }
        all_disjoint = false;
    }

    if (all_disjoint) {
        return Contention::Disjoint;
    }
    if (any_shared) {
        return Contention::Shared;
    }
    return Contention::Unknown;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
