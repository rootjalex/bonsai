#include "SSA/Contention.h"

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
bool depends_on(const std::shared_ptr<Value> &v,
                const std::shared_ptr<Value> &target,
                std::set<const void *> &seen) {
    if (!v) {
        return false;
    }
    if (same_value(v, target)) {
        return true;
    }
    const Instruction *instr = as_instruction(v);
    if (instr == nullptr) {
        // An argument that is not the target, or a constant. A block parameter
        // could in principle carry the index in from a predecessor; treating
        // that as independent is the unsafe direction, so callers only use
        // this to *keep* an atomic, never to drop one.
        return false;
    }
    if (!seen.insert(instr).second) {
        return false;
    }
    for (const auto &operand : instr->operands) {
        if (depends_on(operand, target, seen)) {
            return true;
        }
    }
    return false;
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
                [](const Terminator::Return &) {},
            },
            block.terminator.data);
    }
    return enclosing;
}

Contention contention_of(const std::shared_ptr<Value> &ptr,
                         const std::vector<ParallelLoop> &enclosing) {
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
        bool mentions = depends_on(address.base, loop.index_value, seen);
        for (const auto &index : address.indices) {
            if (mentions) {
                break;
            }
            seen.clear();
            mentions = depends_on(index, loop.index_value, seen);
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

        // Tier three: derived from the index by arithmetic -- `out[i / 2]`,
        // `out[permutation[i]]`. Whether that is injective over the loop's
        // iteration space is an arithmetic question this does not attempt.
        // An affine test would slot in here, using the bounds the loop carries
        // and the def chain reaching the subscript; anything non-affine stays
        // Unknown, which is the answer that costs speed rather than
        // correctness.
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
