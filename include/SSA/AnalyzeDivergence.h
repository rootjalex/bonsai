#pragma once

#include "SSA/SSA.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// The result of the uniform/varying analysis over one ParFor body region: a
// value is *varying* (divergent) when the lanes of a gang may disagree about
// it, and *uniform* otherwise.
//
// A pointer has two variabilities, not one -- ispc's four kinds of pointer
// (Pharr & Mark, section 5.3). The pointer itself is varying when the lanes
// hold different addresses, which is a vector of pointers and a gather or
// scatter to go through; that is the ordinary sense of "varying" here and is
// recorded in `instrs`/`args` like any other value. Separately, a pointer the
// lanes share may point at memory of which the gang keeps one copy *per lane*
// -- ispc's `varying T * uniform`: a `mut` local written by each lane with
// its own value, whose address is then handed to a callee. Such a pointer is
// *uniform* -- one address -- but *points to varying*: a load through it gives
// each lane its own slot's value, and a store through it writes every lane's
// slot. Those are recorded in `pointee_instrs`/`pointee_args`.
//
// Values are keyed two ways because the SSA form uses block arguments in
// place of phis, and an argument's name is only unique within its block (both
// arms of an `if` may take an argument named `i`). Instructions are keyed by
// identity instead, since a block's instruction list owns them.
struct Divergence {
    // Instructions whose result is varying.
    std::set<const Instruction *> instrs;
    // Varying block arguments, as (block name, argument name).
    std::set<std::pair<std::string, std::string>> args;
    // Pointers -- uniform ones -- whose pointee has one slot per lane.
    std::set<const Instruction *> pointee_instrs;
    std::set<std::pair<std::string, std::string>> pointee_args;
    // Blocks whose Dispatch condition is varying: the lanes of a gang may
    // disagree about which successor to take, so this branch cannot survive
    // into vector code.
    std::set<std::string> branches;
    // Blocks that may execute with some lanes disabled, i.e. that are control
    // dependent -- transitively -- on a divergent branch. These are exactly
    // the blocks whose side effects have to be masked.
    std::set<std::string> masked;

    // For each loop header whose lanes may be at different iterations, why:
    // the latch found to leave divergently and the branch inside the loop
    // that decides it, or the seed that said so. Read by explain_varying.
    std::map<std::string, std::string> divergent_headers;

    // Which varying argument names are in scope in each block: its own, and
    // those of every block that dominates it.
    //
    // A block does not only refer to its own arguments. Linearization drops
    // the arguments it turns into blends, after which the blocks that used to
    // receive them refer to the dominating definition directly -- and a loop
    // header's arguments, which survive, are referred to that way from
    // everywhere in the loop.
    std::map<std::string, std::set<std::string>> in_scope;
    // The same, for the arguments in `pointee_args`.
    std::map<std::string, std::set<std::string>> pointee_in_scope;

    // Is `v`, as referenced from `block`, varying?
    bool is_varying(const std::string &block, const Value &v) const;
    // Is `v`, as referenced from `block`, a shared pointer to per-lane memory?
    bool points_to_varying(const std::string &block, const Value &v) const;
};

// Solves divergence over the region of `func` rooted at `entry`, seeded by
// the entry-block arguments named in `varying_seeds` (for a ParFor body, the
// loop index), by the instructions in `varying_instrs`, by the block
// arguments in `varying_args`, and by the entry-block pointer arguments named
// in `pointee_seeds`, which point at per-lane memory of the caller's.
//
// The second seed is for after linearization, when the index is no longer a
// block argument threaded through the region but a single instruction the
// whole region refers to. The third is for arguments of blocks other than the
// entry, which is what the live and exit masks of a uniformized loop are (see
// SSA/UniformizeLoops.h): nothing they are computed from is varying, and yet
// they hold one bool per lane by construction. The fourth is how a function
// specialized for a gang learns that a `mut` parameter is the gang's own
// per-lane local rather than one value the lanes share.
//
// `entry_mask` is the mask the whole region runs under, when it is a function
// specialized for a call made under one (see SSA/Vectorize.cpp). Within the
// region it is the set of lanes that are there at all, so a store or a call
// predicated by exactly it narrows nothing -- it is how linearization spells
// "every lane" here -- and it is not counted as varying by the rules below
// that ask whether a write is made by some lanes only. A mask computed from
// it inside the region is narrower and is counted.
//
// `masked_seeds` are blocks known to run with some lanes off for a reason
// control dependence cannot see: the blocks of a loop the lanes leave at
// different iterations, once SSA/UniformizeLoops.h has folded its exits into a
// live mask, since the only branch left deciding them is the uniform one on
// `any`. They are as masked as a block a divergent branch decides, and what
// follows from that -- the joins inside them, the writes and calls made in
// them -- follows here too.
//
// A value is varying if it is seeded, if any operand is varying, if it is
// loaded through a pointer to per-lane memory, or -- for a block argument --
// if the block is a join that some lanes may reach along one predecessor and
// others along another.
//
// A pointer points to varying memory if the memory is the gang's own -- a
// local allocation of the region, or a seeded parameter -- and any pointer
// derived from the same allocation (its fields, its elements, the block
// arguments that thread it) is written per lane: stored through with a
// varying value or under a mask, accumulated into likewise, or handed to a
// call that has a varying argument or is made under a mask, since such a call
// may write something different into each lane's slot. Pointers derived from
// one allocation stand or fall together: a struct whose field is per lane is
// per lane as a whole, and two pointers a block argument merges have to be
// laid out the same way for whatever follows the join. The address of a
// varying value points to varying memory too.
Divergence analyze_divergence(
    const Function &func, const std::string &entry,
    const std::set<std::string> &varying_seeds,
    const std::set<const Instruction *> &varying_instrs = {},
    const std::set<std::pair<std::string, std::string>> &varying_args = {},
    const std::set<std::string> &pointee_seeds = {},
    const std::shared_ptr<Value> &entry_mask = nullptr,
    const std::set<std::string> &masked_seeds = {});

// Prints to stderr why `v`, as referenced from `block`, is varying: the
// value, and beneath it the varying operands it is computed from, down to the
// arguments and seeds the divergence started at -- and for an argument, the
// blocks that declare it varying and what each of their predecessors passes
// for it. For reading a diagnostic: the assertion that linearization left
// nothing divergent, and BONSAI_EXPLAIN_LOOP in SSA/UniformizeLoops.cpp.
void explain_varying(const Function &func, const Divergence &div,
                     const std::string &block, const Value &v, int depth = 0);

} // namespace ssa
} // namespace ir
} // namespace bonsai
