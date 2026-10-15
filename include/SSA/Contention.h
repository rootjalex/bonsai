#pragma once

#include "SSA/Definitions.h"
#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

// Whether two iterations that run at the same time can write the same place.
//
// This is the question an atomic accumulate exists to answer. A `parfor`
// promises its iterations are independent, and an accumulate to a location two
// of them share is not -- so the program says `atomic`, and what that costs
// depends on whether the schedule actually parallelised the loop around it. An
// atomic under a loop nothing was bound to is a plain accumulate wearing a
// lock, and taking the lock off is this analysis's job.
//
// The general question is undecidable: an address can be any computed value,
// so deciding whether two of them ever coincide is deciding equality of
// arbitrary functions. Even restricted to affine subscripts it is integer
// linear feasibility, which is NP-complete in general. So this does not try to
// be complete -- it answers cheaply where it can and says so when it cannot,
// and a caller that cannot tell keeps the atomic. Under scheduling guarantees
// that permit reassociation, being unsure costs speed rather than correctness.
//
// What bonsai's design buys is the expensive half for free. In C one must
// *infer* which loops are parallel, which is what dependence analysis is for;
// here `parfor` plus a `bind` says so outright, and SSA makes what an index is
// derived from a walk back along def chains rather than a fixed point.

#include "SSA/SSA.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// What is known about the addresses two concurrent iterations write.
//
// Three-valued on purpose. "Cannot tell" is a different fact from "they
// collide", and keeping them apart is what lets a stronger test be added later
// without changing any caller: an affine solver would move cases out of
// Unknown and leave the other two alone. Collapsing to a bool now would bake
// in the conservatism.
enum class Contention {
    // No two concurrently running iterations write the same address, so an
    // atomic is unnecessary.
    Disjoint,
    // They provably do. The atomic is doing real work -- and had the program
    // not asked for one, this is a race worth reporting.
    Shared,
    // Neither could be shown. Keep the atomic.
    Unknown,
};

// One enclosing loop that the schedule made parallel.
//
// The bounds are carried even though the check below does not read them: an
// affine test needs the iteration space, and collecting it here means adding
// that test does not mean revisiting how the nest is walked. Same for the
// index's Value, which is what a def-chain walk compares against.
struct ParallelLoop {
    std::string index;
    std::shared_ptr<Value> index_value;
    std::shared_ptr<Value> start, end, stride;
    // The block the loop's body begins at: what an allocation has to be
    // inside to be each iteration's own (see contention_of).
    std::string body;
};

// The parallel loops enclosing each block of a function, outermost first.
//
// Only loops the schedule bound to hardware are here. An unbound parfor still
// says its iterations *may* run in any order, but nothing is going to run two
// of them at once, so nothing can collide.
std::map<std::string, std::vector<ParallelLoop>>
parallel_loops_by_block(const Function &f);

// Where a subscript that reaches the analysis as a block parameter came from.
// A pixel computed before a call and used after it arrives at the accumulate
// as an argument threaded through every block between, and an argument by
// itself says nothing; followed back through the jumps that pass it
// (SSA/Definitions.h) it is the instruction that computed it, or a loop's
// index, or a parameter of the function -- and the last two are independent
// of any loop but their own. Built once per function by a caller that wants
// the affine tier to see through parameters; without it a parameter is
// opaque, which is the answer that keeps the atomic.
struct Origins {
    explicit Origins(const Function &f);
    const Function &func;
    Definitions defs;
    // The first parameter of every parfor's body: the loops' indices.
    std::set<std::string> loop_indices;

    // The block where the name `name`, as `block` refers to it, is bound:
    // `block` itself when it takes the name as an argument, else the
    // nearest block dominating it that does, else empty. A block may name
    // a value from above without taking it -- a loop header made by
    // loopify() names the function's parameters it does not carry
    // (SSA/Rewrite.cpp) -- and the definitions analysis leaves such a name
    // at "unknown here" on purpose (defer() captures it for a continuation
    // that way; SSA/Simplify.cpp says so); this analysis wants the
    // parameter it is, so `resolve` goes on from the block that binds it.
    // The graph and its dominator tree are built on first use.
    std::string binding_block(const std::string &block, const std::string &name);

  private:
    std::optional<Cfg> cfg;
    DomTree dom;
};

// Whether the address `ptr` is written by at most one concurrent iteration.
//
// `ptr` is the pointer operand of an accumulate: a chain of GEPs from some
// base. `enclosing` is what the map above holds for the block it lives in;
// `block` names that block and `origins` the function's, for following a
// parameter of it back to its definition (both optional).
Contention contention_of(const std::shared_ptr<Value> &ptr,
                         const std::vector<ParallelLoop> &enclosing,
                         Origins *origins = nullptr,
                         const std::string &block = "");

// The functions that run inside a loop the schedule made parallel without a
// loop of their own to say so: whatever a bound loop's body calls, and
// whatever those call in turn. Each call of such a function is one
// iteration's, and two iterations running at once are two calls running at
// once -- so an accumulate inside it is under its callers' loops as surely as
// one written in the loop's body, though parallel_loops_by_block, which reads
// one function, lists none for it. A function reached from a bound loop only
// through another is here, and so is one called from a bound loop and from a
// sequential one too: one parallel caller is enough. Whole-program, since the
// answer is about the callers; computed once, before the per-function pass.
std::set<std::string> functions_called_in_parallel(
    const std::map<std::string, std::shared_ptr<Function>> &funcs);

// Whether `ptr`, the address an accumulate in `block` writes, is storage the
// function itself made: its base, followed back through the parameters that
// pass it along (origins), is an allocation of the function. Each call makes
// its own, so two calls running at once -- the iterations of a caller's bound
// loop -- write different ones whatever the subscripts, and only the
// function's own loops can make two writes to it collide, which
// parallel_loops_by_block sees. For a function called in parallel this is the
// one way an atomic can be shown unneeded from inside: an address that
// arrives as a parameter is some caller's storage, and whether two iterations
// hand down the same one -- the run's reducer, say -- is not visible here.
bool allocated_by_function(const std::shared_ptr<Value> &ptr, Origins &origins,
                           const std::string &block);

// The parameter of the function that `ptr`, the address an accumulate in
// `block` writes, is an address into: its base, followed back through the
// parameters that pass it along (origins), is the k-th parameter of the
// function's entry block. Nothing when the base is anything else.
std::optional<size_t> parameter_of_address(const std::shared_ptr<Value> &ptr,
                                           Origins &origins,
                                           const std::string &block);

// For each function some bound loop calls (functions_called_in_parallel) and
// each of its parameters that is an address: whether the calls that run at
// once hand it storage of their own. Two calls of such a function run at
// once when they are two iterations of a bound loop, directly or through
// the calls between, so the question is asked at every call site of the
// function over the whole program: an argument that is storage the
// iteration made for itself (an allocation inside the loop's body), an
// element at a subscript one-to-one in the iteration's index, or a record
// slot a drain's entry holds (Function::injective_slots), is Disjoint
// there; one word reaching every iteration from outside the loop -- the
// run's reducer handed down -- is Shared; a call made from a function that
// is itself called in parallel, with no loop of its own around the call,
// hands what its own call made (Disjoint) or what it was handed (that
// parameter's answer, which is why this is a fixed point over the call
// graph, started from Disjoint and lowered until it holds); and a call made
// where nothing runs at once constrains nothing. The worst answer over the
// sites is the parameter's, Shared below Unknown below Disjoint. Only a
// parameter found Disjoint lets an atomic accumulate into it come off
// inside the function (SSA/DemoteAtomics.h), and only when the function's
// own loops agree.
//
// This is what lets a reducer declared per iteration of the renderer's
// sample loop, handed down the calls of a path, keep plain accumulates at
// every level: each call's `l` is its sample's, and the analysis follows it
// back up the call graph to the allocation that says so.
std::map<std::string, std::vector<Contention>> parameter_contention(
    const std::map<std::string, std::shared_ptr<Function>> &funcs,
    const std::set<std::string> &called_in_parallel);

// The common case as a predicate, for callers that only want to know whether
// they may drop the atomic. Unknown counts as unsafe.
inline bool is_parallel_safe(const std::shared_ptr<Value> &ptr,
                             const std::vector<ParallelLoop> &enclosing) {
    return contention_of(ptr, enclosing) == Contention::Disjoint;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
