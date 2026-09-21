#pragma once

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// A queue as the schedule declared it (ir::Queue), with its owner resolved:
// `paths = render.queue(p)` is owner `render`, loop `p`; `.queue(root)` is
// the function itself.
struct QueueSpec {
    std::string name;
    std::string owner;
    std::string loop;
    // A constant capacity the schedule gave, if any. Without one the size is
    // inferred (see below).
    std::optional<uint64_t> capacity;
};

// Turns the calls of `callee` inside `func` into entries on a queue, and gives
// the queue's owner a loop that drains it: `f.defer(callee, q)`.
//
// A recursion's pending calls can be run in two orders. loopify() runs them
// depth first -- a tail call becomes a back edge, a branching call an entry on
// a stack. This runs them breadth first: the call becomes an entry on a
// queue, and the loop that runs the entries is the drain. Both replace a call
// with a representation of the call that has not happened yet, and the
// difference is only the container. In the literature this is
// defunctionalization of the continuation (Reynolds, "Definitional
// Interpreters for Higher-Order Programming Languages", 1972; Danvy and
// Nielsen, "Defunctionalization at Work", PPDP 2001): the entry is the data a
// continuation closes over, and the drain is the `apply` function. A path
// tracer run this way is a wavefront (Laine, Karras and Aila, "Megakernels
// Considered Harmful", HPG 2013; Lee et al., "Vectorized Production Path
// Tracing", HPG 2017), which is what this is for: `full_path_step.defer(
// full_path_step, paths)` makes every path in flight an entry, so that the
// drain's parfor -- named `paths`, so a later directive can split, vectorize
// or bind it -- advances them together.
//
// A queue is logically an array of continuations with a fill count. It is
// stored as a struct of arrays: `Queue_<name> { count : u32; <scalar> :
// T[]; ... }`, the count and a handle to an array per scalar of the entry
// -- an entry's aggregates taken down to their scalars, a `ray.o` to
// `ray_o_x`, `ray_o_y`, `ray_o_z`, as pbrt's wavefront `SOA` structs are --
// each sized by the producer count, all `mut` locals of the owner made once
// per owner iteration and named after the queue and the scalar
// (`paths_queue`, `paths_ray_o_x`). That is what lets a gang of entries read
// each scalar of its continuations as one dense vector load and push each
// with one compress-store. The layout language may later be given a say in
// it; nothing here promises more than the default.
//
// Whether there is one such queue or two follows from the queue graph. When
// running an entry can push onto the queue it came from -- the deferred call
// is a self-recursion -- there are two, as pbrt's wavefront integrator has
// two ray queues (`rayQueues[depth & 1]`): round r reads queue r & 1 and its
// successors go to the other. When the drain's call cannot reach a push
// onto this queue -- the deferred call is to a function off the chain, or in
// a pipeline of stages each pushing onto the next -- one queue and one pass
// over it are all there is. The generated code is meant to read as pbrt's
// wavefront path tracer does, or to do less.
//
// What an entry holds is the continuation of the deferred call, by value.
// First the call's arguments -- not every one. An argument that is one value
// for the whole owner iteration, passed down unchanged from the owner (the
// camera, the scene, the depth limit) and, for a recursion, passed through
// untouched by every recursive call, is in scope where the drain runs and is
// not stored. A pointer argument that points at a mutable local of the
// producer's iteration -- a sampler's state, a surface record the callee
// fills in -- is stored as what it points at: the callee stores the contents
// when it pushes, and the drain gives the entry a local of its own to run
// with. Everything else is stored as itself. Then the producer's own part:
// the values of its iteration that the rest of the iteration, after the
// call, still needs -- a sample's filter weight, its wavelengths, the record
// the callee filled -- which the producer writes into the entry once the call
// returns saying it saved, and which the drain hands the copy of that rest
// when the entry finishes. Each field is named by the parameter or the value
// it holds, and a frame value that is the same value as a stored argument
// shares the argument's field. This is a plain analysis, not an optimal one:
// a value is left out when it can be seen to be one value at every call on
// the way down, and stored otherwise.
//
// The push is made by the callee -- the function that has the arguments in
// hand -- at the deferred call site, onto the queue it is handed as an extra
// parameter, threaded down the chain of calls from the owner. The call site
// becomes the push and a return that says "saved", with the slot it saved to
// when the frame has something to add there, in place of a value; every
// function on the chain returns that beside its value, and the owner acts on
// it: "if the return says we saved state, save this data in the queue
// location, otherwise act as normal".
//
// The owner runs the drain right after the loop that produced the entries --
// the producer loop, a parfor between the owner's loop and the call -- and
// before anything that follows it, so that everything the program ordered
// after the producer still comes after the deferred calls have run. What is
// reordered is only the producer's iterations against one another, which a
// parfor permits and a sequential loop would not; the producer being a parfor
// is what makes the deferral legal, and a `for` between the owner and the
// call is refused. For a self-recursion the drain is
//
//     round = 0
//     while queue[round & 1].count > 0:
//         cur = round & 1; nxt = cur ^ 1
//         queue[nxt].count = 0
//         parfor <queue name> in 0 : queue[cur].count:
//             e = <each field of the entry, from queue[cur].<scalar>[<queue name>]>
//             r = callee(<e's fields, and the arguments in scope>, &queue[nxt])
//             if r.saved: <the frame's scalars of queue[nxt].<scalar>[r.slot]> = <e's>
//             else:       <the rest of the producer's iteration, with r.value>
//         round = round + 1
//
// a round per bounce. Each queue holds the number of entries an owner
// iteration can push, one per iteration of the producer loop, since a
// deferred call is in tail position and each entry's run pushes at most one
// successor: a runtime constant, the producer's trip count, with no bound
// checks and no allocator. A capacity the schedule gives is checked against
// the count where both are constants and refused where the count is not,
// since an overflow would be silent; flushing a full queue by draining it
// before the write (MoonRay's policy) is what a capacity below the count
// would need, and is not built.
//
// Values the drain needs that the owner computes inside the producer's
// iteration -- the scene read through a parameter, a scale from the sample
// count, the depth limit a chain function unwraps from the integrator -- are
// moved: a copy of the computation, when it is recomputable from what is
// available before the loop, goes before the loop and is what the drain and
// the entry analysis use. Loop-invariant code motion, for exactly the values
// the deferral asks about; nothing is traded off against storing, since a
// moved value is computed once per owner iteration rather than once per
// entry. BONSAI_EXPLAIN_DEFER=1 prints what the entry holds and why.
//
// Not supported yet, each refused with an error rather than approximated:
//
//   * A deferred call that is not in tail position, or a frame between it and
//     the owner that does more than return the callee's value: the work after
//     the call is state on the call stack that would have to be saved, and
//     only the producer's own frame saves any.
//   * A branching recursion, where a step makes several deferred calls (a tree
//     traversal): the number of entries in flight would grow, and the queue
//     could not be sized by the producer count. Linear deferral only.
//   * A pointer among the stored arguments that is not a mutable local of the
//     producer's iteration: it may point into a frame that has returned by
//     the time the entry runs.
//   * A function on the chain that is also called from outside it, or a chain
//     that is itself recursive apart from the deferred call; and a queue
//     whose producer is not one call, or one call inside one parfor, of the
//     owner's iteration.
//
// The drain is a parfor like any other, and a schedule that then writes
// `owner.split(<queue>, gang, lane, 16, true).vectorize(lane)` runs sixteen
// entries as one gang -- the packet schedule's gang made from a queue rather
// than from a pixel's samples, and what a wavefront is for. The push inside
// the gang is then made by some lanes and not others, and compacts: the
// count advances once by the number of lanes that push, and each takes the
// slot at its rank among them (Instruction::Op::Rank), so the next round's
// queue is dense. The split's tail is what lets a batch of any size be
// drained by whole gangs, the last one partly full.
//
// `func` is the function the calls are in, `callee` the function they call
// (the same function, for a recursion), and `queue` the queue. Returns the
// struct types it made -- the entry, the queue, the flag -- for the program
// to declare.
std::vector<Type> defer(FuncMap &funcs, const std::string &func,
                        const std::string &callee, const QueueSpec &queue);

// Replaces every Push with what it stands for: a fetch-and-add of the queue's
// count, atomic when the push is, and a store of the entry into the slot that
// claimed. A gang's push -- one whose value is a slot per lane -- adds the
// number of lanes that push and scatters their entries to the slots at their
// ranks. Run once the schedule is applied, right before code generation,
// since until then a push has to stay one instruction for the vectorizer to
// recognize (see Instruction::Op::Push).
void lower_pushes(Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
