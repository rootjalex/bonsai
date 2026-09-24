#pragma once

#include "IR/Program.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// How a queue stores its entries. The entry `Entry_<q>` is the continuation
// as a value -- the deferred call's arguments -- and the queue `Queue_<q>`
// keeps it as a struct of arrays behind a count, one array per scalar of
// the entry that it stores. An entry field that is an aggregate is taken
// down to its scalars, a struct to its fields and a short vector to its
// components, so that a gang of entries reads each scalar of its
// continuations as one dense vector load and the compacting push writes
// each with one compress-store; and so that the queue reads as pbrt's
// wavefront queues do, whose `SOA<Ray>` holds an `SOA<Point3f> o` that
// holds `float *x, *y, *z` (pbrt's `soac`). Each scalar is a Leaf, named
// after the entry's field and the path down to it, in the order the push
// takes an entry apart and the drain puts one back together.
//
// A leaf that is not stored is one the drain can supply without reading
// it back: the padding an ADT's storage puts between its tag and its
// payload, which is zero at every construction (Lower/ADTs.cpp) and read by
// nothing. And a queue of a split (QueueSpec::split) whose callee copy
// never reads a parameter -- the material kernel for a diffuse surface
// reads no previous-hit context -- leaves that parameter's leaves out of
// its own arrays (`unread`, by queue of the split then by leaf; the drain
// hands the copy nothing there), while the queue type, shared by the
// split, keeps the array's place. What is left out is decided here, where
// the entry is made, and recorded by the queue type's name for
// lower_pushes(), which meets the push after every directive has run. A
// layout for the queue that a schedule asks for may replace this; nothing
// here is the layout language's promise.
struct QueueLayout {
    struct Leaf {
        std::string name;
        Type type;
        enum class Kind {
            Stored, // in the queue's array number `array`
            Pad,    // an ADT's padding: zero, stored nowhere
        } kind = Kind::Stored;
        size_t array = 0; // among the queue's arrays, for a stored leaf
    };
    Type entry;
    Type queue;
    std::vector<Leaf> leaves;
    // unread[q][l]: leaf l is not stored in queue q of the split (queue 0
    // when there is no split), its callee never reading it. Empty when
    // every leaf is read everywhere.
    std::vector<std::vector<bool>> unread;
};
// The layouts of a program's queues, by the queue type's name.
using QueueLayouts = std::map<std::string, QueueLayout>;

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
    // Whether the owner's own call into the chain -- the producer's -- is a
    // push of the initial entry rather than a call that runs the first step:
    // the schedule wrote `owner.defer(callee, q)` beside `callee.defer(callee,
    // q)`, and every step then runs from the drain, as pbrt's camera rays go
    // through the ray queue and are traced by nothing else (see the Convert
    // pass, which folds the two directives into one deferral).
    bool initial_push = false;
    // Chain functions other than the callee whose calls to the callee are
    // pushes too: `f.defer(callee, q)` beside `callee.defer(callee, q)`, for
    // the function that makes the recursion's first call when that is not
    // the owner -- `li_vol_path.defer(vol_path_step, rays)`, where the owner
    // `render` reaches the step through `li` and `li_vol_path` -- and for a
    // function the callee calls that calls the callee back: pbrt's material
    // kernel, handed the hit by the trace kernel and handing the next ray
    // back to it (`vol_surface.defer(vol_path_step, rays)`). The two are a
    // recursion between two functions, and deferring both of its ends is
    // what breaks it: once the calls back are pushes, the chain from the
    // owner to the callee has no recursion in it but the callee's own.
    std::vector<std::string> also_from;
    // Whether running an entry can push onto this queue: whether the drain's
    // callee reaches one of the queue's pushers along calls that stay calls
    // once every directive is applied (see the Convert pass, which walks the
    // call graph less the deferred edges). True for a self-recursion the
    // drain still makes, and for a scattering that runs inside the trace's
    // drain and hands the turned ray back. False when every push comes from
    // another queue's drain, after this one's pass -- rays to hits to rays,
    // the medium scatter a queue of its own -- so that nothing writes this
    // queue while it is read, and one buffer, its count read for the pass
    // and reset before it, serves every round in place of the two a
    // self-feeding queue has.
    bool drain_pushes_self = true;
    // The queue split by a value into one queue per variant, and those
    // split further: `hits.specialize(isect); hits[Some].specialize(
    // material);` is a split on `isect` whose `Some` is split on `material`
    // (ir::QueueSpecialize). `key` names a value of the drained function --
    // an instruction, by the program's name for it, or a parameter -- of
    // variant type (an ADT, or an optional with variants None and Some).
    // The leaves of the tree are the queues: each has storage of its own,
    // named by its path (`hits!Some!Diffuse`), a drain loop of that name,
    // and a copy of the drained function in which every key on the path has
    // its variant's tag (SSA/Specialize.h); a push computes the keys from
    // the entry and goes to the leaf they select. Only for a queue drained
    // in one pass so far.
    struct Split {
        std::string key;
        std::map<std::string, Split> under;
    };
    std::optional<Split> split;
    // How the program's variant types are stored, for the split's keys and
    // for the padding the entry leaves out.
    const std::map<std::string, ir::Program::AdtStorage> *adt_storages = nullptr;
    // Where the queue's layout is recorded once made, for lower_pushes().
    QueueLayouts *layouts = nullptr;
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
// successors go to the other. When the successors are pushed by another
// queue's drain instead, after this one's pass is over -- the recursion's
// call is in the staged rest of the callee, so the cycle is rays to hits to
// rays -- there is one, emptied once its count is read for the pass and
// filled again by the other drain, round after round. When the drain's call
// cannot reach a push onto this queue at all -- the deferred call is to a
// function off the chain, or in a pipeline of stages each pushing onto the
// next -- one queue and one pass over it are all there is. The generated
// code is meant to read as pbrt's wavefront path tracer does, or to do less.
//
// What an entry holds is the continuation of the deferred call, by value.
// First the call's arguments -- not every one. An argument that is one value
// for the whole owner iteration, passed down unchanged from the owner (the
// camera, the scene, the depth limit) and, for a recursion, passed through
// untouched by every recursive call, is in scope where the drain runs and is
// not stored. A pointer argument that points at a mutable local of the
// producer's iteration -- a sampler's state, a surface record the callee
// fills in -- is stored as the address of the local's slot in the record
// (below). Everything else is stored as itself. Each field is named by the
// parameter it holds. This is a plain analysis, not an optimal one: a value
// is left out when it can be seen to be one value at every call on the way
// down, and stored otherwise.
//
// The record is pbrt's `pixelSampleState`: one slot per producer iteration,
// in an array per value beside the queues, for what the iteration owns and
// the path reads or writes -- its reducer locals (`L`), the mutable locals it
// hands the chain (the sampler's state, the visible surface), the values the
// rest of the iteration after the call still needs (a sample's filter weight,
// its wavelengths), and the deferred call's value when that rest uses it.
// The locals are given their slots' addresses in place; the values are
// written before the call; the call's value is written where the path ends.
// An entry then carries a slot's address and nothing of the frame, as pbrt's
// work items carry `pixelIndex`. And the rest of the producer's iteration --
// the film write -- leaves the producer for good: it runs once per iteration
// in a pass over the record after the drain (`<queue>_rest`, pbrt's
// UpdateFilm), with the record's values and the slots' addresses, and what
// is in scope. So no drain runs any continuation, and nothing comes back up
// the chain: a chain function returns nothing.
//
// The push is made by the callee -- the function that has the arguments in
// hand -- at the deferred call site, onto the queue it is handed as an extra
// parameter, threaded down the chain of calls from the owner, with the path's
// result slot beside it when the rest uses the call's value. The call site
// becomes the push and a return; a return that ends the path writes the
// path's value into its slot and returns; a tail call of one chain function
// by another drops the value it passed back. In the literature the entry is
// a defunctionalized continuation reduced to a reference into a store of
// frames (Reynolds 1972; the "pixel state" of every wavefront renderer since
// Laine et al. 2013), and the pass is the apply of the one continuation
// every path shares.
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
//     parfor i in <the producer loop>:
//         <the record's values of iteration i> = <the values the rest needs>
//         callee(<the call's arguments>, &queue[0], &result[i])
//     round = 0
//     while queue[round & 1].count > 0:
//         cur = round & 1; nxt = cur ^ 1
//         queue[nxt].count = 0
//         parfor <queue name> in 0 : queue[cur].count:
//             e = <each field of the entry, from queue[cur].<scalar>[<queue name>]>
//             callee(<e's fields, and the arguments in scope>, &queue[nxt], e._result)
//         round = round + 1
//     parfor <queue name>_rest in 0 : <the producer's count>:
//         <the rest of the producer's iteration, with result[i] and the record's values>
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
//     only the producer's own rest is kept, in the record.
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
// struct types it made -- the entry and the queue -- for the program to
// declare.
std::vector<Type> defer(FuncMap &funcs, const std::string &func,
                        const std::string &callee, const QueueSpec &queue);

// The functions `f` calls, by name: the call graph's edges out of `f`.
std::set<std::string> callees_of(const Function &f);

// Replaces every Push with what it stands for: a fetch-and-add of the queue's
// count, atomic when the push is, and a store of each stored scalar of the
// entry (QueueLayout) into its array at the slot that claimed. A gang's push
// -- one whose value is a slot per lane -- adds the number of lanes that
// push and scatters their entries to the slots at their ranks. Run once the
// schedule is applied, right before code generation, since until then a
// push has to stay one instruction for the vectorizer to recognize (see
// Instruction::Op::Push).
void lower_pushes(Function &func, const QueueLayouts &layouts);

} // namespace ssa
} // namespace ir
} // namespace bonsai
