#pragma once

#include <map>
#include <string>
#include <variant>

#include "Layout.h"
#include "Provenance.h"
#include "Resource.h"
#include "Type.h"

namespace bonsai {
namespace ir {

// Temporary, until we can be more sophisticated.
struct Location {
    // loop in `func` over `names[0].names[1].<. . .>`
    // for e.g. dense loops, this is just `names[0] = "i"`
    // but for trees, this can be something like
    // tris.interior.children
    std::vector<std::string> names;
};

// Collapse two for-loops (io, ii) into a single for-loop (i).
// Note that io is shorthand for index in the outer loop, and ii
// is shorthand for index in the inner loop.
struct Collapse {
    Location io;
    Location ii;
    Location i;
};

// A work queue, owned by one loop of one function:
//
//     paths = render.queue(p);          // one queue per iteration of `p`
//     paths = render.queue(root);       // one queue per call of `render`
//     paths = render.queue(p, 4096);    // with a capacity the schedule states
//
// The queue is storage the owner allocates at the start of each of its
// iterations, and a drain loop -- a `parfor` named after the queue, so that
// later directives can split, vectorize or bind it -- that the owner runs
// once every entry of the iteration has been pushed. What an entry is, and
// which calls push one, is said by `defer` below. Without a capacity the size
// is inferred: the number of pushes an owner iteration can make, which for a
// linear deferral is the trip count of the producer loop (see
// SSA/Defer.h). A capacity is a constant for now.
struct Queue {
    std::string owner;
    Location loop;
    std::optional<Expr> capacity;
};

// f.defer(callee, queue)
//
// Every call of `callee` inside `f` becomes a push of the call's varying
// arguments onto `queue` and a return that says the call was saved rather
// than made; the queue's drain makes the call later. Turning a tail
// recursion into a wavefront -- `full_path_step.defer(full_path_step,
// paths)` -- is the case this is for, and `loopify` is the other thing one
// can do with the same recursion: the two are the depth-first and
// breadth-first ways of running the same pending calls. Applied at the SSA
// level; see SSA/Defer.h for what is and is not supported yet.
struct Defer {
    Location callee;
    std::string queue;
};

// A stage boundary after a call: `f.stage(g, q)`. The call to `g` in `f`
// is made where it is; its value and everything `f` still needs after it
// are pushed onto `q`, and `q`'s drain runs the rest of `f` from there.
// Where `defer` queues a call to be made later, this queues the *return*
// of one -- pbrt's boundary after `IntersectClosest`, where the trace has
// run and the material kernel continues from its result:
// `vol_path_step.stage(trace, hits)` beside `vol_path_step.defer(
// vol_path_step, rays)` is pbrt's bounce. Applied at the SSA level
// (SSA/Stage.h) as the compiler's factoring of `f` at the call -- the rest
// becomes a function of the call's value and what is live, and the call
// site tail-calls it -- followed by the deferral of that tail call.
struct Stage {
    Location callee;
    std::string queue;
};

// Turn recursion into iteration.
// For tail-call recursion, generates a DoWhile loop over the recursion
// condition.
// For branching recursion, generates a DoWhile loop over a queue with
// a maximum size of `queue_size`.
struct Loopify {
    // This is only used in the branching recursion case, hence the optionality.
    std::optional<Expr> queue_size;
};

// Bind a cursor to a piece of hardware.
//
// Every resource but RTCore binds the index of a `parfor`: the loop says its
// iterations may run in any order, and the bind says what to run them on. An
// unbound parfor is emitted as an ordinary sequential loop, so this is the
// only thing that makes one actually parallel. RTCore binds a function rather
// than a loop, since what it stands for is a traversal rather than an
// iteration.
//
// This replaces the earlier Parallelize transform, whose `CPUVector` strategy
// the parser never produced -- vectorising is vectorize()'s job.
struct Bind {
    Location i;
    Resource resource;
    // A function bound to a hardware unit that computes it in place of its
    // body -- `texture_filter.bind(TextureUnit, |t : ImageTexture, st :
    // vec2f, dstdx : vec2f, dstdy : vec2f| ...)` -- says what the unit
    // computes as a lambda: its parameters the function's, in order and
    // type, its value the function's result, and inside it the intrinsic
    // that unit is reached by (`tex_sample_grad_2d`). The program keeps the
    // function as the algorithm, exact and runnable anywhere; the schedule
    // says that on this hardware it is this instead. Undefined for a loop's
    // bind, where `i` names the loop. Applied by lower::LowerHardwareBinds,
    // first, so that everything after sees the function as the unit runs
    // it.
    Expr lambda;
};

// Sort the children of `loc` via a lambda applied to each index.
// For now, `loc` is assumed to be something like spheres.Interior
// TODO(ajr): also support queue sorting.
// Note: lambda arguments must always start with the index into
// the children list. All other arguments must be things in scope,
// e.g. the ray.
struct Sort {
    Location loc;
    Expr lambda;
};

// For-loop `i` becomes for-loop `io` over `i`'s range with stride=factor, and
// a nested for-loop `ii` over [0, factor) whose body is `i`'s at io + ii.
// If generate_tail is set, the body is guarded by `io + ii < i.end` (the
// GuardWithIf tail of Halide's split), so a range that is not a whole number
// of chunks runs exactly its own iterations and no more; if it is not set,
// the range is asserted to divide by the factor and no guard is made. Written
// `f.split(i, io, ii, factor, true)` for the guard and `false` for none.
struct Split {
    Location i;
    Location io;
    Location ii;
    Expr factor;
    bool generate_tail;
};

// Vectorize for-loop `i`, which must have a constant extent, turning it into
// a single SIMD "gang" of that width (see Pharr & Mark, "ispc: A SPMD
// Compiler for High-Performance CPU Programming"). Applied at the SSA level,
// not by LoopTransforms.
struct Vectorize {
    Location i;
};

// One copy of the function per variant of a parameter of algebraic type,
// the parameter's tag a constant in each, and the function itself a
// dispatcher that reads the tag once and calls the copy -- written
// `render.specialize(integrator)`. Halide's `specialize` from a boolean
// condition to a variant: what the scene fixes for a whole render (its
// integrator, its sampler) is fixed for the compiler too, so every `match`
// on it folds to one arm in the copy, and under a GPU bind each copy is its
// own kernel, allocated registers for the arm it runs and not for all of
// them (pbrt's GPU build is one integrator by construction). The copies
// are the function as scheduled up to this directive; it goes last among
// the function's directives. Applied at the SSA level (SSA/Specialize.h).
struct Specialize {
    std::string param;
};

using Transform = std::variant<Bind, Collapse, Defer, Loopify, Split, Sort,
                               Specialize, Stage, Vectorize>;

// The arms of a function's branches that a directive points at:
//
//     intersect.skip(Shape.Sphere);    // the arm taking that variant
//     intersect.skip(Shape);           // every arm of a match on a Shape
//     intersect.skip();                // every arm the function has
//
// A named arm is one of a match *written in* the function: the name is
// matched against the provenance lowering leaves on match arms (see
// ir::Provenance), so the arm is found wherever inlining and specialization
// have carried it, and a match in an `[[inline]]` helper is named by the
// helper, not by whatever it was inlined into. The bare form is about the
// function as compiled: every arm in it, an `if`'s arms -- which carry no
// provenance -- and inlined arms included.
struct ArmCursors {
    bool all = false;
    std::vector<Location> arms;

    bool covers(const Provenance &arm_of) const {
        if (all) {
            return true;
        }
        for (const Location &arm : arms) {
            if (arm_of.matches(arm.names)) {
                return true;
            }
        }
        return false;
    }
    bool empty() const { return !all && arms.empty(); }
};

// What a schedule says about the dynamic tests a gang's control flow gets
// beyond the ones it must have, per function (see SSA/Linearize.h).
//
// A vectorized branch is linearized: every arm is computed, under a mask of
// the lanes in it. An arm no lane is in is still paid for unless a uniform
// test of `any(mask)` is put in front of it -- the BOSCC gadget -- and the
// linearizer puts one there only where it must, before an arm that touches
// memory or makes a call. Whether an arm of pure arithmetic is worth a test
// depends on how often a gang finds no lane in it, which is a fact about the
// program's data, so it is the schedule's to say: `skip` names the arms that
// get a test anyway.
struct BranchPolicy {
    ArmCursors skip;
};

// Keyed by the function the schedule named.
using BranchPolicyMap = std::map<std::string, BranchPolicy>;

// Does the schedule put a test in front of an arm found in `func` -- the name
// the schedule knows the function by, before specialization renamed it --
// that was `arm` in the source? The bare `func.skip()` covers it wherever it
// was written; a named cursor covers it under the name of the function it
// was written in (see ArmCursors).
inline bool skips(const BranchPolicyMap &policies, const std::string &func,
                  const Provenance &arm) {
    if (const auto here = policies.find(func);
        here != policies.end() && here->second.skip.all) {
        return true;
    }
    if (arm.defined()) {
        if (const auto origin = policies.find(arm.func());
            origin != policies.end() && origin->second.skip.covers(arm)) {
            return true;
        }
    }
    return false;
}

// How a value of an ADT is stored.
//
//     layout Shape = tagged_index;
//
// A scheduling decision rather than a property of the type, for the same
// reason a tree's node layout is one: which representation a variant gets
// changes how much memory a traversal streams and how many loads it takes to
// reach a field, and changes nothing a program can observe. It lives in the
// Schedule so that it is chosen per target -- an index that travels to a
// device is not the same trade as a pointer that does not.
enum class AdtLayout {
    // A tag beside a union of the variants: Rust's repr(C) enum. Every value
    // is as large as the largest variant, and reaching a field is one load.
    Inline,
    // A tag naming one array per variant and an index into it. The value is
    // small and the variants are stored contiguously by kind, at the cost of
    // an indirection and of the arrays having to exist -- so a variant can
    // only be built where its array can be appended to.
    TaggedIndex,
    // A tag packed into the spare bits of a pointer to the variant, which is
    // what pbrt's TaggedPointer is. Eight bytes whatever the variant, and the
    // variant lives wherever it was allocated.
    //
    // Which is the catch: constructing one has to allocate, so this cannot be
    // combined with `--no-heap`. pbrt pays that cost knowingly -- it threads a
    // per-thread ScratchBuffer through its integrator and resets it every
    // sample, because `BSDF` holds a TaggedPointer to a BxDF and so every
    // intersection has to allocate one.
    TaggedPtr,
};

// What becomes of each arm of a variant type, keyed by the arm's name. The
// choice is per arm because the arms differ: pbrt's Primitive is twenty bytes
// as a GeometricPrimitive and a hundred and thirty as a TransformedPrimitive,
// and a tree's leaf that holds the first inline and the second by index is a
// word or so per primitive with no indirection in front of the common one.
// `layout Shape = tagged_index;` says the same thing of every arm; `layout
// Primitive { Geom = inline; Inst = tagged_index; }` says it arm by arm.
using AdtArmLayouts = std::map<std::string, AdtLayout>;

// Keyed by the ADT's type name.
using AdtLayoutMap = std::map<std::string, AdtArmLayouts>;

// Keys are function names.
using TransformMap = std::map<std::string, std::vector<Transform>>;

// Every transform of a schedule in the order the schedule wrote it, as (the
// function it names, its index in that function's list). The map above keeps
// each function's directives in order but says nothing about the order
// between functions, and that order matters: `render.vectorize(s)` before
// `trace.loopify(64)` puts the gang's traversal on a stack -- a packet
// traversal -- where the other way round each lane keeps a stack of its own.
using TransformOrder = std::vector<std::pair<std::string, size_t>>;

// https://en.cppreference.com/w/cpp/utility/variant/visit
template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

struct Schedule {
    TypeMap tree_types;
    LayoutMap tree_layouts;
    AdtLayoutMap adt_layouts;
    // How an extern array is stored; see ir::ArrayLayout.
    ArrayLayoutMap array_layouts;
    TransformMap func_transforms;
    // The order `func_transforms` was written in; see TransformOrder.
    TransformOrder transform_order;
    // Not transforms: nothing about the order they were written in matters,
    // and they are read by vectorize() wherever it runs (see BranchPolicy).
    BranchPolicyMap branch_policies;
    // The queues the schedule declares, by name (see Queue). A declaration
    // rather than a transform: a `defer` names one, and where in the
    // schedule the queue was declared changes nothing.
    std::map<std::string, Queue> queues;
    // Which group backs a tree held in a field, keyed the same way
    // `tree_types` is: `Instance.blas -> BlasNodes`.
    //
    // A top-level tree has a layout of its own and needs no such map. A nested
    // one has no layout of its own on purpose: its nodes are rows of a group
    // in the layout of whatever holds it, so that every value of the element
    // shares one pool and two of them naming the same row share a subtree.
    // That group is also what gives the field its stored type -- a reference
    // into it -- which is what lets an element with a tree in it be stored at
    // all.
    std::map<std::string, std::string> tree_groups;
};

} // namespace ir
} // namespace bonsai
