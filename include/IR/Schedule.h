#pragma once

#include <map>
#include <string>
#include <variant>

#include "Layout.h"
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

// consumer.defer(producer, i, queue)
// says "when `producer()` is called in `consumer`", instead
// write it to a queue that is allocated at loop level `i`.
// This turns the body of `i` into a do-while loop that
// iterates as long as any queues are non-empty.
// Note that loops over queues (which are forall loops)
// should also be scheduled.
// TODO(ajr): support `continue` parameter!
// TODO(ajr): support type-specialization e.g. trace(Treelet)
struct Defer {
    Location producer;
    Location loop;
    Location queue;
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

// Allocate a queue at a loop of the func, with a constant maximum size.
// TODO(ajr): support dynamic queue sizes.
// If the queue already exists, tag it with the output size.
struct MakeQueue {
    Location queue;
    Location loop;
    std::optional<Expr> queue_size;
    // TODO(ajr): might also want AoS vs. SoA control
    // TODO(ajr): memory type? e.g. Shared/Register/Global/Heap/Stack?
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

// For-loop `i` with extent `n` becomes for-loop `io`
// with start=i.start end=(i.end / factor) * factor, stride=factor and
// nested for-loop `ii` with start=io, end=io+factor,
// stride=1
// if generate_tail is set, no tail strategy is generated
// if it is not set, a tail for-loop `i` with
// start=(i.end / factor) * factor, end=i.end stride=1 is generated.
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

using Transform = std::variant<Bind, Collapse, Defer, Loopify, MakeQueue, Split,
                               Sort, Vectorize>;

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

// Keyed by the ADT's type name.
using AdtLayoutMap = std::map<std::string, AdtLayout>;

// Keys are function names.
using TransformMap = std::map<std::string, std::vector<Transform>>;

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
    TransformMap func_transforms;
};

} // namespace ir
} // namespace bonsai
