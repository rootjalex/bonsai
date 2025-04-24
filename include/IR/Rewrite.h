#pragma once

#include <string>
#include <variant>

#include "Expr.h"

namespace bonsai {
namespace ir {
namespace schedule {

// Placeholder type
struct Location {
    // loop in `func` over `names[0].names[1].<. . .>`
    // for e.g. dense loops, this is just `names[0] = "i"`
    // but for trees, this can be something like
    // tris.interior.children
    std::vector<std::string> names;
};

// For for-loops:
//   Loop `j` directly nested below loop `i` becomes
//   Loop `i` directly nested below loop `j`
// For for-while loops:
//   recursive traversal `j` in loop body `i` becomes
//   grouped-recursive traversal
struct Reorder {
    Location i;
    Location j;
};

// For-loop `i` with extent `n` becomes for-loop `io`
// with start=i.start end=(i.end / factor) * factor, stride=factor and
// nested for-loop `ii` with start=io, end=io+factor,
// stride=1
// if exact is set, no tail strategy is generated
// if it is not set, a tail for-loop `i` with
// start=(i.end / factor) * factor, end=i.end stride=1 is generated.
struct Split {
    Location i;
    Location io;
    Location ii;
    Expr factor;
    bool exact;
};

// For-loops `i` and `j` are fused into loop `ij`
struct Collapse {
    Location i;
    Location j;
    Location ij;
};

// TODO: figure out how to do this.
struct Prefetch {
    Location at;
    Location value;
};

// Unroll loop `i`
struct Unroll {
    Location i;
};

// Parallelize `i` via some strategy.
struct Parallelize {
    enum Strategy { CPUVector, CPUThread, GPUThread, GPUBlock };

    Location i;
    Strategy strategy;
};

// In any Match statement on type `i`,
// perform loop synchronization:
/*
top = root
stack = alloc()
do {
    match top
    | case `i`: f();
    | . . . other cases . . .
} while (top != Sentry);
->
top = root
stack = alloc()
do {
    while (top is `i`) { f(); }
    match top
    | . . . other cases . . .
} while (top != Sentry);
*/
// struct Synchronize {
//     Location i;
// };

// In the Match arm `i`,
// sort the children based on cost `metric`.
struct Sort {
    Location i;
    Expr metric; // lambda
};

// In the Match arm `i`, do not recurse,
// write to a task-queue `queue`
// struct Defer {
//     Location i;
//     Location queue;
// };

// In the labelled recursive func,
// keep a stack size of only `value`.
// TODO: generalize to other stack
// optimizations?
struct Bound {
    Expr value;
};

struct Rewrite {
    std::string func;
    using Cmd = std::variant<Bound, Collapse, /*Defer,*/ Parallelize, Prefetch,
                             Reorder, Sort, Split, /*Synchronize,*/ Unroll>;
    Cmd rewrite;

    Rewrite(std::string func, Cmd rewrite)
        : func(std::move(func)), rewrite(std::move(rewrite)) {}
};

} // namespace schedule
} // namespace ir
} // namespace bonsai