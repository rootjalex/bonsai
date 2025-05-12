#pragma once

#include <map>
#include <string>
#include <variant>

#include "Layout.h"
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

// Turn recursion into iteration.
// For tail-call recursion, generates a DoWhile loop over the recursion
// condition.
// For branching recursion, generates a DoWhile loop over a queue with
// a maximum size of `queue_size`.
struct Loopify {
    // This is only used in the branching recursion case, hence the optionality.
    std::optional<Expr> queue_size;
};

// Parallelize `i` via some strategy.
struct Parallelize {
    enum Strategy { CPUVector, CPUThread, GPUThread, GPUBlock };

    Location i;
    Strategy strategy;
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

// Collapse two for-loops (io, ii) into a single for-loop (i).
// Note that io is shorthand for index in the outer loop, and ii
// is shorthand for index in the inner loop.
struct Collapse {
    Location io;
    Location ii;
    Location i;
};

using Transform = std::variant<Collapse, Loopify, Parallelize, Split, Sort>;

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
    TransformMap func_transforms;
};

} // namespace ir
} // namespace bonsai
