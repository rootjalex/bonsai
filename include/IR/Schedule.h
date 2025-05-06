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

// Parallelize `i` via some strategy.
struct Parallelize {
    enum Strategy { CPUVector, CPUThread, GPUThread, GPUBlock };

    Location i;
    Strategy strategy;
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

using Transform = std::variant<Parallelize, Split>;

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
