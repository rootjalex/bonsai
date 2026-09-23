#pragma once

#include <iostream>
#include <map>
#include <memory>

#include "Function.h"
#include "Schedule.h"
#include "Target.h"
#include "Type.h"

namespace bonsai {
namespace ir {

namespace ssa {
struct Function;
}

using FuncMap = std::map<std::string, std::shared_ptr<Function>>;
using ScheduleMap = std::map<Target, Schedule>;
using ExternList = std::vector<TypedVar>;

struct Program {
    // TODO: more things?

    // Intentionally ordered, this will be the order of arguments to the
    // executable.
    ExternList externs;
    // All function declarations except for main()
    FuncMap funcs;
    // All types (including aliases).
    TypeMap types;
    // TODO: what is the right interface for this?
    ScheduleMap schedules;

    // `with extent = <expr>` on an element, keyed by the element's type name,
    // recorded as a function of the element: a Lambda of one argument of the
    // element's type, whichever way the source spelled it (over the fields of
    // a struct, or `|p| ..` for a variant). Readers apply it to the element
    // they are asking about.
    //
    // What a node's bounds augmentation says about a subtree, said about a
    // single element: everything reachable through a value of this type lies
    // within this geometry. It is what makes a compound element a geometric
    // object, so a tree over such elements can carry a bounds augmentation at
    // all -- and, like a node's volume, it is a promise owed by whoever builds
    // the tree rather than a fact the compiler derives.
    //
    // PBRT's TransformedPrimitive is the case it exists for. Its `Bounds()` is
    // `renderFromPrimitive(primitive.Bounds())`, computed from the transform
    // and the held tree's own root bound, and PBRT calls it only from the BVH
    // constructor -- never from `Intersect`. The same is true here: this is
    // read by the build specification and, symbolically, by predicate
    // analysis, and it is never evaluated per query.
    //
    // Beside `types` rather than inside the Struct_t because a Struct_t is
    // hash-consed on its fields: two elements with the same fields are the
    // same type, while their extents are not interchangeable.
    std::map<std::string, Expr> extents;
    // TODO: interfaces / inheritance?

    // Where a tree keeps its leaf elements, once its layout has been applied:
    // the array they are rows of, as an expression reading it out of the
    // tree's storage, and the type of an index into it. Keyed by the tree's
    // name. This is what a reference to one of the tree's elements (an
    // ElementRef_t) is lowered to -- the index -- and what reading through the
    // reference indexes; see Lower/ElementReferences.cpp. Recorded by
    // Lower/Layouts.cpp, which is the one place that knows.
    struct ElementStorage {
        Expr container;
        Type index_t;
    };
    std::map<std::string, ElementStorage> element_storage;

    // What a variant type became once its layout was applied, for a pass
    // that meets the storage after the fact and has to tell its variants
    // apart: the schedule's `specialize()` copies a loop per variant of a
    // parameter and sets the tag in each copy (SSA/Specialize.h). Keyed by
    // the variant type's name, which is the storage struct's name. An
    // `Inline` storage is the struct of a tag, padding to the payload's
    // alignment where the tag is narrower, and the payload's words, named
    // by the fields here; a `TaggedIndex` storage is one word with the tag
    // in its top bits and has no fields to name. Recorded by
    // Lower/ADTs.cpp, which is the one place that knows.
    struct AdtStorage {
        // Each variant's name and the number its tag holds, in declaration
        // order.
        std::vector<std::pair<std::string, uint64_t>> variants;
        bool inline_storage = true;
        std::string tag_field;
        std::string pad_field; // empty when the storage has no padding
        std::string payload_field;
        Type tag_type;
    };
    std::map<std::string, AdtStorage> adt_storages;

    // The SSA form of whichever functions are to be lowered to the backend
    // straight from it, rather than from the statements the relooper builds.
    // Keyed by the same names as `funcs`, which still holds a statement form
    // of every one of them: the relooper runs regardless, because being able
    // to read what a schedule did as ordinary statements is worth the pass
    // whether or not code is generated from it.
    std::map<std::string, std::shared_ptr<ssa::Function>> ssa_funcs;

    Program() {}

    Program(ExternList externs, FuncMap funcs, TypeMap types,
            ScheduleMap schedules)
        : externs(std::move(externs)), funcs(std::move(funcs)),
          types(std::move(types)), schedules(std::move(schedules)) {}

    ~Program() = default;

    // Defaulted rather than written out. These used to name the four members
    // there were at the time, so a program copied or moved -- which is once
    // per pass, since lowering assigns the result of each back -- silently
    // lost anything added afterwards. The compiler cannot forget a member.
    Program(const Program &other) = default;
    Program &operator=(const Program &other) = default;
    Program(Program &&other) noexcept = default;
    Program &operator=(Program &&other) noexcept = default;
};

} // namespace ir
} // namespace bonsai
