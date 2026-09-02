#pragma once

#include "IR/Program.h"
#include "IR/Type.h"

#include <map>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

// How a value of an ADT is stored.
//
// A description, not a strategy: it says where the tag lives, what number each
// variant is, and what the payload looks like, and Lower/ADTs.cpp does the
// lowering by reading it. Choosing a layout and applying one are separate so
// that a schedule can eventually choose differently -- the same division the
// tree `layout` block already has -- but the applying stays in the pass, since
// that is what lowering is.
//
// Two of these exist. `Inline` is a tag beside a union of the variants, which
// is Rust's repr(C) enum. `TaggedIndex` is a single word: the tag in its top
// bits and, in the rest, an index into one pool per variant. Fields keep the
// order they were declared in either way, since nothing reorders them.
struct ADTLayout {
    ir::AdtLayout kind = ir::AdtLayout::Inline;

    // What a value of the ADT becomes: the struct, for `Inline`, and the
    // handle's unsigned integer type for `TaggedIndex`.
    ir::Type storage;

    // `Inline` only: the union inside the struct, and the two field names.
    ir::Type payload;
    std::string tag_field;
    std::string payload_field;

    ir::Type tag_type;

    // `TaggedIndex` only. A handle is one `u64` with the tag above
    // `tag_shift` and the index below it, so a variant is named in the top
    // byte and there are 2^56 of each.
    //
    // Byte-aligned, unlike pbrt's TaggedPointer, which puts a seven-bit tag at
    // bit 57 because the low 57 bits have to stay a usable pointer. An index
    // is under no such obligation, and a shift of 56 makes the tag a byte the
    // generated code can extract with one shift and no mask.
    static constexpr uint64_t tag_shift = 56;

    // Where the fields of each variant live, and how many of that variant have
    // been built. Both are externs: the pool is the caller's memory and the
    // caller's capacity, which is what lets a variant be built without an
    // allocator. Keyed by variant name.
    std::map<std::string, std::string> pool_of;
    std::map<std::string, std::string> fill_of;

    const std::string &pool(const std::string &variant) const;
    const std::string &fill(const std::string &variant) const;

    // Which number each variant is, and the struct of its fields. The union's
    // members are named for their variants, so one name reaches both.
    std::map<std::string, uint64_t> tag_of;
    std::map<std::string, ir::Type> variant_type;

    // In declaration order, for registering them as types.
    std::vector<ir::Type> variants;

    uint64_t tag(const std::string &variant) const;
    ir::Type variant(const std::string &variant) const;
};

// The layout an ADT gets when nothing says otherwise.
ADTLayout default_adt_layout(const ir::ADT_t &adt);

// The layout a schedule asked for.
//
// This is the hook the header above describes: a `layout Shape = tagged_index;`
// in a schedule block reaches here, and the pass that applies a layout does not
// change. `TaggedPtr` is not built, and reports what is missing rather than
// silently giving the default, since a layout that was asked for and not given
// would be a schedule that changed nothing and said so nowhere.
ADTLayout adt_layout(const ir::ADT_t &adt, ir::AdtLayout kind);

} // namespace lower
} // namespace bonsai
