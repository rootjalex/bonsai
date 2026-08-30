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
// Today there is one of these and every ADT gets it: a tag beside a union of
// the variants, which is Rust's repr(C) enum. Fields keep the order they were
// declared in, since nothing reorders them.
struct ADTLayout {
    // The struct a value of the ADT becomes, and the union inside it.
    ir::Type storage;
    ir::Type payload;

    ir::Type tag_type;
    std::string tag_field;
    std::string payload_field;

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
//
// Where a schedule directive would eventually hook in: it would produce one of
// these instead, and the pass would not change.
ADTLayout default_adt_layout(const ir::ADT_t &adt);

} // namespace lower
} // namespace bonsai
