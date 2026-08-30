#include "Lower/ADTLayout.h"

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

// The smallest unsigned type that can tell the variants apart.
//
// Not a fixed width: a sum of two costs a byte, and how much padding follows
// is decided by the payload's alignment either way, so a wider tag is often
// free -- but never better, and not always free.
Type tag_type_for(size_t variants) {
    internal_assert(variants > 0) << "A type with no variants has no tag";
    if (variants <= (1ull << 8)) {
        return UInt_t::make(8);
    } else if (variants <= (1ull << 16)) {
        return UInt_t::make(16);
    }
    return UInt_t::make(32);
}

} // namespace

uint64_t ADTLayout::tag(const std::string &variant) const {
    const auto found = tag_of.find(variant);
    internal_assert(found != tag_of.end())
        << "No tag for variant " << variant;
    return found->second;
}

ir::Type ADTLayout::variant(const std::string &name) const {
    const auto found = variant_type.find(name);
    internal_assert(found != variant_type.end())
        << "No type for variant " << name;
    return found->second;
}

ADTLayout default_adt_layout(const ir::ADT_t &adt) {
    ADTLayout layout;
    layout.tag_field = "tag";
    layout.payload_field = "payload";
    layout.tag_type = tag_type_for(adt.variants.size());
    layout.variants = adt.variants;

    Union_t::Map members;
    members.reserve(adt.variants.size());
    for (size_t i = 0; i < adt.variants.size(); i++) {
        const std::string &name = adt.variant_name(i);
        members.push_back(TypedVar{name, adt.variants[i]});
        layout.tag_of[name] = i;
        layout.variant_type[name] = adt.variants[i];
    }
    layout.payload = Union_t::make(adt.name + "_payload", std::move(members));

    // The tag first and the payload after it. With the union's own alignment
    // that is Rust's repr(C) enum: the size is the largest variant rounded up
    // to the strictest alignment, and nothing is moved.
    Struct_t::Map fields;
    fields.push_back(TypedVar{layout.tag_field, layout.tag_type});
    fields.push_back(TypedVar{layout.payload_field, layout.payload});
    layout.storage = Struct_t::make(adt.name, std::move(fields));

    return layout;
}

} // namespace lower
} // namespace bonsai
