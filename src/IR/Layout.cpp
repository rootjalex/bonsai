#include "IR/Layout.h"
#include "IR/Equality.h"

#include "IR/Operators.h"
#include "IR/Printer.h"

#include "Utils.h"

#include <functional>

namespace bonsai {
namespace ir {

uint64_t Layout::bits() const {
    switch (node_type()) {
    case IRLayoutEnum::Name: {
        // The width of the storage, which for a vector is every lane:
        // Type::bits() answers for one lane, being asked about arithmetic,
        // and a layout is asking about bytes. A `vec3f` is 96 bits here, as
        // the layout that pads it out to a 32-byte node has always taken it
        // to be.
        const Type &type = as<Name>()->type;
        // A vector of vectors -- `vector[vec3f, 8]`, a node's eight child
        // boxes -- is every lane of every component: 8 * 3 * 32 bits.
        const std::function<uint64_t(const Type &)> storage_bits =
            [&](const Type &t) -> uint64_t {
            if (const auto *vector = t.as<Vector_t>()) {
                return uint64_t(vector->lanes) * storage_bits(vector->etype);
            }
            return t.bits();
        };
        return storage_bits(type);
    }
    case IRLayoutEnum::Pad: {
        return as<Pad>()->bits;
    }
    case IRLayoutEnum::Switch: {
        uint64_t bits = 0;
        for (const auto &arm : as<Switch>()->arms) {
            bits = std::max(bits, arm.layout.bits());
        }
        return bits;
    }
    case IRLayoutEnum::Chain: {
        uint64_t bits = 0;
        for (const auto &l : as<Chain>()->layouts) {
            bits += l.bits();
        }
        return bits;
    }
    case IRLayoutEnum::Group: {
        const Group *node = as<Group>();
        internal_assert(!node->size.defined() || !is_const(node->size))
            << "TODO: should a constant-sized group be inlined? " << *this;
        return 64; // pointer
    }
    case IRLayoutEnum::Materialize: {
        return 0; // computed field, not stored.
    }
    case IRLayoutEnum::Lookup: {
        // The row lives in the group looked up, not here. What is stored at
        // this arm is the index, and that is stored by whoever built the
        // reference -- so this arm contributes nothing of its own.
        return 0;
    }
    }
    internal_error << "TODO: Layout::bits()";
}

Expr Layout::count() const {
    if (const Group *node = as<Group>()) {
        ir::Expr icount = node->inner.count();
        if (!is_const_one(icount)) {
            return node->size * node->inner.count();
        } else {
            return node->size;
        }
    }
    // TODO: should this always be a u64?
    static Expr u64_1 = UIntImm::make(UInt_t::make(64), 1);
    return u64_1;
}

Layout Lookup::make(std::string group_name, Expr index, Layout shape) {
    internal_assert(!group_name.empty())
        << "Lookup::make received an empty group name";
    internal_assert(index.defined())
        << "Lookup::make received an undefined index into " << group_name;

    Lookup *node = new Lookup;
    node->group_name = std::move(group_name);
    node->index = std::move(index);
    node->shape = std::move(shape);
    return node;
}

Layout Pad::make(uint32_t bits) {
    internal_assert(bits > 0) << "0 bits in Pad::make";
    Pad *node = new Pad;
    node->bits = bits;
    return node;
}

namespace {

// What a layout may store, as far as this level can tell.
//
// Weaker than `is_primitive` in one place: a set-typed *field* of an element
// passes, because a schedule may have bound it to a tree, in which case what
// is stored is a reference into that tree's pool rather than a tree. Whether a
// schedule did bind it is not knowable here -- it is a fact about the
// schedule, not about the type -- so the front end checks that (see the
// parser's `storable_in_layout`) and this checks what remains.
bool storable(const Type &type) {
    if (const auto *as_struct = type.as<Struct_t>()) {
        for (const auto &field : as_struct->fields) {
            if (!field.type.is<Set_t>() && !storable(field.type)) {
                return false;
            }
        }
        return true;
    }
    // A variant is stored when each of its arms is: it becomes a tag beside
    // them, or a handle naming one.
    if (const auto *as_adt = type.as<ADT_t>()) {
        return std::all_of(as_adt->variants.cbegin(), as_adt->variants.cend(),
                           [](const Type &variant) { return storable(variant); });
    }
    if (const auto *as_array = type.as<Array_t>()) {
        return storable(as_array->etype);
    }
    return type.is_primitive();
}

} // namespace

Layout Name::make(std::string name, Type type) {
    internal_assert(!name.empty())
        << "empty name in Name::make with Type: " << type;
    internal_assert(type.defined())
        << "Undefined type in Name::make with name: " << name;
    internal_assert(storable(type))
        << "Non-primitive type in Name::make: " << type;

    Name *node = new Name;
    node->name = std::move(name);
    node->type = std::move(type);
    return node;
}

// Layout Star::make(Layout inner) {
//     internal_assert(inner.defined()) << "empty layout in Star::make";

//     Star *node = new Star;
//     node->inner = std::move(inner);
//     node->count = 0; // all
//     return node;
// }

Layout Switch::make(std::string field, std::vector<Switch::Arm> arms) {
    internal_assert(!field.empty()) << "empty field in Switch::make";
    internal_assert(!arms.empty())
        << "empty arms in Switch::make for field: " << field;

    Switch *node = new Switch;
    node->field = std::move(field);
    node->arms = std::move(arms);
    return node;
}

Layout Chain::make(std::vector<Layout> layouts) {
    internal_assert(!layouts.empty()) << "Empty layouts in Chain::make";
    for (const auto &l : layouts) {
        internal_assert(l.defined()) << "Undefined layout in Chain::make";
    }
    Chain *node = new Chain;
    node->layouts = std::move(layouts);
    return node;
}

Layout Group::make(Expr size, std::string name, std::string declared_name,
                   ir::Type index_t, Layout inner, Group::Type type,
                   ir::Type element, Expr start) {
    // A group of the tree's elements may have no size when it is the run of
    // tiles that begins where a looked-up row does, bounded by the range
    // over it (see Lookup::shape); every other group says how many rows it
    // has.
    internal_assert(size.defined() ||
                    (type == Group::Type::Indirect && element.defined()))
        << "Cannot make Group with undefined size, named: " << name;
    // Groups can have no label, name can be empty and index_t can be undefined
    // (default: u32).
    internal_assert(name.empty() != index_t.defined())
        << "Cannot have name without index_t and vice versa: " << name << " : "
        << index_t;
    internal_assert(inner.defined())
        << "Cannot make Group with undefined inner, named: " << name;
    // An indirect group, or a pointer group, is reached only by being named,
    // so one without a name is storage nothing can ever read.
    internal_assert(type == Group::Type::Direct || !declared_name.empty())
        << "An indirect or ptr group has to be named: nothing can look it up "
           "otherwise.";

    internal_assert(!element.defined() || type == Group::Type::Indirect)
        << "A group of elements is reached by the leaves that range over it, "
           "not walked: it is indirect. Group " << declared_name;
    Group *node = new Group;
    node->size = std::move(size);
    node->name = std::move(name);
    node->declared_name = std::move(declared_name);
    node->index_t = std::move(index_t);
    node->inner = std::move(inner);
    node->type = type;
    node->element = std::move(element);
    internal_assert(!start.defined() ||
                    (type == Group::Type::Direct && node->index_t.defined() &&
                     equals(start.type(), node->index_t)))
        << "A group's starting reference is a value of its direct index's "
           "type: "
        << start << " for " << node->name << " : " << node->index_t;
    node->start = std::move(start);
    return node;
}

const Group *tile_of(const Layout &inner) {
    const Group *group = inner.as<Group>();
    if (const Chain *chain = inner.as<Chain>()) {
        if (chain->layouts.size() != 1) {
            return nullptr;
        }
        group = chain->layouts.front().as<Group>();
    }
    if (group == nullptr || !group->name.empty() ||
        !group->declared_name.empty() || !group->size.defined() ||
        !is_const(group->size)) {
        return nullptr;
    }
    return group;
}

uint32_t tile_width(const Group &tile) {
    const std::optional<uint64_t> width = get_constant_value(tile.size);
    internal_assert(width.has_value() && *width > 1 && *width <= 1024)
        << "A tile holds a constant number of elements, more than one: "
        << Layout(&tile);
    return static_cast<uint32_t>(*width);
}

bool holds_fields_of(const Layout &inner, const Struct_t &element) {
    const Chain *chain = inner.as<Chain>();
    std::vector<Layout> members;
    if (chain != nullptr) {
        members = chain->layouts;
    } else {
        members.push_back(inner);
    }
    if (members.size() != element.fields.size()) {
        return false;
    }
    for (size_t i = 0; i < members.size(); i++) {
        const Name *name = members[i].as<Name>();
        if (name == nullptr || name->name != element.fields[i].name ||
            !equals(name->type, element.fields[i].type)) {
            return false;
        }
    }
    return true;
}

Layout Materialize::make(std::string name, Expr value) {
    internal_assert(!name.empty()) << "Materialize::make received empty name";
    internal_assert(value.defined())
        << "Materialize::make received undefined value for name: " << name;

    Materialize *node = new Materialize;
    node->name = std::move(name);
    node->value = std::move(value);
    return node;
}

} // namespace ir
} // namespace bonsai
