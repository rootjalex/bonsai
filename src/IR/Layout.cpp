#include "IR/Layout.h"

#include "IR/Operators.h"
#include "IR/Printer.h"

#include "Utils.h"

namespace bonsai {
namespace ir {

uint64_t Layout::bits() const {
    switch (node_type()) {
    case IRLayoutEnum::Name: {
        return as<Name>()->type.bits();
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

Layout Pad::make(uint32_t bits) {
    internal_assert(bits > 0) << "0 bits in Pad::make";
    Pad *node = new Pad;
    node->bits = bits;
    return node;
}

Layout Name::make(std::string name, Type type) {
    internal_assert(!name.empty())
        << "empty name in Name::make with Type: " << type;
    internal_assert(type.defined())
        << "Undefined type in Name::make with name: " << name;
    internal_assert(type.is_primitive())
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

Layout Group::make(Expr size, std::string name, Type index_t, Layout inner) {
    internal_assert(size.defined())
        << "Cannot make Group with undefined size, named: " << name;
    // Groups can have no label, name can be empty and index_t can be undefined
    // (default: u32).
    internal_assert(name.empty() != index_t.defined())
        << "Cannot have name without index_t and vice versa: " << name << " : "
        << index_t;
    internal_assert(inner.defined())
        << "Cannot make Group with undefined inner, named: " << name;

    Group *node = new Group;
    node->size = std::move(size);
    node->name = std::move(name);
    node->index_t = std::move(index_t);
    node->inner = std::move(inner);
    return node;
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
