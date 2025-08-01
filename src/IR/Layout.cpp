#include "IR/Layout.h"

#include "IR/Operators.h"
#include "IR/Printer.h"

#include "Utils.h"

namespace bonsai {
namespace ir {

uint64_t Member::bits() const {
    switch (node_type()) {
    case IRLayoutEnum::Field: {
        return as<Field>()->type.bits();
    }
    case IRLayoutEnum::Pad: {
        return as<Pad>()->bits;
    }
    case IRLayoutEnum::Split: {
        uint64_t bits = 0;
        for (const auto &arm : as<Split>()->arms) {
            bits = std::max(bits, arm.member.bits());
        }
        return bits;
    }
    case IRLayoutEnum::Chain: {
        uint64_t bits = 0;
        for (const auto &l : as<Chain>()->members) {
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
    internal_error << "TODO: Member::bits()";
}

Expr Member::count() const {
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

Member Pad::make(uint32_t bits) {
    internal_assert(bits > 0) << "0 bits in Pad::make";
    Pad *node = new Pad;
    node->bits = bits;
    return node;
}

Member Field::make(std::string name, Type type) {
    internal_assert(!name.empty())
        << "empty name in Field::make with Type: " << type;
    internal_assert(type.defined())
        << "Undefined type in Field::make with name: " << name;
    internal_assert(type.is_primitive())
        << "Non-primitive type in Field::make: " << type;

    Field *node = new Field;
    node->name = std::move(name);
    node->type = std::move(type);
    return node;
}

Member Split::make(Member field, std::vector<ir::Arm> arms) {
    internal_assert(field.defined()) << "empty field in Split::make";
    internal_assert(!arms.empty())
        << "empty arms in Split::make for field: " << field;

    Split *node = new Split;
    node->field = std::move(field);
    node->arms = std::move(arms);
    return node;
}

std::string Split::field_name() const {
    const ir::Field *field = this->field.as<ir::Field>();
    internal_assert(field);
    return field->name;
}

Member Chain::make(std::vector<Member> members) {
    internal_assert(!members.empty()) << "Empty members in Chain::make";
    for (const auto &l : members) {
        internal_assert(l.defined()) << "Undefined member in Chain::make";
    }
    Chain *node = new Chain;
    node->members = std::move(members);
    return node;
}

Member Group::make(std::string name, Expr size, Expr index, Member inner,
                   Group::Type type) {
    internal_assert(size.defined())
        << "Cannot make Group with undefined size, named: " << name;
    internal_assert(!name.empty());
    // internal_assert(index.defined());
    internal_assert(inner.defined());

    Group *node = new Group;
    node->size = std::move(size);
    node->name = std::move(name);
    node->index = std::move(index);
    node->inner = std::move(inner);
    node->type = type;
    return node;
}

Member Materialize::make(std::string name, Expr value) {
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
