#include "IR/Layout.h"

#include "IR/Equality.h"
#include "IR/Operators.h"
#include "IR/Printer.h"

#include "Utils.h"

namespace bonsai {
namespace ir {
namespace {

// TODO(cgyurgyik): there is an underlying assumption that every layout is a
// chain. This seems in general brittle, and breaks for arms with lookups.
// https://www.youtube.com/watch?v=C6ZnwuhqALY&ab_channel=2ChainzVEVO
const ir::Chain *to_chainz(const ir::Member &member) {
    const ir::Chain *chain = member.as<ir::Chain>();
    if (chain == nullptr) {
        static ir::Chain *m = new ir::Chain;
        m->members = {member};
        return m;
    }
    return chain;
}

bool contains_field(const std::string &field_name, const ir::Member &member) {
    const ir::Chain *chain = to_chainz(member);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const auto *field = m.as<ir::Field>();
            if (field->name == field_name) {
                return true;
            }
            continue;
        }
        // TODO(cgyurgyik): Handle nested groups.
        case ir::IRLayoutEnum::Group: {
            const auto *group = m.as<ir::Group>();
            return contains_field(field_name, group->inner);
        }
        default:
            continue;
        }
    }
    return false;
}

void collect_fields(const ir::Member &member, std::vector<ir::Member> &fields) {
    if (const ir::Field *field = member.as<ir::Field>()) {
        fields.push_back(field);
    }
    if (const ir::Chain *chain = member.as<ir::Chain>()) {
        for (const ir::Member &member : chain->members) {
            collect_fields(member, fields);
        }
    }
    if (const ir::Group *group = member.as<ir::Group>()) {
        collect_fields(group->inner, fields);
    }
    if (const ir::Split *split = member.as<ir::Split>()) {
        for (const auto &arm : split->arms) {
            collect_fields(arm.member, fields);
        }
    }
}

} // namespace

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
        if (node->inner.bits() == 0) {
            return 0;
        }
        return 64; // pointer
    }
    case IRLayoutEnum::Lookup:
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

std::string Member::name() const {
    if (const auto *field = as<ir::Field>()) {
        return field->name;
    }
    if (const auto *group = as<ir::Group>()) {
        return group->name;
    }
    internal_error << "[unimplemented] name: " << *this;
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

Member Split::make(ir::Expr expr, std::vector<ir::Arm> arms) {
    internal_assert(expr.defined()) << "empty expr in Split::make";
    internal_assert(!arms.empty())
        << "empty arms in Split::make for expr: " << expr;

    Split *node = new Split;
    node->expr = std::move(expr);
    node->arms = std::move(arms);
    return node;
}

namespace {
std::string get_field_name(ir::Expr e) {
    if (const auto *v = e.as<ir::Var>()) {
        return v->name;
    }
    if (const auto *s = e.as<ir::Slice>()) {
        return get_field_name(s->value);
    }
    if (const auto *ex = e.as<ir::Extract>()) {
        return get_field_name(ex->vec);
    }
    if (const auto *bo = e.as<ir::BinOp>()) {
        internal_assert(is_const(bo->a) ^ is_const(bo->b));
        if (is_const(bo->a)) {
            return get_field_name(bo->b);
        }
        if (is_const(bo->b)) {
            return get_field_name(bo->a);
        }
    }
    internal_error << "failed to get field name from: " << e;
}
} // namespace

std::string Split::field_name() const { return get_field_name(this->expr); }

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
    switch (type) {
    case Group::Type::Indirect:
        break;
    case Group::Type::Direct:
        if (!index.defined()) {
            // TODO(cgyurgyik): is this too restrictive? What is a realistic
            // case where this is necessary?
            internal_assert(is_const(size))
                << "[unexpected] undefined index with non-constant size: "
                << size;
        }
        break;
    }
    internal_assert(inner.defined())
        << "Cannot make Group: " << name << " with undefined body";

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

Member Lookup::make(std::string group_name, Expr index) {
    internal_assert(!group_name.empty()) << "Lookup::make received empty name";
    internal_assert(index.defined())
        << "Lookup::make received undefined index for group name: "
        << group_name;

    Lookup *node = new Lookup;
    node->group_name = std::move(group_name);
    node->index = std::move(index);
    return node;
}

[[maybe_unused]] std::ostream &operator<<(std::ostream &os,
                                          const LayoutMap &map) {
    os << "name -> layout {\n";
    for (const auto &[member, type] : map) {
        os << member << " : " << type << "\n";
    }
    os << "}\n";
    return os;
}

std::vector<ir::BVH_t::Variant> Layout::variants() const {
    const BVH_t *bvh_t = type.as<BVH_t>();
    internal_assert(bvh_t);
    return bvh_t->variants;
}

std::vector<ir::Member> Layout::find_all_groups() const {
    if (const ir::Group *group = body.as<ir::Group>()) {
        return {group};
    }
    std::vector<ir::Member> groups;
    if (const ir::Chain *chain = body.as<ir::Chain>()) {
        for (const ir::Member &member : chain->members) {
            if (const ir::Field *field = member.as<ir::Field>()) {
                if (field->type.is<ir::Array_t>()) {
                    // An array is syntactic sugar for an indirect group.
                    groups.push_back(field);
                }
            } else if (const ir::Group *group = member.as<ir::Group>()) {
                groups.push_back(group);
            }
        }
    }
    return groups;
}

std::vector<ir::Member> Layout::find_all_fields() const {
    std::vector<ir::Member> fields;
    collect_fields(body, fields);
    return fields;
}

std::vector<ir::Member> Layout::find_direct_groups() const {
    if (const ir::Group *group = body.as<ir::Group>()) {
        if (group->type == ir::Group::Type::Direct) {
            return {group};
        }
    }
    std::vector<ir::Member> direct_groups;
    if (const ir::Chain *chain = body.as<ir::Chain>()) {
        for (const ir::Member &member : chain->members) {
            if (const ir::Group *group = member.as<ir::Group>()) {
                if (group->type == ir::Group::Type::Direct) {
                    direct_groups.push_back(group);
                }
            }
        }
    }
    return direct_groups;
}

ir::Type Layout::get_index_type() const {
    std::set<ir::Expr, ir::ExprLessThan> indexes;
    std::vector<ir::Member> groups = find_direct_groups();
    for (const ir::Member &member : groups) {
        const auto *group = member.as<ir::Group>();
        internal_assert(group) << member;
        indexes.insert(group->index);
    }
    internal_assert(indexes.size() == 1)
        << "[unexpected] multiple indexes, expected 1 but found: "
        << indexes.size();
    auto it = indexes.begin();
    return it->type();
}

ir::Member Layout::find_primitives_group() const {
    const auto *bvh_t = type.as<ir::BVH_t>();
    internal_assert(bvh_t);
    ir::Type primitive_type = bvh_t->primitive;
    const ir::Chain *chain = to_chainz(body);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const auto *field = m.as<ir::Field>();
            if (!field->type.is_iterable()) {
                continue;
            }
            if (!ir::equals(field->type.element_of(), bvh_t->primitive)) {
                continue;
            }
            return field;
        }
        case ir::IRLayoutEnum::Group: {
            // TODO(cgyurgyik): a group may contain the primitives too!
        }
        default:
            continue;
        }
    }
    return ir::Member();
}

ir::Member Layout::find_group_for(const std::string &field_name) const {
    const auto *bvh_t = type.as<ir::BVH_t>();
    internal_assert(bvh_t);
    ir::Type primitive_type = bvh_t->primitive;
    const ir::Chain *chain = to_chainz(body);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field:
            // TODO(cgyurgyik): what about members in the base layout?
            continue;
        case ir::IRLayoutEnum::Group:
            if (contains_field(field_name, m)) {
                return m;
            }
            [[fallthrough]];
        default:
            continue;
        }
    }
    return ir::Member();
}

} // namespace ir
} // namespace bonsai
