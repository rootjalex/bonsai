#include "Lower/Layouts.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Layout.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/ValidateLayout.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Log.h"
#include "Utils.h"

#include <functional>
#include <ranges>
#include <string>
#include <unordered_map>

namespace bonsai {
namespace lower {

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

class LayoutTypeMap {
  public:
    ir::Type insert_struct_layout(const ir::Member &member,
                                  const std::string &name,
                                  const ir::Struct_t::Map &fields,
                                  const ir::Member &parent) {
        std::optional<int64_t> alignment;
        if (const auto *group = parent.as<ir::Group>()) {
            if (group->alignment.defined()) {
                alignment = get_constant_value<int64_t>(group->alignment);
            }
        }
        std::vector<ir::Struct_t::Attribute> attributes = {};
        attributes.push_back(ir::Struct_t::Attribute::packed);
        ir::Type type = ir::Struct_t::make(name, fields, std::move(attributes),
                                           std::move(alignment));

        {
            auto [_, inserted] = layout_to_type.try_emplace(member, type);
            internal_assert(inserted)
                << "duplicate insertion of member: " << member
                << "with type: " << type;
        }
        {
            auto [_, inserted] = layout_to_name.try_emplace(member, name);
            internal_assert(inserted)
                << "duplicate insertion of member: " << member
                << "with name: " << name;
        }
        return type;
    }

    ir::Type insert_group_layout(const ir::Member &member,
                                 const std::string &name,
                                 const ir::Type &type) {
        {
            auto [_, inserted] = layout_to_type.try_emplace(member, type);
            internal_assert(inserted)
                << "duplicate insertion of member: " << member
                << "with type: " << type;
        }
        {
            auto [_, inserted] = layout_to_name.try_emplace(member, name);
            internal_assert(inserted)
                << "duplicate insertion of member: " << member
                << "with name: " << name;
        }
        return type;
    }

    const ir::Type &type(const ir::Member &member) const {
        const auto it = layout_to_type.find(member);
        internal_assert(it != layout_to_type.cend()) << member;
        return it->second;
    }

    // Returns the concretized name used in the struct lowering.
    const std::string &concrete_name(const ir::Member &member) const {
        const auto it = layout_to_name.find(member);
        internal_assert(it != layout_to_name.cend()) << member;
        return it->second;
    }

    const auto &types() const { return layout_to_type; }
    const auto &names() const { return layout_to_name; }

    bool contains_group(const std::string &name) const {
        return group_map.contains(name);
    }

    ir::Member group(const std::string &name) const {
        const auto it = group_map.find(name);
        internal_assert(it != group_map.cend()) << name;
        return it->second;
    }

    void update_group_map(const std::map<std::string, ir::Member> &group_map) {
        for (const auto &[name, group] : group_map) {
            auto [_, inserted] = this->group_map.try_emplace(name, group);
            internal_assert(inserted) << name;
        }
    }

  private:
    std::map<ir::Member, ir::Type, ir::MemberLessThan> layout_to_type;
    std::map<ir::Member, std::string, ir::MemberLessThan> layout_to_name;

    std::map<std::string, ir::Member> group_map;
};

[[maybe_unused]] std::ostream &operator<<(std::ostream &os,
                                          const LayoutTypeMap &map) {
    os << "layout -> type {\n";
    for (const auto &[member, type] : map.types()) {
        os << member << " : " << type << "\n";
    }
    os << "}\n";
    return os;
}

std::string pad_name(uint32_t count) { return "pad" + std::to_string(count); }

std::string split_name(uint32_t count, const std::string &field) {
    return "split" + std::to_string(count) + "on_" + field;
}

struct FindFromType : public ir::Visitor {
    ir::Type from_type;

    void visit(const ir::YieldFrom *node) override {
        if (from_type.defined()) {
            internal_assert(ir::equals(from_type, node->value.type()))
                << "Mismatching types in YieldFrom: " << node->value
                << " is of type " << node->value.type()
                << ", not: " << from_type;
        } else {
            from_type = node->value.type();
        }
    }
};

ir::Expr fill(const ir::MapStack<std::string, ir::Expr> &frames,
              const ir::Expr &expr, const ir::Layout &layout) {
    struct Rewrite : public ir::Mutator {
        const ir::MapStack<std::string, ir::Expr> &frames;
        const ir::Layout &layout;

        Rewrite(const ir::MapStack<std::string, ir::Expr> &frames,
                const ir::Layout &layout)
            : frames(frames), layout(layout) {}

        ir::Expr visit(const ir::Access *node) override {
            const auto *e = node->value.as<ir::Extract>();
            if (e == nullptr) {
                return mutate(node);
            }
            // TODO(cgyurgyik): hack for sorts.
            const auto *v = e->vec.as<ir::Var>();
            if (!(v && v->name == "this")) {
                return mutate(node);
            }
            // ll = this[left].low; (used for sorting)
            return fill(
                frames,
                ir::Var::make(ir::Vector_t::make(ir::Float_t::make_f32(), 3),
                              node->field),
                layout);
        }

        ir::Expr visit(const ir::Var *var) override {
            if (var->name == "this") {
                return var;
            }
            // check if this is a tree-carried dependency.
            if (var->name == "parent") {
                return var;
            }
            // check if this already exists in this scope.
            if (std::optional<ir::Expr> e = frames.from_frames(var->name);
                e.has_value()) {
                return *e;
            }
            // check if this is a root variable.
            if (auto it = std::find_if(layout.root.begin(), layout.root.end(),
                                       [&](const ir::Argument &arg) {
                                           return arg.name == var->name;
                                       });
                it != layout.root.end()) {
                return ir::Var::make(it->type, it->name);
            }

            const auto *bvh_t = layout.type.as<ir::BVH_t>();
            internal_assert(bvh_t) << layout.type;
            ir::Type found_type;
            for (const ir::BVH_t::Variant &variant : bvh_t->variants) {
                for (const auto &[fname, ftype] : variant.fields()) {
                    if (fname != var->name) {
                        continue;
                    }
                    found_type = ftype;
                    break;
                }
            }
            if (found_type.defined()) {
                return ir::Var::make(std::move(found_type), var->name);
            }

            internal_error << "materialization fill cannot find: `" << var->name
                           << "` of type: " << var->type;
        }

        ir::Expr visit(const ir::Call *node) override {
            std::vector<ir::Expr> args = node->args;
            for (ir::Expr &arg : args) {
                arg = mutate(arg);
            }
            return ir::Call::make(node->func, std::move(args));
        }

        ir::Expr visit(const ir::Build *node) override {
            std::vector<ir::Expr> values = node->values;
            for (ir::Expr &value : values) {
                value = mutate(value);
            }
            return ir::Build::make(node->type, std::move(values));
        }
    };
    return Rewrite(frames, layout).mutate(expr);
}

// Groups may be interdependent. This builds fields for the outer layout so that
// materializations can references fields in other groups.
// TODO(cgyurgyik): this is incomplete, e.g., it doesn't support visiting splits
// and lookups. Note that compilation will still gracefully fail if a field
// exists in an unsupported member since it won't be found.
void add_fields(const ir::Expr &base, const ir::Member &member,
                ir::MapStack<std::string, ir::Expr> &frames,
                const LayoutTypeMap &ltmap, const ir::Layout &layout) {
    const ir::Chain *chain = to_chainz(member);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const ir::Field *node = m.as<ir::Field>();
            frames.maybe_add_to_frame(node->name,
                                      ir::Access::make(node->name, base));
            continue;
        }
        case ir::IRLayoutEnum::Group: {
            const ir::Group *node = m.as<ir::Group>();
            std::string field_name = node->name;
            switch (node->type) {
            case ir::Group::Type::Direct: {
                ir::Expr path = base;
                if (m.bits() > 0) {
                    path = ir::Access::make(field_name, base);
                    ir::Expr index;
                    if (!index.defined()) {
                        index = ir::Var::make(ir::Index_t::make(), "<hole>");
                    }

                    path = ir::Extract::make(std::move(path), index);
                    frames.maybe_add_to_frame(node->name, index);
                }
                add_fields(path, node->inner, frames, ltmap, layout);
                break;
            }
            default:
                break;
            }
            continue;
        }
        case ir::IRLayoutEnum::Materialize: {
            const ir::Materialize *node = m.as<ir::Materialize>();
            ir::Expr materialization = fill(frames, node->value, layout);
            frames.maybe_add_to_frame(node->loc.base(),
                                      std::move(materialization));

            continue;
        }
        case ir::IRLayoutEnum::Lookup:
        case ir::IRLayoutEnum::Split:
        case ir::IRLayoutEnum::Pad: {
            continue;
        }
        default:
            internal_error << "[unimplemented] member: " << m;
        }
    }
}

// Returns the struct equivalent for this layout member, and updates the layout
// type map respectively.
ir::Type layout_to_struct(const std::string &name, const ir::Member &member,
                          LayoutTypeMap &ltmap,
                          ir::Member parent = ir::Member()) {
    if (auto it = ltmap.types().find(member); it != ltmap.types().cend()) {
        return it->second;
    }
    uint32_t pad_count = 0, split_count = 0;
    ir::Struct_t::Map fields;
    const ir::Chain *chain = to_chainz(member);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const ir::Field *node = m.as<ir::Field>();
            fields.emplace_back(node->name, node->type);
            continue;
        }
        case ir::IRLayoutEnum::Pad: {
            const ir::Pad *node = m.as<ir::Pad>();
            ir::Type pad_type = ir::UInt_t::make(node->bits);
            fields.emplace_back(pad_name(pad_count++), std::move(pad_type));
            continue;
        }
        case ir::IRLayoutEnum::Group: {
            const ir::Group *node = m.as<ir::Group>();
            // TODO(cgyurgyik): this is incorrect. if two inner groups match,
            // then the second group will received the first group's member,
            // including its name. We need to uniquely match on body *and* name.
            ir::Type base_t =
                         layout_to_struct(node->name, node->inner, ltmap, node),
                     group_t;
            switch (node->type) {
            case ir::Group::Type::Pointer:
                internal_assert(!node->size.defined());
                group_t = ir::Ptr_t::make(std::move(base_t));
                break;
            case ir::Group::Type::Direct:
            case ir::Group::Type::Indirect:
                group_t = ir::Array_t::make(std::move(base_t), node->size);
                break;
            }
            std::string field_name = node->name;
            ltmap.insert_group_layout(m, field_name, group_t);
            if (m.bits() == 0) {
                continue;
            }
            fields.emplace_back(std::move(field_name), std::move(group_t));

            continue;
        }
        case ir::IRLayoutEnum::Split: {
            const ir::Split *node = m.as<ir::Split>();
            // Store as vector of bytes, load and reinterpret to proper
            // type.
            if (uint64_t bits = m.bits(); bits > 0) {
                internal_assert(bits % 8 == 0)
                    << "Split is not byte-aligned (" << bits << "): " << m;
                static const ir::Type u8 = ir::UInt_t::make(8);
                ir::Type byte_vector = ir::Vector_t::make(u8, bits / 8);
                std::string name =
                    split_name(split_count++, node->field_name());
                fields.emplace_back(std::move(name), std::move(byte_vector));
            }
            // Cache the struct-type of each arm.
            for (const ir::Arm &arm : node->arms) {
                std::string name = "arm_";
                internal_assert(arm.name.has_value()) << arm;
                name += *arm.name;
                layout_to_struct(std::move(name), arm.member, ltmap, node);
            }
            continue;
        }
        case ir::IRLayoutEnum::Materialize:
        case ir::IRLayoutEnum::Lookup:
            continue;
        default:
            internal_error << "[unimplemented] member in struct lowering: ";
        }
    }
    return ltmap.insert_struct_layout(member, name.empty() ? "unit" : name,
                                      std::move(fields), parent);
}

// TODO(cgyurgyik): misnomer, fix this.
struct NameSize {
    std::string name;
    ir::Expr size;
};

std::vector<NameSize> name_to_size(ir::Expr e, const LayoutTypeMap &map) {
    struct Visit : public ir::Visitor {
      public:
        Visit(const LayoutTypeMap &map) : map(map) {}
        std::vector<NameSize> mapping;

      private:
        const LayoutTypeMap &map;

        void visit(const ir::Extract *node) override {
            const auto *idx = node->idx.as<ir::Var>();
            if (idx == nullptr) {
                node->vec.accept(this);
                return;
            }
            if (idx->name != "<hole>") {
                node->vec.accept(this);
                return;
            }
            const auto *ac = node->vec.as<ir::Access>();
            internal_assert(ac) << node->vec;
            if (!map.contains_group(ac->field)) {
                ac->value.accept(this);
                return;
            }
            ir::Member group = map.group(ac->field);
            // For non-constants, this should be the index.
            // For constants, this should be size.
            const auto *g = group.as<ir::Group>();
            mapping.push_back({
                ac->field,
                g->size.defined() && is_const(g->size) ? g->size : g->index,
            });
            ac->value.accept(this);
        }
    };

    Visit visit(map);
    internal_assert(e.defined());
    e.accept(&visit);
    return visit.mapping;
}

// Replace <hole> index expressions with the correct offset expression.
//
// TODO(cgyurgyik): HACK. My approach in Builds is way cleaner since I maintain
// a working list of groups while building up the WriteLoc. Do something similar
// here.
ir::Expr fill_index_holes(ir::Expr e, const LayoutTypeMap &map) {
    class ReplaceHole : public ir::Mutator {
      public:
        ReplaceHole(const std::vector<NameSize> &mapping) : mapping(mapping) {}

        ir::Expr visit(const ir::Extract *node) override {
            const auto *index = node->idx.as<ir::Var>();
            if (index == nullptr || index->name != "<hole>") {
                return ir::Mutator::visit(node);
            }
            const auto *access = node->vec.as<ir::Access>();
            if (access == nullptr) {
                return ir::Mutator::visit(node);
            }
            auto [name, size] = mapping.back();
            mapping.pop_back();
            internal_assert(name == access->field)
                << name << " vs " << access->field;
            return ir::Extract::make(ir::Mutator::visit(access),
                                     std::move(size));
        }

      private:
        std::vector<NameSize> mapping;
    };

    if (const auto *node = e.as<ir::Slice>()) {
        return ir::Slice::make(fill_index_holes(node->value, map),
                               fill_index_holes(node->begin, map),
                               fill_index_holes(node->end, map),
                               fill_index_holes(node->step, map));
    }
    if (const auto *node = e.as<ir::Call>()) {
        std::vector<ir::Expr> args;
        for (int i = 0; i < node->args.size(); ++i) {
            args.push_back(fill_index_holes(node->args[i], map));
        }
        return ir::Call::make(node->func, std::move(args));
    }
    if (const auto *node = e.as<ir::Extract>()) {
        return ir::Extract::make(fill_index_holes(node->vec, map),
                                 fill_index_holes(node->idx, map));
    }
    if (const auto *node = e.as<ir::Build>()) {
        std::vector<ir::Expr> values;
        for (const ir::Expr &value : node->values) {
            values.push_back(fill_index_holes(value, map));
        }
        return ir::Build::make(node->type, std::move(values));
    }
    if (const auto *node = e.as<ir::BinOp>()) {
        return ir::BinOp::make(node->op, fill_index_holes(node->a, map),
                               fill_index_holes(node->b, map));
    }

    std::vector<NameSize> mapping = name_to_size(e, map);
    if (mapping.empty()) {
        return e;
    }
    if (mapping.size() == 1) {
        return ReplaceHole(mapping).mutate(e);
    }
    std::reverse(mapping.begin(), mapping.end());
    // Otherwise, we make a bunch of assumptions (for now).
    // 1. There is a single non-constant index.
    // 2. The rest are indexes of an equal constant size.
    ir::Expr constant, nonconstant;
    for (int32_t i = 0, e = mapping.size(); i < e; ++i) {
        const NameSize &ns = mapping[i];
        if (is_const(ns.size)) {
            if (constant.defined()) {
                internal_assert(ir::equals(ns.size, constant));
                continue;
            }
            constant = ns.size;
        } else {
            internal_assert(!nonconstant.defined() ||
                            ir::equals(nonconstant, ns.size))
                << "non-constant already found: `" << nonconstant << "`";
            nonconstant = ns.size;
            continue;
        }
    }
    // Then, we update the indexes.
    for (int32_t i = 0, e = mapping.size(); i < e; ++i) {
        NameSize &ns = mapping[i];
        ns.size =
            is_const(ns.size) ? nonconstant % constant : nonconstant / constant;
    }
    return ReplaceHole(mapping).mutate(e);
}

ir::Expr field_from_layout(const ir::Expr &base, const ir::Member &member,
                           ir::MapStack<std::string, ir::Expr> frames,
                           const std::string &iter_name,
                           const std::string &node_type,
                           const std::string &field, const LayoutTypeMap &ltmap,
                           const ir::Layout &layout, const ir::Expr &root,
                           bool is_lookup = false) {
    uint32_t split_count = 0;
    const ir::Chain *chain = to_chainz(member);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const ir::Field *node = m.as<ir::Field>();
            ir::Expr load = ir::Access::make(node->name, base);
            if (node->name == field) {
                // Found it! Just return a read from the current path.
                return load;
            } else {
                // Otherwise insert into the current frame (it may be used in
                // materialization).
                frames.maybe_add_to_frame(node->name, std::move(load));
            }
            continue;
        }
        case ir::IRLayoutEnum::Group: {
            const ir::Group *node = m.as<ir::Group>();
            std::string field_name = node->name;
            switch (node->type) {
            case ir::Group::Type::Direct: {
                ir::Expr path = base;
                if (m.bits() > 0) {
                    path = ir::Access::make(field_name, base);
                    ir::Expr index;
                    if (!index.defined()) {
                        index = ir::Var::make(ir::Index_t::make(), "<hole>");
                    }
                    path = ir::Extract::make(std::move(path), index);
                    frames.maybe_add_to_frame(node->name, index);
                }
                if (ir::Expr found = field_from_layout(
                        path, node->inner, frames, iter_name, node_type, field,
                        ltmap, layout, root);
                    found.defined()) {
                    return found;
                }
                break;
            }
            case ir::Group::Type::Indirect: {
                if (is_lookup) {
                    if (ir::Expr found = field_from_layout(
                            base, node->inner, frames, iter_name, node_type,
                            field, ltmap, layout, root);
                        found.defined()) {
                        return found;
                    }
                }
                break;
            }
            case ir::Group::Type::Pointer: {
                ir::Expr path = ir::Access::make(field_name, base);
                if (field == field_name) {
                    return path;
                }
                path = ir::Deref::make(path);
                if (ir::Expr found = field_from_layout(
                        path, node->inner, frames, iter_name, node_type, field,
                        ltmap, layout, root);
                    found.defined()) {
                    return found;
                }
                break;
            }
            }
            continue;
        }
        case ir::IRLayoutEnum::Split: {
            const ir::Split *node = m.as<ir::Split>();
            if (field == node->field_name()) {
                // TODO(cgyurgyik): hot fix for values from the root. Fix this.
                return node->expr;
            }
            for (const ir::Arm &arm : node->arms) {
                if (const std::optional<std::string> &name = arm.name;
                    !name.has_value() || *name != node_type) {
                    continue;
                }
                ir::Expr path = base;
                if (m.bits() > 0) {
                    std::string field_name =
                        split_name(split_count++, node->field_name());
                    path = ir::Access::make(std::move(field_name), path);
                }
                ir::Type reinterpret_type = ltmap.type(arm.member);
                path = ir::Cast::make(reinterpret_type, path,
                                      ir::Cast::Mode::Reinterpret);

                frames.push_frame();
                ir::Expr recurse =
                    field_from_layout(path, arm.member, frames, iter_name,
                                      node_type, field, ltmap, layout, root);
                frames.pop_frame();
                if (recurse.defined()) {
                    return recurse;
                }
            }
            continue;
        }
        case ir::IRLayoutEnum::Lookup: {
            const ir::Lookup *node = m.as<ir::Lookup>();
            // from <group_name>[<index>]
            ir::Member group = ltmap.group(node->group_name);
            std::string concretized_name = ltmap.concrete_name(group);
            ir::Expr path = ir::Access::make(concretized_name, root);
            path = ir::Extract::make(path, node->index);
            frames.push_frame();
            ir::Expr recurse = field_from_layout(
                path, group, frames, iter_name, node_type, field, ltmap, layout,
                root, /*is_lookup=*/true);
            frames.pop_frame();
            internal_assert(recurse.defined());
            return recurse;
        }
        case ir::IRLayoutEnum::Materialize: {
            const ir::Materialize *node = m.as<ir::Materialize>();
            ir::Expr materialization = fill(frames, node->value, layout);
            frames.maybe_add_to_frame(node->loc.base(), materialization);
            if (node->loc.base() == field) {
                return materialization;
            }
            continue;
        }
        case ir::IRLayoutEnum::Pad: {
            continue;
        }
        default:
            internal_error << "[unimplemented] member: " << m;
        }
    }
    return ir::Expr();
}

ir::Expr field_in_layout(const ir::Expr &base, const ir::Member &member,
                         ir::MapStack<std::string, ir::Expr> frames,
                         const std::string &iter_name,
                         const std::string &node_type, const std::string &field,
                         const LayoutTypeMap &ltmap, const ir::Layout &layout,
                         const ir::Expr &root, bool is_lookup = false) {
    ir::Expr concretized_field =
        field_from_layout(base, member, frames, iter_name, node_type, field,
                          ltmap, layout, root, is_lookup);
    if (!concretized_field.defined()) {
        if (std::optional<ir::Expr> expr = frames.from_frames(field)) {
            // This was a previously materialized field.
            return ir::Var::make(expr->type(), field);
        }
    }
    internal_assert(concretized_field.defined())
        << "no concretized field found for: `" << field << "`";
    return fill_index_holes(std::move(concretized_field), ltmap);
}

// Removes TCD `parent.` accesses.
class RemoveTreeCarriedDependencies : public ir::Mutator {
  public:
    explicit RemoveTreeCarriedDependencies(
        const std::set<std::string> &parents,
        const std::vector<ir::Argument> &index_list)
        : parents(parents), index_list(index_list) {}

    ir::Expr visit(const ir::Access *node) override {
        if (!parents.contains(node->field)) {
            return ir::Access::make(node->field, mutate(node->value));
        }
        auto it = std::find_if(
            index_list.begin(), index_list.end(),
            [&](const ir::Argument &a) { return a.name == node->field; });
        internal_assert(it != index_list.end())
            << "[unexpected] tree-carried dependency not found in layout "
               "arguments: `"
            << node->field << "`";
        return ir::Var::make(it->type, node->field);
    }

  private:
    const std::set<std::string> &parents;
    const std::vector<ir::Argument> &index_list;
};

ir::Stmt lower_switch_tree(ir::Member member, const std::string &obj_name,
                           const LayoutTypeMap &ltmap, const ir::Layout &layout,
                           const ir::Expr &root) {
    struct FindPaths : public ir::Visitor {
        using Path = std::vector<std::pair<ir::Expr, ir::Arm>>;
        Path current;
        std::vector<std::pair<std::string, Path>> paths;

        void visit(const ir::Split *node) override {
            for (const auto &arm : node->arms) {
                current.emplace_back(node->expr, arm);
                if (!arm.name.has_value()) {
                    arm.member.accept(this); // Check for deeper splits.
                    continue;
                }
                internal_assert(std::find_if(paths.begin(), paths.end(),
                                             [&](const auto &p) {
                                                 return p.first == *arm.name;
                                             }) == paths.end())
                    << "unexpected duplicate variant: " << *arm.name;
                paths.emplace_back(*arm.name, current);
            }
        }
    };
    FindPaths finder;
    member.accept(&finder);

    // TODO: should this be scheduable...?
    // TODO: we want to insert likely() for non-leaves, I think?
    std::vector<std::string> order;
    for (const auto &[node_type, _] : finder.paths) {
        order.push_back(node_type);
    }
    // std::sort(order.begin(), order.end(),
    //           [&](const std::string &a, const std::string &b) {
    //               // TODO: caching this would make this faster,
    //               // but we probably never have a large number.
    //               auto count_non_null = [](const FindPaths::Path &path) {
    //                   return std::count_if(
    //                       path.begin(), path.end(),
    //                       [](const auto &p) { return p.second.has_value();
    //                       });
    //               };
    //               return count_non_null(finder.paths[a]) <
    //                      count_non_null(finder.paths[b]);
    //           });

    ir::Stmt if_chain;
    for (const std::string &node_type : std::views::reverse(order)) {
        // Make a hole for the body of this node type.
        ir::Stmt body = ir::Label::make(node_type, ir::Stmt());
        if (!if_chain.defined()) {
            if_chain = std::move(body);
            continue;
        }
        ir::Expr condition;
        auto it =
            std::find_if(finder.paths.begin(), finder.paths.end(),
                         [&](const auto &p) { return p.first == node_type; });
        internal_assert(it != finder.paths.end()) << node_type;
        const FindPaths::Path &path = it->second;
        for (const auto &[expr, arm] : path) {
            ir::Expr value = field_in_layout(
                /*base=*/root,
                /*member=*/member,
                /*frames=*/{},
                /*iter_name=*/obj_name,
                /*node_type=*/node_type,
                /*field=*/get_field_name(expr),
                /*ltmap=*/ltmap,
                /*layout=*/layout,
                /*root=*/root);
            if (const auto *s = expr.as<ir::Slice>()) {
                value = ir::Slice::make(value, s->begin, s->end, s->step);
            } else {
                // TODO(cgyurgyik): need to ensure additional operations, e.g.,
                // a bitwise operation, are performed on the split field which
                // may now be an expression.
                internal_assert(expr.is<ir::Var>())
                    << "[unimplemented] additional operations on the split "
                       "value: "
                    << expr;
            }

            if (!arm.value.has_value()) {
                // This is a wildcard.
                condition = ir::BoolImm::make(true);
            } else {
                ir::Expr constant = make_const(value.type(), *arm.value);
                switch (arm.comparator) {
                case ir::Arm::Comparator::EQ:
                    condition = std::move(value) == std::move(constant);
                    break;
                case ir::Arm::Comparator::GT:
                    condition = std::move(value) > std::move(constant);
                    break;
                case ir::Arm::Comparator::LT:
                    condition = std::move(value) < std::move(constant);
                    break;
                case ir::Arm::Comparator::GE:
                    condition = std::move(value) >= std::move(constant);
                    break;
                case ir::Arm::Comparator::LE:
                    condition = std::move(value) <= std::move(constant);
                    break;
                case ir::Arm::Comparator::NE:
                    condition = std::move(value) != std::move(constant);
                    break;
                }
            }
        }

        internal_assert(condition.defined());
        if_chain = ir::IfElse::make(std::move(condition), std::move(body),
                                    std::move(if_chain));
    }
    return if_chain;
}

struct LowerUnwrapAccesses : public ir::Mutator {
    const std::string &tree_name;
    const ir::Type &tree_layout;
    const std::string &node_type;
    const std::map<std::string, ir::Expr> &field_map;

    ir::MapStack<std::string, ir::Type> type_repls;

    LowerUnwrapAccesses(const std::string &tree_name,
                        const ir::Type &tree_layout,
                        const std::string &node_type,
                        const std::map<std::string, ir::Expr> &field_map)
        : tree_name(tree_name), tree_layout(tree_layout), node_type(node_type),
          field_map(field_map) {}

    ir::Expr visit(const ir::Var *node) override {
        if (auto new_type = type_repls.from_frames(node->name)) {
            return ir::Var::make(*new_type, node->name);
        }
        return node;
    }

    ir::Stmt visit(const ir::LetStmt *node) override {
        internal_assert(node->loc.accesses.empty());
        ir::Expr value = mutate(node->value);
        if (value.same_as(node->value)) {
            return node;
        }
        if (equals(value.type(), node->loc.base_type())) {
            return ir::LetStmt::make(node->loc, std::move(value));
        }
        ir::WriteLoc new_loc(node->loc.base(), value.type());
        type_repls.maybe_add_to_frame(node->loc.base(), value.type());
        return ir::LetStmt::make(std::move(new_loc), std::move(value));
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        internal_assert(node->loc.accesses.empty());
        ir::Expr value = mutate(node->value);
        internal_assert(value.defined()) << node->value;
        if (value.same_as(node->value)) {
            return node;
        }
        if (equals(value.type(), node->loc.base_type())) {
            return ir::Allocate::make(node->loc, std::move(value),
                                      node->memory);
        }
        ir::WriteLoc new_loc(node->loc.base(), value.type());
        type_repls.maybe_add_to_frame(node->loc.base(), value.type());
        return ir::Allocate::make(std::move(new_loc), std::move(value),
                                  node->memory);
    }

    ir::Expr visit(const ir::Access *node) override {
        const ir::Unwrap *as_unwrap = node->value.as<ir::Unwrap>();
        if (as_unwrap == nullptr) {
            return ir::Mutator::visit(node);
        }
        internal_assert(as_unwrap->value.is<ir::Var>())
            << "[unimplemented] access of Unwrap on non-Var: "
            << ir::Expr(node);
        std::string variable_name = as_unwrap->value.as<ir::Var>()->name;

        if (variable_name == tree_name) {
            const auto *struct_t = as_unwrap->type.as<ir::Struct_t>();
            internal_assert(struct_t) << as_unwrap->type;
            if (struct_t->name == node_type) {
                const auto &it = field_map.find(node->field);
                internal_assert(it != field_map.cend())
                    << "In lowering of " << ir::Expr(node)
                    << ", failed to find field: " << node->field
                    << " in field map of " << tree_name;
                internal_assert(it->second.defined())
                    << "undefined field for access: " << node->field;
                return it->second;
            }
        }

        // Not the rewrite we're looking for.
        return ir::Mutator::visit(node);
    }

    std::pair<std::vector<ir::Expr>, bool>
    visit_list(const std::vector<ir::Expr> &exprs) {
        bool not_changed = true;
        const size_t n = exprs.size();
        std::vector<ir::Expr> new_exprs(n);
        for (size_t i = 0; i < n; i++) {
            new_exprs[i] = mutate(exprs[i]);
            not_changed = not_changed && new_exprs[i].same_as(exprs[i]);
        }
        return {std::move(new_exprs), not_changed};
    }

    ir::Expr visit(const ir::Build *node) override {
        if (!node->type.is<ir::Tuple_t>()) {
            return Mutator::visit(node);
        }
        // Handle the case that YieldFrom / Scan lowering
        // changed the types of Tuples being built.
        auto [values, not_changed] = visit_list(node->values);
        if (not_changed) {
            return node;
        }
        return make_tuple(std::move(values));
    }

    std::string get_tree_name(const ir::Expr &expr) const {
        struct Getter : public ir::Visitor {
            std::string name;
            void visit(const ir::Unwrap *node) override {
                internal_assert(node->value.is<ir::Var>())
                    << "[unimplemented] Unwrap on non-Var: " << ir::Expr(node);

                internal_assert(name.empty())
                    << name << " when finding: " << ir::Expr(node);
                name = node->value.as<ir::Var>()->name;
            }
        };
        Getter getter;
        expr.accept(&getter);
        internal_assert(!getter.name.empty())
            << "get_tree_name failed on: " << expr;
        return getter.name;
    }

    ir::Expr make_new_call(const std::vector<ir::Expr> &args, size_t added_idx,
                           const ir::Expr &call) {
        // Need to change function signature of function
        const ir::Var *var = call.as<ir::Var>();
        internal_assert(var) << call;
        const ir::Function_t *func_t = var->type.as<ir::Function_t>();
        internal_assert(func_t) << call;

        std::vector<ir::Function_t::ArgSig> arg_types(args.size());

        for (size_t i = 0; i < args.size(); i++) {
            arg_types[i].type = args[i].type();
            arg_types[i].is_mutable =
                (added_idx == i)
                    ? false
                    : ((added_idx < i) ? func_t->arg_types[i - 1].is_mutable
                                       : func_t->arg_types[i].is_mutable);
        }

        ir::Type new_func_t =
            ir::Function_t::make(func_t->ret_type, std::move(arg_types));
        return ir::Var::make(std::move(new_func_t), var->name);
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        bool not_changed = true;
        const size_t n = node->args.size();
        std::vector<ir::Expr> new_args(n);
        size_t partition = 0;
        for (size_t i = 0; i < n; i++) {
            ir::Expr repl = mutate(node->args[i]);
            bool changed = !repl.same_as(node->args[i]);
            if (changed && node->args[i].type().is<ir::Ref_t>()) {
                std::string t = get_tree_name(node->args[i]);
                if (t == tree_name) {
                    partition = i + 1;
                }
            }
            new_args[i] = std::move(repl);
            not_changed = not_changed && !changed;
        }

        if (partition) {
            ir::Expr new_arg = ir::Var::make(tree_layout, tree_name);
            new_args.insert(new_args.begin() + partition, new_arg);
            ir::Expr new_func = make_new_call(new_args, partition, node->func);
            return ir::CallStmt::make(std::move(new_func), std::move(new_args));
        }

        if (not_changed) {
            return node;
        }
        // Assume the func can't be mutated.
        return ir::CallStmt::make(node->func, std::move(new_args));
    }
};

struct FillHole : public ir::Mutator {
    const std::string &label_name;
    ir::Stmt repl;

    FillHole(const std::string &label_name, ir::Stmt repl)
        : label_name(label_name), repl(std::move(repl)) {}

    ir::Stmt visit(const ir::Label *node) override {
        if (node->name == label_name) {
            internal_assert(!node->body.defined())
                << "Expected hole when lowering: " << label_name
                << " branch to " << repl;
            internal_assert(repl.defined())
                << "Found multiple holes when lowering: " << label_name;
            return std::move(repl);
        }
        return ir::Mutator::visit(node);
    }
};

ir::Expr flatten_tuple(ir::Expr expr,
                       const std::map<std::string, ir::Expr> &references) {
    std::vector<ir::Expr> exprs;

    std::function<void(const ir::Expr &)> handle_tuple =
        [&](const ir::Expr &t) -> void {
        if (const ir::Build *as_build = t.as<ir::Build>()) {
            for (const ir::Expr &expr : as_build->values) {
                handle_tuple(expr);
            }
            return;
        }
        if (const ir::Var *var = t.as<ir::Var>()) {

            if (const auto &iter = references.find(var->name);
                iter != references.cend()) {
                handle_tuple(iter->second);
                return;
            }
        }
        if (const ir::Extract *ext = t.as<ir::Extract>()) {
            const ir::Build *vec = ext->vec.as<ir::Build>();
            if (vec == nullptr) {
                exprs.push_back(t);
                return;
            }
            internal_assert(vec) << ext->vec;
            std::optional<uint64_t> idx = get_constant_value(ext->idx);
            internal_assert(idx.has_value()) << ext->idx;
            handle_tuple(vec->values[*idx]);
            return;
        }
        internal_assert(!t.type().is<ir::Tuple_t>())
            << "[unimplemented] flatten_tuple of non-Build: " << t;
        exprs.push_back(t);
    };

    handle_tuple(expr);

    internal_assert(!exprs.empty());

    // Base case, no tuple:
    if (exprs.size() == 1) {
        return expr;
    }

    std::vector<ir::Type> etypes;
    etypes.reserve(exprs.size());
    std::transform(exprs.begin(), exprs.end(), std::back_inserter(etypes),
                   [](const ir::Expr &e) { return e.type(); });

    ir::Type tuple = ir::Tuple_t::make(std::move(etypes));
    return ir::Build::make(std::move(tuple), std::move(exprs));
}

ir::Stmt flatten_yield_froms(ir::Stmt body,
                             const std::vector<ir::Argument> &index_list,
                             const std::map<std::string, ir::Expr> &references,
                             const LayoutTypeMap &map) {
    struct FlattenYieldFroms : public ir::Mutator {
        const std::vector<ir::Argument> &index_list;
        const std::map<std::string, ir::Expr> &references;
        const LayoutTypeMap &map;

        FlattenYieldFroms(const std::vector<ir::Argument> &index_list,
                          const std::map<std::string, ir::Expr> &references,
                          const LayoutTypeMap &map)
            : index_list(index_list), references(references), map(map) {}

        ir::Stmt visit(const ir::YieldFrom *node) override {
            auto ids = break_tuple(node->value);
            std::vector<ir::Expr> flat_ids;
            flat_ids.reserve(ids.size());

            for (auto &id : ids) {
                ir::Expr value = flatten_tuple(id, references);
                ir::Type type = value.type();
                if (index_list.size() == 1) {
                    internal_assert(ir::equals(type, index_list[0].type))
                        << "Mismatching YieldFroms, expected type: "
                        << index_list[0].type << " but found type: " << type
                        << " in: " << ir::Stmt(node);
                } else {
                    if (const ir::Tuple_t *tuple = type.as<ir::Tuple_t>()) {
                        internal_assert(tuple->etypes.size() ==
                                        index_list.size())
                            << "expected `" << index_list.size()
                            << "` values, but found `" << tuple->etypes.size()
                            << "` values: `" << type
                            << "` in recursive function of: `" << ir::Stmt(node)
                            << "`\n with type: `" << type
                            << "` of flattened id: `" << id << "`";
                        // TODO(cgyurgyik): the fuck is going on --- when is the
                        // index list being (incorrectly) reversed?
                        std::vector<ir::Argument> index_list2 = index_list;
                        std::reverse(index_list2.begin(), index_list2.end());
                        for (size_t i = 0; i < index_list2.size(); i++) {
                            internal_assert(ir::equals(index_list2[i].type,
                                                       tuple->etypes[i]))
                                << "Mismatching YieldFroms, expected type: "
                                << index_list2[i].type
                                << " but found type: " << tuple->etypes[i]
                                << " at index: " << i
                                << " in: " << ir::Stmt(node);
                        }
                    }
                }
                flat_ids.push_back(std::move(value));
            }
            ir::Expr value = make_tuple(std::move(flat_ids));
            return ir::YieldFrom::make(std::move(value));
        }
    };

    FlattenYieldFroms f(index_list, references, map);
    return f.mutate(std::move(body));
}

struct LowerMatches : public ir::Mutator {
    const ir::LayoutMap &layout_map;
    const ir::TypeMap &structs;
    const LayoutTypeMap &layout_type_map;

    LowerMatches(const ir::LayoutMap &layout_map, const ir::TypeMap &structs,
                 const LayoutTypeMap &layout_type_map)
        : layout_map(layout_map), structs(structs),
          layout_type_map(layout_type_map) {}

    ir::Member get_member(const std::string &tree_name) {
        const auto &iter = layout_map.find(tree_name);
        internal_assert(iter != layout_map.cend())
            << "Failed to find member of: " << tree_name;
        return iter->second.body;
    }

    ir::Layout get_layout(const std::string &tree_name) {
        const auto &iter = layout_map.find(tree_name);
        internal_assert(iter != layout_map.cend())
            << "Failed to find layout of: " << tree_name;
        return iter->second;
    }

    ir::Type get_struct_type(const std::string &tree_name) {
        const auto &iter = structs.find(tree_name);
        internal_assert(iter != structs.cend())
            << "Failed to find type of: " << tree_name;
        return iter->second;
    }

    std::vector<ir::Argument> index_list;
    std::set<std::string> matched_objects;
    std::map<std::string, ir::Expr> references;
    std::string tree_name;
    size_t counter = 0;

    std::string get_unique_loop_label() {
        return "_loop" + std::to_string(counter++);
    }

    ir::Stmt visit(const ir::RecLoop *node) override {
        // Should not be in a match right now.
        internal_assert(references.empty()) << ir::Stmt(node);
        ir::Stmt body = mutate(node->body);
        body = flatten_yield_froms(std::move(body), index_list, references,
                                   layout_type_map);
        references.clear();
        const ir::Layout &layout = get_layout(tree_name);
        std::set<std::string> parents = layout.tree_carried_dependencies();
        RemoveTreeCarriedDependencies rtcd(parents, index_list);
        std::vector<ir::Stmt> statements;
        for (const ir::Argument &index : index_list) {
            ir::Expr v = ir::Var::make(index.type, index.name);
            ir::Expr is_sentinel =
                index.type.is<ir::Ptr_t>()
                    ? ir::UnOp::make(ir::UnOp::OpType::Not, v)
                    : v == make_all_ones(index.type);
            statements.push_back(
                ir::IfElse::make(std::move(is_sentinel), ir::Return::make()));
        }
        statements.push_back(std::move(body));
        body =
            rtcd.mutate(std::move(ir::Sequence::make(std::move(statements))));
        return ir::RecLoop::make(std::move(index_list), std::move(body));
    }

    ir::Stmt visit(const ir::Match *node) override {
        internal_assert(node->loc.is<ir::Var>())
            << "[unimplemented] Match on non-Var: " << ir::Stmt(node);
        tree_name = node->loc.as<ir::Var>()->name;

        ir::Type struct_type = get_struct_type(tree_name);
        ir::Member member = get_member(tree_name);
        ir::Layout layout = get_layout(tree_name);

        ir::Expr root = ir::Var::make(struct_type, tree_name);
        ir::Stmt body =
            lower_switch_tree(member, tree_name, layout_type_map, layout, root);
        if (!body.defined()) {
            // Even the tagged representation is implicitly represented.
            std::vector<ir::Stmt> stmts;
            const std::vector<ir::BVH_t::Variant> variants = layout.variants();
            stmts.reserve(variants.size());
            std::transform(
                variants.begin(), variants.end(), std::back_inserter(stmts),
                [](const ir::BVH_t::Variant &variant) {
                    return ir::Label::make(variant.name(), ir::Stmt());
                });
            body = ir::Sequence::make(std::move(stmts));
        }
        // tree-carried dependencies need to be added to the references for
        // further processing when visiting YieldFrom.
        for (const auto &[name, expr] : layout.tree_carried_updates()) {
            references.insert({name, expr});
        }

        for (const auto &[variant, statement] : node->arms) {
            std::map<std::string, ir::Expr> field_map;
            const std::string &branch_name = variant.name();
            for (const auto &field : variant.fields()) {
                ir::MapStack<std::string, ir::Expr> frames;
                add_fields(root, member, frames, layout_type_map, layout);
                field_map[field.name] = field_in_layout(
                    root, member, frames, tree_name, branch_name, field.name,
                    layout_type_map, layout, root);
            }

            // Lower these Unwraps.
            ir::Stmt branch_body =
                LowerUnwrapAccesses(tree_name, branch_name, field_map)
                    .mutate(statement);
            body = FillHole(branch_name, std::move(branch_body))
                       .mutate(std::move(body));
        }

        // First time we see a tree, add it's type to the type parameters list.
        if (!matched_objects.contains(tree_name)) {
            std::vector<ir::Argument> node_index_list = layout.root;
            for (ir::Argument &arg : node_index_list) {
                if (const auto *ptr_t = arg.type.as<ir::Ptr_t>()) {
                    // Add implicit default argument for pointers.
                    internal_assert(!arg.default_value.defined());
                    const auto *ref_t = ptr_t->etype.as<ir::Ref_t>();
                    internal_assert(ref_t) << arg;
                    arg.default_value = ir::Var::make(ptr_t, ref_t->name);
                }
                if (!arg.default_value.defined()) {
                    continue;
                }
                if (is_const(arg.default_value)) {
                    continue;
                }
                const auto *v = arg.default_value.as<ir::Var>();
                internal_assert(v) << arg.default_value;
                ir::Expr base = ir::Var::make(struct_type, tree_name);
                arg.default_value = field_from_layout(
                    base, layout.body, /*frames=*/{}, "?", "?", v->name,
                    layout_type_map, layout, root);
            }
            std::reverse(node_index_list.begin(), node_index_list.end());
            std::vector<ir::Expr> idxs;
            idxs.reserve(node_index_list.size());
            for (auto &it : node_index_list) {
                idxs.push_back(ir::Var::make(it.type, it.name));
            }
            references[tree_name] = make_tuple(std::move(idxs));
            index_list.insert(index_list.end(),
                              std::make_move_iterator(node_index_list.begin()),
                              std::make_move_iterator(node_index_list.end()));
            matched_objects.insert(tree_name);
        }

        // Now recursively mutate the body, for nested matches.
        return mutate(body);
    }

    ir::Expr visit(const ir::Var *node) override {
        if (structs.contains(node->name)) {
            return ir::Var::make(structs.at(node->name), node->name);
        }
        return node;
    }

    struct MutatedArgSig {
        std::vector<ir::Expr> args;
        ir::Type new_type;
        bool changed;
    };

    MutatedArgSig mutate_call(const ir::Function_t *func_t,
                              const std::vector<ir::Expr> &args) {
        const size_t n = args.size();
        internal_assert(n == func_t->arg_types.size());

        bool changed = false;
        std::vector<ir::Expr> ret_args(n);
        std::vector<ir::Function_t::ArgSig> arg_types(n);

        for (size_t i = 0; i < n; i++) {
            ir::Expr arg = mutate(args[i]);
            changed = changed || !arg.same_as(args[i]);
            arg_types[i].type = arg.type();
            arg_types[i].is_mutable = func_t->arg_types[i].is_mutable;
            ret_args[i] = std::move(arg);
        }
        if (changed) {
            return {
                std::move(ret_args),
                ir::Function_t::make(func_t->ret_type, std::move(arg_types)),
                changed};
        } else {
            return {{}, {}, false};
        }
    }

    template <typename I, typename T>
    I handle(const T *node) {
        // TODO(ajr): do we ever mutate node->func?
        const ir::Function_t *func_t =
            node->func.type().template as<ir::Function_t>();
        internal_assert(func_t);
        auto check = mutate_call(func_t, node->args);
        if (!check.changed) {
            return node;
        }
        // Need to change function signature of node->func
        const ir::Var *var = node->func.template as<ir::Var>();
        internal_assert(var);

        ir::Type new_type = std::move(check.new_type);
        ir::Expr func = ir::Var::make(std::move(new_type), var->name);
        return T::make(std::move(func), std::move(check.args));
    }

    ir::Expr visit(const ir::Call *node) override {
        return handle<ir::Expr>(node);
    }

    ir::Stmt visit(const ir::CallStmt *node) override {
        return handle<ir::Stmt>(node);
    }

    ir::Expr visit(const ir::Build *node) override {
        bool not_changed = true;
        bool not_changed_type = true;
        const size_t n = node->values.size();
        std::vector<ir::Expr> values(n);
        for (size_t i = 0; i < n; i++) {
            values[i] = mutate(node->values[i]);
            not_changed = not_changed && values[i].same_as(node->values[i]);
            not_changed_type =
                not_changed_type &&
                ir::equals(values[i].type(), node->values[i].type());
        }
        if (not_changed) {
            return node;
        }
        if (not_changed_type) {
            return ir::Build::make(node->type, std::move(values));
        }
        internal_assert(node->type.is<ir::Tuple_t>())
            << "Mutated type of non-tuple in member lowering: "
            << ir::Expr(node);
        return make_tuple(std::move(values));
    }
};

std::map<std::string, ir::Member> get_group_map(const ir::Layout &layout) {
    struct GetGroupMap : ir::Visitor {
        void visit(const ir::Group *node) override {
            std::string name = node->name;
            if (node->type == ir::Group::Type::Indirect) {
                name.front() = std::toupper(name.front());
            }
            const auto [_, inserted] = map.insert({std::move(name), node});
            internal_assert(inserted)
                << "unexpected duplicate group name: " << node->name;

            // visit nested groups.
            node->inner.accept(this);
        };
        std::map<std::string, ir::Member> map;
    };

    GetGroupMap ggm;
    layout.body.accept(&ggm);
    return ggm.map;
}

ir::Stmt replace_sentinel(const ir::Stmt &body, const ir::LayoutMap &map) {
    struct Replace : public ir::Mutator {
        const ir::LayoutMap &map;
        Replace(const ir::LayoutMap &map) : map(map) {}

        ir::Expr visit(const ir::BinOp *node) override {
            const ir::Expr &a = node->a;
            const ir::Expr &b = node->b;
            if (!(a.is<ir::Var>() && b.is<ir::Var>())) {
                return node;
            }

            const auto *av = a.as<ir::Var>();
            internal_assert(av) << a;
            const auto *bv = b.as<ir::Var>();
            internal_assert(bv) << b;
            if (bv->name == "nullptr") {
                const auto *bvh_t = bv->type.as<ir::BVH_t>();
                internal_assert(bvh_t) << bv->type;
                internal_assert(map.size() == 1);
                const auto &[_, layout] = *map.begin();
                ir::Expr lhs = ir::Var::make(layout.get_index_type(), av->name);
                if (layout.get_index_type().is<ir::Ptr_t>()) {
                    return lhs == make_zero(layout.get_index_type());
                }
                return lhs == make_all_ones(layout.get_index_type());
            }
            return node;
        }
    };
    return Replace(map).mutate(body);
}

} // namespace

ir::Program LowerLayouts::run(ir::Program program,
                              const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    const ir::LayoutMap &tree_layouts =
        program.schedules[ir::Target::Host].tree_layouts;

    if (tree_layouts.empty()) {
        return program;
    }

    ir::TypeMap types;
    LayoutTypeMap layout_type_map;
    for (const auto &[name, layout] : tree_layouts) {
        layout_type_map.update_group_map(get_group_map(layout));

        ir::Type struct_t =
            layout_to_struct(name, layout.body, layout_type_map);
        types[name] = struct_t;
        program.types.emplace(name, struct_t);

        bool found = false;
        for (auto &[ename, etype] : program.externs) {
            if (name == ename) {
                found = true;
                etype = struct_t;
                break;
            }
        }
        internal_assert(found)
            << "extern `" << name << "` has a layout, but not found";

        for (const auto &[_, type] : layout_type_map.types()) {
            if (const auto *struct_t = type.as<ir::Struct_t>()) {
                if (struct_t->fields.empty()) {
                    continue;
                }
                program.types[struct_t->name] = type;
            }
        }
    }

    // lower all `Access`es on `Unwrap`s
    LowerMatches lower(tree_layouts, types, layout_type_map);

    for (auto &[fname, func] : program.funcs) {
        if (fname.starts_with("_scan")) {

            std::vector<ir::Function::Argument> new_args;

            // All arguments except the last are trees and should be replaced.
            for (size_t i = 0; i + 1 < func->args.size(); ++i) {
                const auto &arg = func->args[i];

                // Replace type if mapped
                auto type_it = types.find(arg.name);
                internal_assert(type_it != types.end())
                    << arg.name << "in _scan has no layout.";

                auto layout = tree_layouts.find(arg.name);
                internal_assert(layout != tree_layouts.end())
                    << arg.name << "in _scan has no layout.";

                // Get index struct and expand its fields as args
                auto index_type = get_index_type(layout->second);

                // Each needs to also accept the arguments returned by
                // `get_index_type(layout)` using the layout associated with
                // that tree type.
                for (const auto &idx_t : index_type) {
                    new_args.emplace_back(arg.name + "_" + idx_t.name,
                                          idx_t.type);
                }

                new_args.emplace_back(arg.name, type_it->second);
            }
            new_args.push_back(func->args.back()); // write location.
            func->args = new_args;
        } else {
            for (auto &arg : func->args) {
                if (types.contains(arg.name)) {
                    arg.type = types.at(arg.name);
                }
            }
        }
        func->body = lower.mutate(func->body);
        func->body = replace_sentinel(func->body, tree_layouts);
    }
    return program;
}

} // namespace lower
} // namespace bonsai
