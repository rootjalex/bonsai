#include "Lower/Layouts.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/ValidateLayout.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Log.h"
#include "Utils.h"

#include <ranges>

namespace bonsai {
namespace lower {

namespace {

class LayoutMap {
  public:
    ir::Type insert_struct_layout(const ir::Member &member,
                                  const std::string &name,
                                  const ir::Struct_t::Map &fields) {
        ir::Type type =
            ir::Struct_t::make(name, fields, {ir::Struct_t::Attribute::packed});
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

    [[maybe_unused]] ir::Type insert_group_layout(const ir::Member &member,
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

    ir::Member group(const std::string &name) const {
        const auto it = group_map.find(name);
        internal_assert(it != group_map.cend()) << name;
        return it->second;
    }

    // Returns a unique name for this mapping. `tl` is short for "tree layout."
    std::string get_unique_name() { return "tl" + std::to_string(counter++); }

    void update_group_map(std::map<std::string, ir::Member> group_map) {
        this->group_map = std::move(group_map);
    }

  private:
    uint64_t counter = 0;
    std::map<ir::Member, ir::Type, ir::MemberLessThan> layout_to_type;
    // TODO(cgyurgyik): This can also just be pulled from the struct types.
    std::map<ir::Member, std::string, ir::MemberLessThan> layout_to_name;

    std::map<std::string, ir::Member> group_map;
};

[[maybe_unused]] std::ostream &operator<<(std::ostream &os,
                                          const LayoutMap &map) {
    os << "layout -> type {\n";
    for (const auto &[member, type] : map.types()) {
        os << member << " : " << type << "\n";
    }
    os << "}\n";
    return os;
}

std::string pad_name(uint32_t count) { return "pad" + std::to_string(count); }

std::string group_name(uint32_t count, const std::string &index,
                       ir::Group::Type type) {
    std::string name;
    if (type == ir::Group::Type::Indirect) {
        name += "indirect_";
    }
    name += "group_" + index + std::to_string(count);
    return name;
}

std::string split_name(uint32_t count, const std::string &field) {
    return "split" + std::to_string(count) + "on_" + field;
}

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

using IndexTList = std::vector<ir::TypedVar>;

IndexTList get_index_type(const ir::Member &member) {
    IndexTList index_ts;
    if (const ir::Chain *chain = to_chainz(member)) {
        ir::Struct_t::Map fields;
        for (const auto &m : chain->members) {
            switch (m.node_type()) {
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = m.as<ir::Group>();
                if (!node->index.defined()) {
                    continue;
                }
                index_ts = get_index_type(node->inner);
                auto *index = node->index.as<ir::Var>();
                index_ts.push_back({index->name, index->type});
                break;
            }
            case ir::IRLayoutEnum::Split: {
                const ir::Split *node = m.as<ir::Split>();
                for (const auto &arm : node->arms) {
                    auto rec = get_index_type(arm.member);
                    internal_assert(rec.empty())
                        << "[unimplemented] groups inside splits: " << member;
                }
                break;
            }
            case ir::IRLayoutEnum::Field:
            case ir::IRLayoutEnum::Pad:
            case ir::IRLayoutEnum::Materialize:
            case ir::IRLayoutEnum::Lookup:
                break;
            case ir::IRLayoutEnum::Chain: {
                internal_error << "[unimplemented] nested chains: " << member;
            }
            }
        }
        return index_ts;
    }
    internal_error << "[unimplemented] handle get_index_type for: " << member;
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

        ir::Expr visit(const ir::Var *var) override {
            if (var->name == "range" && var->type.is<ir::Function_t>()) {
                const ir::Function_t *func = var->type.as<ir::Function_t>();
                internal_assert(func->ret_type.is<ir::Array_t>());
                const ir::Array_t *array = func->ret_type.as<ir::Array_t>();
                ir::Expr size = array->size;
                size = mutate(size);
                ir::Type ret_type =
                    ir::Array_t::make(array->etype, std::move(size));
                return ir::Var::make(
                    ir::Function_t::make(std::move(ret_type), func->arg_types),
                    var->name);
            }
            std::optional<ir::Expr> expr = frames.from_frames(var->name);
            if (!expr.has_value()) {
                auto it = std::find_if(layout.root.begin(), layout.root.end(),
                                       [&](const ir::Argument &arg) {
                                           return arg.name == var->name;
                                       });
                if (it != layout.root.end()) {
                    expr = ir::Var::make(it->type, it->name);
                }
            }
            internal_assert(expr.has_value())
                << "Materialization fill cannot find: " << var->name;
            return *expr;
        }
    };
    return Rewrite(frames, layout).mutate(expr);
}

// Returns the struct equivalent for this layout member, and updates the layout
// type map respectively.
ir::Type layout_to_struct(const ir::Member &member, LayoutMap &lmap) {
    if (auto it = lmap.types().find(member); it != lmap.types().cend()) {
        return it->second;
    }

    uint32_t pad_count = 0;
    uint32_t group_count = 0;
    uint32_t split_count = 0;
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
            ir::Type base_t = layout_to_struct(node->inner, lmap);
            ir::Type group_t = ir::Array_t::make(std::move(base_t), node->size);
            std::string field_name =
                group_name(group_count++, node->name, node->type);
            lmap.insert_group_layout(m, field_name, group_t);
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
                layout_to_struct(arm.member, lmap);
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
    std::string name = lmap.get_unique_name();
    return lmap.insert_struct_layout(member, name, std::move(fields));
}

ir::Expr field_in_layout(const ir::Expr &base, const ir::Member &member,
                         ir::MapStack<std::string, ir::Expr> frames,
                         const std::string &iter_name,
                         const std::string &node_type, const std::string &field,
                         const LayoutMap &lmap, const ir::Layout &layout,
                         const ir::Expr &root, bool is_lookup = false) {
    uint32_t group_count = 0, split_count = 0;
    const ir::Chain *chain = to_chainz(member);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const ir::Field *node = m.as<ir::Field>();
            ir::Expr load = ir::Access::make(node->name, base);
            if (node->name == field) {
                // Found it!
                // Just return a read from the current path.
                return load;
            } else {
                // Otherwise insert into current frame,
                // might be used in materialization.
                frames.add_to_frame(node->name, std::move(load));
            }
            continue;
        }
        case ir::IRLayoutEnum::Group: {
            const ir::Group *node = m.as<ir::Group>();
            std::string field_name =
                group_name(group_count++, node->name, node->type);
            switch (node->type) {
            case ir::Group::Type::Direct: {
                ir::Expr path = ir::Access::make(field_name, base);
                frames.push_frame();
                internal_assert(node->index.defined()) << m;
                ir::Expr index = node->index;
                path = ir::Extract::make(std::move(path), index);
                frames.add_to_frame(node->name, index);
                ir::Expr recurse =
                    field_in_layout(path, node->inner, frames, iter_name,
                                    node_type, field, lmap, layout, root);
                frames.pop_frame();
                if (recurse.defined()) {
                    return recurse;
                }
                break;
            }
            case ir::Group::Type::Indirect:
                if (is_lookup) {
                    frames.push_frame();
                    ir::Expr recurse =
                        field_in_layout(base, node->inner, frames, iter_name,
                                        node_type, field, lmap, layout, root);
                    frames.pop_frame();
                    internal_assert(recurse.defined());
                    return recurse;
                }
                break;
            }
            continue;
        }
        case ir::IRLayoutEnum::Split: {
            const ir::Split *node = m.as<ir::Split>();
            // Stored as vector of bytes, load and reinterpret to proper
            // type.
            if (field == node->field_name()) {
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
                    ir::Expr path =
                        ir::Access::make(std::move(field_name), base);
                }
                ir::Type reinterpret_type = lmap.type(arm.member);
                path = ir::Cast::make(reinterpret_type, path,
                                      ir::Cast::Mode::Reinterpret);

                frames.push_frame();
                ir::Expr recurse =
                    field_in_layout(path, arm.member, frames, iter_name,
                                    node_type, field, lmap, layout, root);
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
            ir::Member group = lmap.group(node->group_name);
            std::string concretized_name = lmap.concrete_name(group);
            ir::Expr path = ir::Access::make(concretized_name, root);
            path = ir::Extract::make(path, node->index);
            frames.push_frame();
            ir::Expr recurse =
                field_in_layout(path, group, frames, iter_name, node_type,
                                field, lmap, layout, root, /*is_lookup=*/true);
            frames.pop_frame();
            internal_assert(recurse.defined());
            return recurse;
        }
        case ir::IRLayoutEnum::Materialize: {
            const ir::Materialize *node = m.as<ir::Materialize>();
            ir::Expr materialization = fill(frames, node->value, layout);
            if (node->name == field) {
                return materialization;
            } else {
                frames.add_to_frame(node->name, std::move(materialization));
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

ir::Stmt lower_switch_tree(ir::Member member, ir::Expr base,
                           const std::string &obj_name, const LayoutMap &lmap,
                           const ir::Layout &layout, const ir::Expr &root) {
    struct FindPaths : public ir::Visitor {
        // TODO(cgyurgyik): enable ADTs to have same children name.
        using Path = std::vector<std::pair<std::string, ir::Arm>>;
        Path current;
        std::vector<std::pair<std::string, Path>> paths;

        void visit(const ir::Split *node) override {
            for (const auto &arm : node->arms) {
                current.emplace_back(node->field_name(), arm);
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
            // TODO: this doesn't work if it's possible to have fully NULL
            // reprs.
            if_chain = std::move(body);
            continue;
        }
        ir::Expr condition;
        auto it =
            std::find_if(finder.paths.begin(), finder.paths.end(),
                         [&](const auto &p) { return p.first == node_type; });
        internal_assert(it != finder.paths.end()) << node_type;
        const FindPaths::Path &path = it->second;
        for (const auto &[field_name, arm] : path) {
            ir::Expr value = field_in_layout(
                /*base=*/base,
                /*member=*/member,
                /*frames=*/{},
                /*iter_name=*/obj_name,
                /*node_type=*/node_type,
                /*field=*/field_name,
                /*lmap=*/lmap,
                /*layout=*/layout,
                /*root=*/root);

            if (arm.value.has_value()) {
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
                continue;
            }
            // This is a wildcard.
            if (!condition.defined()) {
                condition = ir::BoolImm::make(true);
            }
        }

        internal_assert(condition.defined());
        if_chain = ir::IfElse::make(std::move(condition), std::move(body),
                                    std::move(if_chain));
    }
    internal_assert(if_chain.defined());
    return if_chain;
}

struct LowerUnwrapAccesses : public ir::Mutator {
    const std::string &tree_name;
    const std::string &node_type;
    const std::map<std::string, ir::Expr> &field_map;

    ir::MapStack<std::string, ir::Type> type_repls;

    LowerUnwrapAccesses(const std::string &tree_name,
                        const std::string &node_type,
                        const std::map<std::string, ir::Expr> &field_map)
        : tree_name(tree_name), node_type(node_type), field_map(field_map) {}

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
        if (equals(value.type(), node->loc.base_type)) {
            return ir::LetStmt::make(node->loc, std::move(value));
        }
        ir::WriteLoc new_loc(node->loc.base, value.type());
        type_repls.add_to_frame(node->loc.base, value.type());
        return ir::LetStmt::make(std::move(new_loc), std::move(value));
    }

    ir::Stmt visit(const ir::Allocate *node) override {
        internal_assert(node->loc.accesses.empty());
        ir::Expr value = mutate(node->value);
        internal_assert(value.defined()) << node->value;
        if (value.same_as(node->value)) {
            return node;
        }
        if (equals(value.type(), node->loc.base_type)) {
            return ir::Allocate::make(node->loc, std::move(value),
                                      node->memory);
        }
        ir::WriteLoc new_loc(node->loc.base, value.type());
        type_repls.add_to_frame(node->loc.base, value.type());
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
        } else if (const ir::Var *var = t.as<ir::Var>()) {
            if (const auto &iter = references.find(var->name);
                iter != references.cend()) {
                handle_tuple(iter->second);
                return;
            }
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

ir::Stmt
flatten_yield_froms(const IndexTList &index_list, ir::Stmt body,
                    const std::map<std::string, ir::Expr> &references) {
    struct FlattenYieldFroms : public ir::Mutator {
        const IndexTList &index_list;
        const std::map<std::string, ir::Expr> &references;

        FlattenYieldFroms(const IndexTList &index_list,
                          const std::map<std::string, ir::Expr> &references)
            : index_list(index_list), references(references) {}

        ir::Stmt visit(const ir::YieldFrom *node) override {
            std::vector<ir::Expr> ids = break_tuple(node->value);
            std::vector<ir::Expr> flat_ids;
            flat_ids.reserve(ids.size());

            for (ir::Expr &id : ids) {
                ir::Expr value = flatten_tuple(id, references);
                ir::Type type = value.type();
                if (index_list.size() == 1) {
                    internal_assert(ir::equals(type, index_list[0].type))
                        << "Mismatching YieldFroms, expected type: "
                        << index_list[0].type << " but found type: " << type
                        << " in: " << ir::Stmt(node);
                } else {
                    const ir::Tuple_t *tuple = type.as<ir::Tuple_t>();
                    internal_assert(tuple &&
                                    tuple->etypes.size() == index_list.size())
                        << "Expected " << index_list.size()
                        << " values, but found: " << type
                        << " in recursive function of: " << ir::Stmt(node)
                        << "\n with type: " << type
                        << " of flattened id: " << id;

                    for (size_t i = 0; i < index_list.size(); i++) {
                        internal_assert(
                            ir::equals(index_list[i].type, tuple->etypes[i]))
                            << "Mismatching YieldFroms, expected type: "
                            << index_list[i].type
                            << " but found type: " << tuple->etypes[i]
                            << " at index: " << i << " in: " << ir::Stmt(node);
                    }
                }
                flat_ids.push_back(std::move(value));
            }
            ir::Expr value = make_tuple(std::move(flat_ids));
            return ir::YieldFrom::make(std::move(value));
        }
    };

    FlattenYieldFroms f(index_list, references);
    return f.mutate(std::move(body));
}

struct LowerMatches : public ir::Mutator {
    const ir::LayoutMap &members;
    const ir::TypeMap &structs;
    const LayoutMap &lmap;
    ir::Layout layout;

    LowerMatches(const ir::LayoutMap &members, const ir::TypeMap &structs,
                 const LayoutMap &lmap)
        : members(members), structs(structs), lmap(lmap) {}

    std::map<std::string, ir::Type> ref_types;
    IndexTList index_list;
    std::set<std::string> matched_objects;
    std::map<std::string, ir::Expr> references;

    size_t counter = 0;

    std::string get_unique_loop_label() {
        return "_loop" + std::to_string(counter++);
    }

    ir::Stmt visit(const ir::RecLoop *node) override {
        // Should not be in a match right now.
        internal_assert(references.empty()) << ir::Stmt(node);
        ir::Stmt body = mutate(node->body);
        body = flatten_yield_froms(index_list, std::move(body), references);
        references.clear();
        std::vector<ir::Argument> args = layout.root;
        for (int i = 0, e = index_list.size(); i < e; ++i) {
            auto it = std::find_if(args.begin(), args.end(),
                                   [&](const ir::Argument &arg) {
                                       return arg.name == index_list[i].name;
                                   });
            if (it == args.end()) {
                args.push_back(ir::Argument::from(index_list[i]));
            }
        }
        return ir::RecLoop::make(std::move(args), std::move(body));
    }

    ir::Stmt visit(const ir::Match *node) override {
        internal_assert(node->loc.is<ir::Var>())
            << "[unimplemented] Match on non-Var: " << ir::Stmt(node);
        const std::string tree_name = node->loc.as<ir::Var>()->name;

        // Now, based on member, form switch-tree.
        ir::Member member = [&]() {
            const auto &iter = members.find(tree_name);
            internal_assert(iter != members.cend())
                << "Failed to find member of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            return iter->second.body;
        }();

        {
            const auto &iter = members.find(tree_name);
            internal_assert(iter != members.cend())
                << "Failed to find member of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            layout = iter->second;
        }

        ir::Type struct_type = [&]() {
            const auto &iter = structs.find(tree_name);
            internal_assert(iter != structs.cend())
                << "Failed to find type of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            return iter->second;
        }();

        ir::Expr root = ir::Var::make(struct_type, tree_name);
        ir::Stmt body =
            lower_switch_tree(member, root, tree_name, lmap, layout, root);

        for (const auto &[variant, statement] : node->arms) {
            std::map<std::string, ir::Expr> field_map;
            const std::string &branch_name = variant.name();
            for (const auto &field : variant.fields()) {
                field_map[field.name] = field_in_layout(
                    root, member, /*frames=*/{}, tree_name, branch_name,
                    field.name, lmap, layout, root);
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
            IndexTList node_index_list = get_index_type(member);
            std::reverse(node_index_list.begin(), node_index_list.end());
            std::vector<ir::Expr> idxs;
            std::transform(node_index_list.begin(), node_index_list.end(),
                           std::back_inserter(idxs),
                           [](const auto &it) { return ir::Var::from(it); });

            references[tree_name] = make_tuple(std::move(idxs));
            index_list.insert(index_list.end(),
                              std::make_move_iterator(node_index_list.begin()),
                              std::make_move_iterator(node_index_list.end()));
        }
        matched_objects.insert(tree_name);

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
            const auto [_, inserted] = map.insert({node->name, node});
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

} // namespace

ir::Program LowerLayouts::run(ir::Program program,
                              const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    ir::LayoutMap tree_layouts =
        std::move(program.schedules[ir::Target::Host].tree_layouts);

    if (tree_layouts.empty()) {
        return program;
    }

    ir::TypeMap types;
    LayoutMap lmap;
    for (const auto &[name, layout] : tree_layouts) {
        lmap.update_group_map(get_group_map(layout));

        ir::Type struct_t = layout_to_struct(layout.body, lmap);
        types[name] = struct_t;

        bool found = false;
        for (auto &[ename, etype] : program.externs) {
            if (name == ename) {
                found = true;
                etype = struct_t;
                break;
            }
        }
        internal_assert(found)
            << "extern " << name << " has member but not found.\n";

        for (const auto &[_, type] : lmap.types()) {
            if (const auto *struct_t = type.as<ir::Struct_t>()) {
                program.types[struct_t->name] = type;
            }
        }
    }

    // lower all `Access`es on `Unwrap`s
    LowerMatches lower(tree_layouts, types, lmap);

    for (auto &[fname, func] : program.funcs) {
        for (auto &arg : func->args) {
            if (types.contains(arg.name)) {
                arg.type = types.at(arg.name);
            }
        }
        func->body = lower.mutate(func->body);
    }
    return program;
}

} // namespace lower
} // namespace bonsai
