#include "Lower/Layouts.h"

#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/ValidateLayout.h"
#include "IR/Visitor.h"

#include "Opt/Simplify.h"

#include "Error.h"
#include "Utils.h"

#include <functional>
#include <ranges>

namespace bonsai {
namespace lower {

namespace {

struct LayoutTypeMap {
    std::map<ir::Layout, ir::Type, ir::LayoutLessThan> layout_to_type;
    std::map<ir::Layout, std::string, ir::LayoutLessThan> layout_to_name;
    // Named groups, by what the source called them, so that a lookup can be
    // resolved to the storage it names. A group has to be declared before the
    // arm that looks it up, which a chain read in order gives for free.
    struct Named {
        ir::Layout inner;  // the group's element layout
        std::string field; // the field of the enclosing struct holding it
        // The same group as a layout that can be *walked* rather than looked
        // up: one row at a time, advancing its own index. An indirect group is
        // exactly this when something traverses it instead of reading a single
        // row out of it, which is what a tree stored in one is.
        ir::Layout walkable;
        // The struct the group is a field of -- what a walk of it is rooted
        // at.
        ir::Type owner;
        // And what that struct is called where the program can name it: the
        // tree the schedule gave this layout to. A tree held in an element is
        // stored in the same object as the tree that reaches it, so a walk of
        // it reads through that object's name and not through the element's
        // field, which holds only where to start.
        std::string owner_name;
    };
    std::map<std::string, Named> groups;
    // Element fields the schedule bound to a tree, and the type a reference
    // into that tree's group is. Keyed `Element.field`.
    //
    // This is where a nested tree stops being a set and becomes an index: in
    // the *stored* form only. The program keeps saying `set[Triangle]`, which
    // is what the query language reasons about; storage says `u32`, which is
    // what a pool of shared subtrees needs. Keeping the two apart is what lets
    // an instance's tree be stored once and named twice.
    std::map<std::string, ir::Type> field_refs;
    uint64_t counter = 0;
};

// The type an element takes in storage: the same element, with every field the
// schedule bound to a tree replaced by a reference into that tree's group.
ir::Type stored_type(const ir::Type &type,
                     const std::map<std::string, ir::Type> &field_refs) {
    if (field_refs.empty()) {
        return type;
    }
    if (const auto *as_array = type.as<ir::Array_t>()) {
        ir::Type etype = stored_type(as_array->etype, field_refs);
        if (etype.same_as(as_array->etype)) {
            return type;
        }
        return ir::Array_t::make(std::move(etype), as_array->size);
    }
    // A variant whose arm holds a tree -- `Inst(render_from_instance, blas)`
    // beside `Solo(tri)` -- is stored with that arm holding the reference,
    // and keeps its name for the same reason the struct below does.
    if (const auto *as_adt = type.as<ir::ADT_t>()) {
        ir::ADT_t::Variants variants;
        variants.reserve(as_adt->variants.size());
        bool changed = false;
        for (const ir::Type &variant : as_adt->variants) {
            variants.push_back(stored_type(variant, field_refs));
            changed = changed || !variants.back().same_as(variant);
        }
        if (!changed) {
            return type;
        }
        return ir::ADT_t::make(as_adt->name, std::move(variants));
    }
    const auto *as_struct = type.as<ir::Struct_t>();
    if (as_struct == nullptr) {
        return type;
    }
    bool changed = false;
    ir::Struct_t::Map fields = as_struct->fields;
    for (auto &field : fields) {
        const auto ref = field_refs.find(as_struct->name + "." + field.name);
        if (ref != field_refs.cend()) {
            field.type = ref->second;
            changed = true;
            continue;
        }
        ir::Type inner = stored_type(field.type, field_refs);
        if (!inner.same_as(field.type)) {
            field.type = std::move(inner);
            changed = true;
        }
    }
    if (!changed) {
        return type;
    }
    // Keeps the name. There is only ever one element at runtime -- the stored
    // one -- and the set-typed spelling is a fiction the query language needs
    // and the machine never sees. Renaming would leave both live at once, and
    // everything that reads an element out of storage would stop typechecking
    // against everything that was written about it.
    return ir::Struct_t::make(as_struct->name, std::move(fields),
                              as_struct->attributes);
}

// The type a layout's own field takes in storage. A layout says how many bytes
// a field is, and lowers to exactly that: a `vec3f` in a layout is three
// floats, twelve bytes, where the vector the program computes with is the
// machine's sixteen. So a vector field is stored as the packed kind (see
// Vector_t::packed), and a read of it is converted to the unpacked kind where
// the field is read (field_in_layout).
ir::Type packed_storage(const ir::Type &type) {
    if (const auto *vector = type.as<ir::Vector_t>();
        vector != nullptr && !vector->packed) {
        return ir::Vector_t::make(vector->etype, vector->lanes,
                                  /*packed=*/true);
    }
    return type;
}

// Says the above about the whole program, and not only about the arrays a
// layout stores elements in.
//
// It has to be the whole program because `stored_type` keeps the element's
// name: after this there is one Instance -- the one holding an index where the
// set used to be -- so anything still describing the other one would be a
// second type of the same name. `place(i : Instance, ..)` is the case that
// forces it, since a traversal reads an instance out of storage and hands it
// straight to a function the program wrote.
struct RewriteStoredElements : public ir::Mutator {
    const std::map<std::string, ir::Type> &field_refs;

    RewriteStoredElements(const std::map<std::string, ir::Type> &field_refs)
        : field_refs(field_refs) {}

    ir::Type mutate(const ir::Type &type) override {
        // The contents first, so that an element holding an element holding a
        // tree comes out with the index in it either way round.
        return stored_type(ir::Mutator::mutate(type), field_refs);
    }

    using ir::Mutator::mutate;

    // The base Mutator leaves the types inside an expression as it found them;
    // these are the same four Lower/ADTs.cpp has to reach for the same reason.
    ir::Expr visit(const ir::Var *node) override {
        ir::Type type = mutate(node->type);
        if (type.same_as(node->type)) {
            return node;
        }
        return ir::Var::make(std::move(type), node->name);
    }

    std::pair<ir::WriteLoc, bool>
    mutate_writeloc(const ir::WriteLoc &loc) override {
        ir::Type base_type = mutate(loc.base_type);
        bool not_changed = base_type.same_as(loc.base_type);
        ir::WriteLoc new_loc(loc.base, std::move(base_type));
        for (const auto &value : loc.accesses) {
            if (const ir::Expr *expr = std::get_if<ir::Expr>(&value)) {
                ir::Expr new_value = mutate(*expr);
                not_changed = not_changed && new_value.same_as(*expr);
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        return {std::move(new_loc), not_changed};
    }

    ir::Expr visit(const ir::Build *node) override {
        std::vector<ir::Expr> values;
        values.reserve(node->values.size());
        bool not_changed = true;
        for (const ir::Expr &value : node->values) {
            ir::Expr new_value = mutate(value);
            not_changed = not_changed && new_value.same_as(value);
            values.push_back(std::move(new_value));
        }
        ir::Type type = mutate(node->type);
        if (not_changed && type.same_as(node->type)) {
            return node;
        }
        return ir::Build::make(std::move(type), std::move(values));
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        ir::Type type = mutate(node->type);
        if (value.same_as(node->value) && type.same_as(node->type)) {
            return node;
        }
        return ir::Cast::make(std::move(type), std::move(value), node->mode);
    }

    ir::Expr visit(const ir::Lambda *node) override {
        std::vector<ir::TypedVar> args;
        args.reserve(node->args.size());
        bool not_changed = true;
        for (const ir::TypedVar &arg : node->args) {
            ir::Type type = mutate(arg.type);
            not_changed = not_changed && type.same_as(arg.type);
            args.push_back(ir::TypedVar{arg.name, std::move(type)});
        }
        ir::Expr value = mutate(node->value);
        if (not_changed && value.same_as(node->value)) {
            return node;
        }
        return ir::Lambda::make(std::move(args), std::move(value));
    }
};

std::string pad_name(uint32_t count) { return "pad" + std::to_string(count); }

std::string group_name(uint32_t count, const std::string &index) {
    return "group" + std::to_string(count) + "_" + index;
}

std::string split_name(uint32_t count, const std::string &field) {
    return "split" + std::to_string(count) + "on_" + field;
}

using IndexTList = std::vector<ir::TypedVar>;

IndexTList get_index_type(const ir::Layout &layout) {
    IndexTList index_ts;
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                if (node->type == ir::Group::Type::Indirect) {
                    // Auxiliary storage. A lookup into it supplies its own
                    // index, so it is no part of the reference the traversal
                    // carries -- which is also why it may sit beside the
                    // direct group without the two being ambiguous.
                    break;
                }
                internal_assert(index_ts.empty())
                    << "[unimplemented] adjacent groups in layout: " << layout;
                index_ts = get_index_type(node->inner);
                index_ts.push_back({node->name, node->index_t});
                break;
            }
            case ir::IRLayoutEnum::Lookup:
                // Names a row of another group; stores nothing here.
                break;
            case ir::IRLayoutEnum::Switch: {
                const ir::Switch *node = l.as<ir::Switch>();
                for (const auto &arm : node->arms) {
                    auto rec = get_index_type(arm.layout);
                    internal_assert(rec.empty())
                        << "[unimplemented] groups inside splits: " << layout;
                }
                break;
            }
            case ir::IRLayoutEnum::Name:
                break;
            case ir::IRLayoutEnum::Pad:
                break;
            case ir::IRLayoutEnum::Materialize:
                break;
            case ir::IRLayoutEnum::Chain: {
                internal_error << "[unimplemented] nested chains: " << layout;
            }
            }
        }
        return index_ts;
    }
    if (layout.as<ir::Lookup>()) {
        // The row is addressed by the index the lookup carries, which the
        // traversal already holds; it adds no index of its own.
        return {};
    }
    internal_error << "[unimplemented] handle get_index_type for: " << layout;
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
              const ir::Expr &expr) {
    struct Rewrite : public ir::Mutator {
        const ir::MapStack<std::string, ir::Expr> &frames;

        Rewrite(const ir::MapStack<std::string, ir::Expr> &frames)
            : frames(frames) {}

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
            internal_assert(expr.has_value())
                << "Materialization fill cannot find: " << var->name;
            return *expr;
        }
    };
    return Rewrite(frames).mutate(expr);
}

ir::Type layout_to_structs(const ir::Layout &layout, LayoutTypeMap &ltmap) {
    if (const auto in_cache = ltmap.layout_to_type.find(layout);
        in_cache != ltmap.layout_to_type.cend()) {
        return in_cache->second;
    }
    // An arm whose fields are a row of another group is that row: its shape is
    // the group's element shape, wherever the group happens to be stored.
    if (const ir::Lookup *lookup = layout.as<ir::Lookup>()) {
        const auto named = ltmap.groups.find(lookup->group_name);
        internal_assert(named != ltmap.groups.cend())
            << "Lookup names group " << lookup->group_name
            << ", which no group in this layout declares. A group has to be "
               "declared before the arm that looks it up.";
        ir::Type row = layout_to_structs(named->second.inner, ltmap);
        ltmap.layout_to_type[layout] = row;
        return row;
    }
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
        uint32_t pad_count = 0;
        uint32_t group_count = 0;
        uint32_t split_count = 0;
        std::string name = "_tree_layout" + std::to_string(ltmap.counter++);
        std::vector<std::pair<std::string, const ir::Group *>> named_here;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Name: {
                const ir::Name *node = l.as<ir::Name>();
                fields.emplace_back(
                    node->name,
                    packed_storage(stored_type(node->type, ltmap.field_refs)));
                break;
            }
            case ir::IRLayoutEnum::Pad: {
                const ir::Pad *node = l.as<ir::Pad>();
                ir::Type pad_type = ir::UInt_t::make(node->bits);
                fields.emplace_back(pad_name(pad_count++), std::move(pad_type));
                break;
            }
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                ir::Type base_t = layout_to_structs(node->inner, ltmap);
                ir::Type group_t =
                    ir::Array_t::make(std::move(base_t), node->size);
                internal_assert(!node->name.empty());
                std::string field_name = group_name(group_count++, node->name);
                // A named group can be looked up, so remember where its rows
                // live and what shape they are. Declared before the arm that
                // names it, which reading the chain in order gives for free.
                if (!node->declared_name.empty()) {
                    const auto [_, added] = ltmap.groups.emplace(
                        node->declared_name,
                        LayoutTypeMap::Named{node->inner, field_name,
                                             ir::Layout(), ir::Type()});
                    internal_assert(added)
                        << "Two groups named " << node->declared_name
                        << ": a lookup could not say which it meant.";
                    named_here.emplace_back(node->declared_name, node);
                }
                // push back new field type.
                fields.emplace_back(std::move(field_name), std::move(group_t));
                break;
            }
            case ir::IRLayoutEnum::Switch: {
                const ir::Switch *node = l.as<ir::Switch>();
                // Store as vector of bytes, load and reinterpret to proper
                // type.
                //
                // A packed vector of bytes: storage of an exact width, not a
                // value, which is what `packed` says (see Vector_t). The arm
                // that owns the bytes reinterprets them as its own struct.
                const uint64_t bits = l.bits();
                if (bits > 0) {
                    internal_assert(bits % 8 == 0)
                        << "Switch is not byte-aligned: " << l;
                    static const ir::Type u8 = ir::UInt_t::make(8);
                    ir::Type byte_vec =
                        ir::Vector_t::make(u8, bits / 8, /*packed=*/true);
                    std::string name = split_name(split_count++, node->field);
                    fields.emplace_back(std::move(name), std::move(byte_vec));
                }
                // Cache the struct-type of each arm.
                // TODO(ajr): this fails if an arm is ever not a Chain, can that
                // happen?
                for (const auto &arm : node->arms) {
                    layout_to_structs(arm.layout, ltmap);
                }
                break;
            }
            case ir::IRLayoutEnum::Materialize:
                break;
            default: {
                internal_error << "Handle layout in Chain lowering: " << l;
            }
            }
        }

        {
            auto [_, inserted] = ltmap.layout_to_name.try_emplace(layout, name);
            internal_assert(inserted) << layout;
        }
        // Marked as a layout as well as packed, which is what says its bytes
        // may be read at more than one type (see Struct_t::Attribute::layout).
        const std::vector<ir::Struct_t::Attribute> attributes = {
            ir::Struct_t::Attribute::packed, ir::Struct_t::Attribute::layout};
        ir::Type struct_t =
            ir::Struct_t::make(std::move(name), std::move(fields), attributes);
        // A walk of a named group is rooted at the struct the group is a field
        // of, which only exists now that the chain is finished -- and it needs
        // the chain around it, not just the group: a leaf of that group reads
        // the primitive array declared beside it, and dropping the chain would
        // put that out of scope. So the walkable form is this chain with the
        // *other* groups taken out and this one walked directly.
        for (const auto &[declared, group] : named_here) {
            std::vector<ir::Layout> members;
            for (const auto &l : chain->layouts) {
                if (const ir::Group *g = l.as<ir::Group>()) {
                    if (g != group) {
                        continue;
                    }
                    members.push_back(ir::Group::make(
                        g->size, g->name, g->declared_name, g->index_t,
                        g->inner, ir::Group::Type::Direct));
                    continue;
                }
                members.push_back(l);
            }
            LayoutTypeMap::Named &named = ltmap.groups.at(declared);
            named.walkable = ir::Chain::make(std::move(members));
            named.owner = struct_t;
        }
        auto [_, inserted] = ltmap.layout_to_type.try_emplace(layout, struct_t);
        internal_assert(inserted) << layout << " already in cache\n";
        return struct_t;
    }
    internal_error << "Handle layout conversion for: " << layout;
}

ir::Expr field_in_layout(const ir::Expr &base, const ir::Layout &layout,
                         ir::MapStack<std::string, ir::Expr> frames,
                         const std::string &iter_name,
                         const std::string &node_type, const std::string &field,
                         const LayoutTypeMap &ltmap, ir::Expr group) {
    // The arm's fields are a row of another group: index that group and go on
    // looking inside the row.
    if (const ir::Lookup *lookup = layout.as<ir::Lookup>()) {
        const auto named = ltmap.groups.find(lookup->group_name);
        internal_assert(named != ltmap.groups.cend())
            << "Lookup names group " << lookup->group_name
            << ", which no group in this layout declares.";
        const std::optional<ir::Expr> rows =
            frames.from_frames(lookup->group_name);
        internal_assert(rows.has_value())
            << "Group " << lookup->group_name
            << " is not in scope where it is looked up. A group has to be "
               "declared before the arm that names it.";
        ir::Expr row =
            ir::Extract::make(*rows, fill(frames, lookup->index));
        return field_in_layout(std::move(row), named->second.inner, frames,
                               iter_name, node_type, field, ltmap,
                               std::move(group));
    }
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        uint32_t group_count = 0;
        uint32_t split_count = 0;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Name: {
                const ir::Name *node = l.as<ir::Name>();
                ir::Expr load = ir::Access::make(node->name, base);
                // A vector is stored packed and computed with unpacked; what
                // is read out of the layout is converted on the way.
                if (const auto *vector = load.type().as<ir::Vector_t>();
                    vector != nullptr && vector->packed) {
                    load = ir::Cast::make(
                        ir::Vector_t::make(vector->etype, vector->lanes),
                        std::move(load), ir::Cast::Mode::Convert);
                }
                if (node->name == field) {
                    // Found it!
                    // Just return a read from the current path.
                    return load;
                } else {
                    // Otherwise insert into current frame,
                    // might be used in materialization.
                    frames.add_to_frame(node->name, std::move(load));
                }
                break;
            }
            case ir::IRLayoutEnum::Pad: {
                break;
            }
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                std::string field_name = group_name(group_count++, node->name);
                if (node->type == ir::Group::Type::Indirect) {
                    // Not walked into: its rows are reached only through a
                    // lookup, which supplies the index, and descending here
                    // would invent an index variable nothing binds. Put the
                    // rows in scope under the name a lookup uses instead.
                    if (!node->declared_name.empty()) {
                        frames.add_to_frame(
                            node->declared_name,
                            ir::Access::make(field_name, base));
                    }
                    break;
                }
                ir::Expr push_group = ir::Access::make(field_name, base);
                ir::Expr index =
                    ir::Var::make(node->index_t, iter_name + "_" + node->name);
                ir::Expr path = ir::Extract::make(push_group, index);
                frames.push_frame();
                frames.add_to_frame(node->name, index);
                ir::Expr rec =
                    field_in_layout(path, node->inner, frames, iter_name,
                                    node_type, field, ltmap, push_group);
                frames.pop_frame();
                if (rec.defined()) {
                    return rec;
                }
                break;
            }
            case ir::IRLayoutEnum::Switch: {
                const ir::Switch *node = l.as<ir::Switch>();
                // Stored as vector of bytes, load and reinterpret to proper
                // type.
                for (const auto &arm : node->arms) {
                    if (!arm.name.has_value() || (*arm.name == node_type)) {
                        std::string field_name =
                            split_name(split_count++, node->field);

                        auto iter = ltmap.layout_to_type.find(arm.layout);
                        internal_assert(iter != ltmap.layout_to_type.cend())
                            << "Unseen Switch arm layout: " << ir::Layout(node)
                            << " at " << arm.layout;
                        ir::Type reinterpret_type = iter->second;

                        ir::Expr path;
                        if (l.bits() > 0) {
                            path =
                                ir::Access::make(std::move(field_name), base);
                            path = ir::Cast::make(reinterpret_type, path,
                                                  ir::Cast::Mode::Reinterpret);
                        }

                        frames.push_frame();
                        ir::Expr rec =
                            field_in_layout(path, arm.layout, frames, iter_name,
                                            node_type, field, ltmap, group);
                        frames.pop_frame();
                        if (rec.defined()) {
                            return rec;
                        }
                    }
                }
                break;
            }
            case ir::IRLayoutEnum::Materialize: {
                const ir::Materialize *node = l.as<ir::Materialize>();
                ir::Expr value = node->value;
                if (group.defined()) {
                    frames.push_frame();
                    frames.add_to_frame("this", group);
                }
                ir::Expr mat = fill(frames, value);
                if (group.defined()) {
                    frames.pop_frame();
                }
                if (node->name == field) {
                    return mat;
                } else {
                    // Otherwise insert into current frame,
                    // might be used in materialization.
                    frames.add_to_frame(node->name, std::move(mat));
                }
                break;
            }
            default: {
                internal_error << "Handle layout in Chain lowering: " << l;
            }
            }
        }
        return ir::Expr();
    }
    internal_error << "Handle layout field grab for: " << layout;
}

ir::Stmt lower_switch_tree(ir::Layout layout, ir::Expr base,
                           const std::string &obj_name,
                           const LayoutTypeMap &ltmap) {
    struct FindPaths : public ir::Visitor {
        using Path =
            std::vector<std::pair<std::string, std::optional<int64_t>>>;
        Path current;
        std::map<std::string, Path> paths;

        // A walk of this layout never arrives inside an indirect group, so the
        // variants declared there are not this walk's to reach -- and may
        // reuse the names this one uses. An instance's tree calls its nodes
        // Interior and Leaf just as the tree holding the instances does.
        void visit(const ir::Group *node) override {
            if (node->type == ir::Group::Type::Indirect) {
                return;
            }
            node->inner.accept(this);
        }

        void visit(const ir::Switch *node) override {
            for (const auto &arm : node->arms) {
                current.emplace_back(node->field, arm.value);
                if (arm.name.has_value()) {
                    internal_assert(!paths.contains(*arm.name))
                        << "Duplicate path for: " << *arm.name;
                    paths[*arm.name] = current;
                } else {
                    arm.layout.accept(this); // check for deeper splits.
                }
            }
        }
    };
    FindPaths finder;
    layout.accept(&finder);

    // TODO: should this be scheduable...?
    // TODO: we want to insert likely() for non-leaves, I think?
    std::vector<std::string> order;
    for (const auto &pair : finder.paths) {
        order.push_back(pair.first);
    }
    std::sort(order.begin(), order.end(),
              [&](const std::string &a, const std::string &b) {
                  // TODO: caching this would make this faster,
                  // but we probably never have a large number.
                  auto count_non_null = [](const FindPaths::Path &path) {
                      return std::count_if(
                          path.begin(), path.end(),
                          [](const auto &p) { return p.second.has_value(); });
                  };
                  return count_non_null(finder.paths[a]) <
                         count_non_null(finder.paths[b]);
              });

    ir::Stmt if_chain;
    for (const auto &node_name : std::views::reverse(order)) {
        // Make a hole for the body of this node type.
        ir::Stmt body = ir::Label::make(node_name, ir::Stmt());

        if (if_chain.defined()) {
            ir::Expr cond;
            internal_assert(finder.paths.contains(node_name));
            const FindPaths::Path &path = finder.paths.at(node_name);

            for (const auto &pair : path) {
                internal_assert(pair.second.has_value());
                ir::Expr value = field_in_layout(
                    base, layout, ir::MapStack<std::string, ir::Expr>(),
                    obj_name, node_name, pair.first, ltmap,
                    /*outer_group=*/ir::Expr());
                ir::Expr constant = make_const(value.type(), *pair.second);
                // TODO: support non-eq matching? e.g. ranges?
                ir::Expr eq = ir::BinOp::make(ir::BinOp::Eq, std::move(value),
                                              std::move(constant));
                if (cond.defined()) {
                    cond = ir::BinOp::make(ir::BinOp::LAnd, std::move(cond),
                                           std::move(eq));
                } else {
                    cond = std::move(eq);
                }
            }

            internal_assert(cond.defined());

            if_chain = ir::IfElse::make(std::move(cond), std::move(body),
                                        std::move(if_chain));
        } else {
            // TODO: this doesn't work if it's possible to have fully NULL
            // reprs.
            if_chain = std::move(body);
        }
    }
    internal_assert(if_chain.defined());
    return if_chain;
}

struct LowerUnwrapAccesses : public ir::Mutator {
    const std::string &tree_name;
    const ir::Expr &tree_idx;
    // The storage this walk reads, which is the tree itself for one the
    // schedule named and the enclosing object for a tree held in an element.
    const ir::Expr &tree_storage;
    const std::string &node_type;
    const std::map<std::string, ir::Expr> &field_map;

    ir::MapStack<std::string, ir::Type> type_repls;

    LowerUnwrapAccesses(const std::string &tree_name, const ir::Expr &tree_idx,
                        const ir::Expr &tree_storage,
                        const std::string &node_type,
                        const std::map<std::string, ir::Expr> &field_map)
        : tree_name(tree_name), tree_idx(tree_idx), tree_storage(tree_storage),
          node_type(node_type), field_map(field_map) {}

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
        if (!node->value.is<ir::Unwrap>()) {
            return ir::Mutator::visit(node);
        }
        const ir::Unwrap *as_unwrap = node->value.as<ir::Unwrap>();
        internal_assert(as_unwrap->value.is<ir::Var>())
            << "[unimplemented] Access of Unwrap on non-Var: "
            << ir::Expr(node);

        std::string var_name = as_unwrap->value.as<ir::Var>()->name;

        if (var_name == tree_name) {
            internal_assert(as_unwrap->type.is<ir::Struct_t>());
            if (as_unwrap->type.as<ir::Struct_t>()->name == node_type) {
                const auto &iter = field_map.find(node->field);
                internal_assert(iter != field_map.cend())
                    << "In lowering of " << ir::Expr(node)
                    << ", failed to find field: " << node->field
                    << " in field map of " << tree_name;
                return iter->second;
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
            uint32_t count = 0;
            void visit(const ir::Unwrap *node) override {
                internal_assert(node->value.is<ir::Var>())
                    << "[unimplemented] Unwrap on non-Var: " << ir::Expr(node);

                count++;
                name = node->value.as<ir::Var>()->name;
            }
        };
        Getter getter;
        expr.accept(&getter);
        internal_assert(!getter.name.empty())
            << "get_tree_name failed on: " << expr;
        internal_assert(getter.count == 1)
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
                ir::Expr arg = opt::Simplify::simplify(node->args[i]);
                std::string t = get_tree_name(arg);
                if (t == tree_name) {
                    partition = i + 1;
                }
            }
            if (node->args[i].type().is<ir::BVH_t>()) {
                ir::Expr arg = opt::Simplify::simplify(node->args[i]);
                const ir::Var *arg_var = arg.as<ir::Var>();
                internal_assert(arg_var) << arg;
                if (arg_var->name == tree_name) {
                    repl = tree_idx;
                    partition = i + 1;
                }
            }
            new_args[i] = opt::Simplify::simplify(std::move(repl));
            not_changed = not_changed && !changed;
        }

        if (partition) {
            if (new_args[partition - 1].type().is<ir::Tuple_t>()) {
                // Split it up!
                auto splits = break_tuple(new_args[partition - 1]);
                internal_assert(splits.size() == 1)
                    << "TODO: handle tuple index in layout lowering of join: "
                    << new_args[partition - 1];
                new_args[partition - 1] = splits[0];
            }
            new_args.insert(new_args.begin() + partition, tree_storage);
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
            // The keys come through unchanged. This flattens what a branch
            // *is* -- a subtree reference becoming the indices that stand for
            // it -- and a key is not a branch, it is a number computed about
            // one. It still has to be carried, though: a sort() key names the
            // node's split axis, and this pass is what turns that name into a
            // load from the layout.
            std::vector<ir::Expr> keys;
            keys.reserve(node->keys.size());
            for (const auto &key : node->keys) {
                keys.push_back(mutate(key));
            }
            return ir::YieldFrom::make(std::move(value), std::move(keys));
        }
    };

    FlattenYieldFroms f(index_list, references);
    return f.mutate(std::move(body));
}

struct LowerMatches : public ir::Mutator {
    const ir::LayoutMap &layouts;
    const ir::TypeMap &structs;
    const LayoutTypeMap &ltmap;

    const std::map<std::string, std::string> &tree_field_groups;

    LowerMatches(const ir::LayoutMap &layouts, const ir::TypeMap &structs,
                 const LayoutTypeMap &ltmap,
                 const std::map<std::string, std::string> &tree_field_groups)
        : layouts(layouts), structs(structs), ltmap(ltmap),
          tree_field_groups(tree_field_groups) {}

    std::map<std::string, ir::Type> ref_types;
    IndexTList index_list;
    // Where each of those indices starts, in the same order.
    std::vector<ir::Expr> index_starts;
    std::set<std::string> matched_objects;
    std::map<std::string, ir::Expr> references;
    // Trees reached through an element rather than named by the schedule.
    //
    // `let _subtree0 = i.blas in rec(_subtree0) { match _subtree0 ... }` --
    // the match is on a local, so there is no layout under that name. What
    // there is, is the group the schedule said holds that tree's nodes, and a
    // walk of it is a walk of the enclosing structure restricted to that
    // group. So the local is given a layout of exactly that: the group,
    // walked directly, over the same storage.
    std::map<std::string, ir::Layout> nested_layouts;
    std::map<std::string, ir::Type> nested_structs;
    // And the name that storage goes by. A nested tree's nodes are rows of a
    // group in the enclosing object's layout, so its walk reads through the
    // enclosing object -- `instances.group0_bnode[i]`, not `_subtree0.…`.
    // What `_subtree0` holds is only where in that group to start.
    std::map<std::string, std::string> nested_bases;

    // The layout a match on `name` should be lowered against.
    const ir::Layout *layout_of(const std::string &name) const {
        if (const auto iter = nested_layouts.find(name);
            iter != nested_layouts.cend()) {
            return &iter->second;
        }
        if (const auto iter = layouts.find(name); iter != layouts.cend()) {
            return &iter->second;
        }
        return nullptr;
    }

    const ir::Type *struct_of(const std::string &name) const {
        if (const auto iter = nested_structs.find(name);
            iter != nested_structs.cend()) {
            return &iter->second;
        }
        if (const auto iter = structs.find(name); iter != structs.cend()) {
            return &iter->second;
        }
        return nullptr;
    }

    // The name the storage a walk of `name` reads is bound under. For a tree
    // the schedule named, that is the tree itself; for one held in an element,
    // it is whatever holds the group its nodes are rows of.
    const std::string &base_of(const std::string &name) const {
        if (const auto iter = nested_bases.find(name);
            iter != nested_bases.cend()) {
            return iter->second;
        }
        return name;
    }

    // Whether a walk of `name` starts wherever it was told to rather than at
    // the first row of its group.
    bool is_nested(const std::string &name) const {
        return nested_layouts.contains(name);
    }

    // `let <tree> = <element>.<field> in ...` for a field the schedule bound
    // to a tree: remember which group that tree's nodes live in, so the match
    // that follows can be lowered against it.
    ir::Stmt visit(const ir::LetStmt *node) override {
        const ir::Access *access = node->value.as<ir::Access>();
        if (access != nullptr && !node->loc.base.empty()) {
            const auto *element = access->value.type().as<ir::Struct_t>();
            if (element != nullptr) {
                const std::string path =
                    element->name + "." + access->field;
                const auto group = tree_field_groups.find(path);
                if (group != tree_field_groups.cend()) {
                    const auto named = ltmap.groups.find(group->second);
                    internal_assert(named != ltmap.groups.cend())
                        << "No group named " << group->second << " for "
                        << path;
                    internal_assert(!named->second.owner_name.empty())
                        << path << " is stored in group " << group->second
                        << ", which is not a field of anything the program can "
                           "name, so a walk of it has nothing to read through.";
                    // Walked directly: the reference the recursion advances is
                    // an index into this group, which is what an indirect
                    // group is when something is walking it rather than
                    // looking one row up.
                    nested_layouts[node->loc.base] = named->second.walkable;
                    nested_structs[node->loc.base] = named->second.owner;
                    nested_bases[node->loc.base] = named->second.owner_name;
                }
            }
        }
        return ir::Mutator::visit(node);
    }

    size_t counter = 0;

    std::string get_unique_loop_label() {
        return "_loop" + std::to_string(counter++);
    }

    ir::Stmt visit(const ir::RecLoop *node) override {
        // Recursions nest: a query over a tree held in an element walks the
        // outer tree and, at each element it reaches, walks that element's own
        // tree. Each carries its own stack, so each gets its own index
        // parameters and its own references -- the outer ones are put aside
        // and restored, rather than shared, because an inner traversal
        // advances an index into a different pool.
        auto outer_references = std::move(references);
        auto outer_index_list = std::move(index_list);
        auto outer_index_starts = std::move(index_starts);
        references.clear();
        index_list.clear();
        index_starts.clear();

        ir::Stmt body = mutate(node->body);
        body = flatten_yield_froms(index_list, std::move(body), references);

        internal_assert(index_list.size() == index_starts.size());
        std::vector<ir::RecLoop::Arg> args;
        args.reserve(index_list.size());
        for (size_t i = 0; i < index_list.size(); i++) {
            args.push_back(
                ir::RecLoop::Arg{index_list[i], std::move(index_starts[i])});
        }
        ir::Stmt loop = ir::RecLoop::make(std::move(args), std::move(body));

        references = std::move(outer_references);
        index_list = std::move(outer_index_list);
        index_starts = std::move(outer_index_starts);
        return loop;
    }

    ir::Stmt visit(const ir::Match *node) override {
        internal_assert(node->loc.is<ir::Var>())
            << "[unimplemented] Match on non-Var: " << ir::Stmt(node);
        const std::string tree_name = node->loc.as<ir::Var>()->name;

        // Now, based on layout, form switch-tree.
        ir::Layout layout = [&]() {
            const ir::Layout *found = layout_of(tree_name);
            internal_assert(found != nullptr)
                << "Failed to find layout of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            return *found;
        }();

        ir::Type struct_type = [&]() {
            const ir::Type *found = struct_of(tree_name);
            internal_assert(found != nullptr)
                << "Failed to find type of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            return *found;
        }();

        ir::Expr base_struct = ir::Var::make(struct_type, base_of(tree_name));
        ir::Stmt body =
            lower_switch_tree(layout, base_struct, tree_name, ltmap);

        // First time we see a tree, add it's type to the type parameters list.
        if (!matched_objects.contains(tree_name)) {
            IndexTList node_index_list = get_index_type(layout);
            std::reverse(node_index_list.begin(), node_index_list.end());
            std::vector<ir::Expr> idxs;
            std::vector<ir::Expr> starts;
            idxs.reserve(node_index_list.size());
            starts.reserve(node_index_list.size());
            const bool nested = is_nested(tree_name);
            internal_assert(!nested || node_index_list.size() == 1)
                << tree_name << " is reached through a field holding one "
                << "index, but walking it advances " << node_index_list.size();
            for (auto &it : node_index_list) {
                // A tree the schedule named is rooted at the first row of its
                // group. A tree held in an element's field is rooted wherever
                // that field says, which is the value already bound under this
                // name -- the same walk over the same storage, begun somewhere
                // else.
                starts.push_back(nested ? ir::Var::make(it.type, tree_name)
                                        : make_zero(it.type));
                it.name = tree_name + "_" + it.name;
                idxs.push_back(ir::Var::make(it.type, it.name));
            }
            references[tree_name] = make_tuple(std::move(idxs));
            index_list.insert(index_list.end(),
                              std::make_move_iterator(node_index_list.begin()),
                              std::make_move_iterator(node_index_list.end()));
            index_starts.insert(index_starts.end(),
                                std::make_move_iterator(starts.begin()),
                                std::make_move_iterator(starts.end()));
            matched_objects.insert(tree_name);
        }

        const auto tree_idx_iter = references.find(tree_name);
        internal_assert(tree_idx_iter != references.cend()) << tree_name;
        const ir::Expr tree_idx = tree_idx_iter->second;

        for (const auto &arm : node->arms) {
            std::map<std::string, ir::Expr> field_map;
            const std::string &branch_name = arm.first.name();
            for (const auto &field : arm.first.fields()) {
                field_map[field.name] = field_in_layout(
                    base_struct, layout, ir::MapStack<std::string, ir::Expr>{},
                    tree_name, branch_name, field.name, ltmap,
                    /*outer_group=*/ir::Expr());
            }

            // Lower these Unwraps.
            ir::Stmt branch_body =
                LowerUnwrapAccesses(tree_name, tree_idx, base_struct,
                                    branch_name, field_map)
                    .mutate(arm.second);

            body = FillHole(branch_name, std::move(branch_body))
                       .mutate(std::move(body));
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
            << "Mutated type of non-tuple in layout lowering: "
            << ir::Expr(node);
        return make_tuple(std::move(values));
    }
};

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
    LayoutTypeMap ltmap;

    // What a field bound to a tree is, in storage. The schedule said which
    // group holds that tree's nodes; the group says what indexing it costs.
    for (const auto &[path, group_name] :
         program.schedules[ir::Target::Host].tree_groups) {
        ir::Type index_t;
        for (const auto &[_, layout] : tree_layouts) {
            struct FindGroup : public ir::Visitor {
                const std::string &wanted;
                ir::Type index_t;

                FindGroup(const std::string &wanted) : wanted(wanted) {}

                void visit(const ir::Group *node) override {
                    if (node->declared_name == wanted) {
                        index_t = node->index_t;
                    }
                    node->inner.accept(this);
                }
            };
            FindGroup finder(group_name);
            layout.accept(&finder);
            if (finder.index_t.defined()) {
                index_t = finder.index_t;
                break;
            }
        }
        internal_assert(index_t.defined())
            << path << " is stored in group " << group_name
            << ", which no layout declares.";
        ltmap.field_refs[path] = std::move(index_t);
    }

    // Now say that about the program, before anything is lowered against it,
    // so that everything reading an element and everything written about one
    // agree on what an element is.
    if (!ltmap.field_refs.empty()) {
        RewriteStoredElements rewriter(ltmap.field_refs);
        for (auto &[_, type] : program.types) {
            type = rewriter.mutate(type);
        }
        for (auto &[_, type] : program.externs) {
            type = rewriter.mutate(type);
        }
        for (auto &[_, func] : program.funcs) {
            for (auto &arg : func->args) {
                arg.type = rewriter.mutate(arg.type);
                arg.default_value = rewriter.mutate(arg.default_value);
            }
            func->ret_type = rewriter.mutate(func->ret_type);
            func->body = rewriter.mutate(func->body);
        }
    }

    for (const auto &[name, layout] : tree_layouts) {
        ir::Type struct_t = layout_to_structs(layout, ltmap);
        types[name] = struct_t;

        // A group declared at the top of this layout is a field of the object
        // the program knows by this name, so that is the name a walk of it
        // reads through. One declared further in is a field of something the
        // program cannot name, and a walk of it would have nowhere to start.
        for (auto &[_, named] : ltmap.groups) {
            if (named.owner.defined() && named.owner.same_as(struct_t)) {
                named.owner_name = name;
            }
        }

        bool found = false;
        for (auto &[ename, etype] : program.externs) {
            if (name == ename) {
                found = true;
                etype = struct_t;
                break;
            }
        }
        internal_assert(found)
            << "Extern " << name << " has layout but not found.\n";

        for (const auto &[layout, type] : ltmap.layout_to_type) {
            internal_assert(type.is<ir::Struct_t>());
            program.types[type.as<ir::Struct_t>()->name] = type;
        }
    }

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

        LowerMatches lowerer(tree_layouts, types, ltmap,
                             program.schedules[ir::Target::Host].tree_groups);
        func->body = lowerer.mutate(func->body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
