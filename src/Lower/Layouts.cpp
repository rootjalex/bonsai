#include "Lower/Layouts.h"

#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/ValidateLayout.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <ranges>

namespace bonsai {
namespace lower {

namespace {

struct LayoutTypeMap {
    std::map<ir::Layout, ir::Type, ir::LayoutLessThan> layout_to_type;
    std::map<ir::Layout, std::string, ir::LayoutLessThan> layout_to_name;
    uint64_t counter = 0;
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
                internal_assert(index_ts.empty())
                    << "[unimplemented] adjacent groups in layout: " << layout;
                index_ts = get_index_type(node->inner);
                index_ts.push_back({node->name, node->index_t});
                break;
            }
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
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
        uint32_t pad_count = 0;
        uint32_t group_count = 0;
        uint32_t split_count = 0;
        std::string name = "_tree_layout" + std::to_string(ltmap.counter++);
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Name: {
                const ir::Name *node = l.as<ir::Name>();
                fields.emplace_back(node->name, node->type);
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
                // push back new field type.
                fields.emplace_back(std::move(field_name), std::move(group_t));
                break;
            }
            case ir::IRLayoutEnum::Switch: {
                const ir::Switch *node = l.as<ir::Switch>();
                // Store as vector of bytes, load and reinterpret to proper
                // type.
                const uint64_t bits = l.bits();
                internal_assert(bits % 8 == 0)
                    << "Switch is not byte-aligned: " << l;
                static const ir::Type u8 = ir::UInt_t::make(8);
                ir::Type byte_vec = ir::Vector_t::make(u8, bits / 8);
                std::string name = split_name(split_count++, node->field);
                fields.emplace_back(std::move(name), std::move(byte_vec));
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
        constexpr auto P = ir::Struct_t::Attribute::packed;
        ir::Type struct_t =
            ir::Struct_t::make(std::move(name), std::move(fields), {P});
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
                         const LayoutTypeMap &ltmap) {
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        uint32_t group_count = 0;
        uint32_t split_count = 0;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Name: {
                const ir::Name *node = l.as<ir::Name>();
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
                break;
            }
            case ir::IRLayoutEnum::Pad: {
                break;
            }
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                std::string field_name = group_name(group_count++, node->name);
                ir::Expr path = ir::Access::make(field_name, base);
                ir::Expr index =
                    ir::Var::make(node->index_t, iter_name + "_" + node->name);
                path = ir::Extract::make(std::move(path), index);
                frames.push_frame();
                frames.add_to_frame(node->name, index);
                ir::Expr rec =
                    field_in_layout(path, node->inner, frames, iter_name,
                                    node_type, field, ltmap);
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
                        ir::Expr path =
                            ir::Access::make(std::move(field_name), base);
                        auto iter = ltmap.layout_to_type.find(arm.layout);
                        internal_assert(iter != ltmap.layout_to_type.cend())
                            << "Unseen Switch arm layout: " << ir::Layout(node)
                            << " at " << arm.layout;
                        ir::Type reinterpret_type = iter->second;
                        path = ir::Cast::make(reinterpret_type, path,
                                              ir::Cast::Mode::Reinterpret);
                        frames.push_frame();
                        ir::Expr rec =
                            field_in_layout(path, arm.layout, frames, iter_name,
                                            node_type, field, ltmap);
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
                ir::Expr mat = fill(frames, node->value);
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
                    obj_name, node_name, pair.first, ltmap);
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
            return ir::YieldFrom::make(std::move(value));
        }
    };

    FlattenYieldFroms f(index_list, references);
    return f.mutate(std::move(body));
}

struct LowerMatches : public ir::Mutator {
    const ir::LayoutMap &layouts;
    const ir::TypeMap &structs;
    const LayoutTypeMap &ltmap;

    LowerMatches(const ir::LayoutMap &layouts, const ir::TypeMap &structs,
                 const LayoutTypeMap &ltmap)
        : layouts(layouts), structs(structs), ltmap(ltmap) {}

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
        return ir::RecLoop::make(std::move(index_list), std::move(body));
    }

    ir::Stmt visit(const ir::Match *node) override {
        internal_assert(node->loc.is<ir::Var>())
            << "[unimplemented] Match on non-Var: " << ir::Stmt(node);
        const std::string tree_name = node->loc.as<ir::Var>()->name;

        // Now, based on layout, form switch-tree.
        ir::Layout layout = [&]() {
            const auto &iter = layouts.find(tree_name);
            internal_assert(iter != layouts.cend())
                << "Failed to find layout of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            return iter->second;
        }();

        ir::Type struct_type = [&]() {
            const auto &iter = structs.find(tree_name);
            internal_assert(iter != structs.cend())
                << "Failed to find type of: " << tree_name
                << " for Match lowering: " << ir::Stmt(node);
            return iter->second;
        }();

        ir::Expr base_struct = ir::Var::make(struct_type, tree_name);
        ir::Stmt body =
            lower_switch_tree(layout, base_struct, tree_name, ltmap);

        for (const auto &arm : node->arms) {
            std::map<std::string, ir::Expr> field_map;
            const std::string &branch_name = arm.first.name();
            for (const auto &field : arm.first.fields()) {
                field_map[field.name] = field_in_layout(
                    base_struct, layout, ir::MapStack<std::string, ir::Expr>{},
                    tree_name, branch_name, field.name, ltmap);
            }

            // Lower these Unwraps.
            ir::Stmt branch_body = LowerUnwrapAccesses(tree_name, struct_type,
                                                       branch_name, field_map)
                                       .mutate(arm.second);

            body = FillHole(branch_name, std::move(branch_body))
                       .mutate(std::move(body));
        }

        // First time we see a tree, add it's type to the type parameters list.
        if (!matched_objects.contains(tree_name)) {
            IndexTList node_index_list = get_index_type(layout);
            std::reverse(node_index_list.begin(), node_index_list.end());
            std::vector<ir::Expr> idxs;
            idxs.reserve(node_index_list.size());
            for (auto &it : node_index_list) {
                it.name = tree_name + "_" + it.name;
                idxs.push_back(ir::Var::make(it.type, it.name));
            }
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
    for (const auto &[name, layout] : tree_layouts) {
        ir::Type struct_t = layout_to_structs(layout, ltmap);
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

        LowerMatches lowerer(tree_layouts, types, ltmap);
        func->body = lowerer.mutate(func->body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
