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

// ensures unique names in lowering.
static size_t name_counter = 0;
static size_t pad_counter = 0;

std::string unique_struct_name(std::string base) {
    return "_" + base + "_layout" + std::to_string(name_counter++);
}

std::string unique_pad_name() { return "_pad" + std::to_string(pad_counter++); }

std::string get_group_name(const std::string &base, const std::string &index) {
    return base + "__" + index;
}

std::string get_split_field_name(const std::string &base,
                                 const std::string &field) {
    return base + "_spliton_" + field;
}

using IndexTList = std::vector<ir::TypedVar>;

IndexTList get_index_type(const std::string &base_name,
                          const ir::Layout &layout) {
    IndexTList index_ts;
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                internal_assert(index_ts.empty())
                    << "[unimplemented] adjacent groups in layout: " << layout;
                std::string iter_name = get_group_name(base_name, node->name);
                index_ts = get_index_type(base_name, node->inner);
                index_ts.push_back({iter_name, node->index_t});
                break;
            }
            case ir::IRLayoutEnum::Split: {
                const ir::Split *node = l.as<ir::Split>();
                for (const auto &arm : node->arms) {
                    auto rec = get_index_type(base_name, arm.layout);
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

std::vector<ir::Type> layout_to_structs(std::string base,
                                        const ir::Layout &layout) {
    std::vector<ir::Type> rets;
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
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
                fields.emplace_back(unique_pad_name(), std::move(pad_type));
                break;
            }
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                std::vector<ir::Type> rec =
                    layout_to_structs(base, node->inner);
                internal_assert(!rec.empty());
                ir::Type base_t = rec.back();
                ir::Type group_t =
                    ir::Array_t::make(std::move(base_t), node->size);
                // Add to rets.
                rets.insert(rets.end(), std::make_move_iterator(rec.begin()),
                            std::make_move_iterator(rec.end()));
                internal_assert(!node->name.empty());
                std::string field_name = base + "_" + node->name;
                // push back new field type.
                fields.emplace_back(std::move(field_name), std::move(group_t));
                break;
            }
            case ir::IRLayoutEnum::Split: {
                const ir::Split *node = l.as<ir::Split>();
                // Store as vector of bytes, load and reinterpret to proper
                // type.
                const uint64_t bits = l.bits();
                internal_assert(bits % 8 == 0)
                    << "Split is not byte-aligned: " << l;
                static const ir::Type u8 = ir::UInt_t::make(8);
                ir::Type byte_vec = ir::Vector_t::make(u8, bits / 8);
                std::string split_name =
                    get_split_field_name(base, node->field);
                fields.emplace_back(std::move(split_name), std::move(byte_vec));
                break;
            }
            case ir::IRLayoutEnum::Materialize:
                break;
            default: {
                internal_error << "Handle layout in Chain lowering: " << l;
            }
            }
        }
        constexpr auto P = ir::Struct_t::Attribute::packed;
        rets.push_back(ir::Struct_t::make(unique_struct_name(std::move(base)),
                                          std::move(fields), {P}));
        return rets;
    }
    internal_error << "Handle layout conversion for: " << layout;
}

ir::Expr get_field(ir::Expr base, const std::string &obj_name,
                   const ir::Layout &layout, const std::string &node_name,
                   const std::string &field) {
    struct FindPaths : public ir::Visitor {
        std::string base_name;
        const std::string &node_name;
        const std::string &field;
        FindPaths(ir::Expr base, const std::string &obj_name,
                  const std::string &node_name, const std::string &field)
            : base_name(obj_name), node_name(node_name), field(field),
              path(std::move(base)) {
            frames.push_frame();
        }

        ir::MapStack<std::string, ir::Expr> frames;

        ir::Expr path;
        ir::Expr value;

        void visit(const ir::Name *node) override {
            ir::Expr load = ir::Access::make(node->name, path);
            if (node->name == field) {
                // Found it!
                // Just return a read from the current path.
                value = std::move(load);
            } else {
                // Otherwise insert into current frame,
                // might be used in materialization.
                frames.add_to_frame(node->name, std::move(load));
            }
        }

        // No overload for Pad

        void visit(const ir::Split *node) override {
            // TODO(ajr): this is not equivalent w.r.t. naming.
            // Can save by caching this call.

            for (const auto &arm : node->arms) {
                if (!arm.name.has_value() || (*arm.name == node_name)) {
                    ir::Expr old_path = path;
                    std::string field =
                        get_split_field_name(base_name, node->field);
                    path = ir::Access::make(std::move(field), std::move(path));
                    ir::Type reinterpret_type =
                        layout_to_structs("", arm.layout).back();
                    path = ir::Cast::make(reinterpret_type, path);
                    frames.push_frame();
                    arm.layout.accept(this);
                    frames.pop_frame();
                    path = old_path;
                }
            }
        }

        // No overload for Chain

        void visit(const ir::Group *node) override {
            // Path becomes index into array.
            ir::Expr old_path = path;
            std::string iter_name = get_group_name(base_name, node->name);
            ir::Expr var = ir::Var::make(node->index_t, iter_name);
            std::string field_name = base_name + "_" + node->name;
            path = ir::Extract::make(ir::Access::make(field_name, path), var);

            frames.push_frame();
            frames.add_to_frame(node->name, std::move(var));
            ir::Visitor::visit(node);
            frames.pop_frame();
            path = old_path;
        }

        void visit(const ir::Materialize *node) override {
            ir::Expr mat = fill(frames, node->value);
            if (node->name == field) {
                // Found it!
                // Just return a read from the current path.
                value = std::move(mat);
            } else {
                // Otherwise insert into current frame,
                // might be used in materialization.
                frames.add_to_frame(node->name, std::move(mat));
            }
        }
    };
    FindPaths finder(base, obj_name, node_name, field);
    layout.accept(&finder);
    internal_assert(finder.value.defined())
        << "Field: " << field << " not set in layout traversal: " << layout;
    return finder.value;
}

ir::Stmt lower_switch_tree(ir::Layout layout, ir::Expr base,
                           const std::string &obj_name) {
    struct FindPaths : public ir::Visitor {
        using Path =
            std::vector<std::pair<std::string, std::optional<int64_t>>>;
        Path current;
        std::map<std::string, Path> paths;

        void visit(const ir::Split *node) override {
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
                ir::Expr value =
                    get_field(base, obj_name, layout, node_name, pair.first);
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
    const std::string &node_type;
    const std::map<std::string, ir::Expr> &field_map;

    LowerUnwrapAccesses(const std::string &tree_name,
                        const std::string &node_type,
                        const std::map<std::string, ir::Expr> &field_map)
        : tree_name(tree_name), node_type(node_type), field_map(field_map) {}

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

ir::Expr flatten_tuple(ir::Expr expr) {
    std::vector<ir::Expr> exprs;

    std::function<void(const ir::Expr &)> handle_tuple =
        [&](const ir::Expr &t) -> void {
        if (const ir::Build *as_build = t.as<ir::Build>()) {
            for (const ir::Expr &expr : as_build->values) {
                handle_tuple(expr);
            }
        } else {
            internal_assert(!t.type().is<ir::Tuple_t>())
                << "[unimplemented] flatten_tuple of non-Build: " << t;
            exprs.push_back(t);
        }
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

ir::Stmt flatten_yield_froms(const IndexTList &index_list, ir::Stmt body) {
    struct FlattenYieldFroms : public ir::Mutator {
        const IndexTList &index_list;

        FlattenYieldFroms(const IndexTList &index_list)
            : index_list(index_list) {}

        ir::Stmt visit(const ir::YieldFrom *node) override {
            // Base case, single value yield from.
            ir::Expr value = flatten_tuple(node->value);
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
                    << " in recursive function of: " << ir::Stmt(node);

                for (size_t i = 0; i < index_list.size(); i++) {
                    internal_assert(
                        ir::equals(index_list[i].type, tuple->etypes[i]))
                        << "Mismatching YieldFroms, expected type: "
                        << index_list[i].type
                        << " but found type: " << tuple->etypes[i]
                        << " at index: " << i << " in: " << ir::Stmt(node);
                }
            }
            return ir::YieldFrom::make(std::move(value));
        }
    };

    FlattenYieldFroms f(index_list);
    return f.mutate(std::move(body));
}

struct LowerMatches : public ir::Mutator {
    const ir::LayoutMap &layouts;
    const ir::TypeMap &structs;

    LowerMatches(const ir::LayoutMap &layouts, const ir::TypeMap &structs)
        : layouts(layouts), structs(structs) {}

    std::map<std::string, ir::Type> ref_types;
    size_t n_matches = 0;
    IndexTList index_list;

    size_t counter = 0;

    std::string get_unique_loop_label() {
        return "_loop" + std::to_string(counter++);
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
        ir::Stmt body = lower_switch_tree(layout, base_struct, tree_name);

        for (const auto &arm : node->arms) {
            std::map<std::string, ir::Expr> field_map;
            const std::string &branch_name = arm.first.name();
            for (const auto &field : arm.first.fields()) {
                field_map[field.name] = get_field(
                    base_struct, tree_name, layout, branch_name, field.name);
            }

            // Lower these Unwraps.
            ir::Stmt branch_body =
                LowerUnwrapAccesses(tree_name, branch_name, field_map)
                    .mutate(arm.second);

            body = FillHole(branch_name, std::move(branch_body))
                       .mutate(std::move(body));
        }

        // std::cout << "pushing back: " << tree_name << " at index: " <<
        // n_matches << std::endl;
        IndexTList node_index_list = get_index_type(tree_name, layout);
        std::reverse(node_index_list.begin(), node_index_list.end());
        index_list.insert(index_list.end(),
                          std::make_move_iterator(node_index_list.begin()),
                          std::make_move_iterator(node_index_list.end()));

        // Now recursively mutate the body, for nested matches.
        n_matches++;
        body = mutate(body);
        n_matches--;

        // The outermost match is a recursive while loop.
        // To find the types, grab the recursive types from
        // YieldFroms.

        if (n_matches == 0) {
            body = flatten_yield_froms(index_list, std::move(body));

            return ir::RecLoop::make(std::move(index_list), std::move(body));
        }
        return body;
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
    for (const auto &[name, layout] : tree_layouts) {
        auto struct_ts = layout_to_structs(name, layout);
        internal_assert(!struct_ts.empty());
        types[name] = struct_ts.back();

        bool found = false;
        for (auto &[ename, etype] : program.externs) {
            if (name == ename) {
                found = true;
                etype = struct_ts.back();
                break;
            }
        }
        internal_assert(found)
            << "Extern " << name << " has layout but not found.\n";

        for (const auto &type : struct_ts) {
            internal_assert(type.is<ir::Struct_t>());
            program.types[type.as<ir::Struct_t>()->name] = type;
        }
    }

    // lower all `Access`es on `Unwrap`s
    LowerMatches lowerer(tree_layouts, types);

    for (auto &[fname, func] : program.funcs) {
        for (auto &arg : func->args) {
            if (types.contains(arg.name)) {
                arg.type = types.at(arg.name);
            }
        }
        func->body = lowerer.mutate(func->body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
