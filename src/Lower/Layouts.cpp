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
    return "?" + base + "_layout" + std::to_string(name_counter++);
}

std::string unique_pad_name() { return "?pad" + std::to_string(pad_counter++); }

std::string get_group_name(const std::string &base, const std::string &index) {
    return base + "__" + index;
}

std::string get_split_field_name(const std::string &base,
                                 const std::string &field) {
    return base + "_?spliton_" + field;
}

std::vector<std::pair<std::string, ir::Type>>
get_index_type(const ir::Layout &layout) {
    std::vector<std::pair<std::string, ir::Type>> index_ts;
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                internal_assert(index_ts.empty())
                    << "[unimplemented] adjacent groups in layout: " << layout;
                index_ts = get_index_type(node->inner);
                index_ts.emplace_back(node->name, node->index_t);
                break;
            }
            case ir::IRLayoutEnum::Split: {
                const ir::Split *node = l.as<ir::Split>();
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

struct LowerFroms : public ir::Mutator {
    std::string stack_name;
    ir::WriteLoc counter_loc;
    ir::Expr counter;

    ir::Expr one;

    LowerFroms(std::string stack_name, ir::WriteLoc counter_loc,
               ir::Expr counter)
        : stack_name(std::move(stack_name)),
          counter_loc(std::move(counter_loc)), counter(std::move(counter)) {
        one = make_one(this->counter.type());
    }

    ir::Stmt visit(const ir::YieldFrom *node) override {
        return ir::Sequence::make(
            {ir::Accumulate::make(counter_loc, ir::Accumulate::Add, one),
             ir::Store::make(stack_name, counter, node->value)});
    }
};

ir::Expr fill(const ir::FrameStack<ir::Expr> &frames, const ir::Expr &expr) {
    struct Rewrite : public ir::Mutator {
        const ir::FrameStack<ir::Expr> &frames;

        Rewrite(const ir::FrameStack<ir::Expr> &frames) : frames(frames) {}

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
            internal_assert(frames.name_in_scope(var->name))
                << "Materialization fill cannot find: " << var->name;
            if (frames.name_in_scope(var->name)) {
                return frames.from_frames(var->name);
            }
            return var;
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
        rets.push_back(ir::Struct_t::make(unique_struct_name(std::move(base)),
                                          std::move(fields)));
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
            frames.new_frame();
        }

        ir::FrameStack<ir::Expr> frames;

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
                    frames.new_frame();
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

            frames.new_frame();
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

ir::Stmt lower_switch_tree(std::map<std::string, ir::Stmt> bodies,
                           ir::Layout layout, ir::Expr base,
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
    internal_assert(bodies.size() == finder.paths.size())
        << "Incorrect number of labelled paths in layout: " << layout
        << " expected: " << bodies.size()
        << " but found: " << finder.paths.size();

    // TODO: should this be scheduable...?
    // TODO: we want to insert likely() for non-leaves, I think?
    std::vector<std::string> order;
    for (const auto &pair : bodies) {
        internal_assert(finder.paths.contains(pair.first))
            << "No path found for node type: " << pair.first
            << " in layout: " << layout;
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
        internal_assert(bodies.contains(node_name));
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
                    cond = ir::BinOp::make(ir::BinOp::And, std::move(cond),
                                           std::move(eq));
                } else {
                    cond = std::move(eq);
                }
            }

            internal_assert(cond.defined());

            if_chain = ir::IfElse::make(std::move(cond), bodies.at(node_name),
                                        std::move(if_chain));
        } else {
            // TODO: this doesn't work if it's possible to have fully NULL
            // reprs.
            if_chain = bodies.at(node_name);
        }
    }
    internal_assert(if_chain.defined());
    return if_chain;
}

struct LowerUnwrapAccesses : public ir::Mutator {
    const ir::LayoutMap &layouts;
    const ir::TypeMap &structs;

    LowerUnwrapAccesses(const ir::LayoutMap &layouts,
                        const ir::TypeMap &structs)
        : layouts(layouts), structs(structs) {}

    std::map<std::string, ir::Type> ref_types;
    bool in_match = false;

    ir::Stmt visit(const ir::Match *node) override {
        if (in_match) {
            // Just lower to an if-else or switch chain!
            internal_error << "[unimplemented] nested Match lowering";
        }
        in_match = true;
        // TODO: can do stack-top optimizations

        // TODO: this should be taken from the schedule!
        const int stack_size = 64;

        // TODO: should these be unique? What if nested traversals? e.g. TLAS /
        // BLAS?
        const std::string stack_name = "?stack";
        const std::string count_name = "?count";
        internal_assert(node->loc.is<ir::Var>())
            << "[unimplemented] Match on non-Var: " << ir::Stmt(node);
        const std::string node_name = node->loc.as<ir::Var>()->name;
        const std::string top_name = "?top";

        const size_t n_branches = node->arms.size();
        std::map<std::string, ir::Stmt> bodies;

        FindFromType finder;
        for (size_t i = 0; i < n_branches; i++) {
            ir::Stmt body = mutate(node->arms[i].second);
            body.accept(&finder);
            bodies[node->arms[i].first.name()] = std::move(body);
        }
        internal_assert(finder.from_type.defined())
            << "No YieldFroms in Match: " << ir::Stmt(node);

        // TODO: Array_t for dynamic...?
        ir::Type stack_etype = std::move(finder.from_type);
        // ir::Type stack_type = ir::Vector_t::make(stack_etype, stack_size);
        // ir::Stmt alloc = ir::Assign::make(ir::WriteLoc(stack_name,
        // stack_type), ir::Build::make(stack_type, std::vector<ir::Expr>{}),
        // /*mutating=*/false);
        // TODO(ajr): make stack iterable size scheduable?
        static const ir::Type count_type = ir::UInt_t::make(32);
        static const ir::Expr count_zero = make_zero(count_type);
        ir::Expr stack_size_expr = ir::UIntImm::make(count_type, stack_size);
        ir::Type stack_type =
            ir::Array_t::make(stack_etype, std::move(stack_size_expr));
        ir::Stmt alloc = ir::Allocate::make(stack_name, stack_type);
        ir::WriteLoc counter_loc(count_name, count_type);
        ir::Stmt set_count =
            ir::Assign::make(counter_loc, count_zero, /*mutating=*/false);
        ir::Expr counter = ir::Var::make(count_type, count_name);
        ir::Expr stack = ir::Var::make(stack_type, stack_name);

        // TODO(ajr): this doesn't work for nested Matches!! debug with
        // collision detection. This also assumes that the root is always 0, and
        // that there is an INFINITY value for stack_etype.
        ir::WriteLoc top_loc(top_name, stack_etype);
        ir::Stmt set_root = ir::Assign::make(top_loc, make_zero(stack_etype),
                                             /*mutating=*/false);
        ir::Expr inf_value = make_inf(stack_etype);
        ir::Stmt set_canary =
            ir::Store::make(stack_name, make_zero(count_type), inf_value);
        ir::Expr top_var = ir::Var::make(stack_etype, top_name);

        LowerFroms lower_froms(stack_name, counter_loc, counter);
        for (auto &[_, body] : bodies) {
            body = lower_froms.mutate(body);
        }

        // Now, based on layout, form switch-tree.
        ir::Layout layout = [&]() {
            const auto &iter = layouts.find(node_name);
            internal_assert(iter != layouts.cend())
                << "Failed to find layout of: " << node_name
                << " for Match lowering: " << ir::Stmt(node);
            return iter->second;
        }();

        ir::Type struct_type = [&]() {
            const auto &iter = structs.find(node_name);
            internal_assert(iter != structs.cend())
                << "Failed to find type of: " << node_name
                << " for Match lowering: " << ir::Stmt(node);
            return iter->second;
        }();

        // TODO(ajr): this does not handle nested Matches!
        ir::Stmt extract_from_top;
        const auto indexes = get_index_type(layout);

        if (indexes.size() > 1) {
            std::vector<ir::Stmt> extracts(indexes.size());
            int n_extracted = 0;
            for (auto &[name, type] : std::views::reverse(indexes)) {
                std::string iter_name = get_group_name(node_name, name);
                ir::WriteLoc loc(std::move(iter_name), std::move(type));
                ir::Expr value = ir::Extract::make(top_var, n_extracted);
                extracts[n_extracted++] =
                    ir::LetStmt::make(std::move(loc), std::move(value));
            }
            extract_from_top = ir::Sequence::make(std::move(extracts));
        } else {
            internal_assert(indexes.size() == 1);
            std::string iter_name = get_group_name(node_name, indexes[0].first);
            ir::WriteLoc loc(std::move(iter_name),
                             std::move(indexes[0].second));
            extract_from_top = ir::LetStmt::make(std::move(loc), top_var);
        }

        ir::Expr base_struct = ir::Var::make(struct_type, node_name);
        ir::Stmt match_body = lower_switch_tree(
            std::move(bodies), std::move(layout), base_struct, node_name);

        // TODO: perform top optimizations.
        ir::Stmt do_while_body = ir::Sequence::make(
            {// set relevant indices from stack top.
             std::move(extract_from_top),
             // perform match on current `top`
             std::move(match_body),
             // set current top
             ir::Assign::make(top_loc, ir::Extract::make(stack, counter),
                              /*mutating=*/true),
             ir::Accumulate::make(counter_loc, ir::Accumulate::Sub,
                                  make_one(count_type))});

        ir::Expr top_not_empty = (top_var != inf_value);

        ir::Stmt do_while = ir::DoWhile::make(std::move(do_while_body),
                                              std::move(top_not_empty));

        in_match = false;

        std::vector<ir::Stmt> body = {
            std::move(alloc),      // stack = allocate stack
            std::move(set_count),  // count = 0;
            std::move(set_root),   // assign node = 0 // root
            std::move(set_canary), // stack[0] = INFINITY(reference_type)
            std::move(do_while)    // do {
                                   //   extract-from-top;
                                   //   if-else-body;
                                   //   node = stack[count--];
                                   // } while(node != INFINITY(REFERENCE_TYPE))
        };
        return ir::Sequence::make(std::move(body));
    }

    ir::Expr visit(const ir::Access *node) override {
        if (!node->value.is<ir::Unwrap>()) {
            return ir::Mutator::visit(node);
        }
        const ir::Unwrap *as_unwrap = node->value.as<ir::Unwrap>();
        internal_assert(as_unwrap->value.is<ir::Var>())
            << "[unimplemented] Access of Unwrap on non-Var: "
            << ir::Expr(node);

        std::string tree_name = as_unwrap->value.as<ir::Var>()->name;

        internal_assert(structs.contains(tree_name));
        ir::Expr base = ir::Var::make(structs.at(tree_name), tree_name);
        // std::cout << "BASE: " << base << "\n";
        return get_field(base, tree_name, layouts.at(tree_name),
                         as_unwrap->type.as<ir::Struct_t>()->name, node->field);
    }

    ir::Expr visit(const ir::Unwrap *node) override {
        internal_error << "Found Unwrap not handled by Access visitor: "
                       << ir::Expr(node);
    }
};

} // namespace

ir::Program LowerLayouts::run(ir::Program program) const {
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
    LowerUnwrapAccesses lowerer(tree_layouts, types);

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
