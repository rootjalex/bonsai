#include "Lower/Builds.h"

#include "IR/Analysis.h"
#include "IR/Build.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Layout.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Log.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

namespace {

static constexpr char CANONICAL_TREE[] = "CT";
static constexpr char SPECIALIZED_TREE[] = "ST";

// Returns a reference to the "self", used when referencing fields of the
// currently visited BVH node.
ir::Expr self(const ir::Layout &layout) {
    return ir::Var::make(layout.type, "node");
}

std::string build_name(std::string name) { return "build_" + name; }
std::string count_name(std::string name) { return "count_" + name; }
std::string size_name(std::string name) { return "size_" + name; }

std::string get_index_name(const std::string &name) { return name + "_index"; }
std::string get_index_name(const ir::Expr &expr) {
    if (const auto *v = expr.as<ir::Var>()) {
        return get_index_name(v->name);
    }
    if (const auto *a = expr.as<ir::Extract>()) {
        std::optional<uint64_t> idx = get_constant_value(a->idx);
        internal_assert(idx.has_value()) << a->idx;
        return get_index_name(a->vec) + std::to_string(*idx);
    }
    internal_error << "unexpected expression: " << expr;
}

ir::Expr get_index(const ir::Expr &expr) {
    if (const auto *a = expr.as<ir::Extract>()) {
        return a->idx;
    }
    return ir::Expr();
}

ir::Type get_layout_reference_type(const ir::Layout &layout) {
    const auto *bvh_t = layout.type.as<ir::BVH_t>();
    internal_assert(bvh_t) << "expected ADT, received: " << layout.type;
    return ir::Ref_t::make(bvh_t->name);
}

ir::Type get_recursive_build_function_type(const ir::Layout &layout) {
    return ir::Function_t::make(
        layout.get_index_type(),
        {
            ir::Function_t::ArgSig{.type = get_layout_reference_type(layout),
                                   .is_mutable = false},
        });
}

ir::Type get_recursive_count_function_type(const ir::Layout &layout,
                                           const ir::Type &concretized_type) {
    return ir::Function_t::make(
        ir::Void_t::make(), {
                                ir::Function_t::ArgSig{
                                    .type = get_layout_reference_type(layout),
                                    .is_mutable = false,
                                },
                            });
}

std::string get_recursive_build_function_name(const ir::Layout &layout) {
    return "rec_" + build_name(layout.name);
}

std::string get_recursive_count_function_name(const ir::Layout &layout) {
    return "rec_" + count_name(layout.name);
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

std::vector<const ir::BuildRule *> get_build_rules(const ir::BuildIR &ir) {
    struct Visit : public ir::Visitor {
        std::vector<const ir::BuildRule *> rules;
        void visit(const ir::BuildRule *node) { rules.push_back(node); };
    };
    Visit visitor;
    ir.accept(&visitor);
    return visitor.rules;
}

// The size of this member, if it exists.
ir::Expr get_member_size(const ir::Member &member) {
    if (const auto *field = member.as<ir::Field>()) {
        const auto *type = field->type.as<ir::Array_t>();
        internal_assert(type) << field->type;
        return type->size;
    }
    if (const auto *group = member.as<ir::Group>()) {
        return group->size;
    }
    internal_error << "[unimplemented]: " << member;
}

std::string split_name(uint32_t count, const std::string &field) {
    return "split" + std::to_string(count) + "on_" + field;
}

// Similar to the `get_field_in_layout` algorithm, but really only cares about
// the concretized location of fields in the layout. We also keep track of
// visited groups so that when an element is appended, their respective index is
// incremented.
ir::WriteLoc get_write_loc(ir::WriteLoc base, const ir::Member &member,
                           const std::string &field,
                           const ir::BVH_t::Variant &variant,
                           const ir::Layout &layout, const ir::Program &program,
                           std::vector<const ir::Group *> &visited_groups) {
    uint32_t split_count = 0;
    const ir::Chain *chain = to_chainz(member);
    for (const auto &m : chain->members) {
        switch (m.node_type()) {
        case ir::IRLayoutEnum::Field: {
            const ir::Field *node = m.as<ir::Field>();
            if (node->name == field) {
                base.add_struct_access(node->name);
                return base;
            }
            continue;
        }
        case ir::IRLayoutEnum::Group: {
            const ir::Group *node = m.as<ir::Group>();
            std::string field_name = node->name;
            ir::WriteLoc path = base;
            path.add_struct_access(field_name);
            if (node->name == field) {
                return path;
            }
            ir::Expr index =
                ir::Var::make(layout.get_index_type(), get_index_name("this"));
            path.add_index_access(index);
            if (path = get_write_loc(path, node->inner, field, variant, layout,
                                     program, visited_groups);
                path.defined()) {
                if (auto it = std::find_if(visited_groups.begin(),
                                           visited_groups.end(),
                                           [&](const ir::Group *g) {
                                               return g->name == node->name;
                                           });
                    it == visited_groups.end()) {
                    visited_groups.push_back(node);
                }
                return path;
            }
            continue;
        }
        case ir::IRLayoutEnum::Split: {
            const ir::Split *node = m.as<ir::Split>();
            if (field == node->field_name()) {
                base.add_struct_access(node->field_name());
                return base;
            }
            for (const ir::Arm &arm : node->arms) {
                if (const std::optional<std::string> &name = arm.name;
                    !name.has_value() || *name != variant.name()) {
                    continue;
                }
                ir::WriteLoc path = base;
                if (m.bits() > 0) {
                    std::string field_name =
                        split_name(split_count++, node->field_name());
                    path.add_struct_access(field_name);
                }
                if (ir::WriteLoc split_path =
                        get_write_loc(path, arm.member, field, variant, layout,
                                      program, visited_groups);
                    split_path.defined()) {
                    std::string name = "arm_" + variant.name();
                    auto it = program.types.find(name);
                    internal_assert(it != program.types.end()) << name;
                    path.add_cast(ir::Ptr_t::make(it->second),
                                  ir::Cast::Mode::Reinterpret);
                    path.add_struct_access(field);
                    return path;
                }
            }
            continue;
        }
        case ir::IRLayoutEnum::Pad:
        case ir::IRLayoutEnum::Lookup:
        case ir::IRLayoutEnum::Materialize:
            continue;
        default:
            internal_error << "[unimplemented] member: " << m;
        }
    }
    return ir::WriteLoc();
}

class ConcretizeIndex : public ir::Mutator {
  public:
    ConcretizeIndex(const ir::Layout &layout) : layout(layout) {}

  private:
    const ir::Layout &layout;

    ir::Expr visit(const ir::Var *node) override {
        if (!ir::equals(node->type, get_layout_reference_type(layout))) {
            return node;
        }
        return ir::Var::make(layout.get_index_type(),
                             get_index_name(node->name));
    }

    // skip
    ir::BuildIR visit(const ir::BuildRecurse *node) override { return node; }
};

// Adds an access to the node if the value is retrieved from it, e.g., `high` ->
// `node.high`.
class AddSelfAccess : public ir::Mutator {
  public:
    AddSelfAccess(const ir::Layout &layout, const ir::BuildFunction &function)
        : layout(layout), function(function) {}

  private:
    const ir::Layout &layout;
    const ir::BuildFunction &function;

    ir::Expr visit(const ir::Var *node) override {
        const auto &fields = function.variant.fields();
        auto it = std::find_if(
            fields.begin(), fields.end(),
            [&](const ir::TypedVar &v) { return v.name == node->name; });
        if (it == fields.end()) {
            return node;
        }
        return ir::Access::make(node->name, self(layout));
    }
};

// Converts an append to the respective index. The for-loop is added
// during build construction.
class ConcretizeAppend : public ir::Mutator {
  public:
    ConcretizeAppend(const ir::Layout &layout) : layout(layout) {}

  private:
    const ir::Layout &layout;

    ir::Expr visit(const ir::Append *node) override {
        ir::Member primitives_group = layout.find_primitives_group();
        internal_assert(primitives_group.defined());
        std::string name = primitives_group.name();
        return ir::Var::make(ir::Index_t::make(), get_index_name(name));
    }
};

class ConcretizeVar : public ir::Mutator {
  public:
    ConcretizeVar(const ir::Layout &layout, const ir::BVH_t::Variant &variant,
                  const ir::Type &concretized_type, const ir::Program &program)
        : layout(layout), variant(variant), concretized_type(concretized_type),
          program(program) {}

  private:
    const ir::Layout &layout;
    const ir::BVH_t::Variant &variant;
    const ir::Type &concretized_type;
    const ir::Program &program;

    ir::Expr visit(const ir::Var *node) override {
        ir::WriteLoc loc(SPECIALIZED_TREE, concretized_type);
        std::vector<const ir::Group *> _;
        const ir::Struct_t::Map &fields = variant.fields();
        if (std::any_of(fields.begin(), fields.end(),
                        [&](const ir::TypedVar &field) {
                            return field.name == node->name;
                        })) {
            // This is an argument passed in from the canonical tree. We don't
            // need to concretize its location.
            return node;
        }
        if (loc = get_write_loc(loc, layout.body, node->name, variant, layout,
                                program, _);
            loc.defined()) {
            return loc.to_expr();
        }
        return node;
    }
};

class ConstructBuild : public ir::Visitor {
  public:
    ConstructBuild(const ir::Layout &layout, const ir::BVH_t::Variant &variant,
                   const ir::Type &concretized_type, const ir::Program &program)
        : layout(layout), variant(variant), concretized_type(concretized_type),
          program(program) {}

    std::vector<ir::Stmt> statements() {
        if (root.empty()) {
            return stmts;
        }
        // Otherwise, prepend the root.
        ir::Expr is_zero =
            ir::Var::make(ir::Index_t::make(), get_index_name("this")) ==
            ir::IdxImm::make(0);
        stmts.insert(stmts.begin(),
                     ir::IfElse::make(std::move(is_zero),
                                      ir::Sequence::make(std::move(root))));
        return stmts;
    }

    std::map<ir::Expr, std::vector<const ir::Group *>, ir::ExprLessThan>
    groups_by_field() {
        return groups;
    }

  private:
    bool is_root = false;
    const ir::Layout &layout;
    const ir::BVH_t::Variant &variant;
    const ir::Type &concretized_type;
    const ir::Program &program;
    std::vector<ir::Stmt> stmts;
    std::vector<ir::Stmt> root;
    std::map<ir::Expr, std::vector<const ir::Group *>, ir::ExprLessThan> groups;

    void append(ir::Stmt stmt) {
        if (is_root) {
            root.push_back(std::move(stmt));
            return;
        }
        stmts.push_back(std::move(stmt));
    }

    ir::Type get_field_type(const std::string &field) {
        const auto &fields = variant.fields();
        auto it = std::find_if(
            fields.begin(), fields.end(),
            [&](const ir::TypedVar &v) { return v.name == field; });
        if (it == fields.end()) {
            return ir::Type();
        }
        return it->type;
    }

    ir::WriteLoc update_indexes(ir::WriteLoc loc,
                                const std::vector<const ir::Group *> &groups) {
        if (groups.size() <= 1) {
            // Need to update the indexes.
            return loc;
        }
        // TODO(cgyurgyik): generalize this.
        internal_assert(groups.size() == 2)
            << "[unimplemented] " << groups.size() << "-nesting of groups";
        const ir::Group *inner = groups.front();
        const ir::Group *outer = groups.back();
        ir::Expr index = outer->index, size = inner->size;
        ir::WriteLoc new_location(loc.base(), loc.base_type());
        std::string last; // save the last field accessed.
        for (const std::variant<std::string, ir::Expr, ir::WriteLoc::Cast>
                 &access : loc.accesses) {
            if (std::holds_alternative<std::string>(access)) {
                std::string field = std::get<std::string>(access);
                last = field;
                new_location.add_struct_access(field);
                continue;
            }
            if (std::holds_alternative<ir::Expr>(access)) {
                index = std::get<ir::Expr>(access);
                if (outer->name == last) {
                    new_location.add_index_access(index / size);
                    last.clear();
                    continue;
                }
                if (inner->name == last) {
                    new_location.add_index_access(index % size);
                    last.clear();
                    continue;
                }
            }
            if (std::holds_alternative<ir::WriteLoc::Cast>(access)) {
                auto cast = std::get<ir::WriteLoc::Cast>(access);
                new_location.add_cast(cast.type, cast.mode);
                continue;
            }
            internal_error << "[unexpected] WriteLoc access type";
        }
        return new_location;
    }

    void visit(const ir::Append *node) {
        ir::Type index_type = ir::Index_t::make();
        ir::Member primitives_group = layout.find_primitives_group();
        internal_assert(primitives_group.defined())
            << "failed to find primitives collection in layout: `"
            << layout.name << "`";
        std::string name = primitives_group.name();

        internal_assert(!name.empty()) << primitives_group;
        ir::WriteLoc index(get_index_name(name), index_type);

        ir::WriteLoc write(SPECIALIZED_TREE, concretized_type);
        write.add_struct_access(name);
        std::string index_name = "__p";
        ir::Expr i = ir::Var::make(index_type, index_name);
        write.add_index_access(i + index.to_expr());
        ir::Stmt body =
            ir::Store::make(write, ir::Extract::make(node->input, i));
        ir::Type size_type = node->size.type();
        append(ir::ForAll::make(index_name,
                                ir::ForAll::Slice{
                                    .begin = make_zero(size_type),
                                    .end = node->size,
                                    .stride = make_one(size_type),
                                },
                                std::move(body)));
        append(ir::Accumulate::make(index, ir::Accumulate::Add, node->size));
    }

    void visit(const ir::BuildRecurse *node) {
        ir::Expr field = node->field;
        if (field.type().is_iterable()) {
            // f(a: T[n]) { recurse a; }
            // ...is syntactic sugar for:
            //
            // for i in 0..n { recurse a[i]; }
            std::string index_name = "__r";
            ir::WriteLoc let(get_index_name(node->field),
                             layout.get_index_type());
            ir::Expr size = field.type().size();
            ir::Expr index = ir::Var::make(size.type(), index_name);

            std::vector<ir::Stmt> body;
            // Call the recursive function.
            body.push_back(ir::LetStmt::make(
                let, call_recurse(ir::Extract::make(field, index))));
            // Then, update the specialized tree's respective field (...if the
            // field exists).
            ir::WriteLoc loc(SPECIALIZED_TREE, concretized_type);
            std::string name = get_field_name(field);
            std::vector<const ir::Group *> _;
            if (loc = get_write_loc(loc, layout.body, name, variant, layout,
                                    program, _);
                loc.defined()) {
                loc.add_index_access(std::move(index));
                body.push_back(ir::Store::make(std::move(loc), let.to_expr()));
            }
            append(ir::ForAll::make(index_name,
                                    ir::ForAll::Slice{
                                        .begin = make_zero(size.type()),
                                        .end = size,
                                        .stride = make_one(size.type()),
                                    },
                                    ir::Sequence::make(std::move(body))));
            return;
        }

        ir::WriteLoc let(get_index_name(node->field), layout.get_index_type());
        append(ir::LetStmt::make(let, call_recurse(field)));
        std::string name = get_field_name(field);

        ir::WriteLoc loc(SPECIALIZED_TREE, concretized_type);
        std::vector<const ir::Group *> _;
        if (loc = get_write_loc(loc, layout.body, name, variant, layout,
                                program, _);
            loc.defined()) {
            if (ir::Expr index = get_index(node->field); index.defined()) {
                loc.add_index_access(std::move(index));
            }
            append(ir::Store::make(std::move(loc), let.to_expr()));
        }
    }

    void visit(const ir::BuildReturn *node) {
        append(ir::Return::make(node->expr));
        node->expr.accept(this);
    }

    void visit(const ir::BuildLet *node) {
        node->stmt.accept(this);
        append(node->stmt);
    }

    void visit(const ir::BuildRoot *node) {
        is_root = true;
        node->rules.accept(this);
        is_root = false;
    }

    void visit(const ir::BuildRule *node) {
        ir::WriteLoc loc(SPECIALIZED_TREE, concretized_type);
        std::vector<const ir::Group *> visited_groups;
        ir::Expr field = node->field;
        std::string field_name = get_field_name(field);
        loc = get_write_loc(loc, layout.body, field_name, variant, layout,
                            program, visited_groups);
        internal_assert(loc.defined())
            << "did not find concretized location for field: `" << field << "`"
            << " : " << field.type();
        internal_assert(!groups.contains(field)) << field;
        groups[field] = visited_groups;
        ir::Expr expr = node->expr;
        if (!expr.defined()) {
            // This should just retrieve the field from this node.
            expr = node->field;
        }
        loc = update_indexes(std::move(loc), visited_groups);
        append(ir::Store::make(std::move(loc), expr));
        expr.accept(this);
    }

    void visit(const ir::BuildSequence *node) {
        for (const ir::BuildIR &ir : node->sequence) {
            ir.accept(this);
        }
    }

    ir::Expr call_recurse(const ir::Expr &field) {
        ir::Type type = get_recursive_build_function_type(layout);
        const auto *function_t = type.as<ir::Function_t>();
        internal_assert(function_t) << type;
        std::vector<ir::Expr> args = {field};
        std::string name = get_recursive_build_function_name(layout);
        ir::Expr func = ir::Var::make(std::move(type), std::move(name));
        return ir::Call::make(func, args);
    }

    std::string get_field_name(const ir::Expr &field) {
        if (const auto *v = field.as<ir::Var>()) {
            return v->name;
        }
        if (const auto *e = field.as<ir::Extract>()) {
            return get_field_name(e->vec);
        }
        internal_error << "[unexpected] expression: " << field;
    }
};

ir::Stmt construct_build_recursive_body(const ir::BuildFunction &function,
                                        const ir::Type &concretized_type,
                                        const ir::Layout &layout,
                                        const ir::Program &program) {
    ir::BuildIR body = function.body;
    const ir::BVH_t::Variant &variant = function.variant;
    body = ConcretizeIndex(layout).mutate(body);

    ConstructBuild visitor(layout, variant, concretized_type, program);
    body.accept(&visitor);

    std::vector<ir::Stmt> stmts;
    std::set<std::string> groups_visited;
    // There is one unique index per element in a collection; we only need
    // to define it once.
    bool this_index_defined = false;
    for (const auto &[field, groups] : visitor.groups_by_field()) {
        // Add an index statement.
        if (groups.empty()) {
            continue;
        }
        for (const ir::Group *group : groups) {
            internal_assert(!group->name.empty())
                << "[unexpected] empty group name (at this point, each group "
                   "should have a name, whether it be user-provided or "
                   "machine-generated)";
            if (groups_visited.contains(group->name)) {
                continue;
            }
            groups_visited.insert(group->name);
            if (!this_index_defined) {
                ir::Type index_type = ir::Index_t::make();
                std::string group_name = group->name;
                // In the case of SoA, we just refer to the first group we come
                // across (they all share the same index in the collection.)
                ir::WriteLoc assign(get_index_name("this"), index_type);
                stmts.push_back(ir::LetStmt::make(
                    std::move(assign),
                    ir::Var::make(index_type, get_index_name(group_name))));

                // Preemptively increment the unique index for the next element
                // in the collection.
                ir::WriteLoc increment(get_index_name(group_name), index_type);
                stmts.push_back(ir::Accumulate::make(std::move(increment),
                                                     ir::Accumulate::Add,
                                                     make_one(index_type)));
                this_index_defined = true;
            }
        }
    }

    std::vector<ir::Stmt> visitor_stmts = visitor.statements();
    stmts.insert(stmts.end(), visitor_stmts.begin(), visitor_stmts.end());

    ir::Stmt sequence = ir::Sequence::make(std::move(stmts));
    sequence = ConcretizeAppend(layout).mutate(std::move(sequence));
    sequence = ConcretizeVar(layout, variant, concretized_type, program)
                   .mutate(std::move(sequence));
    sequence = AddSelfAccess(layout, function).mutate(std::move(sequence));
    return sequence;
}

std::shared_ptr<ir::Function> construct_build_recursive(
    const ir::BuildLayout &build, const ir::Type &concretized_type,
    const ir::Layout &layout, const ir::Program &program) {
    ir::Match::Arms arms;
    for (const ir::BuildFunction &function : build.functions) {
        arms.push_back({
            function.variant,
            construct_build_recursive_body(function, concretized_type, layout,
                                           program),
        });
    }
    ir::Stmt body = ir::Match::make(self(layout), std::move(arms));
    std::vector<ir::Argument> args = {
        ir::Argument("node", get_layout_reference_type(layout)),
        ir::Argument(SPECIALIZED_TREE, concretized_type,
                     /*default_value=*/ir::Expr(),
                     /*mutating=*/true),
    };
    std::vector<ir::TypedVar> free_variables = gather_free_vars(body);
    for (const auto &[name, type] : free_variables) {
        if (std::any_of(args.begin(), args.end(), [&](const ir::Argument &arg) {
                return arg.name == name;
            })) {
            continue;
        }
        args.push_back(ir::Argument(name, type,
                                    /*default_value=*/ir::Expr(),
                                    /*mutating=*/true));
    }
    std::string name = get_recursive_build_function_name(layout);
    ir::Function::InterfaceList interfaces;
    std::vector<ir::Function::Attribute> attributes;
    return std::make_shared<ir::Function>(
        std::move(name), std::move(args), layout.get_index_type(),
        std::move(body), interfaces, std::move(attributes));
}

ir::WriteLoc get_write_loc(ir::WriteLoc loc, const std::string &name,
                           const ir::BuildLayout &build,
                           const ir::Layout &layout,
                           const ir::Program &program) {
    std::vector<const ir::Group *> _;
    for (const ir::BuildFunction &function : build.functions) {
        ir::WriteLoc base = get_write_loc(loc, layout.body, name,
                                          function.variant, layout, program, _);
        if (!base.defined()) {
            continue;
        }
        return base;
    }
    internal_error << "member name: `" << name << "` not found";
}

// Returns both the `count` body and a list of generated indexes. These must be
// added as arguments to the final `count` function.
std::pair<ir::Stmt, std::set<ir::Expr, ir::ExprLessThan>>
construct_count_recursive_body(const ir::BuildFunction &function,
                               const ir::Type &concretized_type,
                               const ir::Layout &layout,
                               const ir::BuildLayout &build,
                               const ir::Program &program) {
    const auto *bvh_t = layout.type.as<ir::BVH_t>();
    internal_assert(bvh_t) << "expected ADT, received: " << layout.type;

    std::vector<ir::Stmt> stmts;
    std::set<ir::Expr, ir::ExprLessThan> counts_updated;
    std::set<ir::Expr, ir::ExprLessThan> generated_indexes;
    for (const ir::Argument &argument : function.arguments) {
        if (argument.type.is_iterable() &&
            ir::equals(argument.type.element_of(), bvh_t->primitive)) {
            ir::Expr count = argument.type.size();
            internal_assert(count.defined());
            ir::Member group = layout.find_primitives_group();
            internal_assert(group.defined());
            ir::Expr member_size = get_member_size(group);
            if (is_const(member_size)) {
                continue;
            }
            const auto *size_variable = member_size.as<ir::Var>();
            // TODO(cgyurgyik): don't limit this to constants and variables.
            internal_assert(size_variable) << member_size;
            ir::WriteLoc base(SPECIALIZED_TREE, concretized_type);
            ir::WriteLoc loc = get_write_loc(base, size_variable->name, build,
                                             layout, program);
            stmts.push_back(ir::Accumulate::make(
                std::move(loc), ir::Accumulate::OpType::Add, std::move(count)));
        }
        if (ir::equals(argument.type, get_layout_reference_type(layout))) {
            ir::Type function_t =
                get_recursive_count_function_type(layout, concretized_type);
            std::string name = get_recursive_count_function_name(layout);
            ir::Expr func = ir::Var::make(function_t, name);
            ir::Expr arg = ir::Access::make(argument.name, self(layout));
            stmts.push_back(ir::CallStmt::make(func, {arg}));
            continue;
        }
        if (argument.type.is_iterable() &&
            ir::equals(argument.type.element_of(),
                       get_layout_reference_type(layout))) {
            std::string index_name = "__r";
            ir::Expr i = ir::Var::make(ir::Index_t::make(), index_name);

            ir::Type function_t =
                get_recursive_count_function_type(layout, concretized_type);
            std::string name = get_recursive_count_function_name(layout);
            ir::Expr func = ir::Var::make(function_t, name);
            ir::Expr arg = ir::Access::make(argument.name, self(layout));
            arg = ir::Extract::make(arg, i);
            ir::Stmt body = ir::CallStmt::make(func, {arg});
            ir::Expr size = argument.type.size();
            stmts.push_back(
                ir::ForAll::make(index_name,
                                 ir::ForAll::Slice{
                                     .begin = make_zero(size.type()),
                                     .end = size,
                                     .stride = make_one(size.type()),
                                 },
                                 std::move(body)));
        }
    }
    std::vector<const ir::BuildRule *> rules = get_build_rules(function.body);
    for (const ir::BuildRule *rule : rules) {
        const auto *field = rule->field.as<ir::Var>();
        if (field == nullptr) {
            const auto *extract = rule->field.as<ir::Extract>();
            field = extract->vec.as<ir::Var>();
        }
        internal_assert(field) << rule->field;
        ir::Member group = layout.find_group_for(field->name);
        if (!group.defined()) {
            continue;
        }

        ir::Expr size = get_member_size(group);
        if (counts_updated.contains(size)) {
            continue;
        }
        std::string name = group.name();
        counts_updated.insert(size);
        if (!size.defined()) {
            // Some groups may not include a size field. We need to include one
            // for malloc'ing the correct count.
            ir::WriteLoc base(size_name(group.name()), ir::Index_t::make());
            generated_indexes.insert(base.to_expr());
            stmts.push_back(ir::Accumulate::make(
                std::move(base), ir::Accumulate::OpType::Add,
                make_one(ir::Index_t::make())));
            continue;
        }
        if (is_const(size)) {
            continue;
        }
        const auto *size_variable = size.as<ir::Var>();
        // TODO(cgyurgyik): don't limit this to constants and variables.
        internal_assert(size_variable) << size;
        ir::WriteLoc base(SPECIALIZED_TREE, concretized_type);
        ir::WriteLoc loc =
            get_write_loc(base, size_variable->name, build, layout, program);
        stmts.push_back(ir::Accumulate::make(std::move(loc),
                                             ir::Accumulate::OpType::Add,
                                             make_one(size_variable->type)));
    }
    internal_assert(!stmts.empty()) << "[unexpected] empty count function!";
    ir::Stmt sequence = ir::Sequence::make(std::move(stmts));
    sequence = AddSelfAccess(layout, function).mutate(std::move(sequence));
    return {sequence, generated_indexes};
}

std::shared_ptr<ir::Function> construct_count_recursive(
    const ir::Type &concretized_type, const ir::BuildLayout &build,
    const ir::Layout &layout, const ir::Program &program) {

    ir::Match::Arms arms;
    std::set<ir::Expr, ir::ExprLessThan> generated_indexes;
    for (const ir::BuildFunction &function : build.functions) {
        auto [body, indexes] = construct_count_recursive_body(
            function, concretized_type, layout, build, program);
        arms.push_back({
            function.variant,
            std::move(body),
        });
        for (const ir::Expr &index : indexes) {
            generated_indexes.insert(index);
        }
    }

    std::string name = get_recursive_count_function_name(layout);
    ir::Stmt body = ir::Match::make(self(layout), std::move(arms));
    std::vector<ir::Argument> args = {
        ir::Argument("node", get_layout_reference_type(layout)),
        ir::Argument(SPECIALIZED_TREE, concretized_type,
                     /*default_value=*/ir::Expr(),
                     /*mutating=*/true),
    };
    std::vector<ir::TypedVar> free_variables = gather_free_vars(body);
    for (const auto &[name, type] : free_variables) {
        if (std::any_of(args.begin(), args.end(), [&](const ir::Argument &arg) {
                return arg.name == name;
            })) {
            continue;
        }
        args.push_back(ir::Argument(name, type));
    }
    // Capture the generated indexes as well.
    for (const ir::Expr &expr : generated_indexes) {
        const ir::Var *v = expr.as<ir::Var>();
        internal_assert(v) << expr;
        if (std::any_of(args.begin(), args.end(), [&](const ir::Argument &arg) {
                return v->name == name;
            })) {
            continue;
        }
        args.push_back(ir::Argument(v->name, v->type,
                                    /*default_value=*/ir::Expr(),
                                    /*mutating=*/true));
    }
    ir::Function::InterfaceList interfaces;
    std::vector<ir::Function::Attribute> attributes;
    return std::make_shared<ir::Function>(std::move(name), std::move(args),
                                          ir::Void_t::make(), std::move(body),
                                          interfaces, std::move(attributes));
}

std::shared_ptr<ir::Function>
construct_build_full(const ir::Type &concretized_type,
                     const ir::BuildLayout &build, const ir::Layout &layout,
                     const ir::Program &program) {

    std::vector<ir::Stmt> stmts;
    std::vector<ir::Stmt> stack;
    // 1. Initialize the specialized tree.
    ir::WriteLoc specialized_tree(SPECIALIZED_TREE, concretized_type);
    stack.push_back(
        ir::Allocate::make(specialized_tree, ir::Allocate::Memory::Stack));

    // 2. Pre-process: call to function gathering counts for each group.
    {
        std::string name = get_recursive_count_function_name(layout);
        ir::Type type =
            get_recursive_count_function_type(layout, concretized_type);
        ir::Expr func = ir::Var::make(type, name);
        ir::Expr arg1 = ir::Var::make(layout.type, CANONICAL_TREE);
        ir::Expr arg2 = ir::Var::make(concretized_type, SPECIALIZED_TREE);
        stmts.push_back(ir::CallStmt::make(func, {arg1, arg2}));
    }

    // 3. malloc all groups (to include arrays) with the given counts.
    std::vector<ir::Member> groups = layout.find_all_groups();
    for (const ir::Member &member : groups) {
        if (const ir::Field *field = member.as<ir::Field>()) {
            internal_assert(field->type.is<ir::Array_t>()) << field->type;
            ir::Expr size = field->type.size();
            if (const auto *size_variable = size.as<ir::Var>()) {
                ir::WriteLoc base(SPECIALIZED_TREE, concretized_type);
                ir::WriteLoc location = get_write_loc(base, size_variable->name,
                                                      build, layout, program);
                internal_assert(location.defined());
                size = location.to_expr();
            }
            ir::WriteLoc location(
                field->name, ir::Array_t::make(field->type.element_of(), size));
            ir::WriteLoc write_to = get_write_loc(specialized_tree, field->name,
                                                  build, layout, program);
            stmts.push_back(ir::Allocate::make(location));
            stmts.push_back(
                ir::Store::make(std::move(write_to), location.to_expr()));

            // Allocate indexes.
            stack.push_back(ir::Allocate::make(
                /*loc=*/ir::WriteLoc(get_index_name(field->name),
                                     ir::Index_t::make()),
                /*memory=*/ir::Allocate::Memory::Stack));
            continue;
        }
        if (const ir::Group *group = member.as<ir::Group>()) {
            if (member.bits() == 0) {
                continue;
            }
            ir::WriteLoc write_to = get_write_loc(specialized_tree, group->name,
                                                  build, layout, program);
            auto it = program.types.find(group->name);
            internal_assert(it != program.types.end())
                << "no type found for: `" << group->name << "`";
            ir::Expr size = group->size;
            if (!size.defined()) {
                // Not all indirect groups may have a size. We write to a local
                // variable in this case for element counting.
                ir::WriteLoc index(size_name(member.name()),
                                   ir::Index_t::make());
                stack.push_back(ir::Allocate::make(index, ir::Allocate::Stack));
                size = index.to_expr();
                ir::WriteLoc location(group->name,
                                      ir::Array_t::make(it->second, size));
                stmts.push_back(ir::Allocate::make(location));
                stmts.push_back(
                    ir::Store::make(std::move(write_to), location.to_expr()));
                continue;
            }
            if (const auto *size_variable = size.as<ir::Var>()) {
                ir::WriteLoc base(SPECIALIZED_TREE, concretized_type);
                ir::WriteLoc location = get_write_loc(base, size_variable->name,
                                                      build, layout, program);
                internal_assert(location.defined());
                size = location.to_expr();
            }
            ir::WriteLoc location(group->name,
                                  ir::Array_t::make(it->second, size));
            stmts.push_back(ir::Allocate::make(location));
            stmts.push_back(
                ir::Store::make(std::move(write_to), location.to_expr()));

            ir::WriteLoc index(get_index_name(group->name),
                               ir::Index_t::make());
            stack.push_back(
                ir::Allocate::make(index, ir::Allocate::Memory::Stack));
            continue;
        }

        internal_error << "[unexpected] group: " << member;
    }

    // 4. Call `__build_<name>` on CT.root
    {
        ir::Type function_t = get_recursive_build_function_type(layout);
        std::string name = get_recursive_build_function_name(layout);
        stmts.push_back(ir::CallStmt::make(
            ir::Var::make(function_t, name),
            {
                ir::Var::make(get_layout_reference_type(layout),
                              CANONICAL_TREE),
            }));
    }

    // 5. Return `ST`
    stmts.push_back(
        ir::Return::make(ir::Var::make(concretized_type, SPECIALIZED_TREE)));

    std::vector<ir::Argument> args = {
        ir::Argument(CANONICAL_TREE, get_layout_reference_type(layout)),
    };
    // Place stack allocations at the beginning; these will be filled by the
    // recursive `count` function.
    stmts.insert(stmts.begin(), stack.begin(), stack.end());
    ir::Stmt body = ir::Sequence::make(std::move(stmts));
    std::string name = build_name(build.name);
    ir::Function::InterfaceList interfaces;
    std::vector<ir::Function::Attribute> attributes;
    return std::make_shared<ir::Function>(std::move(name), std::move(args),
                                          concretized_type, std::move(body),
                                          interfaces, std::move(attributes));
}

template <typename Out, typename In>
Out visit_call(const In *node, const std::string &name,
               const std::vector<ir::Argument> &arguments,
               const ir::Type &return_type) {
    const auto *v = node->func.template as<ir::Var>();
    if (v == nullptr) {
        return node;
    }
    if (v->name != name) {
        return node;
    }
    const auto *function_t = v->type.template as<ir::Function_t>();
    internal_assert(function_t) << v->type;
    // Leave the first argument alone.
    std::vector<ir::Expr> args = {node->args.front()};
    for (int i = 1, e = arguments.size(); i < e; ++i) {
        args.push_back(ir::Var::make(arguments[i].type, arguments[i].name));
    }

    std::vector<ir::Function_t::ArgSig> types;
    for (int i = 0, e = arguments.size(); i < e; ++i) {
        types.push_back(ir::Function_t::ArgSig{
            .type = arguments[i].type,
            .is_mutable = arguments[i].mutating,
        });
    }
    ir::Type updated_function_t =
        ir::Function_t::make(return_type, std::move(types));
    return In::make(ir::Var::make(std::move(updated_function_t), name),
                    std::move(args));
}

ir::Stmt update_recursive_arguments(ir::Stmt body, const std::string &name,
                                    const std::vector<ir::Argument> &arguments,
                                    const ir::Type &return_type) {
    struct UpdateCalls : ir::Mutator {
        const std::string &name;
        const std::vector<ir::Argument> &arguments;
        const ir::Type &return_type;

        UpdateCalls(const std::string &name,
                    const std::vector<ir::Argument> &arguments,
                    const ir::Type &return_type)
            : name(name), arguments(arguments), return_type(return_type) {}

        ir::Expr visit(const ir::Call *node) override {
            return visit_call<ir::Expr>(node, name, arguments, return_type);
        }

        // TODO(cgyurgyik): exact duplicate of above :(
        ir::Stmt visit(const ir::CallStmt *node) override {
            return visit_call<ir::Stmt>(node, name, arguments, return_type);
        }
    };
    return UpdateCalls{name, arguments, return_type}.mutate(body);
}

} // namespace

ir::Program LowerBuilds::run(ir::Program program,
                             const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "[unimplemented] support selecting a schedule target!\n";
    ir::Schedule &schedule = program.schedules[ir::Target::Host];
    const ir::BuildMap &tree_builds = schedule.tree_builds;
    if (tree_builds.empty()) {
        return program;
    }
    const ir::LayoutMap &tree_layouts = schedule.tree_layouts;
    for (const auto &[name, build] : tree_builds) {
        ir::Type concretized_type;
        {
            auto it = program.types.find(name);
            internal_assert(it != program.types.end())
                << "no concretized type found for tree: `" << name << "`";
            concretized_type = it->second;
        }
        auto it = tree_layouts.find(name);
        // The layout is necessary for determining where values should live.
        internal_assert(it != tree_layouts.end())
            << "no layout found for tree: `" << name << "`";
        std::vector<std::shared_ptr<ir::Function>> functions;
        // First, construct the recursive build algorithm.
        functions.push_back(construct_build_recursive(build, concretized_type,
                                                      it->second, program));
        // Then, construct the recursive count algorithm.
        functions.push_back(construct_count_recursive(concretized_type, build,
                                                      it->second, program));
        // Finally, construct the final build algorithm.
        functions.push_back(
            construct_build_full(concretized_type, build, it->second, program));
        functions.back()->attributes.push_back(
            ir::Function::Attribute::exported);

        // Capture free variables for function calls.
        for (int i = 0, e = functions.size(); i < e; ++i) {
            for (int j = 0; j < e; ++j) {
                functions[i]->body = update_recursive_arguments(
                    std::move(functions[i]->body), functions[j]->name,
                    functions[j]->args, functions[j]->ret_type);
            }
        }

        // Append them to the program.
        for (int i = 0, e = functions.size(); i < e; ++i) {
            std::string name = functions[i]->name;
            const auto [_, inserted] =
                program.funcs.try_emplace(name, std::move(functions[i]));
            internal_assert(inserted)
                << "function: `" << name << "` already exists in program";
        }
    }

    // The layout and build lowering are complete! Send that shit to the void.
    schedule.tree_layouts.clear();
    schedule.tree_builds.clear();
    return program;
}

} // namespace lower
} // namespace bonsai
