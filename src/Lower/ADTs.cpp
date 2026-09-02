#include "Lower/ADTs.h"

#include "Lower/ADTLayout.h"

#include "IR/Expr.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <memory>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

using LayoutMap = std::map<std::string, ADTLayout>;
using FuncMap = std::map<std::string, std::shared_ptr<ir::Function>>;

// The function that puts a `TaggedIndex` variant in its pool and returns the
// handle naming it. Qualified twice over: `Shape_Sph` is already the exported
// constructor and `Sph` is already the struct of the fields.
std::string appender_name(const Type &adt, const std::string &variant) {
    const ADT_t *node = adt.as<ADT_t>();
    internal_assert(node) << "Not a variant type: " << adt;
    return node->name + "_" + variant + "_new";
}

// Rewrites variant types into what their layout says they are stored as.
struct RewriteADTs : public Mutator {
    const LayoutMap &layouts;
    // Prepared for every `TaggedIndex` variant, added to the program only for
    // the ones a Construct below actually reaches.
    const FuncMap &appenders;
    std::set<std::string> needed;

    RewriteADTs(const LayoutMap &layouts, const FuncMap &appenders)
        : layouts(layouts), appenders(appenders) {}

    const ADTLayout &layout_of(const Type &type) const {
        const ADT_t *adt = type.as<ADT_t>();
        internal_assert(adt) << "Not a variant type: " << type;
        const auto found = layouts.find(adt->name);
        internal_assert(found != layouts.end())
            << "No layout chosen for " << adt->name;
        return found->second;
    }

    // A variant type becomes whatever it is stored as.
    Type mutate(const Type &type) override {
        Type rec = Mutator::mutate(type);
        if (rec.is<ADT_t>()) {
            return layout_of(rec).storage;
        }
        return rec;
    }

    using Mutator::mutate;

    // A name of a variant type now names one of whatever it is stored as. The
    // base Mutator leaves a Var's type alone, so this is the same thing
    // Lower/Options.cpp has to do for option types.
    Expr visit(const Var *node) override {
        Type type = mutate(node->type);
        if (type.same_as(node->type)) {
            return node;
        }
        return Var::make(std::move(type), node->name);
    }

    // Similarly for the place a write goes.
    std::pair<WriteLoc, bool> mutate_writeloc(const WriteLoc &loc) override {
        Type base_type = mutate(loc.base_type);
        bool not_changed = base_type.same_as(loc.base_type);
        WriteLoc new_loc(loc.base, std::move(base_type));
        for (const auto &value : loc.accesses) {
            if (const Expr *expr = std::get_if<Expr>(&value)) {
                Expr new_value = mutate(*expr);
                not_changed = not_changed && new_value.same_as(*expr);
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        return {std::move(new_loc), not_changed};
    }

    // The tag of `value`, as the layout's tag type.
    Expr tag_of(const ADTLayout &layout, const Expr &value) const {
        if (layout.kind == ir::AdtLayout::Inline) {
            return Access::make(layout.tag_field, value);
        }
        // The top bits of the handle. A shift and nothing else: the tag is the
        // highest field, so there is nothing above it to mask off.
        return BinOp::make(
            BinOp::OpType::Shr, value,
            UIntImm::make(layout.tag_type, ADTLayout::tag_shift));
    }

    // The index part of a `TaggedIndex` handle: everything below the tag.
    Expr index_of(const ADTLayout &layout, const Expr &value) const {
        const uint64_t mask = (1ull << ADTLayout::tag_shift) - 1ull;
        return BinOp::make(BinOp::OpType::BwAnd, value,
                           UIntImm::make(layout.tag_type, mask));
    }

    // The fields of `value`, read as `variant`.
    //
    // Reading a field of the variant a value is not reads whatever those bytes
    // happen to be, under either layout, which is why every caller of this is
    // already inside an arm that has tested the tag.
    Expr as_variant(const ADTLayout &layout, const Expr &value,
                    const std::string &variant) const {
        if (layout.kind == ir::AdtLayout::Inline) {
            // The union's members are named for their variants, so this is
            // just naming one.
            return Access::make(variant,
                                Access::make(layout.payload_field, value));
        }
        // The pool the tag names, at the index the rest of the handle is. The
        // pool is an extern, so this Var is free here and the LowerExterns that
        // runs after this pass turns it into a parameter of whichever functions
        // reach it.
        const Type pool_type = Array_t::make(layout.variant(variant), Expr());
        return Extract::make(Var::make(pool_type, layout.pool(variant)),
                             index_of(layout, value));
    }

    // A Build of something that holds a variant type -- an array of shapes,
    // say. The base Mutator keeps a Build's type as it found it (its TODO asks
    // whether a mutation can change it; this is one that does), so an array
    // would keep an element type no value in it has any more.
    //
    // A Build *of* a variant type does not reach here: that is a Construct,
    // which names the variant, and this pass turns it into a Build of the
    // storage.
    Expr visit(const Build *node) override {
        std::vector<Expr> values;
        values.reserve(node->values.size());
        bool not_changed = true;
        for (const Expr &value : node->values) {
            Expr new_value = mutate(value);
            not_changed = not_changed && new_value.same_as(value);
            values.push_back(std::move(new_value));
        }
        // No assertion tying the two together, unlike Lower/Options.cpp: a
        // value here is rebuilt whenever anything inside it changed, which is
        // often and says nothing about this Build's own type. Reading a
        // sphere's centre out of a tree whose primitives are variants gives a
        // new expression of the same f32x3 type.
        Type type = mutate(node->type);
        if (not_changed && type.same_as(node->type)) {
            return node;
        }
        return Build::make(std::move(type), std::move(values));
    }

    // A cast's target type, which the base Mutator carries over as it found
    // it -- its own TODO asks whether it should. Dereferencing an option is
    // one of these: `*isect` is `cast<Shape>(isect)`, and once the option
    // holds what a Shape is stored as, the cast has to say so too.
    Expr visit(const Cast *node) override {
        Expr value = mutate(node->value);
        Type type = mutate(node->type);
        if (value.same_as(node->value) && type.same_as(node->type)) {
            return node;
        }
        return Cast::make(std::move(type), std::move(value), node->mode);
    }

    // A lambda's argument types, which the base Mutator also carries over as
    // it found them. `filter(|sh : Shape| .., shapes)` only type-checks while
    // the lambda and the set agree about what an element is, so the argument
    // has to change at the same time the set does.
    Expr visit(const Lambda *node) override {
        std::vector<TypedVar> args;
        args.reserve(node->args.size());
        bool not_changed = true;
        for (const TypedVar &arg : node->args) {
            Type type = mutate(arg.type);
            not_changed = not_changed && type.same_as(arg.type);
            args.push_back(TypedVar{arg.name, std::move(type)});
        }
        Expr value = mutate(node->value);
        if (not_changed && value.same_as(node->value)) {
            return node;
        }
        return Lambda::make(std::move(args), std::move(value));
    }

    Expr visit(const Construct *node) override {
        std::vector<Expr> args;
        args.reserve(node->args.size());
        for (const Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        const ADTLayout &layout = layout_of(node->type);

        if (layout.kind == ir::AdtLayout::TaggedIndex) {
            // Building a handle means putting the fields somewhere and saying
            // where, which is a store and a counter rather than an expression,
            // so it becomes a call to a function that does both. Recorded so
            // that only the variants a program actually builds get one -- an
            // unused appender would still name its pool, and choosing this
            // layout would then oblige the caller to supply storage for
            // variants it never constructs.
            const std::string &fname = appender_name(node->type, node->variant);
            needed.insert(fname);
            const auto found = appenders.find(fname);
            internal_assert(found != appenders.end())
                << "No appender prepared for " << fname;
            return Call::make(
                Var::make(found->second->call_type(), fname), std::move(args));
        }

        std::vector<Expr> whole;
        whole.push_back(
            UIntImm::make(layout.tag_type, layout.tag(node->variant)));
        whole.push_back(UnionOf::make(
            layout.payload, node->variant,
            Build::make(layout.variant(node->variant), std::move(args))));
        return Build::make(layout.storage, std::move(whole));
    }

    // A match becomes a test per arm, in the order they were written.
    //
    // The last arm needs no test: MatchVariant::make has already checked that
    // every variant is named exactly once, so once the others are ruled out
    // this is the only thing left. Testing it anyway would leave a branch
    // nothing can reach and nothing to put in it.
    Stmt visit(const MatchVariant *node) override {
        const ADT_t *adt = node->value.type().as<ADT_t>();
        internal_assert(adt) << "Match on a non-variant type: " << node->value;
        const ADTLayout &layout = layout_of(node->value.type());
        const Expr value = mutate(node->value);

        Stmt result;
        for (size_t i = node->arms.size(); i-- > 0;) {
            const MatchVariant::Arm &arm = node->arms[i];
            const auto index = adt->index_of(arm.variant);
            internal_assert(index.has_value())
                << adt->name << " has no variant " << arm.variant;
            const Struct_t::Map &fields = adt->fields(*index);

            // The names the arm gave the fields, bound to them. Reading a
            // field of the variant a value is not would read whatever those
            // bytes happen to be, which is why these are inside the arm.
            const Expr payload = as_variant(layout, value, arm.variant);
            std::vector<Stmt> body;
            body.reserve(arm.bindings.size() + 1);
            for (size_t f = 0; f < arm.bindings.size(); f++) {
                body.push_back(LetStmt::make(
                    WriteLoc(arm.bindings[f], mutate(fields[f].type)),
                    Access::make(fields[f].name, payload)));
            }
            body.push_back(mutate(arm.body));
            Stmt arm_body = Sequence::make(std::move(body));

            if (!result.defined()) {
                result = std::move(arm_body);
                continue;
            }
            const Expr is_variant = BinOp::make(
                BinOp::OpType::Eq, tag_of(layout, value),
                UIntImm::make(layout.tag_type, layout.tag(arm.variant)));
            result = IfElse::make(is_variant, std::move(arm_body),
                                  std::move(result));
        }
        return result;
    }
};

// The variant types anything in `type` is built out of, by name.
//
// Reachability rather than a direct match: an exported function rarely takes a
// `Shape`, it takes a tree whose primitives are shapes, or an option of one.
struct GatherADTs : public Visitor {
    std::set<std::string> &found;
    std::set<std::string> seen_structs;

    GatherADTs(std::set<std::string> &found) : found(found) {}

    void visit(const ADT_t *node) override {
        found.insert(node->name);
        Visitor::visit(node);
    }

    // A type can name itself through a pointer, so a struct is walked once.
    void visit(const Struct_t *node) override {
        if (seen_structs.insert(node->name).second) {
            Visitor::visit(node);
        }
    }
};

// One constructor per variant of every variant type a caller can see, as an
// exported function whose body is a Construct.
//
// Without these a C++ caller has to write `s.tag = 0; s.payload.Sph = ..`,
// which puts this pass's numbering into the driver -- where nothing would
// catch it drifting, and where it would have to change again as soon as a
// schedule picks a different layout. These are added before the rewrite
// below, so they are lowered by exactly the same code as a Construct written
// in bonsai, and every backend gets them rather than just the C++ one.
void add_variant_constructors(ir::Program &program) {
    std::set<std::string> outward;
    GatherADTs gather(outward);
    for (const auto &[fname, func] : program.funcs) {
        if (!func->is_exported()) {
            continue;
        }
        func->ret_type.accept(&gather);
        for (const auto &arg : func->args) {
            arg.type.accept(&gather);
        }
    }

    std::map<std::string, std::shared_ptr<ir::Function>> made;
    for (const auto &[name, type] : program.types) {
        const ADT_t *adt = type.as<ADT_t>();
        if (!adt || !outward.contains(adt->name)) {
            continue;
        }
        for (size_t v = 0; v < adt->variants.size(); v++) {
            const std::string &variant = adt->variant_name(v);
            const Struct_t::Map &fields = adt->fields(v);

            std::vector<ir::Function::Argument> args;
            std::vector<Expr> values;
            args.reserve(fields.size());
            values.reserve(fields.size());
            for (const TypedVar &field : fields) {
                args.push_back(ir::Function::Argument{field.name, field.type,
                                                      Expr(), false, false});
                values.push_back(Var::make(field.type, field.name));
            }

            // Qualified, because a variant name alone is already the name of
            // the struct of its fields, and C++ would then have a function
            // and a type sharing one name.
            const std::string fname = adt->name + "_" + variant;
            internal_assert(!program.funcs.contains(fname))
                << "Cannot name the constructor for " << variant << " of "
                << adt->name << ": something is already called " << fname;
            made[fname] = std::make_shared<ir::Function>(
                fname, std::move(args), type,
                Return::make(Construct::make(type, variant, std::move(values))),
                ir::Function::InterfaceList{},
                std::vector<ir::Function::Attribute>{
                    ir::Function::Attribute::exported});
        }
    }
    program.funcs.insert(made.begin(), made.end());
}

// One appender per variant of every `TaggedIndex` type, by name.
//
// The body is the whole of what building a handle means:
//
//     func Shape_Sph_new(centre : vec3f, radius : Float) -> u64 {
//         i = atomic_add(&Shape_Sph_fill[0], 1);
//         Shape_Sph_pool[i] = Sph{centre, radius};
//         return (tag << 56) | i;
//     }
//
// Atomic because a schedule may run the code that builds these in parallel, and
// a schedule is not allowed to change the answer. It does not make the *order*
// deterministic -- which index a given value lands at depends on who got there
// first -- but nothing may observe two values at one index, and a plain
// read-modify-write would allow exactly that.
//
// `pool` and `fill` are free here on purpose. They are declared as externs
// beside these, and the LowerExterns that runs after this pass threads them
// into these functions and into everything that calls them, so the caller
// passes in the storage and the counter the same way it passes in any other
// extern. That is what makes a variant buildable without an allocator: the
// memory and the capacity are the caller's.
FuncMap make_appenders(const LayoutMap &layouts) {
    FuncMap appenders;
    for (const auto &[adt_name, layout] : layouts) {
        if (layout.kind != ir::AdtLayout::TaggedIndex) {
            continue;
        }
        const Type handle = layout.storage;
        for (const Type &variant : layout.variants) {
            const Struct_t *fields_of = variant.as<Struct_t>();
            internal_assert(fields_of) << "A variant is a struct: " << variant;
            const std::string &vname = fields_of->name;

            std::vector<ir::Function::Argument> args;
            std::vector<Expr> values;
            for (const TypedVar &field : fields_of->fields) {
                args.push_back(ir::Function::Argument{field.name, field.type,
                                                      Expr(), false, false});
                values.push_back(Var::make(field.type, field.name));
            }

            const Type pool_type = Array_t::make(variant, Expr());
            const Type fill_type = Array_t::make(handle, Expr());
            const std::string &pool = layout.pool(vname);
            const std::string &fill = layout.fill(vname);
            const std::string index = "_index";

            // Where in the pool this one goes. AtomicAdd returns the value
            // before the add, so this is the first free slot and the counter
            // now names the next one.
            WriteLoc fill_slot(fill, fill_type);
            fill_slot.add_index_access(UIntImm::make(handle, 0));
            const Expr claim =
                AtomicAdd::make(PtrTo::make(fill_slot.to_expr()),
                                UIntImm::make(handle, 1));

            WriteLoc slot(pool, pool_type);
            slot.add_index_access(Var::make(handle, index));

            const Expr tag = UIntImm::make(layout.tag_type, layout.tag(vname));
            const Expr handle_value = BinOp::make(
                BinOp::OpType::BwOr,
                BinOp::make(BinOp::OpType::Shl, tag,
                            UIntImm::make(layout.tag_type,
                                          ADTLayout::tag_shift)),
                Var::make(handle, index));

            std::vector<Stmt> body;
            body.push_back(LetStmt::make(WriteLoc(index, handle), claim));
            body.push_back(Store::make(
                std::move(slot), Build::make(variant, std::move(values))));
            body.push_back(Return::make(handle_value));

            const std::string fname = adt_name + "_" + vname + "_new";
            appenders[fname] = std::make_shared<ir::Function>(
                fname, std::move(args), handle,
                Sequence::make(std::move(body)), ir::Function::InterfaceList{},
                std::vector<ir::Function::Attribute>{});
        }
    }
    return appenders;
}

// The storage a `TaggedIndex` layout names, declared so the program has it.
//
// Declared for every variant rather than only the built ones: reading a handle
// names the pool too, and a program that only ever matches on shapes it was
// handed still needs somewhere to read them from. The fill counters are the
// exception -- only an appender touches one -- but they are declared together
// so that a caller sees one pair per variant rather than having to work out
// which halves it owes.
void declare_pools(ir::Program &program, const LayoutMap &layouts) {
    for (const auto &[adt_name, layout] : layouts) {
        if (layout.kind != ir::AdtLayout::TaggedIndex) {
            continue;
        }
        for (const Type &variant : layout.variants) {
            const std::string &vname = variant.as<Struct_t>()->name;
            program.externs.push_back(
                TypedVar{layout.pool(vname), Array_t::make(variant, Expr())});
            program.externs.push_back(TypedVar{
                layout.fill(vname), Array_t::make(layout.storage, Expr())});
        }
    }
}

} // namespace

ir::Program LowerADTs::run(ir::Program program,
                           const CompilerOptions &options) const {
    // What the schedule asked for, if anything. Keyed per target, so the same
    // program can store a variant one way on a CPU and another on a device --
    // an index travels to one and a pointer does not.
    ir::AdtLayoutMap chosen;
    if (const auto it = program.schedules.find(ir::Target::Host);
        it != program.schedules.end()) {
        chosen = it->second.adt_layouts;
    }

    LayoutMap layouts;
    for (const auto &[name, type] : program.types) {
        const ADT_t *adt = type.as<ADT_t>();
        if (adt == nullptr) {
            continue;
        }
        const auto ask = chosen.find(adt->name);
        if (ask == chosen.end()) {
            layouts[adt->name] = default_adt_layout(*adt);
            continue;
        }
        // A boxed variant is allocated when it is constructed, so asking for
        // one and forbidding the heap are contradictory instructions. Said
        // here, where both are in view, rather than as a mallocs-are-forbidden
        // error from inside code generation with nothing to point at.
        if (options.no_heap && ask->second == ir::AdtLayout::TaggedPtr) {
            internal_error
                << "`layout " << adt->name
                << " = tagged_ptr` stores each variant behind a pointer, so "
                   "constructing one allocates it -- which --no-heap forbids. "
                   "Pick `tagged_index`, which indexes arrays the caller owns, "
                   "or drop --no-heap.";
        }
        layouts[adt->name] = adt_layout(*adt, ask->second);
    }
    if (layouts.empty()) {
        return program;
    }

    add_variant_constructors(program);
    declare_pools(program, layouts);

    const FuncMap appenders = make_appenders(layouts);
    RewriteADTs rewriter(layouts, appenders);

    for (auto &[name, type] : program.types) {
        type = rewriter.mutate(std::move(type));
    }
    // The types the layouts introduced. Added after the program's own have
    // been rewritten, so that rewriting does not walk over them.
    for (const auto &[name, layout] : layouts) {
        for (const Type &variant : layout.variants) {
            program.types[variant.as<Struct_t>()->name] = variant;
        }
        program.types[name] = layout.storage;
    }

    for (auto &[fname, func] : program.funcs) {
        std::vector<ir::Function::Argument> args(func->args.size());
        for (size_t i = 0; i < args.size(); i++) {
            const auto &arg = func->args[i];
            args[i] =
                ir::Function::Argument{arg.name, rewriter.mutate(arg.type),
                                       rewriter.mutate(arg.default_value),
                                       arg.mutating, arg.unaliased};
        }
        func = std::make_shared<ir::Function>(
            func->name, std::move(args), rewriter.mutate(func->ret_type),
            rewriter.mutate(func->body), func->interfaces, func->attributes);
    }

    for (auto &extern_var : program.externs) {
        extern_var.type = rewriter.mutate(std::move(extern_var.type));
    }

    // The appenders the rewrite above reached, added last so that rewriting
    // does not walk over bodies that are already written in terms of storage.
    for (const std::string &fname : rewriter.needed) {
        internal_assert(!program.funcs.contains(fname))
            << "Cannot name the appender " << fname
            << ": something is already called that.";
        program.funcs[fname] = appenders.at(fname);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
