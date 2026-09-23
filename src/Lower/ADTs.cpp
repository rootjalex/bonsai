#include "Lower/ADTs.h"

#include "Lower/ADTLayout.h"
#include "Lower/WordStorage.h"

#include "IR/Analysis.h"
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
    // Whose body is being rewritten, for the provenance a match's arms get
    // (see ir::Provenance): the function the arm was written in.
    std::string current_function;

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

    // What a value of an `Inline` type is stored as: the tag, padding up to
    // the payload's alignment, and the payload's words (see
    // Lower/WordStorage.h). Built from the members as *rewritten*, not as
    // the layout recorded them. Layouts are chosen from the program's types
    // before any of them change, so a variant that holds another variant
    // type still names it -- `Light`'s storage holds a `DiffuseArea` holding
    // a `Shape`, and a `tagged_index` Shape is a `u64` -- and it is the
    // rewritten member that has the size the payload has to hold.
    //
    // The payload starts at the largest alignment any member has, so that a
    // member's fields sit where C would put them, and at a word at least; it
    // is as long as the largest member rounded up to that. With a one-byte
    // tag that is Rust's repr(C) enum, less the alignment of the whole,
    // which is a word's rather than the payload's: an array of these steps
    // by the same bytes either way.
    //
    // This terminates because a variant type cannot contain itself: the
    // recursion is over the *contents* of the storage, and each step strips
    // one ADT away.
    std::map<std::string, Type> storages;
    Type storage_of(const std::string &adt_name, const ADTLayout &layout) {
        if (layout.kind != ir::AdtLayout::Inline) {
            return layout.storage;
        }
        if (const auto made = storages.find(adt_name); made != storages.end()) {
            return made->second;
        }
        uint64_t size = 0;
        uint64_t align = 1;
        for (const TypedVar &member : layout.members) {
            const Type stored = mutate(member.type);
            size = std::max(size, layout_bytes(stored));
            align = std::max(align, layout_align(stored));
        }
        const uint64_t at = std::max<uint64_t>(align, 4);
        const uint64_t words = (size + at - 1) / at * at / 4;
        Struct_t::Map fields;
        fields.push_back(TypedVar{layout.tag_field, layout.tag_type});
        const uint64_t tag_bytes = layout_bytes(layout.tag_type);
        internal_assert(tag_bytes <= at) << "a tag wider than the payload's "
                                         << "alignment: " << layout.tag_type;
        if (at > tag_bytes) {
            fields.push_back(TypedVar{
                layout.pad_field,
                Vector_t::make(UInt_t::make(8), uint32_t(at - tag_bytes),
                               /*packed=*/true)});
        }
        fields.push_back(TypedVar{layout.payload_field, words_type(words)});
        Type storage = Struct_t::make(adt_name, std::move(fields));
        storages[adt_name] = storage;
        return storage;
    }

    // A variant type becomes whatever it is stored as.
    Type mutate(const Type &type) override {
        Type rec = Mutator::mutate(type);
        if (const ADT_t *adt = rec.as<ADT_t>()) {
            return storage_of(adt->name, layout_of(rec));
        }
        return rec;
    }

    using Mutator::mutate;

    // The payload's words of an `Inline` value.
    Expr payload_of(const ADTLayout &layout, const Expr &value) const {
        return Access::make(layout.payload_field, value);
    }

    // The zero padding between an `Inline` value's tag and its payload, if
    // its storage has any.
    std::vector<Expr> tag_and_padding(const ADTLayout &layout,
                                      const Type &storage, uint64_t tag) const {
        std::vector<Expr> fields;
        fields.push_back(UIntImm::make(layout.tag_type, tag));
        const Struct_t *s = storage.as<Struct_t>();
        internal_assert(s) << "storage that is not a struct: " << storage;
        if (s->fields.size() == 3) {
            const Vector_t *pad = s->fields[1].type.as<Vector_t>();
            internal_assert(pad) << "padding that is not bytes: " << storage;
            fields.push_back(VecImm::make(
                s->fields[1].type,
                std::vector<Expr>(pad->lanes, UIntImm::make(pad->etype, 0))));
        }
        return fields;
    }

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

    // Where in its pool an arm stored by index keeps `value`'s fields: the
    // low bits of a handle, or the first word of the payload.
    Expr index_of(const ADTLayout &layout, const Expr &value,
                  const std::string &variant) const {
        (void)variant;
        if (layout.kind == ir::AdtLayout::Inline) {
            return read_words(payload_of(layout, value), 0, layout.index_type);
        }
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
                    const std::string &variant) {
        if (!layout.boxed(variant)) {
            // The arm's fields, read out of the payload's words from the
            // start (see Lower/WordStorage.h).
            return read_words(payload_of(layout, value), 0,
                              mutate(layout.variant(variant)));
        }
        // The pool the tag names, at the index the value holds. The pool is an
        // extern, so this Var is free here and the LowerExterns that runs
        // after this pass turns it into a parameter of whichever functions
        // reach it. Its element is the variant as *stored* -- the layout
        // recorded the variant as found, and a field of it that is itself a
        // variant type has since become that type's storage.
        const Type pool_type =
            Array_t::make(mutate(layout.variant(variant)), Expr());
        return Extract::make(Var::make(pool_type, layout.pool(variant)),
                             index_of(layout, value, variant));
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

        // What goes in the value for this arm: its fields, or where they were
        // put.
        Expr member;
        if (layout.boxed(node->variant)) {
            // Putting the fields in a pool and saying where is a store and a
            // counter rather than an expression, so it becomes a call to a
            // function that does both. Recorded so that only the variants a
            // program actually builds get one -- an unused appender would
            // still name its pool, and choosing this layout would then oblige
            // the caller to supply storage for variants it never constructs.
            const std::string &fname = appender_name(node->type, node->variant);
            needed.insert(fname);
            const auto found = appenders.find(fname);
            internal_assert(found != appenders.end())
                << "No appender prepared for " << fname;
            // The appender was prepared from the variant as found; its
            // parameters are what those fields have become (see where the
            // appenders are added to the program, at the end of the pass).
            member = Call::make(
                Var::make(mutate(found->second->call_type()), fname),
                std::move(args));
            if (layout.kind == ir::AdtLayout::TaggedIndex) {
                return member; // the appender answers the whole handle.
            }
        } else {
            // The layouts were chosen from the program's types before any of
            // them were rewritten, so a variant that holds *another* variant
            // type still names it. `Light::DiffuseArea` holds a `Shape`, and a
            // `Shape` under `tagged_index` is a `u64`; building the one from
            // the other without this is a struct whose field types disagree
            // with its values.
            member = Build::make(mutate(layout.variant(node->variant)),
                                 std::move(args));
        }

        // The tag, the padding, and the payload's words with the member
        // written into them from the start (see Lower/WordStorage.h).
        const ADT_t *adt = node->type.as<ADT_t>();
        internal_assert(adt) << "Not a variant type: " << node->type;
        const Type storage = storage_of(adt->name, layout);
        const Type member_type = layout.boxed(node->variant)
                                     ? layout.index_type
                                     : mutate(layout.variant(node->variant));
        const Type payload_type = storage.as<Struct_t>()->fields.back().type;
        const uint64_t n = payload_type.as<Vector_t>()->lanes;
        std::vector<Expr> words(n, UIntImm::make(UInt_t::make(32), 0));
        write_words(words, 0, member, member_type);
        std::vector<Expr> whole =
            tag_and_padding(layout, storage, layout.tag(node->variant));
        whole.push_back(Build::make(payload_type, std::move(words)));
        return Build::make(storage, std::move(whole));
    }

    // A value read as one of its variants: the fields of that variant, at
    // wherever the layout keeps them. Only ever inside the arm of a match that
    // established which variant it is (see ir::Unwrap).
    Expr visit(const Unwrap *node) override {
        const ADT_t *adt = node->value.type().as<ADT_t>();
        if (adt == nullptr) {
            return Mutator::visit(node); // a tree node's arm, not ours.
        }
        const ADTLayout &layout = layout_of(node->value.type());
        return as_variant(layout, mutate(node->value),
                          adt->variant_name(node->index));
    }

    // A match becomes a switch on the tag with an arm per variant, in tag
    // order rather than the order the arms were written in: a variant's tag
    // is its index (see ADTLayout), and arm k of a switch is the one taken on
    // k (see ir::SwitchStmt), which is what lets the switch become one
    // Dispatch in the SSA form rather than a chain of tests.
    //
    // The last arm is the switch's default, which needs no test:
    // MatchVariant::make has already checked that every variant is named
    // exactly once, so once the others are ruled out this is the only thing
    // left.
    Stmt visit(const MatchVariant *node) override {
        const ADT_t *adt = node->value.type().as<ADT_t>();
        internal_assert(adt) << "Match on a non-variant type: " << node->value;
        const ADTLayout &layout = layout_of(node->value.type());
        const Expr value = mutate(node->value);

        std::vector<Stmt> arms(node->arms.size());
        std::vector<Provenance> provenance(node->arms.size());
        for (const MatchVariant::Arm &arm : node->arms) {
            const auto index = adt->index_of(arm.variant);
            internal_assert(index.has_value())
                << adt->name << " has no variant " << arm.variant;
            const Struct_t::Map &fields = adt->fields(*index);

            // The names the arm gave the fields, bound to them. Reading a
            // field of the variant a value is not would read whatever those
            // bytes happen to be, which is why these are inside the arm. An
            // arm stored inline reads each field straight out of the
            // payload's words at its offset, rather than the whole struct
            // and a field of that.
            std::vector<Stmt> body;
            body.reserve(arm.bindings.size() + 1);
            if (!layout.boxed(arm.variant)) {
                const Type stored = mutate(layout.variant(arm.variant));
                const Struct_t *as_struct = stored.as<Struct_t>();
                internal_assert(as_struct && as_struct->fields.size() == fields.size())
                    << adt->name << "::" << arm.variant << " is stored as "
                    << stored;
                const Expr words = payload_of(layout, value);
                for (size_t f = 0; f < arm.bindings.size(); f++) {
                    body.push_back(LetStmt::make(
                        WriteLoc(arm.bindings[f], as_struct->fields[f].type),
                        read_words(words, layout_offset(*as_struct, f),
                                   as_struct->fields[f].type)));
                }
            } else {
                const Expr payload = as_variant(layout, value, arm.variant);
                for (size_t f = 0; f < arm.bindings.size(); f++) {
                    body.push_back(LetStmt::make(
                        WriteLoc(arm.bindings[f], mutate(fields[f].type)),
                        Access::make(fields[f].name, payload)));
                }
            }
            body.push_back(mutate(arm.body));

            const uint64_t tag = layout.tag(arm.variant);
            internal_assert(tag < arms.size() && !arms[tag].defined())
                << "The tag of " << adt->name << "::" << arm.variant << " is "
                << tag << ", not its index among " << arms.size()
                << " variants";
            arms[tag] = Sequence::make(std::move(body));
            provenance[tag] = Provenance::match_arm(current_function,
                                                    adt->name, arm.variant);
        }
        if (arms.size() == 1) {
            // The only variant there is: nothing to switch on.
            return arms[0];
        }
        // Each arm marked with the variant it takes, so that a schedule can
        // name it once the match is a switch, and then a block.
        return SwitchStmt::make(tag_of(layout, value), std::move(arms),
                                std::move(provenance));
    }
};

// A match that is a value becomes a call to a function whose body is the same
// match as a statement, each arm returning its expression.
//
//     x = match p { Geom(g) => a, Inst(m, blas) => b }
//
// is a branch, and a branch is a statement: the arms read fields of the
// variant the value is, so neither may be evaluated unless the tag says so,
// which rules out a `Select`. Rather than teach every kind of statement to
// have one hoisted out of its expressions, the branch is given a function of
// its own -- the value and whatever else the arms mention are its parameters
// -- and the match statement inside it is lowered by the same code as one the
// program wrote. The call is inlined by the backend, so what remains is the
// tag test and the arm, where the expression was.
//
// A set-typed match never gets here: it is part of a query, and LowerTrees
// opened it into the traversal (see ir::MatchExpr).
struct LiftMatchExprs : public Mutator {
    FuncMap lifted;
    size_t counter = 0;

    using Mutator::visit;

    Expr visit(const MatchExpr *node) override {
        internal_assert(!node->type.is<Set_t>())
            << "A set-valued match survived tree lowering: " << Expr(node);
        const Expr value = mutate(node->value);

        // The matched value is the function's first parameter, and the arms
        // are rewritten to read it by that name.
        const std::string name = "_match" + std::to_string(counter++);
        const std::string matched = "_matched";
        const Expr param = Var::make(value.type(), matched);
        std::map<Expr, Expr, ExprLessThan> as_param;
        as_param[value] = param;

        std::vector<Function::Argument> args;
        std::vector<Expr> call_args;
        std::set<std::string> seen;
        args.emplace_back(matched, value.type());
        call_args.push_back(value);
        seen.insert(matched);

        std::vector<MatchVariant::Arm> arms;
        arms.reserve(node->arms.size());
        for (const auto &arm : node->arms) {
            Expr body = replace(as_param, mutate(arm.value));
            // Everything else an arm reads comes in as a parameter, in the
            // order first seen, so the call site names it too.
            for (const TypedVar &var : gather_free_vars(body)) {
                if (seen.insert(var.name).second) {
                    args.emplace_back(var.name, var.type);
                    call_args.push_back(Var::make(var.type, var.name));
                }
            }
            arms.push_back(
                MatchVariant::Arm{arm.variant, {}, Return::make(body)});
        }

        auto func = std::make_shared<Function>(
            name, std::move(args), node->type,
            MatchVariant::make(param, std::move(arms)),
            Function::InterfaceList{}, std::vector<Function::Attribute>{});
        const Expr callee = Var::make(func->call_type(), name);
        lifted[name] = std::move(func);
        return Call::make(callee, std::move(call_args));
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
    // The counter is a `u64` whichever storage the value has; a pool can hold
    // more than a `u32` index in a payload names, and the count is what says
    // whether it has.
    const Type counter = UInt_t::make(64);
    for (const auto &[adt_name, layout] : layouts) {
        for (const Type &variant : layout.variants) {
            const Struct_t *fields_of = variant.as<Struct_t>();
            internal_assert(fields_of) << "A variant is a struct: " << variant;
            const std::string &vname = fields_of->name;
            if (!layout.boxed(vname)) {
                continue;
            }

            std::vector<ir::Function::Argument> args;
            std::vector<Expr> values;
            for (const TypedVar &field : fields_of->fields) {
                args.push_back(ir::Function::Argument{field.name, field.type,
                                                      Expr(), false, false});
                values.push_back(Var::make(field.type, field.name));
            }

            const Type pool_type = Array_t::make(variant, Expr());
            const Type fill_type = Array_t::make(counter, Expr());
            const std::string &pool = layout.pool(vname);
            const std::string &fill = layout.fill(vname);
            const std::string index = "_index";

            // Where in the pool this one goes. AtomicAdd returns the value
            // before the add, so this is the first free slot and the counter
            // now names the next one.
            WriteLoc fill_slot(fill, fill_type);
            fill_slot.add_index_access(UIntImm::make(counter, 0));
            const Expr claim =
                AtomicAdd::make(PtrTo::make(fill_slot.to_expr()),
                                UIntImm::make(counter, 1));

            WriteLoc slot(pool, pool_type);
            slot.add_index_access(Var::make(counter, index));

            // What the caller gets back: the whole handle, tag and index, when
            // the value is one; the index alone, which the payload's words
            // then hold, when the value is a tag beside a payload.
            Expr answer;
            if (layout.kind == ir::AdtLayout::TaggedIndex) {
                const Expr tag =
                    UIntImm::make(layout.tag_type, layout.tag(vname));
                answer = BinOp::make(
                    BinOp::OpType::BwOr,
                    BinOp::make(BinOp::OpType::Shl, tag,
                                UIntImm::make(layout.tag_type,
                                              ADTLayout::tag_shift)),
                    Var::make(counter, index));
            } else {
                answer = Cast::make(layout.index_type, Var::make(counter, index));
            }

            std::vector<Stmt> body;
            body.push_back(LetStmt::make(WriteLoc(index, counter), claim));
            body.push_back(Store::make(
                std::move(slot), Build::make(variant, std::move(values))));
            body.push_back(Return::make(std::move(answer)));

            const std::string fname = adt_name + "_" + vname + "_new";
            appenders[fname] = std::make_shared<ir::Function>(
                fname, std::move(args), layout.index_type,
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
        for (const Type &variant : layout.variants) {
            const std::string &vname = variant.as<Struct_t>()->name;
            if (!layout.boxed(vname)) {
                continue;
            }
            program.externs.push_back(
                TypedVar{layout.pool(vname), Array_t::make(variant, Expr())});
            program.externs.push_back(TypedVar{
                layout.fill(vname), Array_t::make(UInt_t::make(64), Expr())});
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
        // A variant behind a pointer is allocated when it is constructed, so
        // asking for one and forbidding the heap are contradictory
        // instructions. Said here, where both are in view, rather than as a
        // mallocs-are-forbidden error from inside code generation with nothing
        // to point at.
        for (const auto &[arm, kind] : ask->second) {
            if (options.no_heap && kind == ir::AdtLayout::TaggedPtr) {
                internal_error
                    << "`layout " << adt->name << "` stores " << arm
                    << " behind a pointer (`tagged_ptr`), so constructing one "
                       "allocates it -- which --no-heap forbids. Pick "
                       "`tagged_index`, which indexes arrays the caller owns, "
                       "or drop --no-heap.";
            }
        }
        layouts[adt->name] = adt_layout(*adt, ask->second);
    }
    if (layouts.empty()) {
        return program;
    }

    // What each variant type is stored as, for the passes that meet the
    // storage after this one (ir::Program::adt_storages).
    for (const auto &[name, layout] : layouts) {
        ir::Program::AdtStorage storage;
        for (const TypedVar &member : layout.members) {
            storage.variants.emplace_back(member.name, layout.tag(member.name));
        }
        storage.inline_storage = layout.kind == ir::AdtLayout::Inline;
        storage.tag_field = layout.tag_field;
        storage.pad_field = layout.pad_field;
        storage.payload_field = layout.payload_field;
        storage.tag_type = layout.tag_type;
        program.adt_storages[name] = std::move(storage);
    }

    // Matches that are values become functions first, so that the rewrite
    // below lowers the match statement inside each exactly as it lowers one
    // the program wrote.
    {
        LiftMatchExprs lift;
        for (auto &[name, func] : program.funcs) {
            func->body = lift.mutate(func->body);
        }
        for (auto &[name, func] : lift.lifted) {
            internal_assert(!program.funcs.contains(name))
                << "Cannot name the lifted match " << name
                << ": something is already called that.";
            program.funcs[name] = std::move(func);
        }
    }

    add_variant_constructors(program);
    declare_pools(program, layouts);

    const FuncMap appenders = make_appenders(layouts);
    RewriteADTs rewriter(layouts, appenders);

    for (auto &[name, type] : program.types) {
        type = rewriter.mutate(std::move(type));
    }
    // The types the layouts introduced. Added after the program's own have
    // been rewritten, so that rewriting does not walk over them -- but the
    // layouts themselves were built from the types as they were *found*, so a
    // variant naming another variant type still names it here. Rewriting each
    // as it is registered is what makes `Light::DiffuseArea` hold the `u64` a
    // `tagged_index` Shape actually is rather than the `Shape` it was.
    for (const auto &[name, layout] : layouts) {
        for (const Type &variant : layout.variants) {
            program.types[variant.as<Struct_t>()->name] =
                rewriter.mutate(variant);
        }
        program.types[name] = rewriter.storage_of(name, layout);
    }

    for (auto &[fname, func] : program.funcs) {
        rewriter.current_function = fname;
        std::vector<ir::Function::Argument> args(func->args.size());
        for (size_t i = 0; i < args.size(); i++) {
            const auto &arg = func->args[i];
            args[i] =
                ir::Function::Argument{arg.name, rewriter.mutate(arg.type),
                                       rewriter.mutate(arg.default_value),
                                       arg.mutating, arg.unaliased,
                                       arg.reducer};
        }
        func = std::make_shared<ir::Function>(
            func->name, std::move(args), rewriter.mutate(func->ret_type),
            rewriter.mutate(func->body), func->interfaces, func->attributes);
    }

    for (auto &extern_var : program.externs) {
        extern_var.type = rewriter.mutate(std::move(extern_var.type));
    }

    // The appenders the rewrite above reached, added last. They were built
    // from the variants as *found*, so a variant whose fields hold another
    // variant type -- `Geom(g : Geometric)`, where a Geometric holds a `Shape`
    // -- still names it, and the caller hands over the storage the Shape has
    // become. Rewriting them the same way as everything else makes the two
    // agree; on a handle or a payload that names no variant type it changes
    // nothing, which is why it is safe to do to every one of them.
    for (const std::string &fname : rewriter.needed) {
        internal_assert(!program.funcs.contains(fname))
            << "Cannot name the appender " << fname
            << ": something is already called that.";
        const std::shared_ptr<ir::Function> &appender = appenders.at(fname);
        std::vector<ir::Function::Argument> args(appender->args.size());
        for (size_t i = 0; i < args.size(); i++) {
            const auto &arg = appender->args[i];
            args[i] = ir::Function::Argument{
                arg.name, rewriter.mutate(arg.type),
                rewriter.mutate(arg.default_value), arg.mutating,
                arg.unaliased, arg.reducer};
        }
        program.funcs[fname] = std::make_shared<ir::Function>(
            appender->name, std::move(args),
            rewriter.mutate(appender->ret_type),
            rewriter.mutate(appender->body), appender->interfaces,
            appender->attributes);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
