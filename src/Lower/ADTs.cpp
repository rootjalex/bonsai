#include "Lower/ADTs.h"

#include "Lower/ADTLayout.h"

#include "IR/Expr.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <string>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

using LayoutMap = std::map<std::string, ADTLayout>;

// Rewrites variant types into what their layout says they are stored as.
struct RewriteADTs : public Mutator {
    const LayoutMap &layouts;

    RewriteADTs(const LayoutMap &layouts) : layouts(layouts) {}

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
    std::pair<WriteLoc, bool>
    mutate_writeloc(const WriteLoc &loc) override {
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

    // The payload of `value`, read as `variant`. The union's members are named
    // for their variants, so this is just naming one.
    Expr as_variant(const ADTLayout &layout, const Expr &value,
                    const std::string &variant) const {
        return Access::make(variant,
                            Access::make(layout.payload_field, value));
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
        Type type = mutate(node->type);
        if (type.same_as(node->type)) {
            internal_assert(not_changed)
                << "Lowering ADTs rewrote a Build value but not its type: "
                << Expr(node);
            return node;
        }
        return Build::make(std::move(type), std::move(values));
    }

    Expr visit(const Construct *node) override {
        std::vector<Expr> args;
        args.reserve(node->args.size());
        for (const Expr &arg : node->args) {
            args.push_back(mutate(arg));
        }
        const ADTLayout &layout = layout_of(node->type);

        std::vector<Expr> whole;
        whole.push_back(UIntImm::make(layout.tag_type,
                                      layout.tag(node->variant)));
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
                BinOp::OpType::Eq, Access::make(layout.tag_field, value),
                UIntImm::make(layout.tag_type, layout.tag(arm.variant)));
            result = IfElse::make(is_variant, std::move(arm_body),
                                  std::move(result));
        }
        return result;
    }
};

} // namespace

ir::Program LowerADTs::run(ir::Program program,
                           const CompilerOptions &options) const {
    LayoutMap layouts;
    for (const auto &[name, type] : program.types) {
        if (const ADT_t *adt = type.as<ADT_t>()) {
            layouts[adt->name] = default_adt_layout(*adt);
        }
    }
    if (layouts.empty()) {
        return program;
    }

    RewriteADTs rewriter(layouts);

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
            args[i] = ir::Function::Argument{
                arg.name, rewriter.mutate(arg.type),
                rewriter.mutate(arg.default_value), arg.mutating,
                arg.unaliased};
        }
        func = std::make_shared<ir::Function>(
            func->name, std::move(args), rewriter.mutate(func->ret_type),
            rewriter.mutate(func->body), func->interfaces, func->attributes);
    }

    for (auto &extern_var : program.externs) {
        extern_var.type = rewriter.mutate(std::move(extern_var.type));
    }

    return program;
}

} // namespace lower
} // namespace bonsai
