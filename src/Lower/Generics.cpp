#include "Lower/Generics.h"

#include "IR/Equality.h"
#include "IR/Function.h"
#include "IR/Printer.h"

#include "Utils.h"

#include <set>
#include <string>

namespace bonsai {
namespace lower {

using namespace ir;

namespace {

std::pair<FuncMap, FuncMap> partition_generics(const FuncMap &funcs) {
    FuncMap generics, nongenerics;
    for (const auto &[name, func] : funcs) {
        if (func->interfaces.empty()) {
            nongenerics[name] = func;
        } else {
            generics[name] = func;
        }
    }
    return {generics, nongenerics};
}

bool contains_generics(const Type &type) {
    struct ContainsGenerics : Visitor {
        bool found = false;
        void visit(const Generic_t *) override { found = true; }
    };
    ContainsGenerics checker;
    type.accept(&checker);
    return checker.found;
}

std::string short_type_name(const Type &type) {
    switch (type.node_type()) {
    case IRTypeEnum::Int_t:
        return "i" + std::to_string(type.as<Int_t>()->bits);
    case IRTypeEnum::UInt_t:
        return "u" + std::to_string(type.as<UInt_t>()->bits);
    case IRTypeEnum::Float_t: {
        const auto *f = type.as<Float_t>();
        if (f->is_bfloat16()) {
            return "bf" + std::to_string(f->bits());
        }
        std::string name = "f";
        if (f->is_ieee754()) {
            name += std::to_string(f->bits());
            return name;
        }
        name += std::to_string(f->exponent);
        name += "_";
        name += std::to_string(f->mantissa);
        return name;
    }
    case IRTypeEnum::Bool_t:
        return "bool";
    case IRTypeEnum::Ptr_t:
        return "^" + short_type_name(type.as<Ptr_t>()->etype);
    case IRTypeEnum::Ref_t:
        return "&" + type.as<Ref_t>()->name;
    case IRTypeEnum::Vector_t:
        return short_type_name(type.as<Vector_t>()->etype) + "x" +
               std::to_string(type.as<Vector_t>()->lanes);
    case IRTypeEnum::Struct_t:
        return "#" + type.as<Struct_t>()->name;
    case IRTypeEnum::Tuple_t: {
        std::string name = "_";
        for (const Type &t : type.as<Tuple_t>()->etypes) {
            name += short_type_name(t) + "_";
        }
        return name;
    }
    case IRTypeEnum::Array_t: {
        // TODO: this is not unique over size?
        std::string name = "[]";
        name += short_type_name(type.as<Array_t>()->etype);
        return name;
    }
    case IRTypeEnum::Option_t:
        return "o?" + short_type_name(type.as<Ptr_t>()->etype);
    default: {
        internal_error << "No short_type_name for type: " << type;
    }
    }
}

std::string unique_generic_name(const std::string &name, const TypeMap &types) {
    // TypeMap is a std::map, so sorted on key.
    // This gives a unique ordering, and therefore,
    // a unique name.
    std::string generic_name = name + "$";
    bool first = true;
    for (const auto &[_, type] : types) {
        if (!first) {
            generic_name += "_";
        }
        first = false;
        generic_name += short_type_name(type);
    }
    return generic_name;
}

FuncMap handle_instantiations(const FuncMap &funcs) {
    struct FindInstantiations : Mutator {
        std::map<std::string, std::map<Type, TypeMap, TypeLessThan>> instants;
        bool updated = false;
        std::map<std::string, Expr> repls;

        const TypeMap *type_repls = nullptr;

        Expr visit(const Instantiate *node) override {
            if (!node->expr.is<Var>()) {
                internal_error << "TODO: analyze Instantiate of non-Var: "
                               << Expr(node);
            }

            internal_assert(node->type.defined());
            internal_assert(node->type.is<Function_t>());
            internal_assert(!contains_generics(node->type))
                << "TODO: handle nested generics: " << Expr(node);
            const std::string call_name = node->expr.as<Var>()->name;
            const std::string nongeneric_call_name =
                unique_generic_name(call_name, node->types);

            if (!instants[call_name].contains(node->type)) {
                // Not seen before.
                instants[call_name][node->type] = node->types;
                Expr call = Var::make(node->type, nongeneric_call_name);
                repls[nongeneric_call_name] = call;
                updated = true;
                return call;
            } else {
                // Seen before.
                internal_assert(repls.contains(nongeneric_call_name));
                return repls[nongeneric_call_name];
            }
        }

        Expr visit(const Var *node) override {
            if (!type_repls) {
                return Mutator::visit(node);
            }

            Type type = replace(*type_repls, node->type);
            if (type.same_as(node->type)) {
                return node;
            } else {
                return Var::make(std::move(type), node->name);
            }
        }
    };

    // TODO: does visit order matter?
    // there is probably an evaluation order needed for related-generics.
    // We probably need to build a dependent call graph of some kind.

    auto [generics, nongenerics] = partition_generics(funcs);

    if (generics.empty()) {
        return funcs; // Early out
    }

    FuncMap new_funcs;
    FindInstantiations f;

    // Always eval nongenerics first, they are roots of the call tree of
    // generics.
    for (const auto &[name, func] : nongenerics) {
        ir::Stmt body = f.mutate(func->body);
        new_funcs[name] = func->replace_body(std::move(body));
    }

    const auto copy_instants = f.instants;
    f.updated = false;

    for (const auto &[name, _map] : copy_instants) {
        internal_assert(generics.contains(name));
        const auto &func = generics[name];
        for (const auto &[type, _types] : _map) {
            const size_t n_args = func->args.size();
            std::vector<Function::Argument> args(n_args);
            for (size_t i = 0; i < n_args; i++) {
                ir::Type new_type = replace(_types, func->args[i].type);
                internal_assert(!contains_generics(new_type));
                const ir::Function::Argument &before = func->args[i];
                args[i] =
                    Function::Argument(before.name, std::move(new_type),
                                       before.default_value, before.mutating);
            }

            Type ret_type = replace(_types, func->ret_type);
            internal_assert(!contains_generics(ret_type));

            f.type_repls = &_types;
            Stmt body = f.mutate(func->body);
            internal_assert(!f.updated)
                << "TODO: implement recursive template filling for func: "
                << name;
            f.type_repls = nullptr;

            Function::InterfaceList interfaces = {};

            std::string new_name = unique_generic_name(name, _types);

            new_funcs[new_name] = std::make_shared<Function>(
                new_name, std::move(args), std::move(ret_type), std::move(body),
                std::move(interfaces), func->attributes);
        }
    }

    return new_funcs;
}

} // namespace

ir::FuncMap LowerGenerics::run(ir::FuncMap funcs,
                               const CompilerOptions &options) const {
    return handle_instantiations(funcs);
}

} // namespace lower
} // namespace bonsai
