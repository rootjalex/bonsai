#include "Lower/Options.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"

#include "Error.h"
#include "Utils.h"

#include <variant>

namespace bonsai {
namespace lower {

namespace {

struct RewriteOptions : public ir::Mutator {
    ir::Type Bool = ir::Bool_t::make();
    std::map<ir::Type, ir::Type, ir::TypeLessThan> rewrite_map;
    // number of option etypes rewritten
    size_t counter = 0;

    std::string get_unique_option_name() {
        return "?option" + std::to_string(counter++);
    }

    ir::Type construct_option_struct(const ir::Type &etype) {
        std::string struct_name = get_unique_option_name();
        ir::Struct_t::Map fields;
        fields.emplace_back("value", etype);
        fields.emplace_back("set", Bool);
        return ir::Struct_t::make(struct_name, fields);
    }

    ir::Type mutate(const ir::Type &type) override {
        ir::Type rec = ir::Mutator::mutate(type);
        if (rec.is<ir::Option_t>()) {
            const ir::Type &etype = rec.as<ir::Option_t>()->etype;
            auto iter = rewrite_map.find(etype);
            if (iter != rewrite_map.end()) {
                return iter->second;
            }
            ir::Type repl = construct_option_struct(etype);
            rewrite_map[etype] = repl;
            return repl;
        } else {
            return rec;
        }
    }

    // Why are these necessary?? This is dumb.
    using ir::Mutator::mutate;
    // ir::Expr mutate(const ir::Expr &expr) override {
    //     return ir::Mutator::mutate(expr);
    // }
    // ir::Stmt mutate(const ir::Stmt &stmt) override {
    //     return ir::Mutator::mutate(stmt);
    // }

    ir::Expr visit(const ir::Build *node) override {
        ir::Expr expr = ir::Mutator::visit(node);
        node = expr.as<ir::Build>();
        internal_assert(node);
        if (node->type.is<ir::Option_t>()) {
            ir::Type new_type = mutate(node->type);
            if (node->values.empty()) {
                // Build an "empty" struct - this sets the bool `set` to false
                // by default.
                static const std::vector<ir::Expr> empty = {};
                return ir::Build::make(std::move(new_type), empty);
            } else {
                internal_assert(node->values.size() == 1)
                    << "Error in lowering build of Option_t, expected one "
                       "argument by "
                       "received: "
                    << node->values.size();
                std::vector<ir::Expr> args = {node->values[0],
                                              ir::BoolImm::make(true)};
                return ir::Build::make(std::move(new_type), std::move(args));
            }
        } else {
            return expr;
        }
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        ir::Type type = mutate(node->type);
        const ir::Type &vtype = value.type();
        const ir::Type &old_vtype = node->value.type();
        const ir::Type &old_type = node->type;

        if (const ir::Option_t *as_op = old_type.as<ir::Option_t>()) {
            const ir::Type &otype = as_op->etype;
            if (ir::equals(vtype, otype)) {
                // cast<option>(value) -> build<struct_option>(value)
                std::vector<ir::Expr> args = {std::move(value),
                                              ir::BoolImm::make(true)};
                return ir::Build::make(std::move(type), std::move(args));
            }
        }

        // cast<bool>(option) -> option.set
        if (type.is<ir::Bool_t>() && old_vtype.is<ir::Option_t>()) {
            return ir::Access::make("set", std::move(value));
        }

        // cast<T>(option) -> option.value
        if (const ir::Option_t *v_as_op = old_vtype.as<ir::Option_t>()) {
            ir::Expr deref = ir::Access::make("value", std::move(value));
            internal_assert(ir::equals(type, deref.type()))
                << "Lowering of option access: " << ir::Expr(node)
                << " lowered to deref: " << deref
                << " which is the wrong type: " << deref.type()
                << " instead of " << type;
            return deref;
        }

        internal_assert(!type.is<ir::Option_t>() && !vtype.is<ir::Option_t>());

        if (value.same_as(node->value) && type.same_as(node->type)) {
            return node;
        }
        return ir::Cast::make(std::move(type), std::move(value));
    }

    ir::Expr visit(const ir::Var *node) override {
        ir::Type type = mutate(node->type);
        if (type.same_as(node->type)) {
            return node;
        } else {
            return ir::Var::make(std::move(type), node->name);
        }
    }

    // Similar to mutate_writeloc in Mutator.cpp, but also mutates type.
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

    // These three need to mutate the types of the writelocs.
    ir::Stmt visit(const ir::LetStmt *node) override {
        auto [loc, not_changed] = mutate_writeloc(node->loc);
        ir::Expr value = mutate(node->value);
        if (not_changed && value.same_as(node->value)) {
            return node;
        }
        return ir::LetStmt::make(std::move(loc), std::move(value));
    }

    ir::Stmt visit(const ir::Assign *node) override {
        auto [loc, not_changed] = mutate_writeloc(node->loc);
        ir::Expr value = mutate(node->value);
        if (not_changed && value.same_as(node->value)) {
            return node;
        }
        return ir::Assign::make(std::move(loc), std::move(value),
                                node->mutating);
    }

    ir::Stmt visit(const ir::Accumulate *node) override {
        auto [loc, not_changed] = mutate_writeloc(node->loc);
        ir::Expr value = mutate(node->value);
        if (not_changed && value.same_as(node->value)) {
            return node;
        }
        return ir::Accumulate::make(std::move(loc), node->op, std::move(value));
    }

    // TODO: which other relevant nodes are there?
};

bool contains_option(const ir::Type &type) {
    struct ContainsOption : ir::Visitor {
        bool found = false;

        void visit(const ir::Option_t *) override { found = true; }
    };
    ContainsOption check;
    type.accept(&check);
    return check.found;
}

} // namespace

ir::Program LowerOptions::run(ir::Program program) const {
    RewriteOptions rewriter;
    for (auto &[t, type] : program.types) {
        type = rewriter.mutate(std::move(type));
    }

    for (const auto &[name, type] : program.externs) {
        internal_assert(!contains_option(type))
            << "Lowering failure, found option type in extern: " << name
            << " with type: " << type;
    }

    for (auto &[f, func] : program.funcs) {
        std::vector<ir::Function::Argument> args(func->args.size());
        for (size_t i = 0; i < args.size(); i++) {
            const auto &arg = func->args[i];
            args[i] = ir::Function::Argument{
                arg.name,
                rewriter.mutate(arg.type),
                rewriter.mutate(arg.default_value),
                arg.mutating,
            };
        }
        ir::Type ret_type = rewriter.mutate(func->ret_type);
        ir::Stmt body = rewriter.mutate(func->body);

        func = std::make_shared<ir::Function>(
            func->name, std::move(args), std::move(ret_type), std::move(body),
            func->interfaces, /*is_export=*/func->is_export);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
