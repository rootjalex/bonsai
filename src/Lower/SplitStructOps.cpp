#include "Lower/SplitStructOps.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

std::string get_split_name(const std::string &name, const std::string &field) {
    return "__f_" + name + "_" + field;
}

std::string field_concat(const std::string &f0, const std::string &f1) {
    return f0 + "_" + f1;
}

std::string get_split_func_name(const std::string &fname) {
    return "__split_" + fname;
}

void flatten_struct_type_helper(std::string field, Type type,
                                Struct_t::Map &fields,
                                Struct_t::DefMap &defaults) {
    if (const Struct_t *struct_t = type.as<Struct_t>()) {
        for (const auto &[k, v] : struct_t->fields) {
            internal_assert(
                !(struct_t->defaults.contains(k) && v.is<Struct_t>()))
                << "Default value for struct field: " << k;
            if (struct_t->defaults.contains(k)) {
                defaults[k] = struct_t->defaults.at(k);
                continue;
            }
            flatten_struct_type_helper(field_concat(field, k), v, fields,
                                       defaults);
        }
        return;
    }
    internal_assert(!contains<Struct_t>(type))
        << "TODO: weird struct nesting in type: " << type
        << " at field: " << field;
    fields.emplace_back(std::move(field), std::move(type));
}

Type flatten_struct_type(const Struct_t *struct_t) {
    Struct_t::Map fields;
    Struct_t::DefMap defaults;
    fields.reserve(struct_t->fields.size());
    for (const auto &[k, v] : struct_t->fields) {
        internal_assert(!(struct_t->defaults.contains(k) && v.is<Struct_t>()))
            << "Default value for struct field: " << k;
        if (struct_t->defaults.contains(k)) {
            defaults[k] = struct_t->defaults.at(k);
            continue;
        }
        flatten_struct_type_helper(k, v, fields, defaults);
    }
    return Struct_t::make(struct_t->name, std::move(fields));
}

Struct_t::Map flatten_struct_value(const Struct_t *struct_t) {
    Struct_t::Map fields;
    Struct_t::DefMap defaults;
    fields.reserve(struct_t->fields.size());
    for (const auto &[k, v] : struct_t->fields) {
        flatten_struct_type_helper(k, v, fields, defaults);
    }
    internal_assert(defaults.empty());
    return fields;
}

struct SplitStructOpsImpl : public Mutator {
    // The current fields of the struct being split.
    std::map<std::string, Expr> current_fields;

    // Allocations in this function, that have been rewritten.
    std::set<std::string> safe_write_locs;

    // This should always be true except for during returns
    // and writes to external allocations.
    bool split_var_reads = true;

    std::map<std::string, Expr> get_fields() {
        std::map<std::string, Expr> ret = std::move(current_fields);
        current_fields.clear();
        return ret;
    }

    Expr visit(const Var *node) override {
        const Struct_t *struct_t = node->type.as<Struct_t>();
        if (!struct_t) {
            return node;
        }
        current_fields.clear();
        auto fields = flatten_struct_value(struct_t);
        for (auto &[name, type] : fields) {
            current_fields[name] =
                Var::make(std::move(type), field_concat(node->name, name));
        }
        return Expr();
    }

    Expr visit(const Select *node) override {
        internal_assert(!node->type.is<Struct_t>())
            << "TODO: support Select on structs? " << Expr(node);
        return Mutator::visit(node);
    }

    Expr visit(const Cast *node) override {
        internal_assert(!node->type.is<Struct_t>() &&
                        !node->value.type().is<Struct_t>())
            << "TODO: support Cast on structs: " << Expr(node);
        return Mutator::visit(node);
    }

    Expr visit(const Build *node) override {
        const Struct_t *struct_t = node->type.as<Struct_t>();
        if (!struct_t) {
            return Mutator::visit(node);
        }
        internal_assert((struct_t->fields.size() == node->values.size()) ||
                        (node->values.empty()))
            << "TODO: handle default struct fields in struct build: "
            << Expr(node);

        std::vector<Expr> values(struct_t->fields.size());

        // For nestings.
        std::map<std::string, std::map<std::string, Expr>> nested;

        if (node->values.empty()) {
            Type flat_type = flatten_struct_type(struct_t);
            const Struct_t *flat_struct_t = flat_type.as<Struct_t>();
            internal_assert(flat_struct_t);
            current_fields.clear();
            for (const auto &[field, type] : flat_struct_t->fields) {
                current_fields[field] = make_zero(type);
            }
            return Expr();
        }

        for (size_t i = 0; i < node->values.size(); i++) {
            values[i] = mutate(node->values[i]);
            if (!values[i].defined()) {
                auto nested_fields = get_fields();
                internal_assert(!nested_fields.empty())
                    << "Struct lowering broke at index: " << i << " "
                    << node->values[i] << " rewrote to nothing";
                nested[struct_t->fields[i].name] = std::move(nested_fields);
            }
        }

        for (size_t i = 0; i < values.size(); i++) {
            if (values[i].defined()) {
                current_fields[struct_t->fields[i].name] = values[i];
            } else {
                auto iter = nested.find(struct_t->fields[i].name);
                internal_assert(iter != nested.cend())
                    << "Struct lowering broke at index: " << i << " "
                    << node->values[i] << " rewrote to nothing";
                for (const auto &[key, value] : iter->second) {
                    current_fields[field_concat(struct_t->fields[i].name,
                                                key)] = value;
                }
            }
        }
        return Expr();
    }

    Expr visit(const Access *node) override {
        Expr expr = mutate(node->value);
        internal_assert(!expr.defined())
            << "Struct lowering did not expect expression: " << expr
            << " during lowering of " << Expr(node);
        auto fields = get_fields();
        const auto iter = fields.find(node->field);
        internal_assert(iter != fields.cend())
            << "Failed to find: " << node->field
            << " when lowering: " << Expr(node);
        return iter->second;
    }

    Expr visit(const Call *node) override {
        std::vector<Expr> args(node->args.size());
        bool any_changed = false;
        // TODO(ajr): support struct arguments.
        for (size_t i = 0; i < node->args.size(); i++) {
            internal_assert(!contains<Struct_t>(node->args[i].type()))
                << "TODO(ajr): support struct arguments: " << Expr(node);
            args[i] = mutate(node->args[i]);
            internal_assert(args[i].defined())
                << node->args[i] << " got destroyed";
            any_changed = any_changed || !args[i].same_as(node->args[i]);
        }
        if (!any_changed) {
            return node;
        }
        return Call::make(node->func, std::move(args));
    }

    // Don't support (for now?).
    RESTRICT_MUTATOR(Expr, Ramp);
    RESTRICT_MUTATOR(Expr, Unwrap);
    RESTRICT_MUTATOR(Expr, Generator);
    RESTRICT_MUTATOR(Expr, Lambda);
    RESTRICT_MUTATOR(Expr, GeomOp);
    RESTRICT_MUTATOR(Expr, SetOp);
    RESTRICT_MUTATOR(Expr, Instantiate);
    RESTRICT_MUTATOR(Expr, PtrTo);
    RESTRICT_MUTATOR(Expr, Deref);

    // Stmts
    // TODO(ajr)
    RESTRICT_MUTATOR(Stmt, CallStmt);
    RESTRICT_MUTATOR(Stmt, Print);

    Stmt visit(const Return *node) override {
        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        if (!struct_t) {
            return Mutator::visit(node);
        }
        Expr value = mutate(node->value);
        internal_assert(!value.defined())
            << "Struct lowering did not expect expression: " << value
            << " during lowering of " << Stmt(node);
        auto fields = get_fields();
        Type type = flatten_struct_type(struct_t);
        return Return::make(Build::make(std::move(type), std::move(fields)));
    }

    Stmt visit(const LetStmt *node) override {
        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        if (!struct_t) {
            internal_assert(!contains<Struct_t>(node->value.type()))
                << "let " << node->loc << " = " << node->value;
            return Mutator::visit(node);
        }

        Expr value = mutate(node->value);
        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        auto fields = get_fields();
        internal_assert(!fields.empty())
            << node->value << " split into nothing";
        // A LetStmt needs to be broken into fields x LetStmts
        std::vector<Stmt> lets;
        lets.reserve(fields.size());
        for (auto &[field, expr] : fields) {
            WriteLoc loc(get_split_name(node->loc.base, field), expr.type());
            lets.push_back(LetStmt::make(std::move(loc), std::move(expr)));
        }
        return Sequence::make(std::move(lets));
    }

    Stmt visit(const Sequence *node) override {
        bool changed = false;
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());

        auto flatten = [&](const Stmt &stmt) {
            Stmt mut = mutate(stmt);
            changed = changed || !mut.same_as(stmt);
            internal_assert(mut.defined()) << stmt;
            if (const ir::Sequence *seq = mut.as<ir::Sequence>()) {
                stmts.insert(stmts.end(), seq->stmts.begin(), seq->stmts.end());
                changed = true;
            } else {
                stmts.emplace_back(std::move(mut));
            }
        };

        for (const auto &stmt : node->stmts) {
            flatten(stmt);
        }

        if (!changed) {
            return node;
        }
        internal_assert(!stmts.empty());
        return ir::Sequence::make(std::move(stmts));
    }

    Stmt visit(const Allocate *node) override {
        if (!node->value.defined()) {
            internal_assert(!contains<Struct_t>(node->loc.base_type))
                << "alloc " << node->loc.base << " : " << node->loc.base_type;
            return Mutator::visit(node);
        }

        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        if (!struct_t) {
            internal_assert(!contains<Struct_t>(node->value.type()))
                << "alloc " << node->loc << " = " << node->value;
            return Mutator::visit(node);
        }

        internal_assert(node->loc.accesses.empty()) << "TODO: " << Stmt(node);

        Expr value = mutate(node->value);
        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        auto fields = get_fields();
        internal_assert(!fields.empty())
            << node->value << " split into nothing";
        // An Allocate needs to be broken into fields x Allocates
        std::vector<Stmt> allocs;
        allocs.reserve(fields.size());
        for (auto &[field, expr] : fields) {
            WriteLoc loc(get_split_name(node->loc.base, field), expr.type());
            allocs.push_back(
                Allocate::make(std::move(loc), std::move(expr), node->memory));
        }

        return Sequence::make(std::move(allocs));
    }

    Stmt visit(const Store *node) override {
        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        if (!struct_t) {
            internal_assert(!contains<Struct_t>(node->value.type()))
                << node->loc << " = " << node->value;
            return Mutator::visit(node);
        }

        internal_assert(node->loc.accesses.empty()) << "TODO: " << Stmt(node);
        internal_assert(!node->mask.defined()) << "TODO: " << Stmt(node);

        Expr value = mutate(node->value);
        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        auto fields = get_fields();
        internal_assert(!fields.empty())
            << node->value << " split into nothing";
        // An Allocate needs to be broken into fields x Allocates
        std::vector<Stmt> stores;
        stores.reserve(fields.size());
        for (auto &[field, expr] : fields) {
            WriteLoc loc(get_split_name(node->loc.base, field), expr.type());
            stores.push_back(
                Store::make(std::move(loc), std::move(expr), /*mask=*/Expr()));
        }

        return Sequence::make(std::move(stores));
    }

    Stmt visit(const Accumulate *node) override {
        // e.g. vec.x *= 4; (should be canonicalized to vec[0] *= 4).
        internal_assert(node->loc.accesses.empty())
            << "TODO: split with accesses: " << Stmt(node);
        internal_assert(node->op != Accumulate::Argmin &&
                        node->op != Accumulate::Argmax)
            << "TODO: split with accesses: " << Stmt(node);

        Expr value = mutate(node->value);
        if (value.same_as(node->value)) {
            internal_assert(!contains<Struct_t>(value.type()))
                << value << " has type " << value.type();
            return node;
        } else if (value.defined() && !value.type().is_vector()) {
            return Accumulate::make(node->loc, node->op, std::move(value));
        }

        internal_error << "Accumulate on struct_t: " << Stmt(node);
    }

    // Don't support (for now?)
    RESTRICT_MUTATOR(Stmt, Label);
    RESTRICT_MUTATOR(Stmt, RecLoop);
    RESTRICT_MUTATOR(Stmt, Match);
    RESTRICT_MUTATOR(Stmt, Yield);
    RESTRICT_MUTATOR(Stmt, Scan);
    RESTRICT_MUTATOR(Stmt, YieldFrom);
    RESTRICT_MUTATOR(Stmt, ForAll);
    RESTRICT_MUTATOR(Stmt, ForEach);
    RESTRICT_MUTATOR(Stmt, Continue);
    RESTRICT_MUTATOR(Stmt, Launch);
};

} // namespace

std::shared_ptr<Function> split_struct_ops(const Function &func) {
    SplitStructOpsImpl lowerer;
    Stmt body = lowerer.mutate(func.body);
    std::vector<Function::Argument> args;

    for (const auto &arg : func.args) {
        const Struct_t *struct_t = arg.type.as<Struct_t>();
        if (!struct_t) {
            internal_assert(!contains<Struct_t>(arg.type))
                << arg.name << " : " << arg.type;
            args.push_back(arg);
            continue;
        }
        internal_assert(!arg.default_value.defined());
        auto flat = flatten_struct_value(struct_t);
        for (auto &[field, type] : flat) {
            args.emplace_back(field_concat(arg.name, field), std::move(type),
                              /*default_value=*/Expr(), arg.mutating);
        }
    }

    // TODO(ajr): could get rid of structs entirely with RtoP!!
    Type ret_type;
    if (const Struct_t *struct_t = func.ret_type.as<Struct_t>()) {
        ret_type = flatten_struct_type(struct_t);
    } else {
        internal_assert(!contains<Struct_t>(func.ret_type))
            << " ret_type: " << func.ret_type;
        ret_type = func.ret_type;
    }

    return std::make_shared<Function>(
        get_split_func_name(func.name), std::move(args), std::move(ret_type),
        std::move(body), func.interfaces, func.attributes);
}

} // namespace lower
} // namespace bonsai
