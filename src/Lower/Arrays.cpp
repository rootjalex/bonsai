#include "Lower/Arrays.h"

#include "Lower/PredicateAnalysis.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

static size_t counter = 0;

std::string unique_iter_name() { return "?iter" + std::to_string(counter++); }

ir::Stmt build_map(ir::Stmt body, ir::Expr function) {
    struct RewriteMap : public ir::Mutator {
        ir::Expr function;

        RewriteMap(ir::Expr function) : function(std::move(function)) {}

        using ir::Mutator::visit;

        ir::Stmt visit(const ir::Yield *node) override {
            if (const auto *v = function.as<ir::Var>()) {
                // yield x -> yield f(x)
                return ir::Yield::make(ir::Call::make(v, {node->value}));
            }

            const ir::Lambda *lambda = function.as<ir::Lambda>();
            internal_assert(lambda) << "function is not a lambda: " << function;
            const size_t n_args = lambda->args.size();
            std::map<std::string, ir::Expr> repls;

            if (n_args == 1) {
                internal_assert(
                    ir::equals(lambda->args[0].type, node->value.type()))
                    << lambda->args[0].type << ", " << node->value.type();
                repls[lambda->args[0].name] = node->value;
            } else {
                internal_assert(node->value.type().is<ir::Tuple_t>());
                for (size_t i = 0; i < n_args; i++) {
                    // TODO: this needs to simplify or have CSE for it to be
                    // efficient!
                    ir::Expr value = ir::Extract::make(node->value, i);
                    internal_assert(
                        ir::equals(value.type(), lambda->args[i].type));
                    repls[lambda->args[i].name] = std::move(value);
                }
            }
            ir::Expr value = replace(repls, lambda->value);

            return ir::Yield::make(value);
        }

        ir::Stmt visit(const ir::Scan *node) override {
            internal_error << "unexpected: scan over arrays";
        }

        ir::Stmt visit(const ir::YieldFrom *node) override {
            internal_error << "unexpected: yield-from over arrays";
        }
    };

    return RewriteMap(std::move(function)).mutate(body);
}

ir::Stmt build_traversal(const ir::Expr &expr) {
    if (auto *var = expr.as<ir::Var>()) {
        std::string name = unique_iter_name();
        ir::Stmt body =
            ir::Yield::make(ir::Var::make(var->type.element_of(), name));
        return ir::ForEach::make(std::move(name), expr, std::move(body));
    }
    const ir::SetOp *as_set = expr.as<ir::SetOp>();
    if (as_set == nullptr) {
        internal_error << "[unimplemented] Unknown traversal pattern: " << expr;
    }
    switch (as_set->op) {
    case ir::SetOp::map:
        return build_map(build_traversal(as_set->b), as_set->a);
    default:
        internal_error << "[unimplemented] traversal construction: " << expr;
    }
}

// Lowers set operations over arrays to for-each loops.
struct LowerToForEach : public ir::Mutator {
    ir::FuncMap new_funcs;
    size_t counter = 0; // For unique traverse function names.

    std::string new_func_name() {
        return "?traverse" + std::to_string(counter++);
    }

    ir::Expr build_func(const ir::Expr &expr) {
        const std::string func = new_func_name();
        const auto free_vars = ir::gather_free_vars(expr);
        ir::Stmt body = build_traversal(expr);
        internal_assert(body.defined());

        std::vector<ir::Function::Argument> func_args;
        std::transform(free_vars.cbegin(), free_vars.cend(),
                       std::back_inserter(func_args), [&](const auto &var) {
                           return ir::Function::Argument(var->name, var->type);
                       });

        ir::Type ret_type = expr.type();
        auto f = std::make_shared<ir::Function>(
            func, std::move(func_args), std::move(ret_type), std::move(body),
            ir::Function::InterfaceList{});
        ir::Type call_type = f->call_type();
        new_funcs[func] = std::move(f);

        std::vector<ir::Expr> call_args;
        std::transform(free_vars.begin(), free_vars.end(),
                       std::back_inserter(call_args),
                       [&](auto &var) -> ir::Expr { return var; });

        return ir::Call::make(ir::Var::make(std::move(call_type), func),
                              call_args);
    }

    ir::Expr visit(const ir::SetOp *node) override {
        if (!node->b.type().is<ir::Array_t>()) {
            return node;
        }
        switch (node->op) {
        case ir::SetOp::OpType::map:
            return build_func(node);
        case ir::SetOp::OpType::argmin:
        case ir::SetOp::OpType::filter:
        case ir::SetOp::OpType::product:
            internal_error << "unimplemented: " << ir::Expr(node);
        }
    }
};

// Lowers for-each loops to for-all loops.
struct LowerToForAll : public ir::Mutator {
    int64_t acounter = 0; // unique identifier for allocations.
    int64_t icounter = 0; // unique identifier for iterator variable.
    std::string new_alloc_name() {
        return "?alloc" + std::to_string(acounter++);
    }
    std::string new_index_name() {
        return "?index" + std::to_string(icounter++);
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        ir::Expr iter = node->iter;
        const auto *type = iter.type().as<ir::Array_t>();
        if (type == nullptr) {
            return node;
        }
        // The only valid "yield"-like operation for arrays is ir::Yield.
        const auto *body = node->body.as<ir::Yield>();
        if (body == nullptr) {
            return node;
        }
        internal_assert(type->size.defined())
            << "for-all over an array requires a defined size, received: "
            << ir::Expr(iter) << " : " << ir::Type(type);
        ir::ForAll::Slice slice{
            .begin = ir::IntImm::make(ir::Int_t::make(32), 0),
            .end = type->size,
            .stride = ir::IntImm::make(ir::Int_t::make(32), 1),
        };

        // 1. Replace ?iterN with array[?indexN] in the body of the for-each.
        // TODO(cgyurgyik): Add index type.
        ir::Expr iterator =
            ir::Var::make(ir::Index_t::make(), new_index_name());
        std::string iter_name = node->name;
        ir::Expr extracted = ir::Extract::make(iter, iterator);
        std::map<std::string, ir::Expr> repls = {{iter_name, extracted}};
        ir::Expr value = replace(repls, body->value);
        // 2. Create the newly allocated memory.
        std::string name = new_alloc_name();
        ir::WriteLoc allocation(name, type);
        ir::Stmt allocate = ir::LetStmt::make(
            std::move(allocation), ir::Allocate::make(type->etype, type->size));
        // 3. Create the new for-all loop.
        ir::Stmt new_body = ir::Store::make(name, iterator, value);
        ir::Stmt forall =
            ir::ForAll::make(iterator, std::move(slice), std::move(new_body));

        return ir::Sequence::make({
            allocate,
            forall,
        });
    }
};

} // namespace

ir::Program LowerArrays::run(ir::Program program) const {
    // TODO(cgyurgyik): This is run until convergence so that nested ir::SetOp
    // nodes are also visited. There is probably a better way to do this?
    LowerToForEach convert_fe;
    int64_t before, after;
    do {
        before = program.funcs.size();
        for (auto &[_, f] : program.funcs) {
            f->body = convert_fe.mutate(f->body);
        }

        for (auto &[name, f] : convert_fe.new_funcs) {
            auto [_, inserted] = program.funcs.try_emplace(name, std::move(f));
            internal_assert(inserted)
                << "function with name: " << name << " already exists";
        }
        after = program.funcs.size();
        convert_fe.new_funcs.clear(); // reset
    } while (before != after);

    LowerToForAll convert_fa;
    for (auto &[_, f] : program.funcs) {
        f->body = convert_fa.mutate(f->body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
