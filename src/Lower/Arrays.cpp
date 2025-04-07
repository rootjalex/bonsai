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

            // Otherwise this is a lambda; we inline the lambda's body.
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
                for (size_t i = 0; i < n_args; ++i) {
                    // TODO(ajr): this needs to simplify or have CSE for it to
                    // be efficient!
                    ir::Expr value = ir::Extract::make(node->value, i);
                    internal_assert(
                        ir::equals(value.type(), lambda->args[i].type));
                    repls[lambda->args[i].name] = std::move(value);
                }
            }

            if (lambda->value.is<ir::SetOp>()) {
                // TODO: fuse the lowering
            } else {
                internal_assert(!ir::contains<ir::SetOp>(lambda->value))
                    << "[unimplemented] nested setop: " << lambda->value;
            }

            ir::Expr value = replace(repls, lambda->value);
            return ir::Yield::make(std::move(value));
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

// Lowers set operations over arrays to for-each loops.
struct LowerToForEach : public ir::Mutator {
    size_t icounter = 0;   // counter for iterator name.
    size_t tcounter = 0;   // For unique traverse function names.
    ir::FuncMap new_funcs; // Store newly created traversal functions here.

    // Resets state of this lowering phase.
    void reset() { new_funcs.clear(); }

    std::string unique_iter_name() {
        return "?it" + std::to_string(icounter++);
    }

    std::string unique_func_name() {
        return "?traverse_array" + std::to_string(tcounter++);
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
            internal_error << "[unimplemented] unknown traversal pattern: "
                           << expr;
        }
        switch (as_set->op) {
        case ir::SetOp::map:
            return build_map(build_traversal(as_set->b), as_set->a);
        case ir::SetOp::OpType::argmin:
        case ir::SetOp::OpType::filter:
        case ir::SetOp::OpType::product:
            internal_error << "[unimplemented] construction on an array: "
                           << expr;
        }
    }

    ir::Expr build_traversal_function(const ir::Expr &expr) {
        const std::string function_name = unique_func_name();
        const auto free_vars = ir::gather_free_vars(expr);
        ir::Stmt body = build_traversal(expr);
        internal_assert(body.defined())
            << "traversal building undefined for: " << expr;

        std::vector<ir::Function::Argument> func_args;
        std::transform(free_vars.cbegin(), free_vars.cend(),
                       std::back_inserter(func_args), [&](const auto &var) {
                           return ir::Function::Argument(var->name, var->type);
                       });

        ir::Type ret_type = expr.type();
        auto f = std::make_shared<ir::Function>(
            function_name, std::move(func_args), std::move(ret_type),
            std::move(body), ir::Function::InterfaceList{});
        ir::Type call_type = f->call_type();
        new_funcs[function_name] = std::move(f);

        std::vector<ir::Expr> call_args;
        std::transform(free_vars.begin(), free_vars.end(),
                       std::back_inserter(call_args),
                       [&](auto &var) -> ir::Expr { return var; });

        return ir::Call::make(
            ir::Var::make(std::move(call_type), function_name), call_args);
    }

    ir::Expr visit(const ir::SetOp *node) override {
        if (!node->b.type().is<ir::Array_t>()) {
            return ir::Mutator::visit(node);
        }
        switch (node->op) {
        case ir::SetOp::OpType::map:
            return build_traversal_function(node);
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
    int64_t icounter = 0; // unique identifier for index variable.
    int64_t lcounter = 0; // unique identifier for load variable.
    std::string unique_alloc_name() {
        return "?alloc" + std::to_string(acounter++);
    }
    std::string unique_index_name() {
        return "?index" + std::to_string(icounter++);
    }

    std::string unique_load_name() {
        return "?load" + std::to_string(lcounter++);
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

        // 1a. Replace ?iterN with array[?indexN] in the body of the for-each.
        ir::Expr index =
            ir::Var::make(ir::Index_t::make(), unique_index_name());
        std::string iter_name = node->name;
        ir::Expr extracted = ir::Extract::make(iter, index);

        // 1b. Create the for-all header.
        std::string load_name = unique_load_name();
        ir::WriteLoc header_loc(load_name, extracted.type());
        ir::Expr header_var =
            ir::Var::make(header_loc.base_type, header_loc.base);
        ir::Stmt header = ir::LetStmt::make(std::move(header_loc), extracted);

        std::map<std::string, ir::Expr> replacements = {
            {iter_name, header_var}};
        ir::Expr value = replace(replacements, body->value);

        // 2. Create the newly allocated memory.
        std::string allocation_name = unique_alloc_name();
        ir::Stmt allocation = ir::Allocate::make(allocation_name, type);

        // 3. Create the bounds and stride for the for-all loop.
        ir::ForAll::Slice slice{
            .begin = ir::IntImm::make(ir::Int_t::make(32), 0),
            .end = type->size,
            .stride = ir::IntImm::make(ir::Int_t::make(32), 1),
        };

        // 4. Finally, construct the for-all loop. with the respective store
        // into the newly allocated memory.
        ir::Stmt new_body = ir::Store::make(std::move(allocation_name), index,
                                            std::move(value));
        ir::Stmt forall =
            ir::ForAll::make(std::move(index), std::move(header),
                             std::move(slice), std::move(new_body));

        return ir::Sequence::make({
            allocation,
            forall,
        });
    }
};

} // namespace

ir::Program LowerArrays::run(ir::Program program) const {
    // 1. Lower set operations on arrays to for-each loops and yield operations.
    LowerToForEach convert_fe;

    // This needs to run until convergence in order to visit set operations that
    // are moved into the newly built traverse functions.
    for (auto &[_, f] : program.funcs) {
        f->body = convert_fe.mutate(f->body);
    }

    for (auto &[name, f] : convert_fe.new_funcs) {
        auto [_, inserted] = program.funcs.try_emplace(name, std::move(f));
        internal_assert(inserted)
            << "function with name: " << name << " already exists";
    }

    // 2. Lower for-each loops to concrete for-all loops.
    LowerToForAll convert_fa;
    for (auto &[_, f] : program.funcs) {
        f->body = convert_fa.mutate(f->body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
