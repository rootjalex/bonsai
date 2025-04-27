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

struct ReturnsToYields : public ir::Mutator {
    ir::Stmt visit(const ir::Return *node) override {
        return ir::Sequence::make(
            {ir::Yield::make(node->value), ir::Continue::make()});
    }
};

struct RewriteYields : public ir::Mutator {
    std::function<ir::Stmt(ir::Expr)> rewriter;

    RewriteYields(std::function<ir::Stmt(ir::Expr)> rewriter)
        : rewriter(std::move(rewriter)) {}

    ir::Stmt visit(const ir::Yield *node) override {
        ir::Stmt repl = rewriter(node->value);
        internal_assert(repl.defined())
            << "RewriteYields produced empty stmt: " << ir::Stmt(node);
        return repl;
    }
};

struct RmUnnecessaryContinues : public ir::Mutator {
    ir::Stmt rm_continues(const ir::Stmt &stmt) const {
        if (stmt.is<ir::Continue>()) {
            return ir::Stmt();
        } else if (const ir::Sequence *seq = stmt.as<ir::Sequence>()) {
            internal_assert(seq->stmts.size() != 0);
            ir::Stmt last = seq->stmts.back();
            ir::Stmt rec = rm_continues(last);
            if (rec.same_as(last)) {
                return stmt;
            }

            // Removed some `Continue`.
            std::vector<ir::Stmt> stmts = seq->stmts;
            if (rec.defined()) {
                stmts.back() = std::move(rec);
            } else {
                stmts.pop_back();
            }
            // Return new sequence
            if (stmts.size() == 0) {
                return ir::Stmt();
            } else if (stmts.size() == 1) {
                return stmts.back();
            }
            return ir::Sequence::make(std::move(stmts));
        } else if (const ir::IfElse *ifelse = stmt.as<ir::IfElse>()) {
            ir::Stmt then_body = rm_continues(ifelse->then_body);
            ir::Stmt else_body = ifelse->else_body.defined()
                                     ? rm_continues(ifelse->else_body)
                                     : ifelse->else_body;
            if (then_body.same_as(ifelse->then_body) &&
                else_body.same_as(ifelse->else_body)) {
                return stmt;
            }
            internal_assert(then_body.defined());
            return ir::IfElse::make(ifelse->cond, std::move(then_body),
                                    std::move(else_body));
        } else if (const ir::Label *label = stmt.as<ir::Label>()) {
            ir::Stmt body = rm_continues(label->body);
            if (body.same_as(label->body)) {
                return stmt;
            }
            return ir::Label::make(label->name, std::move(body));
        }
        // Don't know how to remove a `Continue`
        return stmt;
    }

    ir::Stmt visit(const ir::ForAll *node) override {
        // First recurse on body.
        ir::Stmt stmt = ir::Mutator::visit(node);
        node = stmt.as<ir::ForAll>();
        internal_assert(node);

        ir::Stmt body = rm_continues(node->body);
        internal_assert(body.defined());
        if (!body.same_as(node->body)) {
            return ir::ForAll::make(node->index, node->header, node->slice,
                                    std::move(body));
        }
        // No `Continue` to remove.
        return stmt;
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        // First recurse on body.
        ir::Stmt stmt = ir::Mutator::visit(node);
        node = stmt.as<ir::ForEach>();
        internal_assert(node);

        ir::Stmt body = rm_continues(node->body);
        internal_assert(body.defined());
        if (!body.same_as(node->body)) {
            return ir::ForEach::make(node->name, node->iter, std::move(body));
        }
        // No `Continue` to remove.
        return stmt;
    }
};

// Applies the lambda or function call to the top-level yield operation.
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
            // Otherwise, this is a lambda. We always inline the lambda's body.
            const ir::Lambda *lambda = function.as<ir::Lambda>();
            internal_assert(lambda) << "function is not a lambda: " << function;
            const size_t n_args = lambda->args.size();
            std::map<std::string, ir::Expr> repls;

            // Replace the name of the lambda argument to match that of the
            // yield operation. This is necessary in cases where multiple set
            // operations are fused together.
            if (n_args == 1) {
                internal_assert(
                    ir::equals(lambda->args[0].type, node->value.type()))
                    << lambda->args[0].name << ": " << lambda->args[0].type
                    << ", " << node->value << ": " << node->value.type();
                repls[lambda->args[0].name] = node->value;
            } else {
                internal_assert(node->value.type().is<ir::Tuple_t>());
                for (size_t i = 0; i < n_args; ++i) {
                    // TODO(ajr): this needs to simplify or have CSE for it to
                    // be efficient!
                    ir::Expr value = ir::Extract::make(node->value, i);
                    internal_assert(
                        ir::equals(lambda->args[i].type, value.type()))
                        << lambda->args[i].name << ": " << lambda->args[i].type
                        << ", " << node->value << ": " << node->value.type();
                    repls[lambda->args[i].name] = std::move(value);
                }
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
    // The functions in this program at the time of lowering.
    const ir::FuncMap &program_functions;
    size_t counter = 0; // For unique traverse function names.
    std::string unique_func_name() {
        return "_traverse_array" + std::to_string(counter++);
    }

    LowerToForEach(const ir::FuncMap &program_functions)
        : program_functions(program_functions) {};
    ir::FuncMap new_funcs; // Store newly created traversal functions here.

    // Returns the name of the argument(s) so that an iterator variable name can
    // be appropriately mapped to its function/lambda argument name.
    std::string get_argument_name(const ir::Expr &expr) {
        // Case 1: lambda
        if (const auto *lambda = expr.as<ir::Lambda>()) {
            internal_assert(lambda->args.size() == 1)
                << "TODO(cgyurgyik): relax this for tuples";
            return lambda->args[0].name;
        }
        // Case 2: function
        if (const auto *v = expr.as<ir::Var>()) {
            auto it = program_functions.find(v->name);
            internal_assert(it != program_functions.end());
            const auto &f = it->second;
            internal_assert(f->args.size() == 1)
                << "TODO(cgyurgyik): relax this for tuples";
            return f->args[0].name;
        }
        internal_error << "expected callable, received: " << expr << " : "
                       << expr.type();
    }

    ir::Stmt build_hierarchy(const ir::Expr &expr) {
        const ir::SetOp *as_set = expr.as<ir::SetOp>();
        ir::Expr function = as_set->a, body = as_set->b;
        // If the lambda of this set operation is also a set operation, build
        // the hierarchical loop to iterate over the next dimension.
        if (const auto *lambda = function.as<ir::Lambda>()) {
            if (ir::Expr value = lambda->value; value.is<ir::SetOp>()) {
                return ir::ForEach::make(
                    /*name=*/get_argument_name(lambda),
                    /*iter=*/body,
                    /*body=*/build_hierarchy(value));
            } else {
                // Currently, we only support the cases where (1) the lambda
                // body is just another isolated set operation, or (2) the
                // lambda body has no set operations at all.
                internal_assert(!ir::contains<ir::SetOp>(value))
                    << "[unimplemented] non-insulated set operations: "
                    << value;
            }
        } else if (const auto *v = function.as<ir::Var>()) {
            // TODO(cgyurgyik): Not sure how often this will occur, but we
            // should probably support this.
            if (program_functions.contains(v->name)) {
                // TODO: this could recursively inline. We do not handle that
                // yet.
                const auto &func = program_functions.at(v->name);
                ir::Stmt func_body = func->body;
                internal_assert(!ir::contains<ir::SetOp>(func_body))
                    << "[unimplemented] map with nested setop in " << v->name
                    << " from " << expr;
                ir::Stmt for_body = ReturnsToYields().mutate(func_body);
                return ir::ForEach::make(
                    /*name=*/get_argument_name(function),
                    /*iter=*/body,
                    /*body=*/std::move(for_body));
            }
            internal_assert(!v->type.is_func())
                << "[unimplemented] non-inlined lambda function while building "
                   "hierarchical loops";
        }
        // Otherwise, fuse set operations in this level.
        return build_level(expr);
    }

    ir::Stmt build_level(const ir::Expr &expr,
                         std::optional<std::string> iterator_name = {}) {
        if (auto *var = expr.as<ir::Var>()) {
            internal_assert(iterator_name.has_value());
            ir::Stmt body = ir::Yield::make(
                ir::Var::make(var->type.element_of(), *iterator_name));
            return ir::ForEach::make(*iterator_name, expr, std::move(body));
        }
        const ir::SetOp *as_set = expr.as<ir::SetOp>();
        if (as_set == nullptr) {
            internal_error << "unknown traversal pattern: " << expr;
        }
        switch (as_set->op) {
        case ir::SetOp::map: {
            ir::Expr function = as_set->a;
            // Note that when multiple set operations are fused, it will just
            // take the last set operation's lambda argument name.
            ir::Stmt body = build_level(as_set->b, get_argument_name(function));
            return build_map(std::move(body), std::move(function));
        }
        case ir::SetOp::OpType::argmin:
        case ir::SetOp::OpType::filter:
        case ir::SetOp::OpType::product:
            internal_error << "[unimplemented] construction on an array: "
                           << expr;
        }
    }

    ir::Expr build_traversal_function(const ir::Expr &expr) {
        const std::string function_name = unique_func_name();
        std::vector<ir::TypedVar> free_vars = ir::gather_free_vars(expr);
        ir::Stmt body = build_hierarchy(expr);
        internal_assert(body.defined())
            << "traversal building undefined for: " << expr;

        std::vector<ir::Function::Argument> func_args;
        std::transform(free_vars.cbegin(), free_vars.cend(),
                       std::back_inserter(func_args), [&](const auto &var) {
                           return ir::Function::Argument(var.name, var.type);
                       });

        auto f = std::make_shared<ir::Function>(
            function_name, std::move(func_args), expr.type(), std::move(body),
            ir::Function::InterfaceList{},
            std::vector<ir::Function::Attribute>{});
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
    int64_t lcounter = 0; // unique identifier for load variables.
    std::string unique_alloc_name() {
        return "_alloc" + std::to_string(acounter++);
    }
    std::string unique_load_name() {
        return "_load" + std::to_string(lcounter++);
    }
    // A list of for-each constructs that have been visited already.
    std::set<ir::Stmt> visited;

    // Builds the for-each with the `toplevel` for-each as the base.
    ir::Stmt build(ir::Stmt node, std::vector<ir::ForAll::Slice> &dimensions,
                   std::vector<std::string> &iterator_names,
                   ir::Stmt toplevel) {
        const auto *foreach = node.as<ir::ForEach>();
        internal_assert(foreach) << node;

        ir::Expr iterable = foreach->iter;
        const auto *array_type = iterable.type().as<ir::Array_t>();
        if (array_type == nullptr) {
            return node;
        }

        ir::Type size_type = array_type->size.type();
        internal_assert(size_type.defined())
            << "for-all over an array requires a defined size, received: "
            << iterable << " : " << array_type;
        dimensions.push_back(ir::ForAll::Slice{
            .begin = make_const(size_type, 0),
            .end = array_type->size,
            .stride = make_const(size_type, 1),
        });
        iterator_names.push_back(foreach->name);

        if (foreach->body.is<ir::ForEach>()) {
            return build(foreach->body, dimensions, iterator_names, toplevel);
        }

        std::vector<std::string> reversed_iterator_names;
        std::copy(std::rbegin(iterator_names), std::rend(iterator_names),
                  std::back_inserter(reversed_iterator_names));

        // Create the header with the proper load index.
        ir::Expr toplevel_iterable = toplevel.as<ir::ForEach>()->iter;
        ir::Expr index =
            ir::Var::make(ir::Index_t::make(), iterator_names.front());
        ir::Expr extracted = ir::Extract::make(toplevel_iterable, index);
        for (int j = 1; j < iterator_names.size(); ++j) {
            index = ir::Var::make(ir::Index_t::make(), iterator_names[j]);
            extracted = ir::Extract::make(extracted, index);
        }
        std::string load_name = unique_load_name();
        ir::WriteLoc header_loc(load_name, extracted.type());
        ir::Expr header_variable =
            ir::Var::make(header_loc.base_type, header_loc.base);
        ir::Stmt header = ir::LetStmt::make(header_loc, extracted);

        // Replace the uses of the iterator with a concrete loaded value.
        std::map<std::string, ir::Expr> replacements = {
            {iterator_names.back(), header_variable}};

        // Create the allocation. The type is currently just inferred from the
        // yielded value.
        ir::Type iter_type = toplevel_iterable.type();

        std::vector<ir::Expr> indices;
        ir::Type index_type = iter_type.as<ir::Array_t>()->size.type();
        std::transform(iterator_names.begin(), iterator_names.end(),
                       std::back_inserter(indices),
                       [&](const std::string &name) {
                           return ir::Var::make(index_type, name);
                       });
        ir::Expr store_index;
        // TODO: fix the lowering for multidimensional indices.
        if (indices.size() == 1) {
            store_index = indices[0];
        } else {
            store_index = ir::Build::make(
                ir::Vector_t::make(index_type, dimensions.size()), indices);
        }

        std::string allocation_name = unique_alloc_name();

        ir::Type yielded_type;
        auto visitor = [&](ir::Expr yielded) {
            if (!yielded_type.defined()) {
                yielded_type = yielded.type();
            } else {
                internal_assert(ir::equals(yielded_type, yielded.type()))
                    << "Mismatch yield types in lowering: " << node;
            }
            return ir::Store::make(allocation_name, /*index=*/store_index,
                                   std::move(yielded));
        };

        ir::Stmt repl_body = replace(replacements, foreach->body);

        ir::Stmt final_body =
            RewriteYields(visitor).mutate(std::move(repl_body));
        internal_assert(yielded_type.defined())
            << "No yields found in: " << foreach->body;

        ir::Type yield_type = iter_type.with_etype(yielded_type);

        for (int i = 0; i < dimensions.size(); ++i) {
            final_body = ir::ForAll::make(
                /*index=*/reversed_iterator_names[i],
                /*header=*/i == 0 ? header : ir::Stmt(),
                /*slice=*/dimensions[i],
                /*body=*/final_body);
        }

        final_body = RmUnnecessaryContinues().mutate(std::move(final_body));

        ir::Stmt allocation = ir::Allocate::make(allocation_name, yield_type);

        ir::Stmt ret =
            ir::Return::make(ir::Var::make(yield_type, allocation_name));
        return ir::Sequence::make({
            std::move(allocation),
            std::move(final_body),
            std::move(ret),
        });
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        // We only want to visit the top-level for-each.
        auto [_, inserted] = visited.insert(node);
        if (!inserted) {
            return node;
        }

        ir::Expr iter = node->iter;
        // Currently, this assumes we are only lowering arrays.
        const auto *type = iter.type().as<ir::Array_t>();
        if (type == nullptr) {
            return node;
        }

        internal_assert(type->size.defined())
            << "for-all over an array requires a defined size, received: "
            << ir::Type(type);

        std::vector<ir::ForAll::Slice> dimensions;
        std::vector<std::string> iterator_names;
        return build(node, dimensions, iterator_names, /*toplevel=*/node);
    }
};

} // namespace

ir::Program LowerArrays::run(ir::Program program) const {
    // 1. Lower set operations on arrays to for-each loops and yield operations.
    LowerToForEach convert_fe(program.funcs);
    for (auto &[_, f] : program.funcs) {
        f->body = convert_fe.mutate(f->body);
    }

    for (auto &[name, f] : convert_fe.new_funcs) {
        auto [_, inserted] = program.funcs.try_emplace(name, std::move(f));
        internal_assert(inserted)
            << "function with name: " << name << " already exists";
    }
    // 2. Lower for-each loops to for-all loops.
    LowerToForAll convert_fa;
    for (auto &[_, f] : program.funcs) {
        f->body = convert_fa.mutate(f->body);
    }
    return program;
}

} // namespace lower
} // namespace bonsai
