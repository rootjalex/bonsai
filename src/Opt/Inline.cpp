#include "Opt/Inline.h"

#include "Error.h"
#include "IR/Argument.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <unordered_map>

namespace bonsai {
namespace opt {

namespace {

class ReturnToAssign : public ir::Mutator {
  public:
    ReturnToAssign(ir::WriteLoc loc) : loc(std::move(loc)) {}

    ir::Stmt visit(const ir::Return *node) override {
        internal_assert(node->value.defined());
        return ir::Store::make(loc, node->value);
    }

  private:
    ir::WriteLoc loc;
};

// A half-implemented statement inliner. Hard-coded for let, store, and
// allocate.
class StmtInliner : public ir::Mutator {
  public:
    StmtInliner(
        std::string name, const ir::FuncMap &functions,
        const std::unordered_map<std::string, ir::Stmt> &function_to_stmt)
        : name(std::move(name)), functions(functions),
          function_to_stmt(function_to_stmt) {}

    ir::Stmt visit(const ir::Sequence *node) override {
        std::vector<ir::Stmt> stmts;
        for (const ir::Stmt &stmt : node->stmts) {
            const auto *let = stmt.as<ir::LetStmt>();
            const auto *store = stmt.as<ir::Store>();
            const auto *allocate = stmt.as<ir::Allocate>();

            if (let == nullptr && store == nullptr && allocate == nullptr) {
                stmts.push_back(mutate(stmt));
                continue;
            }

            const ir::Call *call = nullptr;
            ir::WriteLoc loc;

            if (let) {
                call = let->value.as<ir::Call>();
                loc = let->loc;
            } else if (store) {
                call = store->value.as<ir::Call>();
                loc = store->loc;
            } else if (allocate) {
                call = allocate->value.as<ir::Call>();
                loc = allocate->loc;
            }

            if (call == nullptr) {
                stmts.push_back(mutate(stmt));
                continue;
            }

            const auto *func = call->func.as<ir::Var>();
            if (func == nullptr) {
                stmts.push_back(mutate(stmt));
                continue;
            }

            auto it = function_to_stmt.find(func->name);
            if (it == function_to_stmt.end()) {
                stmts.push_back(mutate(stmt));
                continue;
            }

            const auto *seq = it->second.as<ir::Sequence>();
            if (seq == nullptr) {
                stmts.push_back(mutate(stmt));
                continue;
            }

            if (!loc.type.is_scalar()) {
                stmts.push_back(mutate(stmt));
                continue;
            }

            auto f = functions.find(func->name);
            internal_assert(f != functions.end());
            std::vector<std::string> argument_names;
            const std::vector<ir::Argument> &args = f->second->args;
            std::transform(args.begin(), args.end(),
                           std::back_inserter(argument_names),
                           [](const auto &a) { return a.name; });

            std::map<std::string, ir::Expr> repls;
            internal_assert(argument_names.size() == call->args.size())
                << "mismatch in function argument size: "
                << argument_names.size()
                << " and call argument size: " << call->args.size()
                << " for function: " << func->name;
            for (int i = 0, e = argument_names.size(); i < e; ++i) {
                repls[argument_names[i]] = call->args[i];
            }

            ir::Expr trailing_return = get_trailing_return_value(seq);
            if (trailing_return.defined()) {
                trailing_return = replace(repls, trailing_return);
            }

            if (let || allocate) {
                ir::Expr initial_value = trailing_return.defined()
                                             ? trailing_return
                                             : make_zero(loc.type);
                stmts.push_back(ir::Allocate::make(
                    loc, initial_value, ir::Allocate::Memory::Stack));
            }

            ir::Stmt body = replace(repls, seq);
            body = convert_trailing_return_to_else(body, loc, trailing_return);
            ReturnToAssign m(loc);
            body = m.mutate(body);
            stmts.push_back(std::move(body));
        }
        return ir::Sequence::make(stmts);
    }

  private:
    ir::Expr get_trailing_return_value(const ir::Stmt &stmt) {
        const auto *seq = stmt.as<ir::Sequence>();
        if (seq == nullptr || seq->stmts.empty()) {
            const auto *r = stmt.as<ir::Return>();
            return r ? r->value : ir::Expr();
        }

        const auto *last_stmt = &seq->stmts.back();
        const auto *if_else = last_stmt->as<ir::IfElse>();
        if (if_else && if_else->else_body.defined()) {
            return get_return_value_from(if_else->else_body);
        }

        const auto *last_return = last_stmt->as<ir::Return>();
        if (last_return) {
            return last_return->value;
        }

        return ir::Expr();
    }

    ir::Expr get_return_value_from(const ir::Stmt &node) {
        if (const auto *r = node.as<ir::Return>()) {
            return r->value;
        }

        const auto *seq = node.as<ir::Sequence>();
        if (seq && !seq->stmts.empty()) {
            return get_return_value_from(seq->stmts.back());
        }

        return ir::Expr();
    }

    ir::Stmt convert_trailing_return_to_else(const ir::Stmt &stmt,
                                             const ir::WriteLoc &loc,
                                             const ir::Expr &initial_value) {
        const auto *seq = stmt.as<ir::Sequence>();
        if (seq == nullptr || seq->stmts.size() < 2) {
            return stmt;
        }

        const auto *last_return = seq->stmts.back().as<ir::Return>();
        if (last_return == nullptr) {
            return stmt;
        }

        const auto *second_last =
            seq->stmts[seq->stmts.size() - 2].as<ir::IfElse>();
        if (second_last == nullptr || second_last->else_body.defined()) {
            return stmt;
        }

        if (!ends_with_return(second_last->then_body)) {
            return stmt;
        }

        std::vector<ir::Stmt> new_stmts;
        for (size_t i = 0; i < seq->stmts.size() - 2; ++i) {
            new_stmts.push_back(seq->stmts[i]);
        }

        if (initial_value.defined() &&
            ir::equals(last_return->value, initial_value)) {
            new_stmts.push_back(second_last);
        } else {
            new_stmts.push_back(ir::IfElse::make(
                second_last->cond, second_last->then_body, last_return));
        }

        return ir::Sequence::make(new_stmts);
    }

    bool ends_with_return(const ir::Stmt &stmt) {
        if (stmt.is<ir::Return>()) {
            return true;
        }
        const auto *s = stmt.as<ir::Sequence>();
        return s && !s->stmts.empty() && ends_with_return(s->stmts.back());
    }

    std::string name;
    const ir::FuncMap &functions;
    const std::unordered_map<std::string, ir::Stmt> &function_to_stmt;
};

class ExprInliner : public ir::Mutator {
  public:
    ExprInliner(
        const ir::FuncMap &functions,
        const std::unordered_map<std::string, ir::Expr> &function_to_expr)
        : functions(functions), function_to_expr(function_to_expr) {}

    ir::Expr visit(const ir::Call *node) override {
        const ir::Var *v = node->func.as<ir::Var>();
        if (v == nullptr) {
            return Mutator::visit(node);
        }
        const std::string &function_name = v->name;
        auto it = function_to_expr.find(function_name);
        if (it == function_to_expr.end()) {
            return Mutator::visit(node);
        }
        auto f = functions.find(function_name);
        internal_assert(f != functions.end());
        std::vector<std::string> argument_names;
        const std::vector<ir::Argument> &args = f->second->args;
        std::transform(args.begin(), args.end(),
                       std::back_inserter(argument_names),
                       [](const auto &a) { return a.name; });
        // Replace function arguments with call arguments.
        std::map<std::string, ir::Expr> repls;
        internal_assert(argument_names.size() == node->args.size())
            << "mismatch in function argument size: " << argument_names.size()
            << " and call argument size: " << node->args.size()
            << " for function: " << function_name;
        for (int i = 0, e = argument_names.size(); i < e; ++i) {
            repls[argument_names[i]] = node->args[i];
        }
        return replace(repls, it->second);
    }

  private:
    const ir::FuncMap &functions;
    const std::unordered_map<std::string, ir::Expr> &function_to_expr;
};

} // namespace

ir::FuncMap Inline::run(ir::FuncMap funcs,
                        const CompilerOptions &options) const {
    // If a function simply returns a value, replace the call with said value.
    // We refrain from performing more complex inlining for now to avoid code
    // size blowup.
    std::unordered_map<std::string, ir::Expr> function_to_expr;
    std::unordered_map<std::string, ir::Stmt> function_to_stmt;
    for (const auto &[name, func] : funcs) {
        if (func->is_kernel()) {
            // Don't inline kernels.
            continue;
        }
        if (!func->ret_type.is<ir::Void_t>()) {
            if (const auto *body = func->body.as<ir::Return>()) {
                internal_assert(body->value.defined());
                function_to_expr[name] = body->value;
                continue;
            }
            if (name.starts_with("dist") || name.starts_with("intersect")) {
                function_to_stmt[name] = func->body;
            }
        }
    }

    {
        for (auto &[name, func] : funcs) {
            StmtInliner inliner(name, funcs, function_to_stmt);
            func->body = inliner.mutate(std::move(func->body));
        }
    }
    {
        // We assume the inliner will not change the number of arguments in a
        // function, and thus it is ok to only instantiate it once per program.
        // If the `ExprInliner` class were to break this assumption, this would
        // need to change.
        ExprInliner inliner(funcs, function_to_expr);
        for (auto &[name, func] : funcs) {
            func->body = inliner.mutate(std::move(func->body));
        }
    }

    return funcs;
}

} // namespace opt
} // namespace bonsai
