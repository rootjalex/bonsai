#include "Lower/VerifyOptions.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/TypeEnforcement.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Scope.h"
#include "Utils.h"

#include <string>
#include <unordered_set>

namespace bonsai {
namespace lower {

namespace {

// TODO(cgyurgyik): This is bare bones. Other static analysis can be performed
// here to validate options, e.g.,
//      i: option[i32] = 42;
//      use(i); // Legal, but will result in error.
class OptionVisitor : public ir::Visitor {
  public:
  private:
    // Tracks the (legally) dereferenced options in FIFO manner. We may want
    // some ordered hash set in the future for faster lookup.
    std::vector<std::string> dereferenced_options;

    void visit_statement(const ir::Stmt &node) {
        if (const auto *op = node.as<ir::Sequence>()) {
            for (const ir::Stmt &statement : op->stmts) {
                visit_statement(statement);
            }
        } else if (const auto *op = node.as<ir::IfElse>()) {
            visit(op);
        } else if (const auto *op = node.as<ir::Print>()) {
            visit(op);
        } else if (const auto *op = node.as<ir::Return>()) {
            visit(op);
        } else if (const auto *op = node.as<ir::LetStmt>()) {
            visit(op);
        } else if (const auto *op = node.as<ir::Store>()) {
            visit(op);
        } else if (const auto *op = node.as<ir::Assign>()) {
            visit(op);
        } else if (const auto *op = node.as<ir::Accumulate>()) {
            visit(op);
        }
    }

    void visit_expression(const ir::Expr &node) {
        if (auto *v = node.as<ir::Var>()) {
            visit(v);
        } else if (auto *v = node.as<ir::Cast>()) {
            visit(v);
        }
    }

    void visit(const ir::IfElse *node) override {
        ir::Expr condition = node->cond;
        uint32_t before = dereferenced_options.size();
        if (const ir::Cast *c = condition.as<ir::Cast>()) {
            if (const ir::Var *v = c->value.as<ir::Var>()) {

                if (const ir::Option_t *type = v->type.as<ir::Option_t>()) {
                    dereferenced_options.push_back(v->name);
                }
            }
        }
        // We make the following (strict) assumption: an option is only legally
        // dereferenced in the `then-body` of an `if` statement.
        visit_statement(node->then_body);
        if (uint32_t after = dereferenced_options.size(); before != after) {
            internal_assert(!dereferenced_options.empty());
            dereferenced_options.pop_back();
        }
        if (ir::Stmt else_body = node->else_body; else_body.defined()) {
            visit_statement(else_body);
        }
    }

    void visit(const ir::Print *node) override {
        visit_expression(node->value);
    }

    void visit(const ir::Return *node) override {
        visit_expression(node->value);
    }

    void visit(const ir::LetStmt *node) override {
        visit_expression(node->value);
    }

    void visit(const ir::Store *node) override {
        visit_expression(node->value);
    }

    void visit(const ir::Assign *node) override {
        visit_expression(node->value);
    }

    void visit(const ir::Accumulate *node) override {
        visit_expression(node->value);
    }

    void visit(const ir::Cast *node) override {
        if (node->type.is<ir::Option_t>()) {
            return; // This is a newly created Option.
        }
        if (const ir::Var *v = node->value.as<ir::Var>()) {
            visit(v);
        }
    }

    void visit(const ir::Var *node) override {
        const ir::Option_t *type = node->type.as<ir::Option_t>();
        if (type == nullptr) {
            return;
        }
        std::string_view name = node->name;
        if (auto it = std::find(dereferenced_options.begin(),
                                dereferenced_options.end(), name);
            it == dereferenced_options.end()) {
            internal_error << "illegal dereference of `" << name << ": "
                           << node->type << "`";
        }
    }
};

void verify(const ir::Program &program) {
    for (const auto &[_, f] : program.funcs) {
        if (const ir::Stmt &body = f->body; body.defined()) {
            OptionVisitor visitor;
            body.accept(&visitor);
        }
    }
}

} // namespace

void verify_options(const ir::Program &program) { verify(program); }

} // namespace lower
} // namespace bonsai
