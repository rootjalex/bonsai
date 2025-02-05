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
#include <vector>

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
    std::vector<std::string> frame;

    // Returns the valid dereference count.
    uint32_t valid_dereference_count(const ir::Expr &e) {
        uint32_t count = 0;
        if (const ir::Cast *c = e.as<ir::Cast>()) {
            if (const auto *v = c->value.as<ir::Var>()) {
                if (v->type.is<ir::Option_t>()) {
                    frame.push_back(v->name);
                    return ++count;
                }
            }
        }

        if (const ir::BinOp *node = e.as<ir::BinOp>()) {
            switch (node->op) {
            case ir::BinOp::Or: {
                // We cannot make any assumptions about these.
                return count;
            }
            case ir::BinOp::And: {
                return valid_dereference_count(node->a) +
                       valid_dereference_count(node->b);
            }
            default:
                return count;
            }
        }

        if (const ir::UnOp *node = e.as<ir::UnOp>()) {
            switch (node->op) {
            case ir::UnOp::Not:
                return valid_dereference_count(node->a);
            case ir::UnOp::Neg:
                return count;
            }
        }

        return count;
    }

    void visit(const ir::IfElse *node) override {
        ir::Expr condition = node->cond;
        const uint32_t count = valid_dereference_count(condition);

        // We make the following (strict) assumption: an option is only
        // legally dereferenced in the `then-body` of an `if` statement.
        node->then_body.accept(this);
        for (uint32_t i = 0; i != count; ++i) {
            frame.pop_back();
        }
        if (ir::Stmt else_body = node->else_body; else_body.defined()) {
            else_body.accept(this);
        }
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
        if (auto it = std::find(frame.begin(), frame.end(), name);
            it == frame.end()) {
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
