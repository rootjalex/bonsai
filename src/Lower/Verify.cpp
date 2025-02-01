#include "Lower/Verify.h"

#include "IR/Operators.h"
#include "Utils.h"
#include <unordered_set>

namespace bonsai {
namespace lower {

// TODO(cgyurgyik): We should implement some better error handling, e.g.,
// LogicalResult / Expect in LLVM.
class Verifier : public ir::Visitor {
  public:
    void verify_statement(const ir::Stmt &s) {
        internal_assert(s.defined());
        s.accept(this);
    }

  private:
    // A set of "results" from side-effecting (effectively void) let statements.
    std::unordered_set<std::string> results;

    // Verify this variable is not used.
    void verify_no_use(const ir::Var *v) {
        internal_assert(!results.contains(v->name))
            << "result from side-effecting (effectively void) "
               "instruction used: "
            << v->name;
    }

    void verify_expression(const ir::Expr &e) {
        internal_assert(e.defined());
        e.accept(this);
    }

    void visit(const ir::LetStmt *node) {
        verify_expression(node->value);
        if (node->value.is_side_effecting()) {
            results.insert(node->loc.base);
            return;
        }
    }

    void visit(const ir::Call *call) {
        for (const ir::Expr &arg : call->args) {
            if (const ir::Var *v = arg.as<ir::Var>()) {
                verify_no_use(v);
            }
        }
    }

    void visit(const ir::Assign *assign) {
        if (const ir::Var *v = assign->value.as<ir::Var>()) {
            verify_no_use(v);
        }
    }

    void visit(const ir::Print *print) {
        if (const ir::Var *v = print->value.as<ir::Var>()) {
            verify_no_use(v);
        }
    }
};

void verify(const ir::Program &program) {
    Verifier v;
    for (const auto &[_, f] : program.funcs) {
        v.verify_statement(f->body);
    }
}

} // namespace lower
} // namespace bonsai
