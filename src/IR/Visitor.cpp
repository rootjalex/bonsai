#include "IR/Visitor.h"

#include "IR/Expr.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

namespace bonsai {
namespace ir {

namespace {

template <typename T>
void visit_list(Visitor *v, const std::vector<T> nodes) {
    for (const auto &node : nodes) {
        node.accept(v);
    }
}

// template<typename T>
// void visit_map(Visitor *v, const std::map<std::string, T> fields) {
//     for (const auto& [key, value] : fields) {
//         value.accept(v);
//     }
// }

void visit_writeloc(Visitor *v, const WriteLoc &loc) {
    for (const auto &value : loc.accesses) {
        if (std::holds_alternative<Expr>(value)) {
            std::get<Expr>(value).accept(v);
        }
    }
}
} // namespace

void Visitor::visit(const Int_t *) {}

void Visitor::visit(const UInt_t *) {}

void Visitor::visit(const Float_t *) {}

void Visitor::visit(const Bool_t *) {}

void Visitor::visit(const Ptr_t *node) { node->etype.accept(this); }

void Visitor::visit(const Vector_t *node) { node->etype.accept(this); }

void Visitor::visit(const Struct_t *node) {
    for (const auto &[_, value] : node->fields) {
        value.accept(this);
    }
}

void Visitor::visit(const Tuple_t *node) { visit_list(this, node->etypes); }

void Visitor::visit(const Option_t *node) { node->etype.accept(this); }

void Visitor::visit(const Set_t *node) { node->etype.accept(this); }

void Visitor::visit(const Function_t *node) {
    node->ret_type.accept(this);
    visit_list(this, node->arg_types);
}

void Visitor::visit(const IntImm *) {}

void Visitor::visit(const UIntImm *) {}

void Visitor::visit(const FloatImm *) {}

void Visitor::visit(const BoolImm *) {}

void Visitor::visit(const Var *) {}

void Visitor::visit(const BinOp *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const UnOp *node) { node->a.accept(this); }

void Visitor::visit(const Select *node) {
    node->cond.accept(this);
    node->tvalue.accept(this);
    node->fvalue.accept(this);
}

void Visitor::visit(const Cast *node) {
    // TODO: node->type.accept(this) ?
    node->value.accept(this);
}

void Visitor::visit(const Broadcast *node) { node->value.accept(this); }

void Visitor::visit(const VectorReduce *node) { node->value.accept(this); }

void Visitor::visit(const VectorShuffle *node) {
    node->value.accept(this);
    visit_list(this, node->idxs);
}

void Visitor::visit(const Ramp *node) {
    node->base.accept(this);
    node->stride.accept(this);
}

void Visitor::visit(const Extract *node) {
    node->vec.accept(this);
    node->idx.accept(this);
}

void Visitor::visit(const Build *node) { visit_list(this, node->values); }

void Visitor::visit(const Access *node) { node->value.accept(this); }

void Visitor::visit(const Intrinsic *node) { visit_list(this, node->args); }

void Visitor::visit(const Lambda *node) { node->value.accept(this); }

void Visitor::visit(const GeomOp *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const SetOp *node) {
    node->a.accept(this);
    node->b.accept(this);
}

void Visitor::visit(const Call *node) {
    node->func.accept(this);
    visit_list(this, node->args);
}

void Visitor::visit(const Return *node) { node->value.accept(this); }

void Visitor::visit(const Store *node) {
    if (node->index.defined()) {
        node->index.accept(this);
    }
    node->value.accept(this);
}

void Visitor::visit(const LetStmt *node) {
    node->value.accept(this);
    // TODO: fix this!! bring back SSA
    // node->body.accept(this);
}

void Visitor::visit(const IfElse *node) {
    node->cond.accept(this);
    node->then_body.accept(this);
    if (node->else_body.defined()) {
        node->else_body.accept(this);
    }
}

void Visitor::visit(const Sequence *node) { visit_list(this, node->stmts); }

void Visitor::visit(const Assign *node) {
    visit_writeloc(this, node->loc);
    node->value.accept(this);
    // TODO: fix this!! bring back SSA
    // node->body.accept(this);
}

void Visitor::visit(const Accumulate *node) {
    visit_writeloc(this, node->loc);
    node->value.accept(this);
    // TODO: fix this!! bring back SSA
    // node->body.accept(this);
}

} // namespace ir
} // namespace bonsai
