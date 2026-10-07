#pragma once

#include "IR/Expr.h"
#include "IR/Visitor.h"

namespace bonsai {
namespace opt {

// What a value is allowed to do and still be moved or duplicated at one
// program point. `sinkable` admits the pure computations and the reads;
// `memory` marks the reads, which never cross a statement that could write
// (there is no alias analysis here: a read never crosses a write, whosever
// it is). Shared by Opt/Sink (motion) and Opt/NarrowLoads (duplication at
// the same statement, where `memory` costs nothing and `sinkable` is the
// whole test).
struct ValueClass : ir::Visitor {
    bool sinkable = true;
    bool memory = false;

    void visit(const ir::Call *) override { sinkable = false; }
    void visit(const ir::MatchExpr *) override { sinkable = false; }
    void visit(const ir::Generator *) override { sinkable = false; }
    void visit(const ir::Lambda *) override { sinkable = false; }
    void visit(const ir::GeomOp *) override { sinkable = false; }
    void visit(const ir::SetOp *) override { sinkable = false; }
    void visit(const ir::AggOp *) override { sinkable = false; }
    void visit(const ir::Construct *) override { sinkable = false; }
    void visit(const ir::PtrTo *) override { sinkable = false; }
    void visit(const ir::Deref *node) override {
        memory = true;
        ir::Visitor::visit(node);
    }
    void visit(const ir::StoredElement *node) override {
        memory = true;
        ir::Visitor::visit(node);
    }
    void visit(const ir::Unwrap *node) override {
        // A variant read: of a tree node's bytes as often as of a value.
        memory = true;
        ir::Visitor::visit(node);
    }
    void visit(const ir::Intrinsic *node) override {
        if (ir::Intrinsic::has_effects(node->op)) {
            sinkable = false;
            return;
        }
        if (node->op == ir::Intrinsic::rt_payload ||
            node->op == ir::Intrinsic::tex_sample_grad_2d) {
            memory = true;
        }
        ir::Visitor::visit(node);
    }
};

inline ValueClass classify(const ir::Expr &value) {
    ValueClass c;
    value.accept(&c);
    return c;
}

} // namespace opt
} // namespace bonsai
