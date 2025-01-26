// IRHandle.h
#pragma once

#include "IntrusivePtr.h"

#include "Visitor.h"

#include <cassert>

namespace bonsai {
namespace ir {

/** IR nodes are passed around opaque handles to them. This is a
   base class for those handles. It manages the reference count,
   and dispatches visitors. */
template <typename IRNode>
struct IRHandle : public IntrusivePtr<const IRNode> {
    IRHandle() = default;

    IRHandle(const IRNode *p) : IntrusivePtr<const IRNode>(p) {}

    /** Dispatch to the correct visitor method for this node. E.g. if
     * this node is actually an Add node, then this will call
     * Visitor::visit(const Add *) */
    void accept(Visitor *v) const {
        assert(this->ptr);
        this->ptr->accept(v);
    }

    /** Downcast this ir node to its actual type (e.g. Add, or
     * Select). This returns nullptr if the node is not of the requested
     * type. Example usage:
     *
     * if (const Add *add = node->as<Add>()) {
     *   // This is an add node
     * }
     */
    template <typename T>
    const T *as() const {
        if (this->ptr && this->ptr->node_type == T::_node_type) {
            return (const T *)this->ptr;
        }
        return nullptr;
    }

    template <typename T>
    bool is() const {
        return (this->ptr && this->ptr->node_type == T::_node_type);
    }

    typename IRNode::TypeEnum node_type() const { return this->ptr->node_type; }
};

} // namespace ir
} // namespace bonsai
