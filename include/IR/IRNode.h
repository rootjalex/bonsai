#pragma once

#include "IntrusivePtr.h"

namespace bonsai {
namespace ir {

template <typename IR, typename _TypeEnum>
struct IRNode {
    virtual void accept(Visitor *v) const = 0;
    IRNode(_TypeEnum t) : node_type(t) {}
    virtual ~IRNode() = default;

    /** These classes are all managed with intrusive reference
     * counting, so we also track a reference count. It's mutable
     * so that we can do reference counting even through const
     * references to IR nodes.
     */
    mutable RefCount ref_count;

    /** Each IR node subclass has a unique identifier. We can compare
     * these values to do runtime type identification. We don't
     * compile with rtti because that injects run-time type
     * identification stuff everywhere (and often breaks when linking
     * external libraries compiled without it), and we only want it
     * for IR nodes. One might want to put this value in the vtable,
     * but that adds another level of indirection, and for Exprs we
     * have 32 free bits in between the ref count and the Type
     * anyway, so this doesn't increase the memory footprint of an IR node.
     */
    _TypeEnum node_type;

    using TypeEnum = _TypeEnum;
};

// All instances of IRNode need to implement ref_count and destroy!

} // namespace ir
} // namespace bonsai
