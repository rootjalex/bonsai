#pragma once

#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include <vector>

namespace bonsai {
namespace ir {

struct Interface;

enum class IRInterfaceEnum {
    IEmpty,
    IFloat,
    IVector,
    // IUnion,
    // IUserDefined,
};

using IRInterfaceNode = IRNode<Interface, IRInterfaceEnum>;

/* This is necessary to get mutate() to work properly... */
struct BaseInterfaceNode : public IRInterfaceNode {
    BaseInterfaceNode(IRInterfaceEnum t) : IRInterfaceNode(t) {}
    virtual Interface mutate_interface(Mutator *m) const = 0;
};

template <typename T>
struct InterfaceNode : public BaseInterfaceNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    Interface mutate_interface(Mutator *m) const override;
    InterfaceNode() : BaseInterfaceNode(T::node_type) {}
    ~InterfaceNode() override = default;
};

struct Interface : public IRHandle<IRInterfaceNode> {
    /** Make an undefined Interface */
    Interface() = default;

    /** Make a Interface from a concrete Interface node pointer (e.g. Int_t) */
    Interface(const IRInterfaceNode *n) : IRHandle<IRInterfaceNode>(n) {}

    /** Override get() to return a BaseInterfaceNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseInterfaceNode *get() const {
        return (const BaseInterfaceNode *)ptr;
    }

    // TODO: implement copy/move semantics!

    // Does this interface support numeric operations like +/*/etc?
    bool is_numeric() const;
};

template <typename T>
Interface InterfaceNode<T>::mutate_interface(Mutator *m) const {
    return m->visit((const T *)this);
}

struct IEmpty : InterfaceNode<IEmpty> {
    static Interface make();
    static const IRInterfaceEnum node_type = IRInterfaceEnum::IEmpty;
};

struct IFloat : InterfaceNode<IFloat> {
    static Interface make();
    static const IRInterfaceEnum node_type = IRInterfaceEnum::IFloat;
};

struct IVector : InterfaceNode<IVector> {
    Interface etype;

    static Interface make(Interface etype);
    static const IRInterfaceEnum node_type = IRInterfaceEnum::IVector;
};

// TODO: IUserDefined and IUnion

/*
struct IUnion : InterfaceNode<IUnion> {
    std::vector<Interface> interfaces;
    static Interface make(std::vector<Interface> interfaces);
    static const IRInterfaceEnum node_type = IRInterfaceEnum::IUnion;
};
*/

} // namespace ir

template <>
inline RefCount &
ref_count<ir::IRInterfaceNode>(const ir::IRInterfaceNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRInterfaceNode>(const ir::IRInterfaceNode *t) {
    delete t;
}

} // namespace bonsai
