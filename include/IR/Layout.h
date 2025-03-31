#pragma once

#include "Error.h"
#include "Expr.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include <string>
#include <vector>

namespace bonsai {
namespace ir {

struct Layout;

enum class IRLayoutEnum {
    Name,
    Pad,
    Split,
    Chain,
    Group,
    Materialize,
};

using IRLayoutNode = IRNode<Layout, IRLayoutEnum>;

/* This is necessary to get mutate() to work properly... */
struct BaseLayoutNode : public IRLayoutNode {
    BaseLayoutNode(IRLayoutEnum t) : IRLayoutNode(t) {}
    // virtual Layout mutate_layout(Mutator *m) const = 0;
};

template <typename T>
struct LayoutNode : public BaseLayoutNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    // Layout mutate_layout(Mutator *m) const override;
    LayoutNode() : BaseLayoutNode(T::_node_type) {}
    ~LayoutNode() override = default;
};

struct Layout : public IRHandle<IRLayoutNode> {
    /** Make an undefined Layout */
    Layout() = default;

    /** Make a Layout from a concrete Layout node pointer (e.g. Int_t) */
    Layout(const IRLayoutNode *n) : IRHandle<IRLayoutNode>(n) {}

    /** Override get() to return a BaseLayoutNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseLayoutNode *get() const { return (const BaseLayoutNode *)ptr; }

    // Number of bits of this layout.
    // Assumptions: ptrs are 64 bits, indexes are u32.
    uint64_t bits() const;
    // Number of elements this layout represents.
    Expr count() const;

    // TODO: implement copy/move semantics!
};

// template <typename T>
// Layout LayoutNode<T>::mutate_layout(Mutator *m) const {
//     return m->visit((const T *)this);
// }

struct Name : LayoutNode<Name> {
    std::string name;
    Type type; // primitive type.

    static Layout make(std::string name, Type type);

    static const IRLayoutEnum _node_type = IRLayoutEnum::Name;
};

struct Pad : LayoutNode<Pad> {
    uint32_t bits;

    static Layout make(uint32_t bits);

    static const IRLayoutEnum _node_type = IRLayoutEnum::Pad;
};

// split from https://dl.acm.org/doi/pdf/10.1145/3607858
struct Split : LayoutNode<Split> {
    // TODO: allow switching on unnamed bits?
    std::string field; // switch param
    // TODO: support non-constant or field ranges?
    struct Arm {
        std::optional<int64_t> value;
        Layout layout;
    };
    std::vector<Arm> arms;

    static Layout make(std::string field, std::vector<Arm> arms);

    static const IRLayoutEnum _node_type = IRLayoutEnum::Split;
};

struct Chain : LayoutNode<Chain> {
    std::vector<Layout> layouts;

    static Layout make(std::vector<Layout> layouts);

    static const IRLayoutEnum _node_type = IRLayoutEnum::Chain;
};

struct Group : LayoutNode<Group> {
    Expr size;
    std::string name;
    Type index_t;
    Layout inner;

    static Layout make(Expr size, std::string name, Type index_t, Layout inner);

    static const IRLayoutEnum _node_type = IRLayoutEnum::Group;
};

struct Materialize : LayoutNode<Materialize> {
    std::string name;
    Expr value;

    static Layout make(std::string name, Expr value);

    static const IRLayoutEnum _node_type = IRLayoutEnum::Materialize;
};

using LayoutMap = std::map<std::string, Layout>;

} // namespace ir

template <>
inline RefCount &
ref_count<ir::IRLayoutNode>(const ir::IRLayoutNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRLayoutNode>(const ir::IRLayoutNode *t) {
    delete t;
}

} // namespace bonsai
