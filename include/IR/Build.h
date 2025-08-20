#pragma once

#include <string>

#include "Expr.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include "Type.h"

namespace bonsai {
namespace ir {

struct BuildIR;

enum class IRBuildEnum {

};

using IRBuildNode = IRNode<BuildIR, IRBuildEnum>;

struct BaseBuildNode : public IRBuildNode {
    BaseBuildNode(IRBuildEnum t) : IRBuildNode(t) {}
    virtual BuildIR mutate_build(Mutator *m) const = 0;
};

template <typename T>
struct BuildNode : public BaseBuildNode {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    BuildIR mutate_build(Mutator *m) const override;
    BuildNode() : BaseBuildNode(T::node_type) {}
    ~BuildNode() override = default;
};

struct BuildIR : public IRHandle<IRBuildNode> {
    /** Make an undefined expr */
    BuildIR() = default;

    /** Make an expr from a concrete expr node pointer (e.g. Add) */
    BuildIR(const IRBuildNode *n) : IRHandle<IRBuildNode>(n) {}

    /** Override get() to return a BaseBuildNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseBuildNode *get() const { return (const BaseBuildNode *)ptr; }
};

template <typename T>
BuildIR BuildNode<T>::mutate_build(Mutator *m) const {
    return m->visit((const T *)this);
}

// struct IntImm : BuildNode<IntImm> {
//     int64_t value;

//     static BuildIR make(Type t, int64_t value);

//     static const IRBuildEnum node_type = IRBuildEnum::IntImm;
// };

} // namespace ir

template <>
inline RefCount &ref_count<ir::IRBuildNode>(const ir::IRBuildNode *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRBuildNode>(const ir::IRBuildNode *t) {
    delete t;
}

} // namespace bonsai
