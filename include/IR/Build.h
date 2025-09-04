#pragma once

#include <string>

#include "Argument.h"
#include "Expr.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Stmt.h"
#include "Type.h"
#include "Visitor.h"

namespace bonsai {
namespace ir {

struct BuildIR;

enum class IRBuildEnum {
    BuildLet,
    BuildRecurse,
    BuildReturn,
    BuildRule,
    BuildRoot,
    BuildSequence,
};

using IRBuildNode = ir::IRNode<BuildIR, IRBuildEnum>;

struct BaseBuildNode : public IRBuildNode {
    BaseBuildNode(IRBuildEnum t) : IRBuildNode(t) {}
    virtual BuildIR mutate_build(ir::Mutator *m) const = 0;
};

template <typename T>
struct BuildNode : public BaseBuildNode {
    void accept(ir::Visitor *v) const override {
        return v->visit((const T *)this);
    }
    BuildIR mutate_build(ir::Mutator *m) const override;
    BuildNode() : BaseBuildNode(T::node_type) {}
    ~BuildNode() override = default;
};

struct BuildIR : public ir::IRHandle<IRBuildNode> {
    BuildIR() = default;
    BuildIR(const IRBuildNode *n) : ir::IRHandle<IRBuildNode>(n) {}

    /** Override get() to return a BaseBuildNode * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseBuildNode *get() const { return (const BaseBuildNode *)ptr; }
};

template <typename T>
BuildIR BuildNode<T>::mutate_build(ir::Mutator *m) const {
    return m->visit((const T *)this);
}

struct BuildLet : BuildNode<BuildLet> {
    ir::Stmt stmt;

    static BuildIR make(ir::Stmt stmt);

    static const IRBuildEnum node_type = IRBuildEnum::BuildLet;
};

struct BuildRecurse : BuildNode<BuildRecurse> {
    ir::Expr field;

    static BuildIR make(ir::Expr field);

    static const IRBuildEnum node_type = IRBuildEnum::BuildRecurse;
};

struct BuildRule : BuildNode<BuildRule> {
    ir::Expr field;
    // (optionally defined) expression. If none provided, it must have an
    // argument of the same name in the build function.
    ir::Expr expr;

    static BuildIR make(ir::Expr field, ir::Expr expr);

    static const IRBuildEnum node_type = IRBuildEnum::BuildRule;
};

struct BuildRoot : BuildNode<BuildRoot> {
    ir::BuildIR rules;

    static BuildIR make(ir::BuildIR rules);

    static const IRBuildEnum node_type = IRBuildEnum::BuildRoot;
};

struct BuildSequence : BuildNode<BuildSequence> {
    std::vector<BuildIR> sequence;

    static BuildIR make(std::vector<BuildIR> sequence);

    static const IRBuildEnum node_type = IRBuildEnum::BuildSequence;
};

struct BuildReturn : BuildNode<BuildReturn> {
    ir::Expr expr;

    static BuildIR make(ir::Expr expr);

    static const IRBuildEnum node_type = IRBuildEnum::BuildReturn;
};

struct BuildFunction {
    ir::BVH_t::Variant variant;
    std::vector<ir::Argument> arguments;
    BuildIR body;
    bool is_root; // whether this is a root build function.
};

struct BuildLayout {
    std::string name;
    ir::Type type;
    std::vector<BuildFunction> functions;
};

using BuildMap = std::map<std::string, BuildLayout>;
// std::ostream &operator<<(std::ostream &os, const BuildMap &map);

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
