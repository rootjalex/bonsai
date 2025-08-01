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

struct Member;

enum class IRLayoutEnum {
    Field,
    Pad,
    Split,
    Chain,
    Group,
    Materialize,
};

using IRLayoutMember = IRNode<Member, IRLayoutEnum>;

/* This is necessary to get mutate() to work properly... */
struct BaseLayoutMember : public IRLayoutMember {
    BaseLayoutMember(IRLayoutEnum t) : IRLayoutMember(t) {}
    // virtual Member mutate_layout(Mutator *m) const = 0;
};

template <typename T>
struct LayoutMember : public BaseLayoutMember {
    void accept(Visitor *v) const override { return v->visit((const T *)this); }
    // Member mutate_layout(Mutator *m) const override;
    LayoutMember() : BaseLayoutMember(T::node_type) {}
    ~LayoutMember() override = default;
};

struct Member : public IRHandle<IRLayoutMember> {
    /** Make an undefined Member */
    Member() = default;

    /** Make a Member from a concrete Member node pointer (e.g. Int_t) */
    Member(const IRLayoutMember *n) : IRHandle<IRLayoutMember>(n) {}

    /** Override get() to return a BaseLayoutMember * instead of an IRNode.
     *  This is necessary to get mutate() to work properly. **/
    const BaseLayoutMember *get() const {
        return (const BaseLayoutMember *)ptr;
    }

    // Number of bits of this layout.
    // Assumptions: ptrs are 64 bits, indexes are u32.
    uint64_t bits() const;
    // Number of elements this layout represents.
    Expr count() const;

    // TODO: implement copy/move semantics!
};

struct Layout {
    std::string name;
    Type type; // The algebraic data type associated with this layout.
    std::vector<ir::TypedVar> root;
    Member body;
};

struct Field : LayoutMember<Field> {
    std::string name;
    Type type; // primitive type.

    static Member make(std::string name, Type type);

    static const IRLayoutEnum node_type = IRLayoutEnum::Field;
};

struct Pad : LayoutMember<Pad> {
    uint32_t bits;

    static Member make(uint32_t bits);

    static const IRLayoutEnum node_type = IRLayoutEnum::Pad;
};

struct Arm {
    enum class Comparator {
        EQ = 0,
        NE = 1,
        GT = 2,
        GE = 3,
        LT = 4,
        LE = 5,
    };
    Comparator comparator;
    // The integer value used in the comparison; wildcard if none provided.
    std::optional<int64_t> value;
    // The type associated with this arm, if any.
    std::optional<std::string> name;
    // The body of this split arm.
    Member member;

    bool is_wildcard() const { return !value.has_value(); }
};

// split from https://dl.acm.org/doi/pdf/10.1145/3607858
struct Split : LayoutMember<Split> {
    Member field; // split parameter
    std::vector<Arm> arms;

    static Member make(Member field, std::vector<Arm> arms);

    static const IRLayoutEnum node_type = IRLayoutEnum::Split;

    // Returns the name of the field that is being split.
    std::string field_name() const;
};

struct Chain : LayoutMember<Chain> {
    std::vector<Member> members;

    static Member make(std::vector<Member> members);

    static const IRLayoutEnum node_type = IRLayoutEnum::Chain;
};

struct Group : LayoutMember<Group> {
    Expr size;
    Expr index;
    std::string name;
    Member inner;

    enum class Type { Direct, Indirect };
    Type type;

    static Member make(std::string name, Expr size, Expr index, Member inner,
                       Type type = Type::Direct);

    static const IRLayoutEnum node_type = IRLayoutEnum::Group;
};

struct Materialize : LayoutMember<Materialize> {
    std::string name;
    Expr value;

    static Member make(std::string name, Expr value);

    static const IRLayoutEnum node_type = IRLayoutEnum::Materialize;
};

using LayoutMap = std::map<std::string, Layout>;

} // namespace ir

template <>
inline RefCount &
ref_count<ir::IRLayoutMember>(const ir::IRLayoutMember *t) noexcept {
    return t->ref_count;
}

template <>
inline void destroy<ir::IRLayoutMember>(const ir::IRLayoutMember *t) {
    delete t;
}

} // namespace bonsai
