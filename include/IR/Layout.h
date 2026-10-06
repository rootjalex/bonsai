#pragma once

#include "Error.h"
#include "Expr.h"
#include "IRHandle.h"
#include "IRNode.h"
#include "IntrusivePtr.h"
#include "Mutator.h"
#include "Visitor.h"

#include <map>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {

struct Layout;

enum class IRLayoutEnum {
    Name,
    Pad,
    Switch,
    Chain,
    Group,
    Materialize,
    Lookup,
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
    LayoutNode() : BaseLayoutNode(T::node_type) {}
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

    static const IRLayoutEnum node_type = IRLayoutEnum::Name;
};

struct Pad : LayoutNode<Pad> {
    uint32_t bits;

    static Layout make(uint32_t bits);

    static const IRLayoutEnum node_type = IRLayoutEnum::Pad;
};

// split from https://dl.acm.org/doi/pdf/10.1145/3607858
struct Switch : LayoutNode<Switch> {
    // TODO: allow switching on unnamed bits?
    std::string field; // switch param
    // TODO: support non-constant or field ranges?
    struct Arm {
        std::optional<int64_t> value;
        std::optional<std::string> name; // Tree branch name
        Layout layout;
    };
    std::vector<Arm> arms;

    static Layout make(std::string field, std::vector<Arm> arms);

    static const IRLayoutEnum node_type = IRLayoutEnum::Switch;
};

struct Chain : LayoutNode<Chain> {
    std::vector<Layout> layouts;

    static Layout make(std::vector<Layout> layouts);

    static const IRLayoutEnum node_type = IRLayoutEnum::Chain;
};

struct Group : LayoutNode<Group> {
    // Direct groups are addressed by a component of the reference the
    // traversal already holds -- the node array of a tree, walked by its own
    // index. Indirect groups are auxiliary storage, reached only by a lookup
    // from somewhere else: a foreign table, in the relational reading. They
    // are what lets one tree's storage hold another's, and so what lets two
    // terms share a subtree by naming the same index. Pointer groups are
    // indirect groups whose rows are reached by address rather than by index
    // from the group's base: a lookup `arena[a]` is the row at address `a`,
    // Scion's `ptr` groups, and what a reference that is a pointer -- Embree's
    // NodeRef, an address with the kind in its low bits -- is looked up in.
    // The group's own storage (its size, its buffer) is what the program owns
    // and uploads; the addresses the lookups follow point into it.
    enum class Type { Direct, Indirect, Pointer };

    Expr size;
    // The index variable this group is addressed by, in scope for its body.
    std::string name;
    // What the source called the group, and the group's identity: it is what
    // `<variant> from <group>[<index>]` resolves against. Empty for a group
    // the source left anonymous, which is exactly why such a group cannot be
    // looked up.
    std::string declared_name;
    ir::Type index_t;
    Layout inner;
    Type type = Type::Direct;
    // Defined when the group's rows are the elements of the tree's set: an
    // indirect group whose fields are the element's, which a leaf's `data`
    // ranges over. Stored in tiles where the rows are an anonymous constant
    // inner group -- `indirect group prims[pCount] { group[4] { v0 : vec3f;
    // ... }; }` -- each tile holding every field as a vector over its lanes
    // (see ir::TiledArray). Scion's array-of-structs-of-arrays, spelled as
    // Scion spells it: a group nested in a group, the inner one the tile.
    ir::Type element;
    // Where a walk of a direct group begins -- the root's reference -- when
    // the layout says (`layout tris(ref : u64 = 0u)`, Scion's reference
    // parameter with its default); undefined means zero.
    Expr start;
    // For the tile of a group of elements: which of its lanes hold elements,
    // a predicate over the tile's members (`group[4] { ... } where geomID !=
    // 4294967295u`, Embree's padding slots excluded by their ids). Undefined
    // means every lane does; see ir::TiledArray::valid.
    Expr valid;

    static Layout make(Expr size, std::string name, std::string declared_name,
                       ir::Type index_t, Layout inner,
                       Type type = Type::Direct, ir::Type element = ir::Type(),
                       Expr start = Expr(), Expr valid = Expr());

    static const IRLayoutEnum node_type = IRLayoutEnum::Group;
};

struct Materialize : LayoutNode<Materialize> {
    std::string name;
    Expr value;

    static Layout make(std::string name, Expr value);

    static const IRLayoutEnum node_type = IRLayoutEnum::Materialize;
};

// `<variant> from <group>[<index>]`: this arm's fields are not here, they are
// a row of another group.
//
// A direct group stores a term's fields where the reference lands. A lookup
// says instead that the reference names a row of some other group -- the
// relational reading is a foreign key, and it is what makes storage shareable:
// two terms whose references carry the same index are the same row, so an
// instanced subtree is stored once however many times it appears.
//
// The variant this stands for comes from the arm that holds it, so this node
// carries only where to look: which group, and at what index into it.
//
// `<variant> from <group>[<offset>] { <layout> }`: the row is at that byte
// offset of a group of bytes, laid out as `<layout>` says -- the arm brings
// its own shape, since the bytes have none. One storage then holds rows of
// every shape, each where its reference says, and a reference is the byte
// offset with the variant in its low bits: Embree's `NodeRef`, a pointer
// with the kind tagged into its low four bits, as an offset from the
// storage's start rather than an address, so that the tree is still plain
// data that can be copied or handed to a device. The group named is a
// group of bytes, `indirect group arena[bytes] { byte : u8; }`. A group of
// the tree's elements declared inside the shape without a size --
// `indirect group tiles { group[4] { ... }; }` -- is a run of tiles that
// begins where the row does, as many as the arm's `range` over it asks
// for (Embree's leaf, a pointer and a count).
struct Lookup : LayoutNode<Lookup> {
    std::string group_name;
    Expr index;
    // Defined when the row's layout comes with the arm (see above);
    // undefined when it is the named group's own.
    Layout shape;

    static Layout make(std::string group_name, Expr index,
                       Layout shape = Layout());

    static const IRLayoutEnum node_type = IRLayoutEnum::Lookup;
};

// The tile of a group of elements (see Group::element): the anonymous,
// constant-sized group that is the whole of `inner`, or null where `inner` is
// anything else.
const Group *tile_of(const Layout &inner);
// How many elements a tile holds.
uint32_t tile_width(const Group &tile);
// Whether `inner`, a tile's members, provides every field of `element`: each
// member a stored field (`name : type`) or a derived one (`name = expr`),
// each name once, and every field of the struct among them with its type --
// stored under its own name, or derived from the stored ones (ir::TiledArray).
// The stored members are the tile's bytes, in their own order.
bool provides_fields_of(const Layout &inner, const Struct_t &element);

using LayoutMap = std::map<std::string, Layout>;

// How an extern array is stored: rules applied to a cursor into its element.
//
//     layout positions { tight(root); };
//
// `root` is the element of the topmost array. A rule says something about
// the bytes at its cursor and nothing about their meaning, which is what the
// layout language is for. `tight` stores the element as exactly the bytes its
// fields state -- a `vec3f` as three floats, twelve bytes, where the vector
// the program computes with is the machine's sixteen -- and the element is
// converted to the compute type where it is read. The rules to come, split,
// interleave and deinterleave, will name cursors the same way.
struct ArrayLayout {
    struct Rule {
        enum class Kind { Tight };
        Kind kind;
        // The path from the element down: `root`, and later `root.field`.
        std::vector<std::string> cursor;
    };
    std::vector<Rule> rules;
};

// Keyed by the extern's name.
using ArrayLayoutMap = std::map<std::string, ArrayLayout>;

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
