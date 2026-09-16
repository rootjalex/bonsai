// What a reference to a stored element is made of.
//
// An `argmin` over a tree remembers which element was best (see
// build_arg_extremum in Lower/Trees.cpp). The query lowering says only that:
// the accumulator holds a RefTo of the element, of type ElementRef_t, which
// names the tree the element is stored in and nothing about how. The layout
// language decides how a tree's elements are stored, and so it is only once
// the layout has been applied -- once every element a traversal reaches is
// spelled as a read from the tree's storage -- that a reference can be given
// its representation. This pass does that, and it is the one place the two
// meet.
//
// The representation is the fewest bits that pick the element out of that
// storage: not an address. After LowerLayouts and LowerForEachs an element is
// an access chain from the variable holding the tree's storage, through the
// fields and arrays the layout declared, to the element --
//
//     ts.prims[ts.nodes[n].pOffset + i]            an element of an array
//     ts.nodes[n].data                             a field of a node
//     ts.data[n]                                   the same field, split out
//     ts.group0[i].group1[b].data                  a node in nested groups
//     ts.tris[ts.blas_nodes[m].pOffset + j]        an element of a nested tree
//
// -- and everything in the chain is fixed by the layout except the indices,
// so the indices are the reference: one index as itself, several as a tuple.
// Reading through the reference is the chain again with the indices put back.
// An element that different yield sites reach by different chains (a triangle
// standing on its own beside a triangle inside an instance) needs one more
// thing, which chain: the reference is then a variant type with one arm per
// chain, each carrying that chain's indices, and the read is a match on it --
// the same tag-then-branch that any `match` in the program lowers to, so no
// arm's storage is touched unless the tag says so.
//
// Why never an address. A pointer is a representation the program did not
// write: sixty-four bits bound to one address space, that cannot be handed to
// another device or written to a file, and that a SIMD gang would have to
// carry as a vector of pointers and read through with gathers. An index is
// what the layout already uses to name an element, is as wide as the layout
// said, and is the same on every device the storage is copied to.

#include "Lower/ElementReferences.h"

#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

// One way an element is reached from its tree's storage: the access chain
// from the storage's variable to the element, with each index that varies --
// each one that picks the element out -- replaced by a numbered placeholder.
// Two sites reach elements the same way when their chains are equal, which
// compares the placeholders by number and type.
struct Path {
    ir::Expr chain;
    std::vector<ir::Type> index_ts; // of the placeholders, by number
};

std::string placeholder_name(size_t k) { return "!index" + std::to_string(k); }

// The name of an index inside a reference that is a struct or variant.
std::string index_field(size_t k) { return "idx" + std::to_string(k); }

bool is_constant(const ir::Expr &e) {
    return e.is<ir::IntImm>() || e.is<ir::UIntImm>();
}

// `place` taken apart into the way it is reached and the indices that reach
// it. Everything in the way is the layout's -- field accesses, a variant's
// arm (an `Unwrap`, as Lower/ADTs.cpp reads one), a node's bytes reinterpreted
// at a layout's type, a stored vector converted to the kind computed with, an
// element assembled from several arrays -- and stays as it is; only the
// indices of array elements vary from one element to the next, so those are
// what the reference has to carry. An index that reads storage in several
// places (the three arrays of one struct-of-arrays element, say) is one
// index. Every variable the way reads, other than in an index, must be the
// storage of a tree, which is an extern: the layout language stores every
// element of a tree, including the elements of a tree held in another's
// leaves, in the storage of a tree the program named.
struct TakenApart {
    Path path;
    std::vector<ir::Expr> indices;
};

struct TakeApart : public ir::Mutator {
    const ir::ExternList &externs;
    const ir::Expr &whole;
    TakenApart result;

    TakeApart(const ir::ExternList &externs, const ir::Expr &whole)
        : externs(externs), whole(whole) {}

    using ir::Mutator::visit;

    bool is_storage(const std::string &name) const {
        for (const auto &[ename, _] : externs) {
            if (ename == name) {
                return true;
            }
        }
        return false;
    }

    ir::Expr visit(const ir::Extract *node) override {
        ir::Expr vec = mutate(node->vec);
        ir::Expr idx = node->idx;
        if (!is_constant(idx)) {
            size_t k = 0;
            while (k < result.indices.size() &&
                   !ir::equals(result.indices[k], idx)) {
                k++;
            }
            if (k == result.indices.size()) {
                result.indices.push_back(idx);
                result.path.index_ts.push_back(idx.type());
            }
            idx = ir::Var::make(idx.type(), placeholder_name(k));
        }
        return ir::Extract::make(std::move(vec), std::move(idx));
    }

    ir::Expr visit(const ir::Var *node) override {
        // A function the layout reads a field through is code, not a place.
        if (node->type.is<ir::Function_t>()) {
            return node;
        }
        internal_assert(is_storage(node->name))
            << "A reference to a stored element reads `" << node->name
            << "`, which is not a tree's storage, in: " << whole
            << "\nA reference is the indices along the way from a tree's "
            << "storage to the element, so everything else along the way has "
            << "to be that storage.";
        return node;
    }

    TakenApart take() {
        result.path.chain = mutate(whole);
        return std::move(result);
    }
};

TakenApart take_apart(const ir::Expr &place, const ir::ExternList &externs) {
    return TakeApart(externs, place).take();
}

// What every reference of one ElementRef_t type is made of, from every place
// one is made.
struct Shape {
    // The ways an element of this type is reached, in order of first sight.
    std::vector<Path> paths;
    // Decided once every site has been seen (see `decide`).
    ir::Type reference_t;
    // When there are several paths: the variant type that says which, and
    // the name of the arm for each path.
    std::vector<std::string> arms;

    // Which of `paths` this is, adding it if it is new.
    size_t path_of(const Path &path) {
        for (size_t i = 0; i < paths.size(); i++) {
            if (ir::equals(paths[i].chain, path.chain)) {
                return i;
            }
        }
        paths.push_back(path);
        return paths.size() - 1;
    }

    // The indices of one path, as a type: the one index itself, or a tuple.
    static ir::Type indices_t(const Path &path) {
        if (path.index_ts.size() == 1) {
            return path.index_ts[0];
        }
        return ir::Tuple_t::make(path.index_ts);
    }

    void decide(size_t number, ir::TypeMap &types) {
        internal_assert(!paths.empty());
        if (paths.size() == 1) {
            reference_t = indices_t(paths[0]);
            return;
        }
        // One arm per way the element is reached, holding that way's
        // indices. Named as the ADT lowering needs: a variant's name is
        // unique across the program.
        const std::string name = "_ref" + std::to_string(number);
        ir::ADT_t::Variants variants;
        for (size_t i = 0; i < paths.size(); i++) {
            ir::Struct_t::Map fields;
            for (size_t k = 0; k < paths[i].index_ts.size(); k++) {
                fields.push_back(
                    ir::TypedVar{index_field(k), paths[i].index_ts[k]});
            }
            arms.push_back(name + "_" + std::to_string(i));
            variants.push_back(ir::Struct_t::make(arms.back(), fields));
        }
        reference_t = ir::ADT_t::make(name, std::move(variants));
        internal_assert(!types.contains(name))
            << "Cannot name the reference type " << name
            << ": something is already called that.";
        types[name] = reference_t;
    }

    // A reference along path `i` made of `indices`.
    ir::Expr make(size_t i, std::vector<ir::Expr> indices) const {
        if (paths.size() > 1) {
            return ir::Construct::make(reference_t, arms[i],
                                       std::move(indices));
        }
        if (indices.size() == 1) {
            return indices[0];
        }
        return ir::Build::make(reference_t, std::move(indices));
    }

    // The element `ref` refers to: each path's chain with the indices the
    // reference carries put back in place of the placeholders.
    ir::Expr read(const ir::Expr &ref) const {
        auto along = [&](size_t i, const auto &index_k) {
            std::map<std::string, ir::Expr> values;
            for (size_t k = 0; k < paths[i].index_ts.size(); k++) {
                values[placeholder_name(k)] = index_k(k);
            }
            return replace(values, paths[i].chain);
        };
        if (paths.size() > 1) {
            std::vector<ir::MatchExpr::Arm> match_arms;
            for (size_t i = 0; i < paths.size(); i++) {
                ir::Expr arm = ir::Unwrap::make(i, ref);
                match_arms.push_back(ir::MatchExpr::Arm{
                    arms[i], along(i, [&](size_t k) {
                        return ir::Access::make(index_field(k), arm);
                    })});
            }
            return ir::MatchExpr::make(ref, std::move(match_arms));
        }
        return along(0, [&](size_t k) {
            if (paths[0].index_ts.size() == 1) {
                return ref;
            }
            return ir::Extract::make(ref, static_cast<int>(k));
        });
    }
};

using Shapes = std::map<ir::Type, Shape, ir::TypeLessThan>;

// Every place a reference is made, by the type of reference.
struct CollectSites : public ir::Visitor {
    Shapes &shapes;
    const ir::ExternList &externs;

    CollectSites(Shapes &shapes, const ir::ExternList &externs)
        : shapes(shapes), externs(externs) {}

    using ir::Visitor::visit;

    void visit(const ir::RefTo *node) override {
        shapes[node->type].path_of(take_apart(node->place, externs).path);
        ir::Visitor::visit(node);
    }
};

// Spells every reference as its shape says, and everything typed as one too.
struct Resolve : public ir::Mutator {
    const Shapes &shapes;
    const ir::ExternList &externs;

    Resolve(const Shapes &shapes, const ir::ExternList &externs)
        : shapes(shapes), externs(externs) {}

    const Shape &shape_of(const ir::Type &type) const {
        const auto found = shapes.find(type);
        internal_assert(found != shapes.end())
            << "No reference of type " << type
            << " is ever made, so there is nothing to say what one is made "
            << "of.";
        return found->second;
    }

    ir::Type mutate(const ir::Type &type) override {
        ir::Type inner = ir::Mutator::mutate(type);
        if (inner.is<ir::ElementRef_t>()) {
            return shape_of(inner).reference_t;
        }
        return inner;
    }

    using ir::Mutator::mutate;

    ir::Expr visit(const ir::RefTo *node) override {
        TakenApart apart = take_apart(mutate(node->place), externs);
        Shape shape = shape_of(node->type);
        const size_t i = shape.path_of(apart.path);
        internal_assert(i < shape_of(node->type).paths.size())
            << "A reference was made along a path no site was seen to take: "
            << ir::Expr(node);
        return shape.make(i, std::move(apart.indices));
    }

    ir::Expr visit(const ir::Deref *node) override {
        if (!node->expr.type().is<ir::ElementRef_t>()) {
            return ir::Mutator::visit(node);
        }
        internal_assert(!node->mask.defined())
            << "A masked read through a reference to a stored element: "
            << ir::Expr(node);
        const Shape &shape = shape_of(node->expr.type());
        return shape.read(mutate(node->expr));
    }

    // The base Mutator leaves the types inside an expression as it found
    // them; these are the ones that carry a type of their own, the same set
    // Lower/Layouts.cpp and Lower/ADTs.cpp reach for the same reason.
    ir::Expr visit(const ir::Var *node) override {
        ir::Type type = mutate(node->type);
        if (type.same_as(node->type)) {
            return node;
        }
        return ir::Var::make(std::move(type), node->name);
    }

    std::pair<ir::WriteLoc, bool>
    mutate_writeloc(const ir::WriteLoc &loc) override {
        ir::Type base_type = mutate(loc.base_type);
        bool not_changed = base_type.same_as(loc.base_type);
        ir::WriteLoc new_loc(loc.base, std::move(base_type));
        for (const auto &value : loc.accesses) {
            if (const ir::Expr *expr = std::get_if<ir::Expr>(&value)) {
                ir::Expr new_value = mutate(*expr);
                not_changed = not_changed && new_value.same_as(*expr);
                new_loc.add_index_access(std::move(new_value));
            } else {
                new_loc.add_struct_access(std::get<std::string>(value));
            }
        }
        return {std::move(new_loc), not_changed};
    }

    ir::Expr visit(const ir::Build *node) override {
        std::vector<ir::Expr> values;
        values.reserve(node->values.size());
        bool not_changed = true;
        for (const ir::Expr &value : node->values) {
            ir::Expr new_value = mutate(value);
            not_changed = not_changed && new_value.same_as(value);
            values.push_back(std::move(new_value));
        }
        ir::Type type = mutate(node->type);
        if (not_changed && type.same_as(node->type)) {
            return node;
        }
        return ir::Build::make(std::move(type), std::move(values));
    }

    ir::Expr visit(const ir::Cast *node) override {
        ir::Expr value = mutate(node->value);
        ir::Type type = mutate(node->type);
        if (value.same_as(node->value) && type.same_as(node->type)) {
            return node;
        }
        return ir::Cast::make(std::move(type), std::move(value), node->mode);
    }

    ir::Expr visit(const ir::Lambda *node) override {
        std::vector<ir::TypedVar> args;
        args.reserve(node->args.size());
        bool not_changed = true;
        for (const ir::TypedVar &arg : node->args) {
            ir::Type type = mutate(arg.type);
            not_changed = not_changed && type.same_as(arg.type);
            args.push_back(ir::TypedVar{arg.name, std::move(type)});
        }
        ir::Expr value = mutate(node->value);
        if (not_changed && value.same_as(node->value)) {
            return node;
        }
        return ir::Lambda::make(std::move(args), std::move(value));
    }
};

} // namespace

ir::Program LowerElementReferences::run(ir::Program program,
                                        const CompilerOptions &options) const {
    // Every site first: what a reference is made of depends on every way the
    // element it refers to is reached, and the type is threaded through the
    // traversal ahead of any site.
    Shapes shapes;
    CollectSites collect(shapes, program.externs);
    for (const auto &[_, func] : program.funcs) {
        func->body.accept(&collect);
    }
    if (shapes.empty()) {
        return program;
    }
    size_t number = 0;
    for (auto &[_, shape] : shapes) {
        shape.decide(number++, program.types);
    }

    Resolve resolve(shapes, program.externs);
    for (auto &[_, type] : program.types) {
        type = resolve.mutate(type);
    }
    for (auto &[_, type] : program.externs) {
        type = resolve.mutate(type);
    }
    for (auto &[_, func] : program.funcs) {
        for (auto &arg : func->args) {
            arg.type = resolve.mutate(arg.type);
            arg.default_value = resolve.mutate(arg.default_value);
        }
        func->ret_type = resolve.mutate(func->ret_type);
        func->body = resolve.mutate(func->body);
    }
    return program;
}

} // namespace lower
} // namespace bonsai
