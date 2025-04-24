#include "IR/ValidateLayout.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"

#include <set>

namespace bonsai {
namespace ir {

using Path = std::map<std::string, Type>;

namespace {

std::vector<Path> get_paths(const Layout &layout) {
    struct GetPaths : public Visitor {
        std::vector<Path> paths = {{}}; // start with one empty path.

        void visit(const Name *node) override {
            for (auto &path : paths) {
                const auto [_, inserted] =
                    path.try_emplace(node->name, node->type);
                internal_assert(inserted); // TODO: descriptive error message of
                                           // duplicate field in path.
            }
        }

        void visit(const Pad *node) override {}

        void visit(const Split *node) override {
            // All paths are split.
            std::vector<Path> old_paths = std::move(paths);
            // can't prealloc old_paths.size() * node->arms.size()
            // due to possible inner splits!
            std::vector<Path> split_paths;
            for (const auto &arm : node->arms) {
                paths = {{}};
                arm.layout.accept(this);
                std::vector<Path> new_paths = std::move(paths);

                // Yes, this is exponential explosion.
                // As expected for nested splits.
                for (const auto &op : old_paths) {
                    for (const auto &np : new_paths) {
                        Path together = op;
                        for (const auto &[name, type] : np) {
                            const auto [_, inserted] =
                                together.try_emplace(name, type);
                            internal_assert(
                                inserted); // TODO: descriptive error message of
                                           // duplicate field in path.
                        }
                        split_paths.emplace_back(std::move(together));
                    }
                }
            }
            paths = std::move(split_paths);
        }

        // default behavior is good.
        // void visit(const Chain *node) override {}
        // void visit(const Group *node) override {}

        void visit(const Materialize *node) override {
            for (auto &path : paths) {
                const auto [_, inserted] =
                    path.try_emplace(node->name, node->value.type());
                internal_assert(inserted); // TODO: descriptive error message of
                                           // duplicate field in path.
            }
        }
    };

    GetPaths getter;
    layout.accept(&getter);
    return getter.paths;
}

bool equal_paths(const Path &p0, const Path &p1) {
    // TODO: must be a std:: library function for this
    if (p0.size() != p1.size()) {
        return false;
    }
    for (const auto &[name, type] : p0) {
        const auto &iter = p1.find(name);
        if (iter == p1.cend()) {
            return false;
        }
        internal_assert(
            equals(type, iter->second)); // TODO: error message for paths with
                                         // same names but mismatching types?
    }
    return true;
}

bool valid_path(const Path &path, const BVH_t::Node &node) {
    for (const auto &param : node.fields()) {
        const auto &iter = path.find(param.name);
        if (iter == path.cend()) {
            return false;
        }
        if (!equals(param.type, iter->second)) {
            if (param.type.is<ir::Ref_t>() && (iter->second.is_int_or_uint() ||
                                               iter->second.is_int_tuple())) {
                // TODO: figure out how to validate references as indexes into
                // groups!
                continue;
            }
            return false;
        }
    }
    return true;
}

struct ValidateSplits : public Visitor {
    // void visit(const Name *node) override {}
    // void visit(const Pad *node) override {}
    TypeMap defined;

    void visit(const Split *node) override {
        const auto &iter = defined.find(node->field);
        internal_assert(iter != defined.cend())
            << "Switch does not have access to field: " << node->field;
        internal_assert(iter->second.is_int_or_uint())
            << "Switch on non-integer field: " << node->field;
        TypeMap parent = defined;
        for (const auto &arm : node->arms) {
            arm.layout.accept(this);
            defined = parent; // erase arm scope.
        }
    }

    void visit(const Chain *node) override {
        // Two pass: gather all fields, then check nested layouts.
        TypeMap parent = defined;
        for (const auto &layout : node->layouts) {
            if (const Name *name = layout.as<Name>()) {
                const auto [_, inserted] =
                    defined.try_emplace(name->name, name->type);
                internal_assert(inserted)
                    << "Name: " << name->name << " is duplicated in layout";
            } else if (const Materialize *mat = layout.as<Materialize>()) {
                const auto [_, inserted] =
                    defined.try_emplace(mat->name, mat->value.type());
                internal_assert(inserted)
                    << "Name: " << name->name << " is duplicated in layout";
            }
        }

        for (const auto &layout : node->layouts) {
            layout.accept(this);
        }

        defined = parent;
    }
    // void visit(const Group *node) override {}
    // void visit(const Materialize *node) override {}
};

void validate_splits(const Layout &layout) {
    ValidateSplits validator;
    layout.accept(&validator);
}

} // namespace

std::map<std::string, Path> validate_layout(const Layout &layout,
                                            const Type &bvh_t) {
    internal_assert(layout.defined() && bvh_t.defined())
        << "Cannot validate with undefined layout or bvh_t: " << layout << "\n"
        << bvh_t;
    const BVH_t *bvh_node = bvh_t.as<BVH_t>();
    internal_assert(bvh_node)
        << "Cannot validate layout of non-BVH_t: " << bvh_t;

    // Assert all Split fields are accessible at Split level.
    validate_splits(layout);

    std::vector<Path> paths = get_paths(layout);
    // for (const auto &path : paths) {
    //     std::cout << "path {\n";
    //     for (const auto &[name, type] : path) {
    //         std::cout << "  " << name << " : " << type << "\n";
    //     }
    //     std::cout << "}\n";
    // }
    internal_assert(paths.size() == bvh_node->nodes.size())
        << "Layout: " << layout << "\nhas " << paths.size()
        << " paths. BVH type: " << bvh_t << "\nhas " << bvh_node->nodes.size()
        << " node options.";

    // Check paths are unique.
    // TODO: must be a faster way than n^2, but n is small (probably) so this is
    // fine for now.
    for (size_t i = 0; i < paths.size(); i++) {
        const Path &pi = paths[i];
        for (size_t j = i + 1; j < paths.size(); j++) {
            const Path &pj = paths[j];
            internal_assert(
                !equal_paths(pi, pj)); // TODO: error message for equal paths?
        }
    }

    // TODO(ajr): use Split::Arm::name.

    std::map<std::string, Path> pathmap;
    // Check each node has one equivalent path!
    for (const auto &node : bvh_node->nodes) {
        Path node_path;
        for (auto &path : paths) {
            if (!path.empty() && valid_path(path, node)) {
                internal_assert(node_path.empty())
                    << "Ambiguous path for node: " << node.name()
                    << " in layout: " << layout;
                node_path = std::move(path);
            }
        }
        internal_assert(!node_path.empty())
            << "No path for node: " << node.name() << " in layout: " << layout;
        pathmap[node.name()] = std::move(node_path);
    }
    return pathmap;
}

} // namespace ir
} // namespace bonsai
