#include "IR/ValidateLayout.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Log.h"
#include "Utils.h"

#include <set>
#include <unordered_set>

namespace bonsai {
namespace ir {

std::ostream &operator<<(std::ostream &os, const std::vector<Path> &paths) {
    os << "\n[";
    for (int i = 0, e = paths.size(); i < e; ++i) {
        os << paths[i];
        if (i + 1 == e) {
            continue;
        }
        os << ",\n";
    }
    os << "]\n";
    return os;
}

namespace {

// A map from group name to Group.
using GroupMap = std::map<std::string, Member>;

std::vector<Path> get_paths(const ir::Layout &layout,
                            const GroupMap &group_map) {
    class GetPaths : public Visitor {
      public:
        GetPaths(const GroupMap &group_map) : group_map(group_map) {}

        std::vector<Path> get_paths() const { return paths; }

      private:
        const GroupMap &group_map;
        std::vector<Path> paths = {{}}; // start with one empty path.

        void visit(const Field *node) override {
            for (auto &path : paths) {
                const auto &[_, inserted] =
                    path.try_emplace(node->name, node->type);
                internal_assert(inserted)
                    << "field found twice in the same path: " << node->name
                    << " : " << node->type << "\n"
                    << path;
            }
        }

        void visit(const Pad *node) override {}

        void visit(const Lookup *node) override {
            auto it = group_map.find(node->group_name);
            internal_assert(it != group_map.end())
                << "lookup to non-existent group: " << node->group_name;
            const ir::Group *group = (it->second).as<Group>();
            internal_assert(group);
            internal_assert(group->type == Group::Type::Indirect)
                << "unexpected lookup in direct group: " << it->second;
            group->inner.accept(this);
        }

        void visit(const Split *node) override {
            std::vector<Path> old_paths = std::move(paths);
            std::vector<Path> split_paths; // All paths are split.
            for (const ir::Arm &arm : node->arms) {
                paths = {{}};
                arm.member.accept(this);
                std::vector<Path> new_paths = std::move(paths);
                // This is an (expected) exponential explosion.
                for (const Path &old_path : old_paths) {
                    for (const Path &new_path : new_paths) {
                        Path together = old_path;
                        for (const auto &[name, type] : new_path) {
                            const auto [_, inserted] =
                                together.try_emplace(name, type);
                            internal_assert(inserted)
                                << "field found twice in same path: " << name
                                << " : " << type;
                        }
                        split_paths.emplace_back(std::move(together));
                    }
                }
            }
            paths = std::move(split_paths);
        }

        void visit(const Group *node) override {
            switch (node->type) {
            case Group::Type::Direct:
                node->inner.accept(this);
                break;
            case Group::Type::Indirect:
                break; // This can only be accessed via a lookup.
            }
        }

        void visit(const Materialize *node) override {
            for (auto &path : paths) {
                const auto [_, inserted] =
                    path.try_emplace(node->name, node->value.type());
                internal_assert(inserted)
                    << "field found twice in same path: " << node->name;
            }
        }
    };

    GetPaths gp(group_map);
    layout.body.accept(&gp);
    return gp.get_paths();
}

// Returns whether this path is valid for each of the variant's parameters.
bool is_valid_path(const Path &path, const BVH_t::Variant &variant,
                   const ir::Layout &layout) {
    for (auto &parameter : variant.fields()) {
        Type parameter_type = parameter.type;
        const auto &it = path.find(parameter.name);
        if (it == path.cend()) {
            // If no path is found for this parameter, check to see if the
            // parameter exists in the root.
            auto it =
                std::find_if(layout.root.begin(), layout.root.end(),
                             [&](const ir::Argument &arg) {
                                 return arg.name == parameter.name &&
                                        ir::equals(arg.type, parameter_type);
                             });
            if (it != layout.root.end()) {
                continue;
            }
            return false;
        }
        const Type &concrete_type = it->second;
        if (equals(parameter_type, concrete_type)) {
            continue;
        }
        if (parameter_type.is_iterable()) {
            if (!concrete_type.is_iterable()) {
                continue; // Type mismatch.
            }
            if (!ir::equals(parameter_type.size(), concrete_type.size())) {
                continue; // Size mismatch.
            }
            parameter_type = parameter_type.element_of();
        }
        if (parameter_type.is<ir::Ref_t>()) {
            if (concrete_type.is_int_or_uint() ||
                concrete_type.is_int_tuple()) {
                // References to children in the variant may be represented by a
                // unique index into an array.
                continue;
            }
        }
        return false;
    }
    return true;
}

GroupMap get_group_map(const Layout &layout) {
    struct GetGroupMap : Visitor {
        void visit(const Group *node) override {
            const auto [_, inserted] = map.insert({node->name, node});
            internal_assert(inserted)
                << "unexpected duplicate group name: " << node->name;
            node->inner.accept(this); // visit nested groups.
        };
        GroupMap map;
    };

    GetGroupMap ggm;
    layout.body.accept(&ggm);
    return ggm.map;
}

// Returns whether these are equivalent paths.
bool equal_paths(const Path &p0, const Path &p1) {
    if (p0.size() != p1.size()) {
        return false;
    }
    return std::equal(
        p0.begin(), p0.end(), p1.begin(), [](const auto &a, const auto &b) {
            return a.first == b.first && equals(a.second, b.second);
        });
}

std::map<std::string, Path> get_unambiguous_paths(const ir::Layout &layout) {
    GroupMap group_map = get_group_map(layout);
    std::vector<Path> paths = get_paths(layout, group_map);

    // Check paths are unique.
    for (size_t i = 0; i < paths.size(); ++i) {
        const Path &pi = paths[i];
        for (size_t j = i + 1; j < paths.size(); ++j) {
            const Path &pj = paths[j];
            internal_assert(!equal_paths(pi, pj))
                << "unexpected equal paths for " << layout.name << ": " << pi
                << " vs " << pj;
        }
    }

    const BVH_t *bvh_t = layout.type.as<BVH_t>();
    internal_assert(bvh_t) << layout.type;

    std::map<std::string, Path> map;
    // Verify each node has one equivalent path.
    for (const BVH_t::Variant &variant : bvh_t->variants) {
        const std::string &variant_name = variant.name();
        Path path_to_variant;
        for (const Path &path : paths) {
            if (!path.empty() && is_valid_path(path, variant, layout)) {
                internal_assert(path_to_variant.empty())
                    << "two or more paths found for variant: " << variant_name
                    << " in layout: `" << layout.name << "`";
                path_to_variant = path;
            }
        }
        internal_assert(!path_to_variant.empty())
            << "no path found for variant: " << variant_name << " in layout: `"
            << layout.name << "`";
        map[variant_name] = std::move(path_to_variant);
    }
    return map;
}

// Represents the range [min, max].
template <typename T>
struct Range {
    T min;
    T max;

    Range(T min, T max) : min(min), max(max) {}

    bool operator<(const Range &other) const {
        return min < other.min || (min == other.min && max < other.max);
    }

    bool overlaps(const Range &other) const {
        return !(max < other.min || other.max < min);
    }

    bool is_empty() const { return min > max; }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Range<T> &range) {
    os << '[' << range.min << ", " << range.max << ']';
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<Range<T>> &ranges) {
    for (int i = 0, e = ranges.size(); i < e; ++i) {
        os << ranges[i];
        if (i + 1 == e)
            continue;
        os << ", ";
    }
    return os;
}

template <typename T>
std::vector<Range<T>> arm_to_ranges(const Arm &arm) {
    if (!arm.value.has_value()) {
        return {};
    }

    const T tmin = std::numeric_limits<T>::min();
    const T tmax = std::numeric_limits<T>::max();
    T val = static_cast<T>(*arm.value);
    switch (arm.comparator) {
    case Arm::Comparator::EQ:
        if (tmin <= val && val <= tmax) {
            return {Range(val, val)};
        }
        // value is outside the given bounds.
        return {};

    case Arm::Comparator::NE: {
        // Everything except the specific value.
        std::vector<Range<T>> ranges;
        if (val > tmin) {
            ranges.emplace_back(tmin, val - 1);
        }
        if (val < tmax) {
            ranges.emplace_back(val + 1, tmax);
        }
        return ranges;
    }
    case Arm::Comparator::GT:
        if (val < tmax) {
            return {Range(std::max<T>(val + 1, tmin), tmax)};
        }
        return {};

    case Arm::Comparator::GE:
        if (val <= tmax) {
            return {Range(std::max(val, tmin), tmax)};
        }
        return {};

    case Arm::Comparator::LT:
        if (val > tmin) {
            return {Range(tmin, std::min<T>(val - 1, tmax))};
        }
        return {};

    case Arm::Comparator::LE:
        if (val >= tmin) {
            return {Range(tmin, std::min(val, tmax))};
        }
        return {};
    }
}

template <typename T>
bool is_exclusive(const std::vector<std::vector<Range<T>>> &ranges) {
    for (size_t i = 0, e = ranges.size(); i < e; ++i) {
        for (size_t j = i + 1; j < e; ++j) {
            for (const Range<T> &r1 : ranges[i]) {
                for (const Range<T> &r2 : ranges[j]) {
                    if (!r1.overlaps(r2)) {
                        continue;
                    }
                    internal_error << "split arms range overlap: " << r1 << ", "
                                   << r2;
                }
            }
        }
    }
    return true;
}

template <typename T>
bool is_exhaustive(const Range<T> &range, T tmin, T tmax) {
    return range.min == tmin && range.max == tmax;
}

template <typename T>
std::vector<Range<T>> merge_ranges(std::vector<Range<T>> ranges) {
    if (ranges.empty())
        return {};
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(),
                                [](const Range<T> &r) { return r.is_empty(); }),
                 ranges.end());

    if (ranges.empty())
        return {};

    std::sort(ranges.begin(), ranges.end());
    std::vector<Range<T>> merged;
    merged.push_back(ranges.front());

    for (int32_t i = 1; i < ranges.size(); ++i) {
        Range<T> &last = merged.back();
        const Range<T> &current = ranges[i];

        // Check if ranges can be merged (overlapping or adjacent).
        if (current.min <= last.max + 1) {
            last.max = std::max(last.max, current.max);
        } else {
            merged.push_back(current);
        }
    }

    return merged;
}

bool contains_wildcard(const std::vector<Arm> &arms) {
    return std::any_of(arms.begin(), arms.end(),
                       [](const Arm &arm) { return arm.is_wildcard(); });
}

template <typename T>
void validate_arms(const Split &split) {
    const std::vector<Arm> &arms = split.arms;
    internal_assert(!arms.empty());
    internal_assert(
        std::count_if(arms.begin(), arms.end(),
                      [](const Arm &arm) { return arm.is_wildcard(); }) <= 1)
        << "[unexpected] two or more wildcard arms";
    const T tmin = std::numeric_limits<T>::min(),
            tmax = std::numeric_limits<T>::max();

    std::vector<std::vector<Range<T>>> arm_ranges;
    std::vector<Range<T>> all_ranges;

    for (const Arm &arm : arms) {
        std::vector<Range<T>> ranges = arm_to_ranges<T>(arm);
        arm_ranges.push_back(ranges);
        all_ranges.insert(all_ranges.end(), ranges.begin(), ranges.end());
    }

    // Check mutual exclusivity first.
    internal_assert(is_exclusive(arm_ranges));
    // Merge all ranges and check exhaustiveness.
    std::vector<Range<T>> merged_ranges = merge_ranges<T>(all_ranges);
    bool exhaustive = merged_ranges.size() == 1 &&
                      is_exhaustive(merged_ranges.front(), tmin, tmax);
    if (exhaustive && contains_wildcard(arms)) {
        internal_error
            << "[unexpected] exhaustive range provided with a wildcard.";
    }
    if (!exhaustive && !contains_wildcard(arms)) {
        internal_error << "split arms are not collectively exhaustive: "
                       << merged_ranges;
    }
}

// TODO(cgyurgyik): This needs to handle arbitrary nesting.
struct ValidateSplits : public Visitor {
    TypeMap defined;

    void visit(const Split *node) override {
        ir::Expr expr = node->expr;
        internal_assert(expr.type().is_scalar()) << expr.type();
        if (expr.type().is<Bool_t>()) {
            validate_arms<bool>(*node);
        } else if (expr.type().is<Int_t>()) {
            switch (expr.type().bits()) {
            case 1:
                validate_arms<bool>(*node);
                break;
            case 16:
                validate_arms<int16_t>(*node);
                break;
            case 32:
                validate_arms<int32_t>(*node);
                break;
            case 64:
                validate_arms<int64_t>(*node);
                break;
            default:
                internal_error << "[unimplemented] split field count: "
                               << expr.type();
            }
        } else if (expr.type().is<UInt_t>()) {
            switch (expr.type().bits()) {
            case 1:
                validate_arms<bool>(*node);
                break;
            case 16:
                validate_arms<uint16_t>(*node);
                break;
            case 32:
                validate_arms<uint32_t>(*node);
                break;
            case 64:
                validate_arms<uint64_t>(*node);
                break;
            default:
                internal_error << "[unimplemented] split field bit count: "
                               << expr.type();
            }
        } else {
            internal_error << "[unimplemented] split field type: "
                           << expr.type();
        }

        auto it = defined.find(node->field_name());
        internal_assert(it != defined.cend())
            << "Split does not have access to field: `" << node->expr
            << "`. Currently defined fields:\n"
            << defined;
        TypeMap parent = defined;
        for (const ir::Arm &arm : node->arms) {
            arm.member.accept(this);
            defined = parent; // erase arm scope.
        }
    }

    void visit(const Chain *node) override {
        // Two pass: gather all fields, then check nested members.
        TypeMap parent = defined;
        for (const auto &member : node->members) {
            if (const Field *field = member.as<Field>()) {
                const auto [_, inserted] =
                    defined.try_emplace(field->name, field->type);
                internal_assert(inserted)
                    << "field: " << field->name << " is duplicated in member";
            } else if (const Materialize *materialization =
                           member.as<Materialize>()) {
                const auto [_, inserted] = defined.try_emplace(
                    materialization->name, materialization->value.type());
                internal_assert(inserted)
                    << "materialization: " << materialization->name
                    << " is duplicated in member";
            }
        }

        for (const auto &member : node->members) {
            member.accept(this);
        }
        defined = parent;
    }
    void visit(const Group *node) override {
        if (node->index.defined()) {
            const auto *v = node->index.as<ir::Var>();
            internal_assert(v) << node->index;
            // (duplicate index use is fine, e.g., in the case of SoA).
            defined.insert({v->name, v->type});
        }

        TypeMap parent = defined;
        node->inner.accept(this);
        defined = std::move(parent);
    }
    // void visit(const Materialize *node) override {}
};

void validate_splits(const ir::Layout &layout) {
    ValidateSplits validator;
    layout.body.accept(&validator);
}

// Validates all indirect groups are defined at the root.
void validate_indirect_groups(const ir::Layout &layout) {
    // Collect all indirect groups defined anywhere in the layout.
    struct CollectAllIndirectGroups : public Visitor {
        std::set<std::string> indirect_groups;
        void visit(const Group *node) override {
            if (node->type == Group::Type::Indirect) {
                indirect_groups.insert(node->name);
            }
            node->inner.accept(this);
        }

        void visit(const Chain *node) override {
            for (const auto &member : node->members) {
                member.accept(this);
            }
        }
        void visit(const Split *node) override {
            for (const ir::Arm &arm : node->arms) {
                arm.member.accept(this);
            }
        }
    };
    struct CollectRootIndirectGroups : public Visitor {
        std::set<std::string> indirect_groups;
        bool at_root_level = true;

        void visit(const Group *node) override {
            if (node->type == Group::Type::Indirect && at_root_level) {
                indirect_groups.insert(node->name);
            }
            bool was_at_root = at_root_level;
            at_root_level = false;
            node->inner.accept(this);
            at_root_level = was_at_root;
        }
        void visit(const Chain *node) override {
            for (const ir::Member &member : node->members) {
                member.accept(this);
            }
        }

        void visit(const Split *node) override {
            bool was_at_root = at_root_level;
            at_root_level = false;
            for (const ir::Arm &arm : node->arms) {
                arm.member.accept(this);
            }
            at_root_level = was_at_root;
        }
    };

    CollectAllIndirectGroups all;
    layout.body.accept(&all);
    CollectRootIndirectGroups root;
    layout.body.accept(&root);

    // Assert that all indirect groups are defined at root level.
    for (const std::string &indirect_group : all.indirect_groups) {
        internal_assert(root.indirect_groups.contains(indirect_group))
            << "indirect group: `" << indirect_group
            << "` is not defined at root level in layout `" << layout.name
            << "`";
    }
}

void validate_root(const Layout &layout) {
    std::unordered_set<std::string> names;
    for (const ir::Argument &arg : layout.root) {
        internal_assert(!names.contains(arg.name))
            << "unexpected duplicate argument name in root: " << layout.root;
        names.insert(arg.name);
        internal_assert(!arg.mutating)
            << "unexpected mutable argument in root: " << arg;
    }
}

void validate_tcd(const ir::Layout &layout) {
    struct ParentFieldCollector : public Visitor {
        void visit(const Access *node) override {
            if (const Struct_t *struct_type =
                    node->value.type().as<Struct_t>()) {
                if (struct_type->name == "parent_t") {
                    parent_fields.insert(node->field);
                }
            }
        }

        std::set<std::string> parent_fields;
    };

    ParentFieldCollector pfc;
    layout.body.accept(&pfc);
    for (const ir::Argument &arg : layout.root) {
        if (!pfc.parent_fields.contains(arg.name)) {
            // Root arguments may also be indexes used by groups.
            // We can safely ignore these.
            continue;
        }
        internal_assert(arg.default_value.defined())
            << "tree-carried dependency: `" << arg.name
            << "` must define a base case";
    }
}

int32_t count_children(const ir::Layout &layout,
                       const BVH_t::Variant &variant) {
    const ir::BVH_t *bvh = layout.type.as<ir::BVH_t>();
    internal_assert(bvh) << layout.type;
    ir::Type tree_reference = ir::Ref_t::make(bvh->name);

    int32_t count = 0;
    for (const ir::TypedVar &field : variant.fields()) {
        const Type &field_type = field.type;
        if (ir::equals(field_type, tree_reference)) {
            ++count;
            continue;
        }
        if (field_type.is_iterable() &&
            ir::equals(field_type.element_of(), tree_reference)) {
            std::optional<uint32_t> size =
                get_constant_value(field_type.size());
            internal_assert(size.has_value())
                << "cannot determine children count for "
                   "non-constant array size in field: "
                << field.name << " : " << field_type;
            count += *size;
        }
    }
    return count;
}

int32_t count_variant_fields(const ir::BVH_t::Variant &variant,
                             const BVH_t::Volume &volume, int32_t child_count,
                             uint32_t index) {
    const auto *volume_t = volume.struct_type.as<ir::Struct_t>();
    const std::string &initializer = volume.initializers[index];
    ir::Type volume_type = volume_t->fields[index].type;
    ir::Type variant_type =
        ir::Access::make(initializer,
                         ir::Var::make(variant.struct_type, variant.name()))
            .type();

    if (ir::equals(volume_type, variant_type)) {
        return child_count;
    }
    if (variant_type.is_iterable() &&
        equals(variant_type.element_of(), volume_type)) {
        std::optional<uint32_t> size = get_constant_value(variant_type.size());
        internal_assert(size.has_value()) << variant_type.size();
        return *size;
    }
    internal_error << "[unimplemented] variant fields that aren't T or T[]: "
                   << variant_type;
}

void validate_volumes(const ir::Layout &layout) {
    const BVH_t *bvh_t = layout.type.as<BVH_t>();
    internal_assert(bvh_t) << layout.type;
    for (uint32_t i = 0; i < bvh_t->variants.size(); i++) {
        const BVH_t::Variant &variant = bvh_t->variants[i];
        if (!variant.volume.has_value()) {
            continue;
        }
        const BVH_t::Volume &volume = *variant.volume;
        // Verify initializers can actually be traced back to the variants.
        for (size_t i = 0, e = volume.initializers.size(); i < e; ++i) {
            const std::string &name = volume.initializers[i];
            auto it = std::find_if(
                variant.fields().begin(), variant.fields().end(),
                [&](const ir::TypedVar &p) { return p.name == name; });
            internal_assert(it != variant.fields().end())
                << "could not find field for initializer: " << name << '\n';
        }

        switch (volume.bound_type) {
        case BVH_t::Volume::BoundType::Enclosing:
            continue; // skip
        case BVH_t::Volume::BoundType::Childwise:
            break;
        }
        const int32_t child_count = count_children(layout, variant);
        if (child_count == 0) {
            continue;
        }
        const Struct_t *volume_t = volume.struct_type.as<Struct_t>();
        internal_assert(volume_t) << volume.struct_type;
        for (uint32_t i = 0, e = volume_t->fields.size(); i < e; ++i) {
            int32_t field_count =
                count_variant_fields(variant, volume, child_count, i);
            internal_assert(field_count == child_count)
                << "mismatch in child count: " << child_count
                << " and field count: " << field_count << " for volume field: `"
                << volume_t->fields[i] << "`, initialized by variant field: `"
                << volume.initializers[i] << '`';
        }
    }
}

void validate_type(const ir::Layout &layout) {
    const ir::Type &type = layout.type;
    const auto *bvh_t = type.as<ir::BVH_t>();
    internal_assert(bvh_t) << "expected ADT type for `" << layout.name
                           << "`, received: " << type;
    internal_assert(std::any_of(
        bvh_t->variants.begin(), bvh_t->variants.end(),
        [&](const BVH_t::Variant &variant) {
            return std::find_if(
                       variant.fields().begin(), variant.fields().end(),
                       [&](const ir::TypedVar &v) {
                           return ir::equals(v.type, bvh_t->primitive) ||
                                  (v.type.is_iterable() &&
                                   ir::equals(v.type.element_of(),
                                              bvh_t->primitive));
                       }) != variant.fields().end();
        }))
        << "primitive type: `" << bvh_t->primitive
        << "` not found in BVH type: " << type;
    ;
}

} // namespace

// Performs well-formedness checks of the layout, and returns a mapping from
// variant name to path for each variant in the layout type.
std::map<std::string, Path> validate_layout(const ir::Layout &layout) {
    internal_assert(layout.body.defined())
        << "undefined body in layout: `" << layout.name << "`";
    internal_assert(layout.type.defined())
        << "undefined type in layout: `" << layout.name << "`";

    validate_type(layout);
    validate_root(layout);
    validate_tcd(layout);
    validate_volumes(layout);
    validate_splits(layout);
    validate_indirect_groups(layout);
    return get_unambiguous_paths(layout);
}

} // namespace ir
} // namespace bonsai
