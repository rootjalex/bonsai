#include "IR/ValidateLayout.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Log.h"

#include <set>
#include <unordered_set>

// TODO(cgyurgyik): verify that if a variant is bounded childwise, the field
// counts are sensical, e.g., 4 children should have 4 bounding volumes.

// TODO(cgyurgyik): parent variables must be defined in the root.
namespace bonsai {
namespace ir {
namespace {
// A map from group name to Group.
using GroupMap = std::map<std::string, Member>;

// TODO: assert that all volumes only have initializers from
// parent.params or variant.params BVH_t::make asserts this. we should
// catch that failure, and report a backtrace.
std::vector<Path> get_paths(const Member &member, const GroupMap &group_map) {
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
                internal_assert(inserted); // TODO: descriptive error message of
                                           // duplicate field in path.
            }
        }

        // default behavior is good.
        // void visit(const Chain *node) override {}
    };

    GetPaths getter(group_map);
    member.accept(&getter);
    return getter.get_paths();
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

// Returns whether this path is valid for each of the variant's parameters.
bool is_valid_path(const Path &path, const BVH_t::Variant &variant) {
    for (auto &parameter : variant.fields()) {
        Type parameter_type = parameter.type;
        const auto &it = path.find(parameter.name);
        if (it == path.cend()) {
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

    const T type_min = std::numeric_limits<T>::min();
    const T type_max = std::numeric_limits<T>::max();
    T val = static_cast<T>(*arm.value);
    switch (arm.comparator) {
    case Arm::Comparator::EQ:
        if (type_min <= val && val <= type_max) {
            return {Range(val, val)};
        }
        // value is outside the given bounds.
        return {};

    case Arm::Comparator::NE: {
        // Everything except the specific value.
        std::vector<Range<T>> ranges;
        if (val > type_min) {
            ranges.emplace_back(type_min, val - 1);
        }
        if (val < type_max) {
            ranges.emplace_back(val + 1, type_max);
        }
        return ranges;
    }
    case Arm::Comparator::GT:
        if (val < type_max) {
            return {Range(std::max<T>(val + 1, type_min), type_max)};
        }
        return {};

    case Arm::Comparator::GE:
        if (val <= type_max) {
            return {Range(std::max(val, type_min), type_max)};
        }
        return {};

    case Arm::Comparator::LT:
        if (val > type_min) {
            return {Range(type_min, std::min<T>(val - 1, type_max))};
        }
        return {};

    case Arm::Comparator::LE:
        if (val >= type_min) {
            return {Range(type_min, std::min(val, type_max))};
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
bool is_exhaustive(const Range<T> &range, T type_min, T type_max) {
    return range.min == type_min && range.max == type_max;
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
    const T type_min = std::numeric_limits<T>::min(),
            type_max = std::numeric_limits<T>::max();

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
                      is_exhaustive(merged_ranges.front(), type_min, type_max);
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
    // void visit(const Field *node) override {}
    // void visit(const Pad *node) override {}
    TypeMap defined;

    void visit(const Split *node) override {
        ir::Expr expr = node->expr;
        internal_assert(expr.type().is_scalar()) << expr.type();
        // TODO(cgyurgyik): generalize this for n-bit fields.
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

void validate_splits(const Member &member) {
    ValidateSplits validator;
    member.accept(&validator);
}

GroupMap get_group_map(const Layout &layout) {
    struct GetGroupMap : Visitor {
        void visit(const Group *node) override {
            const auto [_, inserted] = map.insert({node->name, node});
            internal_assert(inserted)
                << "unexpected duplicate group name: " << node->name;

            // visit nested groups.
            node->inner.accept(this);
        };
        GroupMap map;
    };

    GetGroupMap ggm;
    layout.body.accept(&ggm);
    return ggm.map;
}

} // namespace

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

// Validates all indirect groups are defined at the root.
void validate_indirect_groups(const Layout &layout) {
    // Collect all indirect groups defined anywhere in the layout.
    struct GetAllIndirectGroups : public Visitor {
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

    struct GetRootIndirectGroups : public Visitor {
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

    GetAllIndirectGroups all;
    layout.body.accept(&all);

    GetRootIndirectGroups root;
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

std::map<std::string, Path> validate_layout(const Layout &layout) {
    const Member &body = layout.body;
    const Type &bvh_t = layout.type;

    internal_assert(body.defined() && bvh_t.defined())
        << "Cannot validate with undefined member or bvh_t: " << body << "\n"
        << bvh_t;
    const BVH_t *bvh_node = bvh_t.as<BVH_t>();
    internal_assert(bvh_node)
        << "Cannot validate member of non-BVH_t: " << bvh_t;

    // Assert root is well-formed.
    validate_root(layout);
    // Assert all Split fields are accessible at Split level.
    validate_splits(body);
    // Assert all Indirect groups are defined at root level.
    validate_indirect_groups(layout);

    GroupMap group_map = get_group_map(layout);
    std::vector<Path> paths = get_paths(body, group_map);
    // internal_assert(paths.size() == bvh_node->variants.size())
    //     << "layout `" << layout.name << "` has " << paths.size()
    //     << " paths, while the BVH ADT " << bvh_t << " has "
    //     << bvh_node->variants.size() << " node variants.";

    // Check paths are unique.
    for (size_t i = 0; i < paths.size(); ++i) {
        const Path &pi = paths[i];
        for (size_t j = i + 1; j < paths.size(); ++j) {
            const Path &pj = paths[j];
            internal_assert(!equal_paths(pi, pj))
                << "unexpected equal paths for " << layout << ": " << pi
                << " vs " << pj;
        }
    }

    // TODO(ajr): use Arm::name.
    std::map<std::string, Path> map;
    // Verify each node has one equivalent path.
    for (const BVH_t::Variant &variant : bvh_node->variants) {
        Path path_to_variant;
        for (const auto &path : paths) {
            if (!path.empty() && is_valid_path(path, variant)) {
                internal_assert(path_to_variant.empty())
                    << "ambiguous path for variant: " << variant.name()
                    << " in layout: `" << layout.name << "`";
                path_to_variant = path;
            }
        }
        internal_assert(!path_to_variant.empty())
            << "no path for variant: " << variant.name() << " in layout: `"
            << layout.name << "`";
        map[variant.name()] = std::move(path_to_variant);
    }
    return map;
}

} // namespace ir
} // namespace bonsai
