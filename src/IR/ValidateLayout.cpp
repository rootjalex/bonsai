#include "IR/ValidateLayout.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"

#include <set>

namespace bonsai {
namespace ir {

std::ostream &operator<<(std::ostream &os, const Path &path) {
    for (const auto &[name, type] : path) {
        os << "  " << name << " : " << type << "\n";
    }
    return os;
}

namespace {

std::vector<Path> get_paths(const Member &member) {
    struct GetPaths : public Visitor {
        std::vector<Path> paths = {{}}; // start with one empty path.

        void visit(const Field *node) override {
            for (auto &path : paths) {
                const auto [_, inserted] =
                    path.try_emplace(node->name, node->type);
                internal_assert(inserted)
                    << "field found twice in the same path: " << node->name
                    << " : " << node->type;
            }
        }

        void visit(const Pad *node) override {}

        void visit(const Split *node) override {
            // All paths are split.
            std::vector<Path> split_paths, old_paths = std::move(paths);
            for (const auto &arm : node->arms) {
                paths = {{}};
                arm.member.accept(this);
                std::vector<Path> new_paths = std::move(paths);

                // Yes, this is exponential explosion.
                // As expected for nested splits.
                for (const auto &old_path : old_paths) {
                    for (const auto &new_path : new_paths) {
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
    member.accept(&getter);
    return getter.paths;
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

// Represents a range [min, max].
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
        // TODO(cgyurgyik): if a wildcard is provided, the range must not be
        // exhaustive.
        return {};
    }

    const T type_min = std::numeric_limits<T>::min();
    const T type_max = std::numeric_limits<T>::max();
    T val = static_cast<T>(*arm.value);
    switch (arm.comparator) {
    case Arm::Comparator::EQ:
        if (val >= type_min && val <= type_max) {
            return {Range(val, val)};
        }
        return {}; // Empty - value is outside type bounds

    case Arm::Comparator::NE: {
        // Everything except the specific value
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
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
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

    const T type_min = std::numeric_limits<T>::min();
    const T type_max = std::numeric_limits<T>::max();

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
    std::vector<Range<T>> merged = merge_ranges<T>(all_ranges);
    bool exhaustive =
        merged.size() == 1 && is_exhaustive(merged.front(), type_min, type_max);
    if (exhaustive && contains_wildcard(arms)) {
        internal_error
            << "[unexpected] exhaustive range provided with a wildcard.";
    }
    if (!exhaustive && !contains_wildcard(arms)) {
        internal_error << "split arms are not collectively exhaustive: "
                       << merged;
    }
}

struct ValidateSplits : public Visitor {
    // void visit(const Field *node) override {}
    // void visit(const Pad *node) override {}
    TypeMap defined;

    void visit(const Split *node) override {
        const auto *field = node->field.as<Field>();
        Type field_type = field->type;
        internal_assert(field_type.is_scalar()) << field_type;
        if (field_type.is<Int_t>()) {
            switch (field_type.bits()) {
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
                               << field_type;
            }
        } else if (field_type.is<UInt_t>()) {
            switch (field_type.bits()) {
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
                               << field_type;
            }
        } else {
            internal_error << "[unimplemented] split field type: "
                           << field_type;
        }
        auto it = defined.find(node->field_name());
        internal_assert(it != defined.cend())
            << "Split does not have access to field: " << node->field;
        internal_assert(it->second.is_int_or_uint())
            << "Split on non-integer field: " << node->field;
        TypeMap parent = defined;
        for (const auto &arm : node->arms) {
            arm.member.accept(this);
            defined = parent; // erase arm scope.
        }
    }

    void visit(const Chain *node) override {
        // Two pass: gather all fields, then check nested members.
        TypeMap parent = defined;
        for (const auto &member : node->members) {
            if (const Field *name = member.as<Field>()) {
                const auto [_, inserted] =
                    defined.try_emplace(name->name, name->type);
                internal_assert(inserted)
                    << "Field: " << name->name << " is duplicated in member";
            } else if (const Materialize *mat = member.as<Materialize>()) {
                const auto [_, inserted] =
                    defined.try_emplace(mat->name, mat->value.type());
                internal_assert(inserted)
                    << "Field: " << name->name << " is duplicated in member";
            }
        }

        for (const auto &member : node->members) {
            member.accept(this);
        }

        defined = parent;
    }
    // void visit(const Group *node) override {}
    // void visit(const Materialize *node) override {}
};

void validate_splits(const Member &member) {
    ValidateSplits validator;
    member.accept(&validator);
}

} // namespace

std::map<std::string, Path> validate_layout(const Layout &layout) {
    const Member &body = layout.body;
    const Type &bvh_t = layout.type;
    internal_assert(body.defined() && bvh_t.defined())
        << "Cannot validate with undefined member or bvh_t: " << body << "\n"
        << bvh_t;
    const BVH_t *bvh_node = bvh_t.as<BVH_t>();
    internal_assert(bvh_node)
        << "Cannot validate member of non-BVH_t: " << bvh_t;

    // Assert all Split fields are accessible at Split level.
    validate_splits(body);

    std::vector<Path> paths = get_paths(body);
    internal_assert(paths.size() == bvh_node->nodes.size())
        << "Layout body: " << body << "\nhas " << paths.size()
        << " paths. BVH type: " << bvh_t << "\nhas " << bvh_node->nodes.size()
        << " node options.";

    // Check paths are unique.
    for (size_t i = 0; i < paths.size(); ++i) {
        const Path &pi = paths[i];
        for (size_t j = i + 1; j < paths.size(); ++j) {
            const Path &pj = paths[j];
            internal_assert(
                !equal_paths(pi, pj)); // TODO: error message for equal paths?
        }
    }

    // TODO(ajr): use Arm::name.

    std::map<std::string, Path> pathmap;
    // Check each node has one equivalent path!
    for (const auto &node : bvh_node->nodes) {
        Path node_path;
        for (auto &path : paths) {
            if (!path.empty() && valid_path(path, node)) {
                internal_assert(node_path.empty())
                    << "Ambiguous path for node: " << node.name()
                    << " in body: " << body;
                node_path = std::move(path);
            }
        }
        internal_assert(!node_path.empty())
            << "No path for node: " << node.name() << " in body: " << body;
        pathmap[node.name()] = std::move(node_path);
    }
    return pathmap;
}

} // namespace ir
} // namespace bonsai
