#include "Lower/VerifyLayouts.h"

#include "IR/ValidateLayout.h"

namespace bonsai {
namespace lower {

ir::Schedule VerifyLayouts::run(ir::Schedule schedule,
                                const CompilerOptions &options) const {
    for (const auto &[name, bvh_t] : schedule.tree_types) {
        // The hardware's tree has no nodes to lay out: its layout is its
        // element arrays and nothing a node walk would check.
        if (bvh_t.template as<ir::BVH_t>()->hardware) {
            continue;
        }
        const auto &iter = schedule.tree_layouts.find(name);
        // TODO: do we want to check this? Some tests might want to not use
        // layouts but still validate them. internal_assert(iter !=
        // schedule.tree_layouts.cend())
        //     << "Tree: " << name << " does not have an associated
        //     layout.\n";
        if (iter != schedule.tree_layouts.cend())
            ir::validate_layout(iter->second, bvh_t);
    }
    return schedule;
}

} // namespace lower
} // namespace bonsai
