#include "Lower/VerifyLayouts.h"

#include "IR/ValidateLayout.h"

namespace bonsai {
namespace lower {

ir::ScheduleMap VerifyLayouts::run(ir::ScheduleMap schedules,
                                   const CompilerOptions &options) const {
    for (const auto &[target, schedule] : schedules) {
        for (const auto &[name, _] : schedule.tree_types) {
            const auto &it = schedule.tree_layouts.find(name);
            // TODO: do we want to check this? Some tests might want to not use
            // layouts but still validate them. internal_assert(iter !=
            // schedule.tree_layouts.cend())
            //     << "Tree: " << name << " does not have an associated
            //     layout.\n";
            if (it == schedule.tree_layouts.cend())
                continue;
            ir::validate_layout(it->second);
        }
    }
    return schedules;
}

} // namespace lower
} // namespace bonsai
