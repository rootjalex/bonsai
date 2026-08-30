#include "Lower/VerifyBuilds.h"

#include "IR/ValidateBuild.h"

namespace bonsai {
namespace lower {

ir::ScheduleMap VerifyBuilds::run(ir::ScheduleMap schedules,
                                  const CompilerOptions &options) const {
    for (const auto &[target, schedule] : schedules) {
        for (const auto &[name, _] : schedule.tree_types) {
            const auto &it = schedule.tree_builds.find(name);
            if (it == schedule.tree_builds.cend())
                continue;
            ir::validate_build(it->second);
        }
    }
    return schedules;
}

} // namespace lower
} // namespace bonsai
