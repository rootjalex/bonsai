#include "Lower/Defers2.h"

#include "Lower/TopologicalOrder.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {
using namespace ir;

namespace {} // namespace

Program LowerDefers2::run(Program program,
                          const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }

    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    TransformMap &transforms = program.schedules[Target::Host].func_transforms;

    if (transforms.empty()) {
        return program;
    }
    // Need simplification to run to avoid unnecessary saved variables
    // TODO(ajr): might also want LICM/CSE here...
    program.funcs = opt::Simplify().run(std::move(program.funcs), options);
    for (const auto &[name, function] : program.funcs) {
        // if (!name.starts_with("_traverse_tree")) {
        // continue;
        // }
        // std::cerr << *function << "\n";
    }

    return program;
}

} // namespace lower
} // namespace bonsai
