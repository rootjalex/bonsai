#include "Lower/Lower.h"

#include "IR/Mutator.h"
#include "Lower/Arrays.h"
#include "Lower/Canonicalize.h"
#include "Lower/Externs.h"
#include "Lower/ForEachs.h"
#include "Lower/Generics.h"
#include "Lower/Geometrics.h"
#include "Lower/Lambdas.h"
#include "Lower/Layouts.h"
#include "Lower/Options.h"
#include "Lower/ReturnToOutParameter.h"
#include "Lower/Trees.h"
#include "Lower/Tuples.h"
#include "Lower/VerifyLayouts.h"
#include "Lower/VerifyOptions.h"
#include "Lower/Yields.h"
#include "Opt/DCE.h"
#include "Opt/Inline.h"
#include "Opt/Simplify.h"
#include "Opt/Unswitch.h"

#include "CompilerOptions.h"
#include "Error.h"
#include "Utils.h"

#include <memory>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

void lower(ir::Program &program, const CompilerOptions &options) {
    // Register passes.
    PassManager pm = register_passes();

    std::vector<Pass *> passes;
    for (const std::string &name : options.passes) {
        if (pm.has_alias(name)) {
            std::vector<Pass *> ps = pm.get_alias_passes(name);
            passes.insert(passes.end(), ps.begin(), ps.end());
            continue;
        }
        Pass *p = pm.get_pass(name);
        passes.push_back(p);
    }

    // Run the passes.
    for (Pass *pass : passes) {
        program = pass->run(std::move(program));
    }
}

// TODO(s):
//  Lower spatial queries
//  Perform first round of scheduling.
//  Lower data structures.
//  Perform second round of scheduling + bit data lowering.
//  Perform final code generation
PassManager register_passes() {
    PassManager manager;
    // Lowering pass registration.
    manager.register_pass<Canonicalize>();
    manager.register_pass<LowerLambdas>();
    manager.register_pass<LowerOptions>();
    manager.register_pass<VerifyOptions>();
    manager.register_pass<LowerGenerics>();
    manager.register_pass<VerifyLayouts>();
    manager.register_pass<LowerTrees>();
    manager.register_pass<LowerArrays>();
    manager.register_pass<LowerForEachs>();
    manager.register_pass<LowerGeometrics>();
    manager.register_pass<LowerLayouts>();
    manager.register_pass<LowerTuples>();
    manager.register_pass<LowerYields>();
    manager.register_pass<LowerExterns>();
    manager.register_pass<ReturnToOutParameter>();
    // Optimizing pass registration.
    manager.register_pass<opt::DCE>();
    manager.register_pass<opt::Simplify>();
    manager.register_pass<opt::Unswitch>();
    manager.register_pass<opt::Inline>();

    // Core: the minimal set of passes required to legally lower Bonsai IR
    // (this should *not* include optimizations).
    std::vector<std::unique_ptr<Pass>> core;
    core.push_back(std::make_unique<Canonicalize>());
    core.push_back(std::make_unique<VerifyOptions>());
    core.push_back(std::make_unique<VerifyLayouts>());
    core.push_back(std::make_unique<LowerArrays>());
    core.push_back(std::make_unique<LowerTrees>());
    core.push_back(std::make_unique<LowerExterns>());
    core.push_back(std::make_unique<LowerGeometrics>());
    core.push_back(std::make_unique<LowerLayouts>());
    core.push_back(std::make_unique<LowerForEachs>());
    core.push_back(std::make_unique<LowerYields>());
    core.push_back(std::make_unique<LowerLambdas>());
    core.push_back(std::make_unique<LowerOptions>());
    core.push_back(std::make_unique<LowerTuples>());
    core.push_back(std::make_unique<LowerGenerics>());
    core.push_back(std::make_unique<ReturnToOutParameter>());
    manager.register_alias("core", core);

    // Default: the default work flow (with optimizations).
    std::vector<std::unique_ptr<Pass>> d;
    d.push_back(std::make_unique<Canonicalize>());
    d.push_back(std::make_unique<VerifyOptions>());
    d.push_back(std::make_unique<VerifyLayouts>());
    d.push_back(std::make_unique<LowerArrays>());
    d.push_back(std::make_unique<LowerTrees>());
    d.push_back(std::make_unique<LowerExterns>());
    d.push_back(std::make_unique<LowerGeometrics>());
    d.push_back(std::make_unique<LowerLayouts>());
    d.push_back(std::make_unique<LowerForEachs>());
    d.push_back(std::make_unique<LowerYields>());
    d.push_back(std::make_unique<LowerLambdas>());
    d.push_back(std::make_unique<LowerOptions>());
    d.push_back(std::make_unique<LowerTuples>());
    d.push_back(std::make_unique<LowerGenerics>());
    d.push_back(std::make_unique<ReturnToOutParameter>());
    d.push_back(std::make_unique<opt::Simplify>());
    d.push_back(std::make_unique<opt::DCE>());
    d.push_back(std::make_unique<opt::Unswitch>());
    d.push_back(std::make_unique<opt::Inline>());
    manager.register_alias("default", d);

    return manager;
}

} // namespace lower
} // namespace bonsai
