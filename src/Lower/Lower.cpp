#include "Lower/Lower.h"

#include "IR/Mutator.h"
#include "Lower/Canonicalize.h"
#include "Lower/Generics.h"
#include "Lower/Lambdas.h"
#include "Lower/Options.h"
#include "Lower/Trees.h"
#include "Lower/VerifyOptions.h"
#include "Opt/DCE.h"

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
    // Pass registration.
    manager.register_pass<Canonicalize>();
    manager.register_pass<LowerLambda>();
    manager.register_pass<LowerOption>();
    manager.register_pass<VerifyOptions>();
    manager.register_pass<LowerGeneric>();
    manager.register_pass<opt::DCE>();
    manager.register_pass<LowerTrees>();

    // Core: the minimal set of passes required to legally lower Bonsai IR
    // (this should *not* include optimizations).
    std::vector<std::unique_ptr<Pass>> core;
    core.push_back(std::make_unique<Canonicalize>());
    core.push_back(std::make_unique<VerifyOptions>());
    core.push_back(std::make_unique<LowerTrees>());
    core.push_back(std::make_unique<LowerLambda>());
    core.push_back(std::make_unique<LowerOption>());
    core.push_back(std::make_unique<LowerGeneric>());
    manager.register_alias("core", core);

    // Default: the default work flow (with optimizations).
    std::vector<std::unique_ptr<Pass>> d;
    d.push_back(std::make_unique<Canonicalize>());
    d.push_back(std::make_unique<VerifyOptions>());
    d.push_back(std::make_unique<LowerTrees>());
    d.push_back(std::make_unique<LowerLambda>());
    d.push_back(std::make_unique<LowerOption>());
    d.push_back(std::make_unique<LowerGeneric>());
    d.push_back(std::make_unique<opt::DCE>());
    manager.register_alias("default", d);

    return manager;
}

} // namespace lower
} // namespace bonsai
