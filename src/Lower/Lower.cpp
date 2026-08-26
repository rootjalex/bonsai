#include "Lower/Lower.h"

#include "IR/Mutator.h"
#include "Lower/Canonicalize.h"
#include "Lower/Defers.h"
#include "Lower/DynamicArrays.h"
#include "Lower/DynamicSets.h"
#include "Lower/Externs.h"
#include "Lower/ForEachs.h"
#include "Lower/Generics.h"
#include "Lower/Geometrics.h"
#include "Lower/Lambdas.h"
#include "Lower/Layouts.h"
#include "Lower/LogicalOperations.h"
#include "Lower/LoopTransforms.h"
#include "Lower/Maps.h"
#include "Lower/Mutability.h"
#include "Lower/Options.h"
#include "Lower/Random.h"
#include "Lower/RecLoops.h"
#include "Lower/RenamePointerToExpr.h"
#include "Lower/ReturnToOutParameter.h"
#include "Lower/Scans.h"
#include "Lower/Sorts.h"
#include "Lower/Trees.h"
#include "Lower/Tuples.h"
#include "Lower/VerifyLayouts.h"
#include "Lower/VerifyOptions.h"
#include "Lower/Yields.h"
#include "Opt/CSE.h"
#include "Opt/DCE.h"
#include "Opt/Fusion.h"
#include "Opt/Inline.h"
#include "Opt/Simplify.h"
#include "Opt/Unswitch.h"
#include "SSA/Convert.h"
#include "SSA/DumpAnalysis.h"

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
    PassManager pm = register_passes(options);

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
        program = pass->run(std::move(program), options);
    }
}

// TODO(s):
//  Lower spatial queries
//  Perform first round of scheduling.
//  Lower data structures.
//  Perform second round of scheduling + bit data lowering.
//  Perform final code generation
PassManager register_passes(const CompilerOptions &options) {
    PassManager manager;
    // Lowering pass registration.
    manager.register_pass<Canonicalize>();
    manager.register_pass<LowerLambdas>();
    manager.register_pass<LowerOptions>();
    manager.register_pass<VerifyOptions>();
    manager.register_pass<LowerGenerics>();
    manager.register_pass<VerifyLayouts>();
    manager.register_pass<LowerTrees>();
    manager.register_pass<LowerSorts>();
    manager.register_pass<LowerDefers>();
    manager.register_pass<LoopTransforms>();
    manager.register_pass<LowerForEachs>();
    manager.register_pass<LowerMaps>();
    manager.register_pass<LowerDynamicSets>();
    manager.register_pass<LowerGeometrics>();
    manager.register_pass<LowerLayouts>();
    manager.register_pass<LowerTuples>();
    manager.register_pass<LowerDynamicArrays>();
    manager.register_pass<LowerYields>();
    manager.register_pass<LowerExterns>();
    manager.register_pass<LowerLogicalOperations>();
    manager.register_pass<LowerRandom>();
    manager.register_pass<LowerScans>();
    manager.register_pass<LowerRecLoops>();
    manager.register_pass<ReturnToOutParameter>();
    manager.register_pass<RenamePointerToExpr>();
    manager.register_pass<Mutability>();
    // Optimizing pass registration.
    manager.register_pass<opt::CSE>();
    manager.register_pass<opt::DCE>();
    manager.register_pass<opt::Fusion>();
    manager.register_pass<opt::Inline>();
    manager.register_pass<opt::Simplify>();
    manager.register_pass<opt::Unswitch>();
    manager.register_pass<ir::ssa::ConvertToSSA>();
    manager.register_pass<ir::ssa::DumpSSAAnalysis>();

    // Core: the minimal set of passes required to legally lower Bonsai IR
    // (this should *not* include optimizations).
    std::vector<std::unique_ptr<Pass>> core;
    core.push_back(std::make_unique<Canonicalize>());
    core.push_back(std::make_unique<VerifyOptions>());
    core.push_back(std::make_unique<VerifyLayouts>());
    // Fusion must always run before Array or Tree lowering!
    core.push_back(std::make_unique<opt::Fusion>());
    core.push_back(std::make_unique<LowerMaps>());
    core.push_back(std::make_unique<LowerTrees>());
    // This must always run after LowerTrees and before LowerLayouts
    core.push_back(std::make_unique<LowerSorts>());
    core.push_back(std::make_unique<LowerDefers>());
    core.push_back(std::make_unique<LowerExterns>());
    core.push_back(std::make_unique<LowerGeometrics>());
    core.push_back(std::make_unique<LowerLayouts>());
    core.push_back(std::make_unique<LowerForEachs>());
    // TODO(ajr): figure out the right placement of transforms.
    core.push_back(std::make_unique<LoopTransforms>());
    // This must *always* go after parallelization,
    // and before Mutability
    core.push_back(std::make_unique<LowerRandom>());
    core.push_back(std::make_unique<LowerDynamicSets>());
    core.push_back(std::make_unique<LowerYields>());
    core.push_back(std::make_unique<LowerScans>());
    core.push_back(std::make_unique<LowerRecLoops>());
    core.push_back(std::make_unique<LowerLambdas>());
    core.push_back(std::make_unique<LowerOptions>());
    core.push_back(std::make_unique<LowerTuples>());
    core.push_back(std::make_unique<LowerDynamicArrays>());
    core.push_back(std::make_unique<LowerLogicalOperations>());
    core.push_back(std::make_unique<LowerGenerics>());
    // This should always run last! It duplicates the exported functions.
    core.push_back(std::make_unique<ReturnToOutParameter>());
    core.push_back(std::make_unique<Mutability>());
    if (options.target == BackendTarget::CUDA) {
        // This must go after Mutability, since it requires PtrTo.
        core.push_back(std::make_unique<RenamePointerToExpr>());
    }
    manager.register_alias("core", core);

    // SSA: like `core`, but skips LoopTransforms (the Stmt-level scheduler,
    // being replaced by the SSA rewriter in src/SSA/Rewrite.cpp) so that
    // schedule transforms which only make sense post-SSA (e.g. `vectorize`)
    // survive to reach ConvertToSSA, which reads them directly off the
    // Program's schedule.
    std::vector<std::unique_ptr<Pass>> ssa;
    ssa.push_back(std::make_unique<Canonicalize>());
    ssa.push_back(std::make_unique<VerifyOptions>());
    ssa.push_back(std::make_unique<VerifyLayouts>());
    // Fusion must always run before Array or Tree lowering!
    ssa.push_back(std::make_unique<opt::Fusion>());
    ssa.push_back(std::make_unique<LowerMaps>());
    ssa.push_back(std::make_unique<LowerTrees>());
    // This must always run after LowerTrees and before LowerLayouts
    ssa.push_back(std::make_unique<LowerSorts>());
    ssa.push_back(std::make_unique<LowerDefers>());
    ssa.push_back(std::make_unique<LowerExterns>());
    ssa.push_back(std::make_unique<LowerGeometrics>());
    ssa.push_back(std::make_unique<LowerLayouts>());
    ssa.push_back(std::make_unique<LowerForEachs>());
    ssa.push_back(std::make_unique<LowerRandom>());
    ssa.push_back(std::make_unique<LowerDynamicSets>());
    ssa.push_back(std::make_unique<LowerYields>());
    ssa.push_back(std::make_unique<LowerScans>());
    ssa.push_back(std::make_unique<LowerRecLoops>());
    ssa.push_back(std::make_unique<LowerLambdas>());
    ssa.push_back(std::make_unique<LowerOptions>());
    ssa.push_back(std::make_unique<LowerTuples>());
    ssa.push_back(std::make_unique<LowerDynamicArrays>());
    ssa.push_back(std::make_unique<LowerLogicalOperations>());
    ssa.push_back(std::make_unique<LowerGenerics>());
    // This should always run last! It duplicates the exported functions.
    ssa.push_back(std::make_unique<ReturnToOutParameter>());
    ssa.push_back(std::make_unique<Mutability>());
    if (options.target == BackendTarget::CUDA) {
        // This must go after Mutability, since it requires PtrTo.
        ssa.push_back(std::make_unique<RenamePointerToExpr>());
    }
    ssa.push_back(std::make_unique<ir::ssa::ConvertToSSA>());
    manager.register_alias("ssa", ssa);

    // Same lowering as "ssa", but dumps the SSA control-flow analyses instead
    // of rewriting and generating code. Used to test those analyses directly.
    std::vector<std::unique_ptr<Pass>> ssa_analysis;
    ssa_analysis.push_back(std::make_unique<Canonicalize>());
    ssa_analysis.push_back(std::make_unique<VerifyOptions>());
    ssa_analysis.push_back(std::make_unique<VerifyLayouts>());
    ssa_analysis.push_back(std::make_unique<opt::Fusion>());
    ssa_analysis.push_back(std::make_unique<LowerMaps>());
    ssa_analysis.push_back(std::make_unique<LowerTrees>());
    ssa_analysis.push_back(std::make_unique<LowerSorts>());
    ssa_analysis.push_back(std::make_unique<LowerDefers>());
    ssa_analysis.push_back(std::make_unique<LowerExterns>());
    ssa_analysis.push_back(std::make_unique<LowerGeometrics>());
    ssa_analysis.push_back(std::make_unique<LowerLayouts>());
    ssa_analysis.push_back(std::make_unique<LowerForEachs>());
    ssa_analysis.push_back(std::make_unique<LowerRandom>());
    ssa_analysis.push_back(std::make_unique<LowerDynamicSets>());
    ssa_analysis.push_back(std::make_unique<LowerYields>());
    ssa_analysis.push_back(std::make_unique<LowerScans>());
    ssa_analysis.push_back(std::make_unique<LowerRecLoops>());
    ssa_analysis.push_back(std::make_unique<LowerLambdas>());
    ssa_analysis.push_back(std::make_unique<LowerOptions>());
    ssa_analysis.push_back(std::make_unique<LowerTuples>());
    ssa_analysis.push_back(std::make_unique<LowerDynamicArrays>());
    ssa_analysis.push_back(std::make_unique<LowerLogicalOperations>());
    ssa_analysis.push_back(std::make_unique<LowerGenerics>());
    ssa_analysis.push_back(std::make_unique<ReturnToOutParameter>());
    ssa_analysis.push_back(std::make_unique<Mutability>());
    if (options.target == BackendTarget::CUDA) {
        ssa_analysis.push_back(std::make_unique<RenamePointerToExpr>());
    }
    ssa_analysis.push_back(std::make_unique<ir::ssa::DumpSSAAnalysis>());
    manager.register_alias("ssa-analysis", ssa_analysis);

    // Default: the default work flow (with optimizations).
    std::vector<std::unique_ptr<Pass>> d;
    d.push_back(std::make_unique<Canonicalize>());
    d.push_back(std::make_unique<VerifyOptions>());
    d.push_back(std::make_unique<VerifyLayouts>());
    // Fusion must always run before Array or Tree lowering!
    d.push_back(std::make_unique<opt::Fusion>());
    d.push_back(std::make_unique<LowerMaps>());
    d.push_back(std::make_unique<LowerTrees>());
    // This must always run after LowerTrees and before LowerLayouts
    d.push_back(std::make_unique<LowerSorts>());
    d.push_back(std::make_unique<LowerDefers>());
    d.push_back(std::make_unique<LowerExterns>());
    d.push_back(std::make_unique<LowerGeometrics>());
    d.push_back(std::make_unique<LowerLayouts>());
    d.push_back(std::make_unique<LowerForEachs>());
    // TODO(ajr): figure out the right placement of transforms.
    d.push_back(std::make_unique<LoopTransforms>());
    // This must *always* go after parallelization,
    // and before Mutability
    d.push_back(std::make_unique<LowerRandom>());
    d.push_back(std::make_unique<LowerDynamicSets>());
    d.push_back(std::make_unique<LowerYields>());
    d.push_back(std::make_unique<LowerScans>());
    d.push_back(std::make_unique<LowerRecLoops>());
    d.push_back(std::make_unique<LowerLambdas>());
    d.push_back(std::make_unique<LowerOptions>());
    d.push_back(std::make_unique<LowerTuples>());
    d.push_back(std::make_unique<LowerDynamicArrays>());
    d.push_back(std::make_unique<opt::Unswitch>());
    d.push_back(std::make_unique<LowerLogicalOperations>());
    d.push_back(std::make_unique<LowerGenerics>());
    d.push_back(std::make_unique<opt::Simplify>());
    // TODO(cgyurgyik): Right now, we don't update functions that are "dead" in
    // the LowerRandom pass because we need to propagate through live functions
    // to get the analysis correct. Ideally we could run this DCE pass much
    // earlier, but this has caused issues that need to be investigated.
    d.push_back(std::make_unique<opt::DCE>());
    d.push_back(std::make_unique<opt::Inline>());
    // Clean up any dead functions after inlining.
    d.push_back(std::make_unique<opt::DCE>());
    // This should always run last! It duplicates the exported functions.
    d.push_back(std::make_unique<ReturnToOutParameter>());
    d.push_back(std::make_unique<Mutability>());
    if (options.target == BackendTarget::CUDA) {
        // This must go after Mutability, since it requires PtrTo.
        d.push_back(std::make_unique<RenamePointerToExpr>());
    }

    manager.register_alias("default", d);

    return manager;
}

} // namespace lower
} // namespace bonsai
