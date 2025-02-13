#include "Lower/Lower.h"

#include "IR/Mutator.h"
#include "Lower/Canonicalize.h"
#include "Lower/Generics.h"
#include "Lower/Lambdas.h"
#include "Lower/Options.h"

#include "CompilerOptions.h"
#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace lower {

void lower(ir::Program &program, const CompilerOptions &options) {
    Canonicalize().run(program);
    LowerLambda().run(program);
    LowerOption().run(program);
    LowerGeneric().run(program);
    // TODO(s):
    //  Lower spatial queries
    //  Perform first round of scheduling.
    //  Lower data structures.
    //  Perform second round of scheduling + bit data lowering.
    //  Perform final code generation
}

} // namespace lower
} // namespace bonsai
