#include "Lower/RenamePointerToExpr.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Program.h"
#include "IR/Type.h"
#include "Lower/TopologicalOrder.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {} // namespace

ir::FuncMap RenamePointerToExpr::run(ir::FuncMap functions,
                                     const CompilerOptions &options) const {
    return functions;
}

} // namespace lower
} // namespace bonsai
