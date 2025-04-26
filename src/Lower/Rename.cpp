#include "Lower/Rename.h"

#include "Error.h"
#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {} // namespace

ir::FuncMap Rename::run(ir::FuncMap funcs) const {
    for (auto &[name, func] : funcs) {
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
