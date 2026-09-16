#pragma once

#include "SSA/SSA.h"

#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

// Gives a loop a single latch. A loop whose body reaches its header along
// several back edges -- a do-while with more than one way of going round again
// -- gets a fresh block that all of them jump to instead, and that alone jumps
// to the header, handing on whatever each back edge brought. Returns the new
// latch's name.
//
// Partial linearization wants this shape: it takes the back edges out of the
// graph to run over, and puts one back at the loop's latch (Moll & Hack, PLDI
// 2018, sections 2.1 and 3.3), so there has to be one. Uniformizing a divergent
// loop wants it too, since the pure latch it makes carries the loop's masks
// round, and every way round has to go through it.
std::string merge_latches(Function &func, const std::string &header,
                          const std::set<std::string> &latches);

} // namespace ssa
} // namespace ir
} // namespace bonsai
