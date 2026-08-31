#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::set;
using std::string;

// May a loop bound to `inner` sit inside one bound to `outer`?
//
// The orderings a schedule has to respect. A GPUThread runs inside a GPUBlock,
// never the other way round, and the two kinds of machine do not nest inside
// each other at all. Anything not named here is allowed: the point is to catch
// a schedule that asks for something no hardware does, not to enumerate every
// pairing that happens to be sensible.
bool may_nest(Resource outer, Resource inner) {
    switch (outer) {
    case Resource::GPUBlock:
        return inner == Resource::GPUThread;
    case Resource::GPUThread:
        // Nothing goes inside a thread but a gang of lanes, which vectorize()
        // makes and which is not a bind.
        return false;
    case Resource::CPUThread:
        return inner == Resource::CPUThread;
    case Resource::RTCore:
    case Resource::OptixThread:
        return false;
    }
    return false;
}

// The blocks a parfor's body can reach, which is where anything nested inside
// it lives. A parfor body ends at a yield, so this stops at the loop.
set<string> body_of(const Function &func, const Terminator::ParFor &loop) {
    return reachable_from(loop.body.name, compute_successors(func));
}

} // namespace

void bind(FuncMap &funcs, string func, string index, Resource resource) {
    internal_assert(funcs.contains(func))
        << "bind applied to unknown func: " << func;
    auto f = funcs[func];

    Terminator::ParFor *bound = nullptr;
    for (const auto &block : f->blocks) {
        auto *parfor = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->index == index) {
            internal_assert(bound == nullptr)
                << "Two loops named " << index << " in " << func;
            bound = parfor;
        }
    }
    internal_assert(bound != nullptr)
        << "bind(" << index << ", " << to_string(resource) << ") on " << func
        << ": no parfor named " << index;

    internal_assert(!bound->binding.has_value())
        << "bind(" << index << ", " << to_string(resource) << ") on " << func
        << ": that loop is already bound to " << to_string(*bound->binding)
        << ", and a loop runs on one hardware resource";

    bound->binding = resource;

    // Every other bound loop this one contains, or is contained by. Checked
    // after the fact rather than as the schedule is read, because the schedule
    // may bind the inner loop first and either order should mean the same
    // thing.
    const set<string> inside = body_of(*f, *bound);
    for (const auto &block : f->blocks) {
        const auto *other =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (other == nullptr || other == bound || !other->binding.has_value()) {
            continue;
        }
        if (inside.count(block->name)) {
            internal_assert(may_nest(resource, *other->binding))
                << "bind(" << index << ", " << to_string(resource) << ") on "
                << func << ": " << other->index << " is inside it and bound to "
                << to_string(*other->binding)
                << ", which does not run inside a " << to_string(resource);
        } else if (body_of(*f, *other)
                       .count(
                           // the block holding the loop being bound
                           [&] {
                               for (const auto &b : f->blocks) {
                                   if (std::get_if<Terminator::ParFor>(
                                           &b->terminator.data) == bound) {
                                       return b->name;
                                   }
                               }
                               return string();
                           }())) {
            internal_assert(may_nest(*other->binding, resource))
                << "bind(" << index << ", " << to_string(resource) << ") on "
                << func << ": it is inside " << other->index << ", which is "
                << "bound to " << to_string(*other->binding) << ", and a "
                << to_string(resource) << " does not run inside that";
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
