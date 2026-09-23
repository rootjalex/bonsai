#pragma once

#include "SSA/Analysis.h"
#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace bonsai {
namespace ir {
namespace ssa {

// Where a value a block refers to is actually defined.
//
// A block's arguments are how this form carries a value from where it is
// computed to where it is used, one argument per block on the way, so the
// argument `weight` of some block deep in a loop body says nothing about
// where `weight` was computed. Following the arguments back through the
// jumps that pass them finds out: the definition is the instruction, the
// function parameter, or the block whose predecessors disagree about the
// value -- a merge, or a loop index, or a call's result -- which is then the
// definition itself. Constants are defined nowhere in particular.
//
// defer() asks this of every argument a deferred call passes, to tell what
// is one value at every call from what differs (SSA/Defer.cpp); stage() asks
// it of what the rest of a function is handed, to tell a parameter of the
// function from an address it computed (SSA/Stage.cpp).
struct Definition {
    std::shared_ptr<Value> value; // the value as its defining block refers to it
    std::string block;            // its block; empty for a constant
};

bool same_definition(const Definition &a, const Definition &b);

class Definitions {
  public:
    explicit Definitions(const Function &func);

    // The definition of `v`, as the block named `block` refers to it.
    Definition of(const std::string &block, const std::shared_ptr<Value> &v);

    // The parameter of the function that `v`, as `block` refers to it, is
    // or is threaded from; null when it is an instruction's value, a
    // constant, or a merge of different values.
    const Argument *parameter(const std::string &block,
                              const std::shared_ptr<Value> &v);

  private:
    Definition of_argument(const std::string &block_name, const Argument &a);

    // The value `pred` passes to `block`'s argument `k` along its edge there,
    // or null when the edge defines that argument itself.
    static std::shared_ptr<Value> passed_to(const Block &pred,
                                            const Block &block, size_t k);

    const Function &func;
    BlockMap bmap;
    std::map<std::pair<std::string, std::string>, Definition> memo;
    std::set<std::pair<std::string, std::string>> visiting;
};

} // namespace ssa
} // namespace ir
} // namespace bonsai
