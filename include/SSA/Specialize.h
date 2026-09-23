#pragma once

// specialize(): a bound loop's body copied per variant of a parameter.
//
// `render.specialize(integrator)`: `integrator` is a parameter of variant
// type -- which integrator the scene runs, fixed for the whole render -- and
// every `match` on it inside the sample loop is a switch on a value the
// compiler could have known. This directive makes it known: the body of
// each of the function's outermost parallel loops is copied once per
// variant of the type, the copy's `integrator` is the given one with its
// tag set to that variant's number, and the loop is entered through a
// dispatch on the tag, so each copy is one loop -- one kernel under a GPU
// bind, one thread body under a CPU bind, one sequential loop unbound -- in
// which the tag is a constant and the backend folds every dispatch on it to
// the one arm and deletes the rest.
// What ptxas then allocates registers for, and what the instruction cache
// holds, is the integrator the scene runs and not all four. pbrt's GPU
// build is one integrator by construction (the wavefront volpath); this is
// how a program that keeps all of pbrt's integrators compares like with
// like.
//
// It is Halide's `specialize()` -- a copy of a stage's loop nest under a
// condition, the general nest kept for the rest -- with the condition a
// variant's tag rather than a boolean, which is what a variant type makes
// natural: the copies are exhaustive, so there is no general nest to keep.
// In the language's terms it changes how the program runs and nothing of
// what it computes, each copy computing exactly what the original would
// have with that tag.
//
// The copies are of the loop as scheduled up to this directive: binds are
// already on the loop (they are tags set at conversion), and a loopify or
// vectorize written before it is in the body copied. A directive written
// after it that names one of the function's loops finds several of that
// name, so `specialize()` goes last among a function's directives.
//
// Only `Inline` storage is specialized (a tag beside the payload), which is
// how a parameter's variant type is stored; a `tagged_index` handle carries
// its tag in the top bits of one word and is not handled yet.

#include "IR/Program.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include <map>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

void specialize_loops(FuncMap &fmap, const std::string &fname,
                      const std::string &param,
                      const std::map<std::string, ir::Program::AdtStorage>
                          &storages);

// A queue's specialize (ir::QueueSpecialize, QueueSpec::split): the drained
// function copied per variant of a value it computes, and the value's tag
// computed where the entry is pushed. The same idea as above with the
// dispatch at the push rather than at the loop, since a queue's entries
// each have their own variant.

// One variant of the value a queue is split on: its label, which names the
// sub-queue (`hits[Some]`, `hits[Diffuse]`), and its tag.
struct KeyVariant {
    std::string label;
    uint64_t tag;
};

// The variants a value of `type` has, for a split on it: an ADT's, by name
// and tag, or an optional's `None` and `Some` -- its `set` field false and
// true (Lower/Options.cpp stores an optional as a struct of `value` and
// `set`). Empty when the type is neither, which a split refuses.
std::vector<KeyVariant>
key_variants(const Type &type,
             const std::map<std::string, ir::Program::AdtStorage> &storages);

// The value named `key` in `func`: an instruction, by the program's name
// for it (a `let`'s Set keeps its name through every copy), or a parameter.
// Null when there is none.
std::shared_ptr<Value> key_value(const Function &func, const std::string &key);

// A copy of `func`, named `name`, in which `key` has the variant `v`: right
// after the key's definition the copy reads it with its tag set to `v`'s
// and uses that everywhere the key was used, so every match on it folds to
// the one arm and the rest goes (SSA/Simplify.h). The copy is what a
// sub-queue's drain calls.
std::shared_ptr<Function>
specialize_function(const Function &func, const std::string &name,
                    const std::string &key, const KeyVariant &v,
                    const std::map<std::string, ir::Program::AdtStorage> &storages);

// The tag of `key` as `func` computes it, made in `block` from `params`, the
// values `block` has for `func`'s parameters by name: the instructions the
// key is computed by, copied there in order. Only a key computed by pure
// instructions from the parameters -- arithmetic, a field of a value, an
// element of an array a parameter names, a load through a parameter nothing
// writes -- since the push has the entry's values and nothing else; a key
// depending on more is refused. The tag is `func`'s to read: an ADT's tag
// field, or an optional's `set`.
std::shared_ptr<Value>
key_tag_at(Block &block, const Function &func, const std::string &key,
           const std::map<std::string, std::shared_ptr<Value>> &params,
           const std::map<std::string, ir::Program::AdtStorage> &storages);

// Follows a specialized value into the callees `blocks` pass it to: a call
// whose argument is the value named `value`, to a parameter of variant type,
// is redirected to a copy of the callee (`callee!Variant`) in which that
// parameter has `v`'s tag (specialize_function), and the copy is searched
// the same way for where it passes the parameter on. What makes a
// specialization reach the matches below the loop it was written on --
// `integrator_li`'s, once `render` is specialized on the integrator -- so
// that later directives, and the compiled code, see one variant's program.
void specialize_callees(FuncMap &fmap,
                        const std::vector<std::shared_ptr<Block>> &blocks,
                        const std::string &value, const KeyVariant &v,
                        const std::map<std::string, ir::Program::AdtStorage> &storages);

} // namespace ssa
} // namespace ir
} // namespace bonsai
