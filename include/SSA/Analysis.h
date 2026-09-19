#pragma once

#include "SSA/SSA.h"

#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// block -> [arg0.is_mutable, arg1.is_mutable, ...]
using ArgMutabilityMap = std::map<std::string, std::vector<bool>>;

ArgMutabilityMap get_mutability_map(const ssa::Function &func);

using BlockMap = std::map<std::string, std::shared_ptr<Block>>;

BlockMap make_block_map(const std::shared_ptr<Function> &func);
BlockMap make_block_map(const Function &func);

//===--------------------------------------------------------------------===//
// Control flow graph
//===--------------------------------------------------------------------===//

// A control flow edge, (from, to).
using Edge = std::pair<std::string, std::string>;

// block name -> adjacent block names, in a deterministic order.
using AdjacencyMap = std::map<std::string, std::vector<std::string>>;

// Intraprocedural successors of `block`, in terminator order.
//
// A Call's successor is its continuation only: the callee belongs to another
// function's CFG. Return and Yield have no successors -- Yield ends a ParFor
// body region, and the implicit back edge to the ParFor header is not a
// successor of the body. A ParFor's successors are its body and continuation.
std::vector<std::string> successors(const Block &block);

// Every jump out of `block`, in terminator order, as mutable references so
// that an edge can be retargeted or given more arguments in place. A Call's
// jump to its callee is not one of them: where a call goes is a matter of
// which function is called, not of this function's control flow.
std::vector<Terminator::Jump *> jumps_of(Block &block);

// Successors of every block, keyed by block name.
AdjacencyMap compute_successors(const Function &func);

// Inverts `succs`. Predecessors are always recomputed this way rather than
// read from Block::preds, which rewrites are free to leave stale.
AdjacencyMap compute_predecessors(const AdjacencyMap &succs);

// Rebuilds every Block::preds from the terminators. Rewrites that retarget
// edges have to call this before anything reads those lists again --
// Block::get_value walks them to thread a value back to where it is defined.
void refresh_preds(Function &func);

// Drops every block the entry cannot reach, and rebuilds the predecessor
// lists. A rewrite that redirects an edge -- loopify() turning a recursion's
// call into a jump, say -- can leave the block the edge went to with no way
// in; such a block still names the blocks it jumps to as their predecessor,
// and still passes them arguments that a later rewrite of the reachable graph
// may have removed. Returns how many blocks were dropped.
size_t remove_unreachable_blocks(Function &func);

// Does `func` call itself? Through a Call or a run of them: a branching
// recursion terminates its block with a MultiCall, and that is exactly the
// case loopify() needs to find in order to queue it.
bool is_recursive(const Function &func);

// Points every use of the instruction `of` at `with` -- operands, the values
// a terminator reads or passes, the names a block looks up -- and removes
// `of` from its block. `with` has to be defined wherever `of` was visible,
// which one of its own operands is.
void replace_uses(Function &func, const Instruction *of,
                  const std::shared_ptr<Value> &with);

// Blocks reachable from `entry`, in reverse postorder. Every analysis below
// iterates in this order, which is what makes the iterative dataflow solvers
// converge in few passes.
std::vector<std::string> reverse_postorder(const std::string &entry,
                                           const AdjacencyMap &succs);

// Blocks reachable from `from` following `succs`, including `from` itself.
std::set<std::string> reachable_from(const std::string &from,
                                     const AdjacencyMap &succs);

//===--------------------------------------------------------------------===//
// Dominance
//===--------------------------------------------------------------------===//

// A dominator tree, as a map from block name to immediate dominator. The root
// maps to itself. Blocks unreachable from the root are absent.
//
// The tree is numbered in preorder when it is built (see `number`), so that
// a dominance query is two lookups and a comparison: `a` dominates `b` iff
// `b`'s number lies in the range `a`'s subtree occupies. This is the interval
// labelling every production dominator tree answers queries with (LLVM's
// DominatorTree, Tarjan's "Finding Dominators in Directed Graphs" for the
// idea); walking the chain of immediate dominators instead costs a map lookup
// per level, and linearization asks the question for every block of a region
// against every join, which made those walks most of a vectorized compile.
// `idom` stays public because passes walk the chain by name; a tree is not to
// be edited after it is numbered.
struct DomTree {
    std::string root;
    std::map<std::string, std::string> idom;

    // Does `a` dominate `b`? Every block dominates itself.
    bool dominates(const std::string &a, const std::string &b) const;

    // Nearest common ancestor of `a` and `b` in the tree.
    std::string nearest_common_ancestor(const std::string &a,
                                        const std::string &b) const;

    // Children of each node, in the order `idom` lists them. By value, so
    // that it can be taken off a tree that is a temporary.
    AdjacencyMap children() const { return kids; }

    // The blocks `a` dominates, `a` itself first, in preorder of the tree.
    // Empty when `a` is not in the tree.
    std::vector<std::string> subtree(const std::string &a) const;

    // Numbers the tree from `idom`. compute_dominator_tree calls this once
    // the immediate dominators are settled; a tree built any other way has
    // to call it before it is queried.
    void number();

  private:
    AdjacencyMap kids;
    // Per block: its preorder number, and one past the last number in its
    // subtree.
    std::unordered_map<std::string, std::pair<size_t, size_t>> range;
};

// Cooper/Harvey/Kennedy iterative dominator construction over `rpo`.
DomTree compute_dominator_tree(const std::string &entry,
                               const AdjacencyMap &succs,
                               const AdjacencyMap &preds,
                               const std::vector<std::string> &rpo);

// Post-dominators: dominators of the reverse CFG.
//
// A function generally has several exits (Return / Yield blocks), so the
// reverse CFG is rooted at a synthetic exit node named by `virtual_exit()`
// that all real exits flow into. Callers should expect that name to appear in
// the resulting tree.
std::string virtual_exit();

DomTree compute_post_dominator_tree(const std::string &entry,
                                    const AdjacencyMap &succs,
                                    const AdjacencyMap &preds);

// Dominance frontier: DF(b) is the set of blocks that b dominates a
// predecessor of, but does not strictly dominate. Placing a block argument
// for a value defined in b at every block of DF(b) -- iterated to a fixed
// point -- is exactly Cytron et al.'s minimal phi placement.
AdjacencyMap compute_dominance_frontier(const AdjacencyMap &preds,
                                        const DomTree &dom);

// The iterated dominance frontier of `defs`: where block arguments have to be
// introduced for a value assigned in every block of `defs`.
std::set<std::string>
iterated_dominance_frontier(const std::set<std::string> &defs,
                            const AdjacencyMap &frontier);

//===--------------------------------------------------------------------===//
// Loops
//===--------------------------------------------------------------------===//

// A natural loop, identified by its header. Partial linearization requires
// reducible control flow, which is exactly the condition that every back edge
// target dominates its source; `compute_loop_forest` asserts this.
struct Loop {
    std::string header;
    // Sources of the back edges into `header`.
    std::set<std::string> latches;
    // Every block in the loop body, including the header and latches.
    std::set<std::string> blocks;
    // Edges leaving the loop: (inside, outside).
    std::set<Edge> exits;
    // Header of the immediately enclosing loop, if any.
    std::optional<std::string> parent;
};

// header name -> Loop.
using LoopForest = std::map<std::string, Loop>;

LoopForest compute_loop_forest(const AdjacencyMap &succs,
                               const AdjacencyMap &preds, const DomTree &dom,
                               const std::vector<std::string> &rpo);

// Header of the innermost loop containing `block`, if any.
std::optional<std::string> innermost_loop(const LoopForest &loops,
                                          const std::string &block);

//===--------------------------------------------------------------------===//
// Control dependence
//===--------------------------------------------------------------------===//

// block -> the set of edges it is control dependent on.
//
// `k` is control dependent on edge (a, b) iff k post-dominates b but does not
// strictly post-dominate a. A block whose set is empty executes whenever the
// function does, so it never needs a mask.
using ControlDependence = std::map<std::string, std::set<Edge>>;

ControlDependence compute_control_dependence(const AdjacencyMap &succs,
                                             const DomTree &pdom);

//===--------------------------------------------------------------------===//
// Block index
//===--------------------------------------------------------------------===//

// A topological ordering of blocks (back edges removed) that is additionally
// *dominance compact* and *loop compact*: the blocks of any dominance region
// and of any loop each occupy a contiguous range of indices.
//
// Partial linearization (Moll & Hack, PLDI 2018, section 3.1) requires both
// properties -- their figure 8 shows the algorithm producing incorrect code
// when the index is not compact.
//
// `last` names blocks to place after every other block that could go in
// their place -- the blocks that call the function itself. Linearization
// runs the blocks of a folded branch one after another, in this order, so
// the order decides what follows a recursive call: a leaf's work placed
// after it would be work deferred past the call once the recursion is put
// on a stack, which SSA/QueueRecursion.h has to reject. Placed last, the call
// keeps the tail position it had in the program. The choice is free where
// the order is, and only there: compactness and topological order come
// first.
std::map<std::string, size_t>
compute_block_index(const std::string &entry, const AdjacencyMap &succs,
                    const DomTree &dom, const LoopForest &loops,
                    const std::set<std::string> &last = {});

//===--------------------------------------------------------------------===//
// Generic dataflow
//===--------------------------------------------------------------------===//

enum class Direction { Forward, Backward };

// Solves a monotone dataflow problem over the block CFG by worklist iteration.
//
// `transfer` maps a block's in-state to its out-state; `meet` combines the
// out-states of a block's predecessors (Forward) or successors (Backward).
// `boundary` is the in-state of the entry (Forward) or of each exit
// (Backward), and `init` the initial state everywhere else.
//
// Returns the in-state of every block. `Lattice` must be copyable and
// equality-comparable, and `transfer`/`meet` must be monotone for this to
// terminate.
template <typename Lattice>
std::map<std::string, Lattice> solve_dataflow(
    const std::string &entry, const AdjacencyMap &succs,
    const AdjacencyMap &preds, Direction direction, const Lattice &init,
    const Lattice &boundary,
    const std::function<Lattice(const std::string &, const Lattice &)>
        &transfer,
    const std::function<Lattice(const Lattice &, const Lattice &)> &meet) {
    const bool forward = direction == Direction::Forward;
    const AdjacencyMap &along = forward ? succs : preds;
    const AdjacencyMap &against = forward ? preds : succs;

    const std::vector<std::string> order = reverse_postorder(entry, succs);

    std::map<std::string, Lattice> in;
    std::map<std::string, Lattice> out;
    for (const auto &name : order) {
        // A node with no incoming edges in the direction of travel takes the
        // boundary value; everything else starts at `init` and is refined.
        const auto it = against.find(name);
        const bool is_boundary = it == against.end() || it->second.empty();
        in[name] = is_boundary ? boundary : init;
        out[name] = transfer(name, in.at(name));
    }

    std::deque<std::string> worklist(order.begin(), order.end());
    std::set<std::string> queued(order.begin(), order.end());
    if (!forward) {
        std::reverse(worklist.begin(), worklist.end());
    }

    while (!worklist.empty()) {
        const std::string name = worklist.front();
        worklist.pop_front();
        queued.erase(name);

        const auto in_it = against.find(name);
        if (in_it != against.end() && !in_it->second.empty()) {
            std::optional<Lattice> merged;
            for (const auto &other : in_it->second) {
                const auto o = out.find(other);
                if (o == out.end()) {
                    continue; // unreachable neighbour
                }
                merged =
                    merged.has_value() ? meet(*merged, o->second) : o->second;
            }
            if (merged.has_value()) {
                in[name] = std::move(*merged);
            }
        }

        Lattice next = transfer(name, in.at(name));
        if (next == out.at(name)) {
            continue;
        }
        out[name] = std::move(next);

        const auto out_it = along.find(name);
        if (out_it == along.end()) {
            continue;
        }
        for (const auto &other : out_it->second) {
            if (in.count(other) && queued.insert(other).second) {
                worklist.push_back(other);
            }
        }
    }

    return in;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
