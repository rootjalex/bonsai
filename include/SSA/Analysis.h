#pragma once

#include "SSA/SSA.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

// block -> [arg0.is_mutable, arg1.is_mutable, ...]
using ArgMutabilityMap = std::map<std::string, std::vector<bool>>;

ArgMutabilityMap get_mutability_map(const ssa::Function &func);

// The blocks of a function by name, for a pass that works by name.
using BlockMap = std::unordered_map<std::string, std::shared_ptr<Block>>;

BlockMap make_block_map(const std::shared_ptr<Function> &func);
BlockMap make_block_map(const Function &func);

//===--------------------------------------------------------------------===//
// Blocks and edges, by name
//===--------------------------------------------------------------------===//

// Intraprocedural successors of `block`, in terminator order.
//
// A Call's successor is its continuation only: the callee belongs to another
// function's CFG. Return and Yield have no successors -- Yield ends a ParFor
// body region, and the implicit back edge to the ParFor header is not a
// successor of the body. A ParFor's successors are its body and continuation.
std::vector<std::string> successors(const Block &block);

// The value `pred` passes to `block`'s argument `k` along its edge there, or
// null when the edge defines that argument itself: a parfor's index, a
// call's kept result (the continuation's first argument, so the call's own
// arguments are offset by one). What a definition is followed through
// (SSA/Definitions.h) and a merge's incoming values are read by
// (SSA/Simplify.cpp).
std::shared_ptr<Value> passed_to(const Block &pred, const Block &block, size_t k);

// Every jump out of `block`, in terminator order, as mutable references so
// that an edge can be retargeted or given more arguments in place. A Call's
// jump to its callee is not one of them: where a call goes is a matter of
// which function is called, not of this function's control flow.
std::vector<Terminator::Jump *> jumps_of(Block &block);

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

// Does anything in `func` read the instruction `of`'s value: an operand, a
// value a terminator reads or passes?
bool has_uses(const Function &func, const Instruction *of);

// Every place `func` holds a value: the operands of its instructions, what
// its terminators read or pass (a dispatch's condition, a return's value, a
// parfor's bounds, every jump's arguments, a run's per-call values and
// keys), and the values its blocks look names up to. `fn` may replace the
// value it is handed, which is what replace_uses does with this.
void for_each_value(Function &func,
                    const std::function<void(std::shared_ptr<Value> &)> &fn);

// The same for one block: the operands of its instructions, what its
// terminator reads or passes, and the values it looks names up to.
void for_each_value(Block &block,
                    const std::function<void(std::shared_ptr<Value> &)> &fn);

// And for one terminator alone: what it reads or passes.
void for_each_value(Terminator &terminator,
                    const std::function<void(std::shared_ptr<Value> &)> &fn);

//===--------------------------------------------------------------------===//
// Control flow graph
//===--------------------------------------------------------------------===//

// The analyses below work on a graph whose nodes are numbered densely from
// zero, and answer their questions in vectors indexed by that number. A block
// is named by a string in this form, and every analysis used to be a map from
// name to answer; the lookups in those maps -- a string comparison per level
// of a tree, at every step of a walk -- were most of a vectorized compile.
//
// A number is only meaningful with the graph it was assigned in. A pass that
// rewrites the function and then asks again builds a new graph, and anything
// it kept from the old one by name is looked up again (see Cfg::id). Results
// that outlive a graph -- the sets a divergence analysis hands the
// linearizer, say -- are kept by name for that reason.
using BlockId = uint32_t;
inline constexpr BlockId NO_BLOCK = std::numeric_limits<BlockId>::max();

// A control flow edge, (from, to).
using Edge = std::pair<BlockId, BlockId>;

// A set of blocks of one graph: a bit per block, iterated in id order.
class BlockSet {
  public:
    BlockSet() = default;
    explicit BlockSet(size_t capacity) : bits(capacity, false) {}

    bool contains(BlockId b) const { return b < bits.size() && bits[b]; }
    // Whether `b` was not already a member. Grows to fit a block numbered
    // past the capacity, for the graphs a pass adds blocks to.
    bool insert(BlockId b) {
        if (b >= bits.size()) {
            bits.resize(size_t(b) + 1, false);
        }
        if (bits[b]) {
            return false;
        }
        bits[b] = true;
        members++;
        return true;
    }
    bool erase(BlockId b) {
        if (!contains(b)) {
            return false;
        }
        bits[b] = false;
        members--;
        return true;
    }
    void clear() {
        bits.assign(bits.size(), false);
        members = 0;
    }
    size_t size() const { return members; }
    bool empty() const { return members == 0; }

    class Iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = BlockId;
        using difference_type = std::ptrdiff_t;
        using pointer = const BlockId *;
        using reference = BlockId;

        Iterator(const std::vector<bool> *bits, size_t at)
            : bits(bits), at(at) {
            settle();
        }
        BlockId operator*() const { return BlockId(at); }
        Iterator &operator++() {
            at++;
            settle();
            return *this;
        }
        Iterator operator++(int) {
            Iterator before = *this;
            ++*this;
            return before;
        }
        bool operator==(const Iterator &o) const { return at == o.at; }
        bool operator!=(const Iterator &o) const { return at != o.at; }

      private:
        void settle() {
            while (at < bits->size() && !(*bits)[at]) {
                at++;
            }
        }
        const std::vector<bool> *bits;
        size_t at;
    };
    Iterator begin() const { return Iterator(&bits, 0); }
    Iterator end() const { return Iterator(&bits, bits.size()); }

  private:
    std::vector<bool> bits;
    size_t members = 0;
};

// The shape of a control flow graph over dense ids: what the analyses below
// read. A successor appears once per edge, so a dispatch with two targets
// that are one block gives that block two entries here and two entries in
// its predecessors, as the terminator has two edges.
struct Graph {
    std::vector<std::vector<BlockId>> succs;
    std::vector<std::vector<BlockId>> preds;
    BlockId entry = NO_BLOCK;
    // The nodes reachable from `entry`, in reverse postorder. Every analysis
    // iterates in this order, which is what makes the iterative solvers
    // converge in few passes.
    std::vector<BlockId> rpo;

    size_t size() const { return succs.size(); }

    // A graph from its successor lists, with `entry` as the root: the
    // predecessors and the reverse postorder follow.
    static Graph from_successors(std::vector<std::vector<BlockId>> succs,
                                 BlockId entry);
};

// Nodes reachable from `from`, in reverse postorder.
std::vector<BlockId> reverse_postorder(const Graph &g, BlockId from);

// Nodes reachable from `from`, including `from` itself.
BlockSet reachable_from(const Graph &g, BlockId from);

// The control flow graph of the blocks of a function, or of the region of
// one -- the blocks reachable from an entry along intraprocedural edges: a
// ParFor body ends at its Yield, and a Call's callee is outside it, so the
// region rooted at a body block is exactly the blocks a gang executes
// together. Ids follow block-name order, so that walking the blocks by id is
// deterministic and independent of where the blocks happen to sit in the
// function. A snapshot: a pass that rewrites the function builds a new one.
class Cfg : public Graph {
  public:
    // Every block of `func`, with the function's entry as the root. Blocks
    // the entry cannot reach are in the graph but not in `rpo`.
    explicit Cfg(const Function &func);
    // The region of `func` reachable from the block named `entry`.
    Cfg(const Function &func, const std::string &entry);

    // The blocks, by id.
    const std::vector<std::shared_ptr<Block>> &blocks() const {
        return held;
    }
    const std::shared_ptr<Block> &block(BlockId b) const { return held[b]; }
    Block &operator[](BlockId b) const { return *held[b]; }
    const std::string &name(BlockId b) const { return held[b]->name; }

    // The id of a block in this graph, by name or by the block itself; the
    // `find`s answer NO_BLOCK for a block that is not in it, where `id`
    // insists that it is.
    BlockId id(const std::string &name) const;
    BlockId id(const Block &block) const;
    BlockId find(const std::string &name) const;
    BlockId find(const Block &block) const;
    bool contains(const std::string &name) const {
        return find(name) != NO_BLOCK;
    }
    bool contains(const Block &block) const {
        return find(block) != NO_BLOCK;
    }

    // Gives a block the pass has just made the next id, so that it can be
    // referred to alongside the snapshot's blocks. It gets no edges: the
    // graph says what the function was when the snapshot was taken, and a
    // pass that has changed the function keeps track of what it changed.
    BlockId add_block(const std::shared_ptr<Block> &block);

  private:
    void build(std::vector<std::shared_ptr<Block>> blocks,
               const std::string &entry);

    std::vector<std::shared_ptr<Block>> held;
    std::unordered_map<std::string, BlockId> by_name;
    std::unordered_map<const Block *, BlockId> by_pointer;
};

//===--------------------------------------------------------------------===//
// Dominance
//===--------------------------------------------------------------------===//

// A dominator tree: each node's immediate dominator. The root maps to
// itself, and a node the root cannot reach to NO_BLOCK.
//
// The tree is numbered in preorder when it is built (see `number`), so that
// a dominance query is two lookups and a comparison: `a` dominates `b` iff
// `b`'s number lies in the range `a`'s subtree occupies. This is the interval
// labelling every production dominator tree answers queries with (LLVM's
// DominatorTree, Tarjan's "Finding Dominators in Directed Graphs" for the
// idea). `idom` stays public because passes walk the chain; a tree is not to
// be edited after it is numbered.
struct DomTree {
    BlockId root = NO_BLOCK;
    std::vector<BlockId> idom;

    size_t size() const { return idom.size(); }
    // Is `b` in the tree -- reachable from the root?
    bool contains(BlockId b) const {
        return b < idom.size() && idom[b] != NO_BLOCK;
    }

    // Does `a` dominate `b`? Every block dominates itself. False when either
    // is not in the tree.
    bool dominates(BlockId a, BlockId b) const {
        if (a >= range.size() || b >= range.size() || !contains(a) ||
            !contains(b)) {
            return false;
        }
        return range[a].first <= range[b].first &&
               range[b].first < range[a].second;
    }

    // The children of `b`, in id order.
    const std::vector<BlockId> &children(BlockId b) const { return kids[b]; }

    // The blocks `a` dominates, `a` itself first, in preorder of the tree.
    // Empty when `a` is not in the tree.
    std::vector<BlockId> subtree(BlockId a) const;

    // Numbers the tree from `idom`. compute_dominator_tree calls this once
    // the immediate dominators are settled; a tree built any other way has
    // to call it before it is queried.
    void number();

  private:
    std::vector<std::vector<BlockId>> kids;
    // Per node: its preorder number, and one past the last number in its
    // subtree.
    std::vector<std::pair<uint32_t, uint32_t>> range;
};

// Cooper/Harvey/Kennedy iterative dominator construction over `g.rpo`.
DomTree compute_dominator_tree(const Graph &g);

// Post-dominators: dominators of the reverse CFG.
//
// A function generally has several exits (Return / Yield blocks), so the
// reverse CFG is rooted at a synthetic exit node that all real exits flow
// into. It is numbered `g.size()`, one past the graph's nodes, and is the
// root of the tree returned; `virtual_exit()` names it for printing.
std::string virtual_exit();

DomTree compute_post_dominator_tree(const Graph &g);

// Is some lane of `mask` known to be on in `block`? It is when a block above
// `block` in the dominator tree branches on `any(mask)` and `block` lies
// below the branch's true side alone -- the shape of every test a gang makes
// before work no lane may want: the guard partial linearization puts in
// front of an arm (SSA/Linearize.h), the one in front of a masked call
// (SSA/Vectorize.cpp), the one in front of a recursion's pushes
// (SSA/QueueRecursion.h). Each of those asks this before installing a test,
// so that a mask a gang has already tested is not tested again on the same
// path; the backend folds some such repeats and not others, and one test per
// mask per path is what a hand-written packet tracer has.
bool known_nonempty(const Cfg &cfg, const DomTree &dom, BlockId block,
                    const Value &mask);

// Dominance frontier: DF(b) is the set of blocks that b dominates a
// predecessor of, but does not strictly dominate. Placing a block argument
// for a value defined in b at every block of DF(b) -- iterated to a fixed
// point -- is exactly Cytron et al.'s minimal phi placement.
using DominanceFrontier = std::vector<std::vector<BlockId>>;

DominanceFrontier compute_dominance_frontier(const Graph &g,
                                             const DomTree &dom);

// The iterated dominance frontier of `defs`: where block arguments have to be
// introduced for a value assigned in every block of `defs`.
BlockSet iterated_dominance_frontier(const BlockSet &defs,
                                     const DominanceFrontier &frontier);

//===--------------------------------------------------------------------===//
// Loops
//===--------------------------------------------------------------------===//

// A natural loop, identified by its header. Partial linearization requires
// reducible control flow, which is exactly the condition that every back edge
// target dominates its source; `compute_loop_forest` asserts this.
struct Loop {
    BlockId header = NO_BLOCK;
    // Sources of the back edges into `header`, in id order.
    std::vector<BlockId> latches;
    // Every block in the loop body, including the header and latches.
    BlockSet blocks;
    // Edges leaving the loop: (inside, outside), in id order.
    std::vector<Edge> exits;
    // Header of the immediately enclosing loop, or NO_BLOCK.
    BlockId parent = NO_BLOCK;
};

// The loops of a graph, by header, with the innermost loop of every block.
class LoopForest {
  public:
    // The loops, in header order.
    const std::vector<Loop> &loops() const { return all; }
    bool empty() const { return all.empty(); }

    // The loop headed by `header`, or null when `header` heads none.
    const Loop *find(BlockId header) const;
    // The innermost loop containing `b`, or null when `b` is in none.
    const Loop *innermost(BlockId b) const;
    // How many loops contain `b`.
    size_t depth(BlockId b) const;

  private:
    friend LoopForest compute_loop_forest(const Graph &g, const DomTree &dom);
    std::vector<Loop> all;
    // Per block: the header of its innermost loop, or NO_BLOCK.
    std::vector<BlockId> innermost_of;
    // Per block: its position in `all` when it heads a loop, or NO_BLOCK.
    std::vector<BlockId> position_of;
};

LoopForest compute_loop_forest(const Graph &g, const DomTree &dom);

//===--------------------------------------------------------------------===//
// Control dependence
//===--------------------------------------------------------------------===//

// Per block, the edges it is control dependent on, in id order.
//
// `k` is control dependent on edge (a, b) iff k post-dominates b but does not
// strictly post-dominate a. A block whose list is empty executes whenever the
// function does, so it never needs a mask.
using ControlDependence = std::vector<std::vector<Edge>>;

ControlDependence compute_control_dependence(const Graph &g,
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
struct BlockIndex {
    // Per block: its position in the order. Unset for a block the entry
    // cannot reach.
    std::vector<size_t> of;
    // The blocks in order.
    std::vector<BlockId> order;
};

// `last` names blocks to place after every other block that could go in
// their place -- the blocks that call the function itself. Linearization
// runs the blocks of a folded branch one after another, in this order, so
// the order decides what follows a recursive call: a leaf's work placed
// after it would be work deferred past the call once the recursion is put
// on a stack, which SSA/QueueRecursion.h has to reject. Placed last, the call
// keeps the tail position it had in the program. The choice is free where
// the order is, and only there: compactness and topological order come
// first.
BlockIndex compute_block_index(const Graph &g, const DomTree &dom,
                               const LoopForest &loops,
                               const BlockSet &last = {});

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
// Returns the in-state of every block; a block the entry cannot reach is left
// at `init`. `Lattice` must be copyable and equality-comparable, and
// `transfer`/`meet` must be monotone for this to terminate.
template <typename Lattice>
std::vector<Lattice> solve_dataflow(
    const Graph &g, Direction direction, const Lattice &init,
    const Lattice &boundary,
    const std::function<Lattice(BlockId, const Lattice &)> &transfer,
    const std::function<Lattice(const Lattice &, const Lattice &)> &meet) {
    const bool forward = direction == Direction::Forward;
    const auto &along = forward ? g.succs : g.preds;
    const auto &against = forward ? g.preds : g.succs;

    std::vector<Lattice> in(g.size(), init);
    std::vector<Lattice> out(g.size(), init);
    std::vector<bool> reached(g.size(), false);
    for (BlockId b : g.rpo) {
        // A node with no incoming edges in the direction of travel takes the
        // boundary value; everything else starts at `init` and is refined.
        in[b] = against[b].empty() ? boundary : init;
        out[b] = transfer(b, in[b]);
        reached[b] = true;
    }

    std::deque<BlockId> worklist(g.rpo.begin(), g.rpo.end());
    std::vector<bool> queued(g.size(), false);
    for (BlockId b : g.rpo) {
        queued[b] = true;
    }
    if (!forward) {
        std::reverse(worklist.begin(), worklist.end());
    }

    while (!worklist.empty()) {
        const BlockId b = worklist.front();
        worklist.pop_front();
        queued[b] = false;

        if (!against[b].empty()) {
            std::optional<Lattice> merged;
            for (BlockId other : against[b]) {
                if (!reached[other]) {
                    continue; // unreachable neighbour
                }
                merged = merged.has_value() ? meet(*merged, out[other])
                                            : out[other];
            }
            if (merged.has_value()) {
                in[b] = std::move(*merged);
            }
        }

        Lattice next = transfer(b, in[b]);
        if (next == out[b]) {
            continue;
        }
        out[b] = std::move(next);

        for (BlockId other : along[b]) {
            if (reached[other] && !queued[other]) {
                queued[other] = true;
                worklist.push_back(other);
            }
        }
    }

    return in;
}

// Whether some parfor of `func` is bound to the GPU (GPUBlock or GPUThread).
// What makes a program need the GPU host and a device module, and every
// function of it lowered straight from SSA (see SSA/Convert.cpp and
// make_llvm_codegen).
bool binds_to_gpu(const Function &func);

} // namespace ssa
} // namespace ir
} // namespace bonsai
