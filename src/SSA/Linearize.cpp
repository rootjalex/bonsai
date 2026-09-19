#include "SSA/Linearize.h"

#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "SSA/PromoteAllocas.h"

#include "Utils.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// One outgoing control-flow edge, in the paper's notation (b, i, s): the i-th
// successor of the terminator of `from`. (SSA/Analysis.h has an `Edge` too,
// which is just a pair of blocks and does not carry the successor index the
// algorithm needs.)
struct CfgEdge {
    BlockId from = NO_BLOCK;
    size_t index = 0;
    BlockId to = NO_BLOCK;
};

// The edges out of `from`, in terminator order, in the graph the region was
// analyzed on -- which is the graph as it stands until the rewiring at the
// end, since nothing before it retargets a jump.
vector<CfgEdge> outgoing_edges(const Cfg &cfg, BlockId from) {
    vector<CfgEdge> edges;
    const vector<BlockId> &succs = cfg.succs[from];
    for (size_t i = 0; i < succs.size(); i++) {
        edges.push_back({from, i, succs[i]});
    }
    return edges;
}

// The result of running the paper's figure 5 over every block: the edges each
// block has in the linearized graph, in the same (b, i, s) form. By block.
using LinearEdges = vector<vector<CfgEdge>>;

// Partial linearization proper (Moll & Hack, figure 5).
//
// Blocks are visited in block index order. A block with a uniform branch
// keeps one edge per original successor, and a block with a divergent branch
// gets a single edge; in both cases an edge goes to the successor of least
// index among those that are available, and the rest are recorded in the
// deferral relation D so that the block they were deferred to picks them up.
//
// Note that the paper's line 18 -- removing b's own entries from D once it
// has been processed -- is left out: appendix A.2 shows it cannot affect the
// result, since D is only ever read for blocks later in the index.
//
// One thing the figure leaves to section 3.3 is spelled out here. The loops
// are all uniform by now, and figure 5 runs on the graph with their back edges
// removed -- which leaves each latch a dead end. A block deferred to a loop's
// header from *outside* the loop (the block the divergent branches before the
// loop reconverge at, say) must therefore not be carried into the body, where
// it would reach the latch with no edge left to follow; the loop runs to
// completion for every lane together, and whatever was pending at its header
// is still pending when it is left. So at the header, targets outside the loop
// are relayed to the blocks the loop exits to, and only targets inside it
// travel along the edges that stay inside. This is what Moll's own linearizer
// (the Region Vectorizer) does with deferred targets at uniform loops.
LinearEdges partial_linearize(const Cfg &cfg, const BlockIndex &index,
                              const BlockSet &divergent_branches,
                              const set<Edge> &back_edges,
                              const LoopForest &loops) {
    LinearEdges linear(cfg.size());
    // The deferral relation: per block, the targets deferred to it.
    vector<set<BlockId>> deferred(cfg.size());

    auto least = [&](const set<BlockId> &candidates) {
        internal_assert(!candidates.empty()) << "no successor to pick";
        return *std::min_element(candidates.begin(), candidates.end(),
                                 [&](BlockId a, BlockId b) {
                                     return index.of[a] < index.of[b];
                                 });
    };

    for (const BlockId b : index.order) {
        // T: what earlier blocks deferred to this one.
        set<BlockId> T = deferred[b];

        // Inside a loop, only the targets inside it may travel along its
        // edges; see above.
        const Loop *in_loop = loops.innermost(b);
        if (in_loop != nullptr) {
            set<BlockId> inside;
            for (BlockId t : T) {
                if (in_loop->blocks.contains(t)) {
                    inside.insert(t);
                } else {
                    internal_assert(b == in_loop->header)
                        << "Block " << cfg.name(b) << " inside loop "
                        << cfg.name(in_loop->header) << " was deferred "
                        << cfg.name(t) << ", which is outside it, "
                        << "and is not the header where that could be relayed";
                    for (const Edge &exit : in_loop->exits) {
                        deferred[exit.second].insert(t);
                    }
                }
            }
            T = std::move(inside);
        }

        // Figure 5 wants an acyclic graph, so the back edges are left out and
        // put back at their latches afterwards (section 3.3).
        vector<CfgEdge> edges;
        for (const CfgEdge &e : outgoing_edges(cfg, b)) {
            if (back_edges.count({b, e.to}) == 0) {
                edges.push_back(e);
            }
        }
        if (edges.empty()) {
            // Either an exit of the region, or a latch whose only edge is the
            // back edge. Both must have nothing outstanding: an exit has
            // nowhere left to go, and a latch that had something deferred
            // would mean a lane left the loop this iteration, which a uniform
            // loop cannot do.
            if (!T.empty()) {
                // The walk so far, since what went wrong is which block was
                // picked ahead of which, and that is only readable off the
                // order and the edges together.
                std::cerr << "--- partial linearization of a region, in block "
                          << "index order:\n";
                for (const BlockId x : index.order) {
                    std::cerr << "  " << cfg.name(x)
                              << (divergent_branches.contains(x) ? " (divergent)"
                                                                 : "")
                              << " ->";
                    for (const CfgEdge &e : outgoing_edges(cfg, x)) {
                        std::cerr << " " << cfg.name(e.to)
                                  << (back_edges.count({x, e.to}) ? "^" : "");
                    }
                    std::cerr << "\n";
                }
                std::cerr << "--- deferred to " << cfg.name(b) << ":";
                for (BlockId t : T) {
                    std::cerr << " " << cfg.name(t);
                }
                std::cerr << "\n";
            }
            internal_assert(T.empty())
                << "Block " << cfg.name(b) << " has deferred successors but "
                << "no way to reach them: the region's exits are not "
                << "post-dominated by everything deferred to them";
            continue;
        }

        if (!divergent_branches.contains(b)) {
            // Uniform: every original successor still gets its own edge, so
            // the branch survives. This is the point of the algorithm.
            for (const CfgEdge &e : edges) {
                set<BlockId> candidates = T;
                candidates.insert(e.to);
                const BlockId next = least(candidates);
                linear[b].push_back({b, e.index, next});
                for (BlockId t : candidates) {
                    if (t != next) {
                        deferred[next].insert(t);
                    }
                }
            }
        } else {
            // Divergent: one edge out, everything else deferred.
            set<BlockId> candidates = T;
            for (const CfgEdge &e : edges) {
                candidates.insert(e.to);
            }
            const BlockId next = least(candidates);
            linear[b].push_back({b, 0, next});
            for (BlockId t : candidates) {
                if (t != next) {
                    deferred[next].insert(t);
                }
            }
        }
    }

    return linear;
}

// What a block's terminator passed to each of its successors, captured before
// the edges are rewritten: (predecessor, values). The values are needed to
// rebuild the block arguments as blends afterwards.
struct Incoming {
    BlockId from = NO_BLOCK;
    // One entry per argument of the target block. A call continuation's
    // result argument has no incoming value here and is left alone.
    vector<shared_ptr<Value>> values;
};

// By target block.
vector<vector<Incoming>> snapshot_arguments(const Cfg &cfg) {
    vector<vector<Incoming>> incoming(cfg.size());
    for (BlockId from = 0; from < cfg.size(); from++) {
        const Block &block = cfg[from];
        const auto record = [&](const Terminator::Jump &j) {
            incoming[cfg.id(j.name)].push_back({from, j.args});
        };
        std::visit(overloads{
                       [&](const std::monostate &) {},
                       [&](const Terminator::Jump &j) { record(j); },
                       [&](const Terminator::Dispatch &d) {
                           for (const auto &target : d.targets) {
                               record(target);
                           }
                       },
                       [&](const Terminator::Return &) {},
                       [&](const Terminator::ParFor &) {
                           internal_error << "Nested ParFor in " << block.name
                                          << " during linearization";
                       },
                       [&](const Terminator::Yield &) {},
                       [&](const Terminator::Call &c) {
                           // The callee is in another function; only the
                           // continuation is an edge of this region. A returned
                           // value arrives as the continuation's first argument
                           // and is not passed here, which the blending
                           // accounts for by matching the values to the *last*
                           // arguments.
                           record(c.cont);
                       },
                       [&](const Terminator::MultiCall &c) {
                           // Like Call: the run's calls all leave the region,
                           // and the only edge inside it is the one after
                           // them.
                           record(c.cont);
                       },
                   },
                   block.terminator.data);
    }
    return incoming;
}

// Builds the boolean values the linearized code needs: an execution mask per
// block, and the mask of each edge.
//
// Following the paper (section 2.4), the predicate of an edge a -> b is the
// predicate of a conjoined with the branch condition that leads to b, and the
// predicate of a block is the disjunction of the predicates of the edges it
// is control dependent on. A block that is not control dependent on any
// divergent branch has a uniform predicate and needs no mask at all
// (theorem 4.1), which is why `Divergence::masked` decides who gets one.
struct Masks {
    // Per block, its execution mask; null when the block is always executed
    // with every lane enabled.
    vector<shared_ptr<Value>> block;
    // (from, to) -> the mask of that edge, for edges out of divergent
    // branches; other edges carry their source block's mask.
    map<Edge, shared_ptr<Value>> edge;
};

// Do two values refer to the same definition? The SSA form threads a
// definition onwards under its own name, so a name is enough to tell -- and
// the name is what is compared, whether a reference spells it as the
// instruction that defined it or as the block argument that carries it: one
// predecessor may pass the instruction and another the argument it arrived
// as, and selecting between the two would be selecting between a value and
// itself, under a mask that makes the result look varying.
// Whether two values passed to a block are one definition: the same
// instruction or argument by name, or equal constants. Equal constants count
// because a select between them on a mask would make a varying value out of
// two uniform ones -- and a branch on that value, which the divergence
// analysis rightly called uniform (see value_key in AnalyzeDivergence.cpp:
// every arm of a switch handing the merge the same `false`), would then be
// left divergent by a linearization that had already happened.
bool same_definition(const Value &a, const Value &b) {
    const auto name_of = [](const Value &v) -> const string * {
        if (const auto *i = std::get_if<shared_ptr<Instruction>>(&v.data)) {
            return &(*i)->name;
        }
        if (const auto *arg = std::get_if<Argument>(&v.data)) {
            return &arg->name;
        }
        return nullptr;
    };
    if (const string *na = name_of(a)) {
        const string *nb = name_of(b);
        return nb != nullptr && *na == *nb;
    }
    if (const auto *ca = std::get_if<Constant>(&a.data)) {
        const auto *cb = std::get_if<Constant>(&b.data);
        return cb != nullptr && equals(ca->type, cb->type) &&
               ca->data == cb->data;
    }
    return false;
}

// Appends an instruction to `block` without the operand rethreading
// make_instruction does: after linearization the blocks form a chain, so a
// value defined in an earlier block is already available here and must not be
// turned into a block argument.
shared_ptr<Value> append(Function &func, const shared_ptr<Block> &block,
                         Type type, Instruction::Op op,
                         vector<shared_ptr<Value>> operands) {
    auto instr =
        std::make_shared<Instruction>(func.get_unique_name(), std::move(type),
                                      op, std::move(operands), block);
    block->instrs.push_back(instr);
    return std::make_shared<Value>(std::move(instr));
}

// The logic a mask is made of: no work worth a branch to skip.
bool is_mask_logic(Instruction::Op op) {
    return op == Instruction::Op::Select || op == Instruction::Op::LAnd ||
           op == Instruction::Op::LOr || op == Instruction::Op::Not ||
           op == Instruction::Op::Any || op == Instruction::Op::Popcount ||
           op == Instruction::Op::Vote;
}

// Work a skipped region must not do at all, rather than merely do for
// nothing: a load in it may be through an address only an active lane made
// valid, and ispc branches around any arm that does such work whatever it
// costs (SafeToRunWithMaskAllOff).
bool touches_memory(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Load:
    case Instruction::Op::Store:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::AtomicAdd:
    case Instruction::Op::Alloc:
    case Instruction::Op::Alloca:
    case Instruction::Op::Append:
        return true;
    default:
        return false;
    }
}

// Whether a region is worth a test to skip: it goes to memory or makes a
// call, or it does enough arithmetic to outweigh the test -- six operations,
// which is where ispc draws the line between predicating both arms straight
// through and branching around them (PREDICATE_SAFE_IF_STATEMENT_COST).
bool worth_skipping(const Cfg &cfg, const BlockSet &region) {
    size_t work = 0;
    for (BlockId b : region) {
        const Block &block = cfg[b];
        if (block.terminator.callee() != nullptr) {
            return true;
        }
        for (const shared_ptr<Instruction> &instr : block.instrs) {
            if (touches_memory(instr->op)) {
                return true;
            }
            work += is_mask_logic(instr->op) ? 0 : 1;
        }
    }
    return work >= 6;
}

// Which block defines `v`, for an instruction; NO_BLOCK for anything else.
BlockId defined_in(const Cfg &cfg, const Value &v) {
    const auto *i = std::get_if<shared_ptr<Instruction>>(&v.data);
    if (i == nullptr) {
        return NO_BLOCK;
    }
    const auto owner = (*i)->owner.lock();
    return owner ? cfg.find(*owner) : NO_BLOCK;
}

// Does `block` declare an argument called `name`?
bool declares(const Block &block, const string &name) {
    return std::any_of(block.args.begin(), block.args.end(),
                       [&](const Argument &arg) { return arg.name == name; });
}

} // namespace

BlockMasks linearize(Function &func, const string &entry_name,
                     const Divergence &divergence,
                     const shared_ptr<Value> &entry_mask,
                     const vector<UniformLoop> &loops_in) {
    if (divergence.branches.empty() && !entry_mask && loops_in.empty()) {
        return {}; // nothing diverges; the control flow is already uniform
    }

    // The region, numbered; the guards below join it as they are made.
    Cfg cfg(func, entry_name);
    const BlockId entry = cfg.entry;
    const DomTree dom = compute_dominator_tree(cfg);
    const DomTree pdom = compute_post_dominator_tree(cfg);
    const ControlDependence cdep = compute_control_dependence(cfg, pdom);
    const LoopForest loops = compute_loop_forest(cfg, dom);

    // What the analysis said about the region, by id.
    BlockSet divergent_branches(cfg.size());
    for (const string &name : divergence.branches) {
        if (const BlockId b = cfg.find(name); b != NO_BLOCK) {
            divergent_branches.insert(b);
        }
    }
    BlockSet masked(cfg.size());
    for (const string &name : divergence.masked) {
        if (const BlockId b = cfg.find(name); b != NO_BLOCK) {
            masked.insert(b);
        }
    }

    // The edges figure 5 is not allowed to see, and that the rewiring below
    // must leave exactly as they are.
    set<Edge> back_edges;
    for (const Loop &loop : loops.loops()) {
        for (BlockId latch : loop.latches) {
            back_edges.insert({latch, loop.header});
        }
    }

    // Every loop still standing has to be uniform, which after
    // uniformize_loops() means every loop: its exits are folded into the live
    // mask and the only branch left is the one on `any`.
    for (const Loop &loop : loops.loops()) {
        for (const Edge &exit : loop.exits) {
            internal_assert(!divergent_branches.contains(exit.first))
                << "Loop " << cfg.name(loop.header)
                << " still leaves divergently from " << cfg.name(exit.first)
                << "; it has to be uniformized first (Moll & Hack section 5)";
        }
    }

    // The live mask governing each block of a uniformized loop, and where each
    // loop's mask is seeded.
    // Innermost first: the uniformization transformed inner loops before the
    // loops around them and lists them in that order, and a block of an inner
    // loop runs under the inner loop's live mask, which was seeded from the
    // outer's and has been narrowed since.
    constexpr size_t NO_LOOP = size_t(-1);
    vector<shared_ptr<Value>> loop_masks(cfg.size());
    vector<size_t> loop_of(cfg.size(), NO_LOOP); // index into loops_in
    vector<BlockSet> loop_blocks;                // per loops_in entry
    vector<const UniformLoop *> loop_seeds(cfg.size(), nullptr);
    vector<BlockId> latch_header(cfg.size(), NO_BLOCK); // pure latch -> header
    for (size_t i = 0; i < loops_in.size(); i++) {
        const UniformLoop &loop = loops_in[i];
        BlockSet members(cfg.size());
        for (const string &name : loop.blocks) {
            const BlockId b = cfg.id(name);
            members.insert(b);
            if (!loop_masks[b]) {
                loop_masks[b] = loop.live;
                loop_of[b] = i;
            }
        }
        loop_blocks.push_back(std::move(members));
        loop_seeds[cfg.id(loop.preheader)] = &loop;
        latch_header[cfg.id(loop.latch)] = cfg.id(loop.header);
    }

    // The index the algorithm walks in, which has to be dominance compact and
    // loop compact for the result to be correct (figure 8 of the paper). A
    // block that calls the function itself goes as late as that allows, so a
    // recursion stays in tail position once its branch is folded (see
    // compute_block_index): a run of recursive calls placed before a leaf's
    // work would be a run that loopify() cannot put on a stack.
    BlockSet recursions(cfg.size());
    for (BlockId b = 0; b < cfg.size(); b++) {
        const auto *call = cfg[b].terminator.callee();
        if (call != nullptr && call->name == func.blocks.front()->name) {
            recursions.insert(b);
        }
    }
    const BlockIndex index = compute_block_index(cfg, dom, loops, recursions);
    const vector<BlockId> &by_index = index.order;

    // Everything the old edges carried, before they are rewritten.
    const vector<vector<Incoming>> incoming = snapshot_arguments(cfg);

    // The conditions of the branches being folded away, kept before their
    // terminators are replaced.
    vector<optional<Terminator::Dispatch>> dispatches(cfg.size());
    for (BlockId b = 0; b < cfg.size(); b++) {
        if (const auto *d = std::get_if<Terminator::Dispatch>(
                &cfg[b].terminator.data)) {
            dispatches[b] = *d;
        }
    }

    LinearEdges linear = partial_linearize(cfg, index, divergent_branches,
                                           back_edges, loops);

    //===------------------------------------------------------------===//
    // Masks
    //===------------------------------------------------------------===//

    const Type bool_type = Bool_t::make();
    Masks masks;
    masks.block.resize(cfg.size());

    // Edge masks first: a divergent branch splits its source's mask by the
    // condition. A dispatch on a bool has target 0 as the false side and
    // target 1 the true side (see the IfElse visitor in SSA/Convert.cpp); a
    // dispatch on an integer is a switch, with target k taken on k (see the
    // SwitchStmt visitor there).
    auto mask_of = [&](BlockId b) -> optional<shared_ptr<Value>> {
        if (b < masks.block.size() && masks.block[b]) {
            return masks.block[b];
        }
        return std::nullopt;
    };

    for (const BlockId b : by_index) {
        const shared_ptr<Block> &block = cfg.block(b);

        // The edges whose masks decide this block's: the edges it is control
        // dependent on, less two kinds that split no lanes.
        //
        // A dependence through a back edge -- a loop header on the edge that
        // keeps its own loop going, and so on the latch -- is a dependence on
        // whether the loop iterates again, which every lane decides together
        // once the loops are uniform. Such an edge comes from a block later in
        // the index, the index being topological with the back edges removed,
        // and that is how it is recognised.
        //
        // Inside a uniformized loop, a dependence on a branch *outside* the
        // loop is already accounted for: the loop's live mask was seeded with
        // the mask the loop was entered under (see loop_seeds below) and has
        // since been narrowed by the lanes that left, so it says everything
        // the outer branch said and more. Only a divergent branch inside the
        // loop can narrow it further, and the masks of its edges start from
        // the live mask.
        const shared_ptr<Value> &loop_mask = loop_masks[b];
        const size_t loop_of_b = loop_of[b];
        vector<Edge> deciding;
        if (masked.contains(b)) {
            for (const auto &[from, to] : cdep[b]) {
                if (index.of[from] >= index.of[b]) {
                    continue; // through a back edge
                }
                if (loop_of_b != NO_LOOP && !loop_blocks[loop_of_b].contains(from)) {
                    continue; // outside the loop the live mask stands for
                }
                deciding.push_back({from, to});
            }
        }

        if (!deciding.empty()) {
            // The block's own mask, from the edges it is control dependent on.
            const size_t before = block->instrs.size();
            shared_ptr<Value> mask;
            for (const auto &[from, to] : deciding) {
                const auto it = masks.edge.find({from, to});
                internal_assert(it != masks.edge.end())
                    << "Control dependence edge " << cfg.name(from) << "->"
                    << cfg.name(to) << " of " << cfg.name(b)
                    << " has no mask yet; the block index is not topological";
                mask = mask ? append(func, block, bool_type,
                                     Instruction::Op::LOr, {mask, it->second})
                            : it->second;
            }
            // Built only from the masks of edges out of other blocks, so it
            // goes first: everything in this block runs under it, and what
            // is predicated on it -- a gather, a store -- may come anywhere.
            std::rotate(block->instrs.begin(), block->instrs.begin() + before,
                        block->instrs.end());
            masks.block[b] = mask;
        } else if (loop_mask) {
            // A block of a uniformized loop runs under that loop's live mask.
            // Control dependence cannot say this: after the transform the
            // only branch deciding whether the loop body runs is the uniform
            // one on `any`, and yet a lane that left on an earlier iteration
            // is still not executing.
            masks.block[b] = loop_mask;
        } else if (entry_mask) {
            // Everything in the region runs under the mask the region was
            // entered with, so a block whose own predicate is uniform still
            // carries it.
            masks.block[b] = entry_mask;
        } else {
            internal_assert(!masked.contains(b))
                << "Masked block " << cfg.name(b)
                << " has no control dependences";
        }

        // A loop is entered under the mask of the block that jumps into it,
        // which is only known now. The transform seeded it with `true`,
        // standing for "every lane that got here".
        if (const UniformLoop *seed = loop_seeds[b]) {
            if (auto m = mask_of(b)) {
                for (Terminator::Jump *jump : jumps_of(*block)) {
                    if (jump->name != seed->header) {
                        continue;
                    }
                    internal_assert(seed->seed_arg < jump->args.size())
                        << "Loop " << seed->header << " has no live "
                        << "mask seed at index " << seed->seed_arg
                        << " in the jump from " << cfg.name(b);
                    jump->args[seed->seed_arg] = *m;
                }
            }
        }

        // The masks of the edges leaving this block.
        const optional<Terminator::Dispatch> &dispatch = dispatches[b];
        const bool folds = divergent_branches.contains(b);
        for (const CfgEdge &e : outgoing_edges(cfg, b)) {
            if (!folds || !dispatch.has_value()) {
                // A uniform branch does not split the lanes: every edge out
                // carries the block's own mask.
                if (auto m = mask_of(b)) {
                    masks.edge[{b, e.to}] = *m;
                }
                continue;
            }

            const shared_ptr<Value> &tag = dispatch->cond;
            const size_t n = dispatch->targets.size();
            shared_ptr<Value> cond;
            if (tag->get_type().is_bool()) {
                cond = tag;
                if (e.index == 0) {
                    // The false side: the lanes where the condition does not
                    // hold. Select rather than a dedicated negation, which
                    // the SSA form does not have (see the UnOp visitor in
                    // SSA/Convert.cpp).
                    auto t = std::make_shared<Value>(Constant{bool_type, true});
                    auto f =
                        std::make_shared<Value>(Constant{bool_type, false});
                    cond = append(func, block, bool_type,
                                  Instruction::Op::Select, {tag, f, t});
                }
            } else {
                // A switch (see ir::SwitchStmt): target k takes the lanes
                // whose tag is k, and the last target the lanes whose tag is
                // none of the others.
                const Type &type = tag->get_type();
                auto key = [&](size_t k) {
                    return std::make_shared<Value>(
                        type.is_uint() ? Constant{type, uint64_t(k)}
                                       : Constant{type, int64_t(k)});
                };
                if (e.index + 1 < n) {
                    cond = append(func, block, bool_type, Instruction::Op::Eq,
                                  {tag, key(e.index)});
                } else {
                    for (size_t k = 0; k + 1 < n; k++) {
                        auto other = append(func, block, bool_type,
                                            Instruction::Op::Ne, {tag, key(k)});
                        cond = cond ? append(func, block, bool_type,
                                             Instruction::Op::LAnd,
                                             {cond, other})
                                    : other;
                    }
                }
            }
            if (auto m = mask_of(b)) {
                cond = append(func, block, bool_type, Instruction::Op::LAnd,
                              {*m, cond});
            }
            masks.edge[{b, e.to}] = cond;
        }
    }

    //===------------------------------------------------------------===//
    // Skipping the arms no lane is on
    //===------------------------------------------------------------===//
    //
    // See the header: Shin's BOSCC, installed as Moll & Hack's gadget, but on
    // the linearized graph. An arm is a target of a divergent branch that has
    // the branch as its only predecessor. Its region is what it dominates,
    // which the fold has laid out as one contiguous run of the path (the
    // index is dominance compact) with one way in -- into the arm, from
    // wherever the fold sent there -- and, for a region worth skipping, one
    // way out: the landing. The guard goes on the way in, and tests the arm's
    // mask, which is the mask of the edge into it and so defined before it.
    //
    // A region reached in more than one way, or leaving in more than one
    // (through a uniform branch of the program's whose sides exit
    // separately), or leaving by a back edge, is left alone: there is no one
    // block for the bypass to land on. So is one whose landing is a loop
    // header, since a bypass into a header would be a way into the loop that
    // is not its preheader.

    struct Gadget {
        BlockId guard;   // the block holding the test and the branch
        BlockId arm;     // the block it stands in front of
        BlockSet region; // the blocks the bypass skips: the arm's region
        BlockId landing; // where the bypass lands: the region's one exit
    };
    vector<Gadget> gadgets;
    BlockSet guards;
    // Every block name in the function, so that a guard's is new.
    std::unordered_set<string> taken;
    for (const auto &block : func.blocks) {
        taken.insert(block->name);
    }
    auto fresh_block_name = [&](const string &stem) {
        string name = stem;
        for (size_t i = 0; taken.count(name); i++) {
            name = stem + "_" + std::to_string(i);
        }
        taken.insert(name);
        return name;
    };

    // For measuring what the skipping is worth on a program: with this set,
    // every gang runs every arm, as linearized code does without it.
    if (std::getenv("BONSAI_NO_BOSCC") == nullptr) {
        for (const BlockId x : by_index) {
            if (!divergent_branches.contains(x) || !dispatches[x].has_value()) {
                continue;
            }
            set<BlockId> seen;
            for (const Terminator::Jump &target : dispatches[x]->targets) {
                const BlockId a = cfg.find(target.name);
                if (a == NO_BLOCK || !seen.insert(a).second) {
                    continue;
                }
                const vector<BlockId> &pa = cfg.preds[a];
                if (pa.size() != 1 || pa[0] != x) {
                    continue; // a join, or a header
                }
                if (loops.innermost(a) != loops.innermost(x)) {
                    continue;
                }
                BlockSet dominated(cfg.size());
                for (BlockId u : dom.subtree(a)) {
                    dominated.insert(u);
                }
                // One way out, forwards.
                BlockId out = NO_BLOCK;
                bool one_exit = true;
                for (BlockId u : dominated) {
                    for (const CfgEdge &e : linear[u]) {
                        if (dominated.contains(e.to)) {
                            continue;
                        }
                        one_exit = one_exit && (out == NO_BLOCK || out == e.to);
                        out = e.to;
                    }
                    for (const Edge &back : back_edges) {
                        if (back.first == u && !dominated.contains(back.second)) {
                            one_exit = false;
                        }
                    }
                }
                if (!one_exit || out == NO_BLOCK || loops.find(out) != nullptr) {
                    continue;
                }
                // One way in, into the arm -- not counting the bypasses of
                // earlier arms that land on it, which the guard takes over:
                // a gang that skips one arm lands on the next arm's test, as
                // it does in ispc.
                size_t ways_in = 0;
                for (BlockId u = 0; u < linear.size(); u++) {
                    if (dominated.contains(u)) {
                        continue;
                    }
                    for (const CfgEdge &e : linear[u]) {
                        if (dominated.contains(e.to)) {
                            internal_assert(e.to == a)
                                << "The fold enters the region of "
                                << cfg.name(a) << " at " << cfg.name(e.to)
                                << ", not at the arm";
                            if (!(guards.contains(u) && e.index == 0)) {
                                ways_in++;
                            }
                        }
                    }
                }
                if (ways_in != 1) {
                    continue;
                }
                const shared_ptr<Value> m = masks.block[a];
                internal_assert(m)
                    << "Arm " << cfg.name(a) << " of the divergent branch in "
                    << cfg.name(x) << " has no mask";
                if (dominated.contains(defined_in(cfg, *m))) {
                    continue; // a mask the arm computes for itself
                }
                if (!worth_skipping(cfg, dominated)) {
                    continue;
                }

                auto guard = std::make_shared<Block>();
                guard->name = fresh_block_name(cfg.name(a) + "!any");
                guard->owner = cfg[a].owner;
                auto any = append(func, guard, bool_type, Instruction::Op::Any,
                                  {m});
                // targets[0] is where a false condition goes: past the arm.
                guard->terminator.data = Terminator::Dispatch{
                    .cond = any,
                    .targets = {
                        Terminator::Jump{.name = cfg.name(out), .args = {}},
                        Terminator::Jump{.name = cfg.name(a), .args = {}}}};
                const BlockId g = cfg.add_block(guard);
                for (vector<CfgEdge> &edges : linear) {
                    for (CfgEdge &e : edges) {
                        if (e.to == a) {
                            e.to = g;
                        }
                    }
                }
                linear.push_back({{g, 0, out}, {g, 1, a}});
                masks.block.push_back(m);
                internal_assert(linear.size() == cfg.size() &&
                                masks.block.size() == cfg.size());
                func.blocks.insert(std::find(func.blocks.begin(),
                                             func.blocks.end(), cfg.block(a)),
                                   guard);
                for (Gadget &earlier : gadgets) {
                    // The guard runs inside every region around the arm...
                    if (earlier.region.contains(a)) {
                        earlier.region.insert(g);
                    }
                    // ...and a bypass that landed on the arm lands on its
                    // test now.
                    if (earlier.landing == a) {
                        earlier.landing = g;
                        std::get<Terminator::Dispatch>(
                            cfg[earlier.guard].terminator.data)
                            .targets[0]
                            .name = guard->name;
                    }
                }
                gadgets.push_back({g, a, dominated, out});
                guards.insert(g);
            }
        }
    }

    // How many arguments each block declared before any landing gained one;
    // the blending below is of these, and only these.
    vector<size_t> original_args(cfg.size());
    for (BlockId b = 0; b < cfg.size(); b++) {
        original_args[b] = cfg[b].args.size();
    }

    // Dominance in the graph as it now is: the path the fold made, the
    // bypasses, and the back edges. This is what decides where a value is
    // in scope from here on; the original graph's dominators do not, since
    // the fold has made every folded arm dominate what follows it, and a
    // bypass has undone that for the arms that are skipped.
    vector<vector<BlockId>> linear_succs(cfg.size());
    for (BlockId from = 0; from < linear.size(); from++) {
        for (const CfgEdge &e : linear[from]) {
            linear_succs[from].push_back(e.to);
        }
    }
    for (const Edge &back : back_edges) {
        linear_succs[back.first].push_back(back.second);
    }
    const DomTree linear_dom = compute_dominator_tree(
        Graph::from_successors(std::move(linear_succs), entry));

    // A value defined inside a skipped region and read outside it has to
    // reach the read along the bypass too, and the bypass computes nothing:
    // so at the landing the value becomes an argument, handed one thing from
    // inside the region and another by each bypass that lands there. A region
    // nested in another that leaves through the same block lands there as
    // well, and its bypass is one more predecessor of the same argument.
    //
    // One entry per argument added to a landing block: what each predecessor
    // of the landing hands it -- the region's exit, each guard whose bypass
    // lands there, and any other way in.
    struct LandingArg {
        shared_ptr<Value> arg;
        map<BlockId, shared_ptr<Value>> from;
    };
    vector<vector<LandingArg>> landing_args(cfg.size());

    // The landings between a definition in `at` and a read in `to`, innermost
    // region first; regions that share a landing are one hop. At each, the
    // block the region leaves from and any other way in -- a landing that is
    // a join of a uniform branch of the program's.
    struct Hop {
        BlockId landing;
        BlockId inside;
        vector<BlockId> others;
    };
    auto hops = [&](BlockId at, BlockId to) {
        vector<Hop> path;
        for (;;) {
            vector<const Gadget *> enclosing;
            for (const Gadget &g : gadgets) {
                if (g.region.contains(at) && !g.region.contains(to)) {
                    enclosing.push_back(&g);
                }
            }
            if (enclosing.empty()) {
                return path;
            }
            const Gadget *inner = enclosing.front();
            for (const Gadget *g : enclosing) {
                if (g != inner && inner->region.contains(g->arm)) {
                    inner = g;
                }
            }
            Hop hop{inner->landing, NO_BLOCK, {}};
            for (BlockId from = 0; from < linear.size(); from++) {
                for (const CfgEdge &e : linear[from]) {
                    if (e.to != hop.landing ||
                        (guards.contains(from) && e.index == 0)) {
                        continue;
                    }
                    if (inner->region.contains(from)) {
                        internal_assert(hop.inside == NO_BLOCK)
                            << "The region of " << cfg.name(inner->arm)
                            << " leaves for " << cfg.name(hop.landing)
                            << " from both " << cfg.name(hop.inside) << " and "
                            << cfg.name(from);
                        hop.inside = from;
                    } else {
                        hop.others.push_back(from);
                    }
                }
            }
            internal_assert(hop.inside != NO_BLOCK)
                << "The region of " << cfg.name(inner->arm)
                << " does not leave for its landing " << cfg.name(hop.landing);
            path.push_back(hop);
            at = hop.landing;
        }
    };

    // Threads `v`, defined in `at`, to `to`: an argument of every landing on
    // the way, handed `v` (or the previous landing's argument) from inside
    // and by every bypass landing there: `skipped(gadget)` from a guard whose
    // region holds the definition, and `v` itself from one whose region does
    // not -- that region is between the definition and the landing, so `v`
    // is in scope at its guard. Along any other way in, `elsewhere`; when
    // there is no such value to hand, nothing: the landings cannot carry
    // `v`, and the caller merges through memory instead. `at` is left naming
    // the block the returned value is defined in. A `memo` shares one
    // argument between every reader of the same value.
    using ThreadMemo =
        map<std::pair<const Instruction *, BlockId>, shared_ptr<Value>>;
    set<string> landing_names;
    map<const Value *, BlockId> landing_arg_block;
    auto thread = [&](shared_ptr<Value> v, BlockId &at, BlockId to,
                      const string &stem,
                      const std::function<shared_ptr<Value>(const Gadget &)>
                          &skipped,
                      const shared_ptr<Value> &elsewhere,
                      ThreadMemo *memo) -> shared_ptr<Value> {
        for (const Hop &hop : hops(at, to)) {
            if (!hop.others.empty() && !elsewhere) {
                return nullptr;
            }
            const auto *instr = std::get_if<shared_ptr<Instruction>>(&v->data);
            const std::pair<const Instruction *, BlockId> key{
                instr ? instr->get() : nullptr, hop.landing};
            if (memo != nullptr && instr != nullptr && memo->count(key)) {
                v = memo->at(key);
                at = hop.landing;
                continue;
            }
            const shared_ptr<Block> &landing = cfg.block(hop.landing);
            // Unique in the function, so that a name says which landing's
            // argument it is (see definition_block).
            string name = stem + "!skip";
            for (size_t i = 0; landing_names.count(name); i++) {
                name = stem + "!skip" + std::to_string(i);
            }
            landing_names.insert(name);
            LandingArg entry_arg{
                landing->add_argument(Argument{v->get_type(), name}), {}};
            entry_arg.from[hop.inside] = v;
            for (const Gadget &g : gadgets) {
                if (g.landing == hop.landing) {
                    entry_arg.from[g.guard] =
                        g.region.contains(at) ? skipped(g) : v;
                }
            }
            for (BlockId other : hop.others) {
                entry_arg.from[other] = elsewhere;
            }
            v = entry_arg.arg;
            at = hop.landing;
            landing_arg_block[v.get()] = at;
            if (memo != nullptr && instr != nullptr) {
                (*memo)[key] = v;
            }
            landing_args[hop.landing].push_back(std::move(entry_arg));
        }
        return v;
    };

    // Where `v`, as `from` passes it, is defined: the block of an
    // instruction; for an argument, the landing that declared it, or else
    // the nearest block up `from`'s dominator chain that declares the name.
    // NO_BLOCK for a constant, or for a parameter -- something in scope
    // everywhere, which no bypass can put out of reach.
    auto definition_block = [&](const shared_ptr<Value> &v,
                                BlockId from) -> BlockId {
        if (std::holds_alternative<shared_ptr<Instruction>>(v->data)) {
            return defined_in(cfg, *v);
        }
        const auto *arg = std::get_if<Argument>(&v->data);
        if (arg == nullptr) {
            return NO_BLOCK;
        }
        const auto landing = landing_arg_block.find(v.get());
        if (landing != landing_arg_block.end()) {
            return landing->second;
        }
        for (BlockId at = from;;) {
            if (declares(cfg[at], arg->name)) {
                return at == entry ? NO_BLOCK : at;
            }
            if (!dom.contains(at) || dom.idom[at] == at) {
                return NO_BLOCK;
            }
            at = dom.idom[at];
        }
    };

    // The masks. A block's mask is the OR of the masks of the edges it is
    // control dependent on, and an early exit from inside an arm -- a return
    // in it, an inner edge deciding what runs after the arm -- makes a block
    // after the region depend on an edge inside it. Along the bypass no lane
    // took that edge, so its mask there is false; and false is what every
    // bypass hands on, so a chain of them lands the same.
    if (!gadgets.empty()) {
        const auto no_lane = std::make_shared<Value>(Constant{bool_type, false});
        ThreadMemo threaded_masks;
        for (BlockId b = 0; b < cfg.size(); b++) {
            const shared_ptr<Block> &block = cfg.block(b);
            auto fix = [&](shared_ptr<Value> &v) {
                if (!v) {
                    return;
                }
                BlockId def = defined_in(cfg, *v);
                if (def == NO_BLOCK || def == b) {
                    return;
                }
                const bool crosses = std::any_of(
                    gadgets.begin(), gadgets.end(), [&](const Gadget &g) {
                        return g.region.contains(def) && !g.region.contains(b);
                    });
                if (!crosses) {
                    return;
                }
                const auto *i = std::get_if<shared_ptr<Instruction>>(&v->data);
                internal_assert(is_mask_logic((*i)->op))
                    << "The value " << (*i)->name << " of " << cfg.name(def)
                    << " is read in " << cfg.name(b) << ", past a bypass of "
                    << "the region that computes it, and is not a mask";
                v = thread(
                    v, def, b, (*i)->name,
                    [&](const Gadget &) { return no_lane; }, no_lane,
                    &threaded_masks);
            };
            for (const auto &instr : block->instrs) {
                for (auto &operand : instr->operands) {
                    fix(operand);
                }
            }
            if (masks.block[b]) {
                fix(masks.block[b]);
            }
            for (auto &[edge, mask] : masks.edge) {
                if (edge.first == b) {
                    fix(mask);
                }
            }
        }
    }

    //===------------------------------------------------------------===//
    // Blending
    //===------------------------------------------------------------===//
    //
    // A block argument picks a value based on which predecessor was taken. If
    // those predecessors have been folded into one path then the choice is no
    // longer made by control flow, so it becomes a select over the masks of
    // the folded edges -- every value has been computed by the time the block
    // runs, and the mask says which one this lane wanted.

    // What each blended-away argument was replaced by, per block and argument
    // name, so that a later block taking that argument as an incoming value
    // picks up the blend instead of a name that no longer exists. Nested
    // branches need this: the outer join's incoming value is the inner join's
    // argument.
    vector<map<string, shared_ptr<Value>>> replaced(cfg.size());

    // Blocks that still merge values through their arguments -- a loop header,
    // say. An edge into one of these has to go on carrying what it passes;
    // dropping it would leave the phi with nothing to merge.
    BlockSet keeps_args(cfg.size());

    // Whether any argument was turned into a slot (see below), which the
    // promotion pass turns back into values once the graph is rewired.
    bool slots_made = false;
    const shared_ptr<Block> &entry_block = cfg.block(entry);
    auto append_store = [&](const shared_ptr<Block> &into,
                            const shared_ptr<Value> &slot,
                            const shared_ptr<Value> &value) {
        into->instrs.push_back(std::make_shared<Instruction>(
            Instruction::Op::Store, vector<shared_ptr<Value>>{slot, value},
            into));
    };

    for (const BlockId b : by_index) {
        const shared_ptr<Block> &block = cfg.block(b);
        if (block->args.empty() || b == entry) {
            continue;
        }

        const vector<Incoming> &sources = incoming[b];
        internal_assert(!sources.empty())
            << "Block " << cfg.name(b)
            << " takes arguments but nothing jumps to it";

        // How many predecessors does this block still have of its own? If the
        // branch into it survived linearization, the argument is still a real
        // phi and has to stay. Back edges count: they were kept out of figure
        // 5 but are still edges, and a loop header reached both from its
        // preheader and from its latch is exactly such a join.
        // A bypass is not a path of the program's: a landing block is still
        // the one path the fold left it, and its argument for the bypass is
        // made below, where the blends are.
        size_t remaining = 0;
        for (BlockId from = 0; from < linear.size(); from++) {
            for (const CfgEdge &e : linear[from]) {
                if (e.to == b && !(guards.contains(from) && e.index == 0)) {
                    remaining++;
                }
            }
        }
        for (const Edge &back : back_edges) {
            if (back.second == b) {
                remaining++;
            }
        }
        // The arguments to blend are the ones the block had before any
        // landing gained an argument (see `thread`): those are appended after
        // them, are handed along the edges rather than blended, and may have
        // been added to this block already, by the joins before it.
        const size_t n_args = original_args[b];
        // A call continuation is handed the returned value as its first
        // argument, which no predecessor passes. Those leading arguments have
        // nothing to blend and stay as they are.
        size_t leading = n_args;
        for (const Incoming &source : sources) {
            internal_assert(source.values.size() <= n_args)
                << "Jump from " << cfg.name(source.from) << " to "
                << cfg.name(b) << " passes " << source.values.size()
                << " arguments to a block taking " << n_args;
            leading = std::min(leading, n_args - source.values.size());
        }

        if (remaining > 1 && remaining == sources.size()) {
            // A genuine join, still selected by control flow. Whether the
            // values it merges have to keep being passed depends on what they
            // are: an argument every edge hands the same definition to is only
            // threading that definition onwards under its own name, and the
            // blocks below can go on naming it whether or not the edges say
            // so -- provided the name still means that definition once the
            // argument is gone, which it does when a dominator declares it (a
            // function parameter, or an argument of a block every path here
            // passes through). One that really does merge different values --
            // what a loop carries, above all -- has nowhere else to get them
            // from, and keeps its arguments and the values every edge passes.
            //
            // Either way the block and its edges must agree: the rewiring
            // below strips the values from a jump to a block that does not
            // keep its arguments, so a block that keeps none must declare
            // none.
            bool threaded = true;
            for (size_t j = leading; j < n_args && threaded; j++) {
                const string &name = block->args[j].name;
                for (const Incoming &source : sources) {
                    const size_t offset = n_args - source.values.size();
                    if (j < offset) {
                        continue;
                    }
                    const Value &v = *source.values[j - offset];
                    if (!std::holds_alternative<Argument>(v.data) ||
                        std::get<Argument>(v.data).name != name) {
                        threaded = false;
                        break;
                    }
                }
                if (!threaded) {
                    break;
                }
                bool declared_above = false;
                for (BlockId at = b;;) {
                    if (!dom.contains(at) || dom.idom[at] == at) {
                        break;
                    }
                    at = dom.idom[at];
                    if (declares(cfg[at], name)) {
                        declared_above = true;
                        break;
                    }
                }
                threaded = declared_above;
            }
            if (threaded) {
                block->args.erase(block->args.begin() + long(leading),
                                  block->args.begin() + long(n_args));
            } else {
                keeps_args.insert(b);
            }
            continue;
        }

        // The value a source hands this block for argument j, as it is now.
        // The snapshot was taken before any argument was blended away, so an
        // argument it names may since have been replaced -- one of the
        // source's own, or one of a dominator's that the source passed on by
        // name, as a loop body passes its header's. The replacement is
        // recorded under the block that declared the argument, so that block
        // is found the way the name would be resolved: up the source's
        // dominator chain, to the nearest block that declared it. A block on
        // the way that still declares the name has not been processed, and
        // its argument is the definition meant.
        auto value_from = [&](const Incoming &source, size_t j) {
            const size_t offset = n_args - source.values.size();
            internal_assert(j >= offset)
                << "Jump from " << cfg.name(source.from) << " to "
                << cfg.name(b) << " passes too few arguments";
            shared_ptr<Value> incoming_value = source.values[j - offset];
            if (std::holds_alternative<Argument>(incoming_value->data)) {
                const string &name =
                    std::get<Argument>(incoming_value->data).name;
                for (BlockId at = source.from;;) {
                    const auto substituted = replaced[at].find(name);
                    if (substituted != replaced[at].end()) {
                        incoming_value = substituted->second;
                        break;
                    }
                    if (declares(cfg[at], name)) {
                        break;
                    }
                    if (!dom.contains(at) || dom.idom[at] == at) {
                        break;
                    }
                    at = dom.idom[at];
                }
            }
            return incoming_value;
        };

        vector<shared_ptr<Value>> blended(n_args);
        set<const Instruction *> blends;

        // A join that kept some of its edges and lost others: a block several
        // uniform paths still reach, and that the lanes of some folded branch
        // reach along one of those paths as well. No single select chain can
        // stand for its arguments, because the folded values are not all in
        // scope at any one predecessor. What each lane brings is instead
        // treated as a variable: every original predecessor assigns it,
        // under the mask of its edge where the edge was folded -- a lane that
        // did not take that edge keeps what it had -- and the block reads it.
        // Written to a slot, so that the promotion pass
        // (SSA/PromoteAllocas.h) can rebuild the merges over the rewired
        // graph, which is exactly the SSA repair the phi needs; the slot
        // itself is gone by the end of this function. This is how Moll's
        // linearizer repairs a phi whose block keeps several predecessors.
        // A bypass changes nothing here: a skipped region stores nothing,
        // and the slot keeps what the sources before it stored.
        auto blend_through_slot = [&](size_t j) {
            const Type &type = block->args[j].type;
            bool same = true;
            for (const Incoming &source : sources) {
                same = same && same_definition(*value_from(sources[0], j),
                                               *value_from(source, j));
            }
            if (same) {
                // Threaded through under its own name: still in scope.
                blended[j] = value_from(sources[0], j);
                return;
            }
            auto slot = append(func, entry_block, Ptr_t::make(type),
                               Instruction::Op::Alloca, {});
            append_store(entry_block, slot,
                         zero_value(type, func, entry_block));
            for (const Incoming &source : sources) {
                const shared_ptr<Block> &from = cfg.block(source.from);
                shared_ptr<Value> value = value_from(source, j);
                const auto edge = masks.edge.find({source.from, b});
                if (edge != masks.edge.end()) {
                    auto held =
                        append(func, from, type, Instruction::Op::Load, {slot});
                    value = append(func, from, type, Instruction::Op::Select,
                                   {edge->second, value, held});
                }
                append_store(from, slot, value);
            }
            auto read = append(func, block, type, Instruction::Op::Load, {slot});
            blends.insert(std::get<shared_ptr<Instruction>>(read->data).get());
            blended[j] = read;
            slots_made = true;
        };

        if (remaining > 1) {
            for (size_t j = leading; j < n_args; j++) {
                blend_through_slot(j);
            }
        } else {
            // The pure latch of a uniformized loop hands the header its
            // carried values under the header's own names (see UniformLoop);
            // a lane that has left the loop is off in every mask here and
            // must keep the header's value, so that is what the blend of
            // such an argument falls through to.
            const Block *loop_header =
                latch_header[b] == NO_BLOCK ? nullptr : &cfg[latch_header[b]];
            // The sources in the order the linearized code runs them, which
            // is the block index's (see compute_block_index).
            vector<const Incoming *> in_order;
            for (const Incoming &source : sources) {
                in_order.push_back(&source);
            }
            std::sort(in_order.begin(), in_order.end(),
                      [&](const Incoming *p, const Incoming *q) {
                          return index.of[p->from] < index.of[q->from];
                      });
            for (size_t j = leading; j < n_args; j++) {
                // The blend of each source goes at the end of that source,
                // where its value and its edge mask are, rather than here at
                // the join: `x = select(mask, value, x)`, source by source in
                // running order, so that the value arriving at the join is
                // the last source's blend and each source's mask and raw
                // value die with the source. This is where ispc puts them --
                // it stores a variable under the current mask as it is
                // assigned (ctx.cpp, maskedStore, lowered to a blend "since
                // it lets us keep values in registers rather than going out
                // to the stack"). Blending at the join, as the Region
                // Vectorizer does, kept every source's mask and value live
                // until here.
                //
                // Where the path from one source to the next passes the
                // landing of a bypass, the running value is handed on as an
                // argument of the landing block (see `thread` above): the
                // blend so far from inside the skipped region, and from the
                // bypass the value as it was before the region -- what a lane
                // outside the arm would have read from the region's blends,
                // so nothing downstream can tell the arm was skipped. When
                // the region holds the join's first source there is no value
                // from before it: the bypass then hands on a zero, which no
                // lane reads. Every lane at the join comes in along one
                // source's edge; a lane of a later source takes that source's
                // value from its select, and a lane of a source in the region
                // is not there when the region is bypassed. This is the
                // Region Vectorizer's undef along a repaired phi's other
                // edge, and the one value here that stands for nothing.
                //
                // A source that does not dominate the next one for any other
                // reason -- reached by a uniform branch of the program's that
                // the next one is not on -- has nowhere to keep the running
                // value, and the argument is merged through a slot instead.
                shared_ptr<Value> value;
                BlockId value_at = NO_BLOCK; // where `value` was blended
                // The running value as each region was entered, by guard.
                map<BlockId, shared_ptr<Value>> before_region;
                const auto skipped = [&](const Gadget &g) {
                    const auto it = before_region.find(g.guard);
                    if (it != before_region.end() && it->second) {
                        return it->second;
                    }
                    return zero_value(block->args[j].type, func,
                                      cfg.block(g.guard));
                };
                if (loop_header != nullptr) {
                    // The header's value of what this argument carries: the
                    // latch hands its argument k on as the header's argument
                    // k, under the header's name for it (see the pure latch
                    // in UniformizeLoops.cpp, whose arguments are the
                    // header's with `.upd` appended). A lane in none of the
                    // sources' masks -- one that left the loop on an earlier
                    // iteration -- has to come out of the blends with this,
                    // whichever source is blended first.
                    const auto *back =
                        std::get_if<Terminator::Jump>(&block->terminator.data);
                    if (back != nullptr && back->name == loop_header->name) {
                        for (size_t k = 0; k < back->args.size() &&
                                           k < loop_header->args.size();
                             k++) {
                            const auto *a =
                                std::get_if<Argument>(&back->args[k]->data);
                            if (a == nullptr || a->name != block->args[j].name) {
                                continue;
                            }
                            const auto held = loop_header->lookups.find(
                                loop_header->args[k].name);
                            if (held != loop_header->lookups.end()) {
                                value = held->second;
                            }
                            break;
                        }
                    }
                }
                bool through_slot = false;
                for (const Incoming *source : in_order) {
                    shared_ptr<Value> incoming_value = value_from(*source, j);
                    // The running value, brought into scope at this source:
                    // threaded through the landings on the way, which is
                    // also what puts it in scope at the guard of any region
                    // this source starts, where the bypass reads it.
                    if (value && value_at != NO_BLOCK) {
                        value = thread(value, value_at, source->from,
                                       block->args[j].name, skipped, nullptr,
                                       nullptr);
                        if (!value ||
                            !linear_dom.dominates(value_at, source->from)) {
                            through_slot = true;
                            break;
                        }
                    }
                    for (const Gadget &g : gadgets) {
                        if (g.region.contains(source->from)) {
                            before_region.emplace(g.guard, value);
                        }
                    }

                    if (!value) {
                        value = incoming_value;
                        value_at = definition_block(value, source->from);
                        continue;
                    }
                    // Every path passing the same definition is the common
                    // case -- a value merely threaded through the region
                    // rather than computed on either side -- and selecting
                    // between a value and itself is both pointless and,
                    // since the argument is about to be replaced by this
                    // select, self-referential.
                    if (same_definition(*value, *incoming_value)) {
                        continue;
                    }
                    const auto edge = masks.edge.find({source->from, b});
                    internal_assert(edge != masks.edge.end())
                        << "Folded edge " << cfg.name(source->from) << "->"
                        << cfg.name(b) << " has no mask to blend on";
                    const shared_ptr<Block> &at = cfg.block(source->from);
                    value = append(func, at, block->args[j].type,
                                   Instruction::Op::Select,
                                   {edge->second, incoming_value, value});
                    value_at = source->from;
                }
                if (!through_slot && value_at != NO_BLOCK) {
                    value = thread(value, value_at, b, block->args[j].name,
                                   skipped, nullptr, nullptr);
                    through_slot = value == nullptr;
                }
                if (through_slot) {
                    blend_through_slot(j);
                    continue;
                }
                blended[j] = value;
            }
        }

        // The blends were appended; move them to the front so they precede
        // the code that uses the arguments. All of them, not just the last of
        // each chain: a blend over three predecessors is a select feeding a
        // select, and moving only the outer one would leave it reading a
        // value defined below it.
        std::stable_partition(block->instrs.begin(), block->instrs.end(),
                              [&](const shared_ptr<Instruction> &i) {
                                  return blends.count(i.get()) > 0;
                              });

        // Replace uses of the arguments with the blends, then drop them. The
        // terminator counts as a use: the value a function returns is the
        // argument of its exit block.
        //
        // Uses in the blocks this one dominates count too. A block may name
        // an argument of a dominator directly rather than take it as an
        // argument of its own -- the header of a loop names its preheader's
        // arguments that way for the values it does not change (see
        // insert_preheader) -- and once the argument is gone, the name would
        // be a reference to nothing: what the analysis reads as a value no
        // block declares, and so uniform, whatever the blend is. A dominated
        // block that declares the name itself, or is dominated by one that
        // does, means something else by it and is left alone.
        for (size_t j = leading; j < n_args; j++) {
            const string name = block->args[j].name;
            auto replace = [&](shared_ptr<Value> &v) {
                if (v && std::holds_alternative<Argument>(v->data) &&
                    std::get<Argument>(v->data).name == name) {
                    v = blended[j];
                }
            };
            auto replace_in = [&](Block &in, bool skip_blends) {
                for (const auto &instr : in.instrs) {
                    // Not inside a blend: its operands are the incoming
                    // values, one of which may well be an argument of this
                    // block with the same name, and rewriting them would make
                    // the blend select between itself.
                    if (skip_blends && blends.count(instr.get())) {
                        continue;
                    }
                    for (auto &operand : instr->operands) {
                        replace(operand);
                    }
                }
                std::visit(
                    overloads{
                        [&](std::monostate &) {},
                        [&](Terminator::Jump &t) {
                            for (auto &a : t.args) {
                                replace(a);
                            }
                        },
                        [&](Terminator::Dispatch &t) {
                            replace(t.cond);
                            for (auto &target : t.targets) {
                                for (auto &a : target.args) {
                                    replace(a);
                                }
                            }
                        },
                        [&](Terminator::Return &t) { replace(t.value); },
                        [&](Terminator::ParFor &) {},
                        [&](Terminator::Yield &) {},
                        [&](Terminator::Call &t) {
                            for (auto &a : t.call.args) {
                                replace(a);
                            }
                            for (auto &a : t.cont.args) {
                                replace(a);
                            }
                        },
                        [&](Terminator::MultiCall &t) {
                            for (auto &a : t.call.args) {
                                replace(a);
                            }
                            for (auto &vs : t.varying) {
                                for (auto &a : vs) {
                                    replace(a);
                                }
                            }
                            for (auto &k : t.keys) {
                                replace(k);
                            }
                            for (auto &a : t.cont.args) {
                                replace(a);
                            }
                        },
                    },
                    in.terminator.data);
            };

            for (BlockId other : dom.subtree(b)) {
                if (other == b) {
                    continue;
                }
                // Does a block between here and `b` -- this one included --
                // declare the name for itself?
                bool shadowed = false;
                for (BlockId at = other; at != b;) {
                    if (declares(cfg[at], name)) {
                        shadowed = true;
                        break;
                    }
                    if (!dom.contains(at) || dom.idom[at] == at) {
                        break;
                    }
                    at = dom.idom[at];
                }
                if (!shadowed) {
                    replace_in(cfg[other], /*skip_blends=*/false);
                }
            }

            replace_in(*block, /*skip_blends=*/true);
            block->lookups[name] = blended[j];
            replaced[b][name] = blended[j];
        }
        // Only the blended arguments go: a call's returned value is still
        // delivered as an argument, and so is what a landing was handed.
        block->args.erase(block->args.begin() + long(leading),
                          block->args.begin() + long(n_args));
    }

    //===------------------------------------------------------------===//
    // Rewiring
    //===------------------------------------------------------------===//

    // Retargets one jump, keeping the values it carries only when the block it
    // now goes to still has arguments to bind them to.
    auto rewire = [&](BlockId b, Terminator::Jump &jump, const CfgEdge &edge) {
        const string &to = cfg.name(edge.to);
        if (!keeps_args.contains(edge.to)) {
            jump.name = to;
            jump.args.clear();
            return;
        }
        internal_assert(jump.name == to)
            << "Linearization sent " << cfg.name(b) << " to " << to
            << ", which kept its arguments, but the edge used to go to "
            << jump.name << ": there are no values for the arguments of " << to
            << " on this path";
        jump.name = to;
    };

    for (const BlockId b : by_index) {
        const shared_ptr<Block> &block = cfg.block(b);
        const vector<CfgEdge> &edges = linear[b];
        if (edges.empty()) {
            continue; // an exit, or a latch whose only edge goes back
        }

        if (divergent_branches.contains(b)) {
            internal_assert(edges.size() == 1)
                << "A folded branch in " << cfg.name(b) << " kept "
                << edges.size() << " edges";
            // A block that still merges values cannot be the target of a fold:
            // the folded branch has no values to hand it.
            internal_assert(!keeps_args.contains(edges[0].to))
                << "A folded branch in " << cfg.name(b) << " goes to "
                << cfg.name(edges[0].to)
                << ", which still merges values through its arguments";
            block->terminator.data = Terminator::Jump{cfg.name(edges[0].to)};
            continue;
        }

        if (auto *d =
                std::get_if<Terminator::Dispatch>(&block->terminator.data)) {
            // Fewer edges than targets means one of them is a back edge, kept
            // out of figure 5 and re-inserted here by being left alone
            // (section 3.3).
            internal_assert(edges.size() <= d->targets.size());
            for (const CfgEdge &e : edges) {
                rewire(b, d->targets[e.index], e);
            }
            continue;
        }
        if (auto *j = std::get_if<Terminator::Jump>(&block->terminator.data)) {
            internal_assert(edges.size() == 1);
            rewire(b, *j, edges[0]);
        }
        if (auto *c = std::get_if<Terminator::Call>(&block->terminator.data)) {
            // Only the continuation moves: where the call goes is a matter of
            // which function is called, not of this region's control flow.
            internal_assert(edges.size() == 1);
            rewire(b, c->cont, edges[0]);
        }
        if (auto *c =
                std::get_if<Terminator::MultiCall>(&block->terminator.data)) {
            // Likewise for a run of calls, which has the one continuation.
            internal_assert(edges.size() == 1);
            rewire(b, c->cont, edges[0]);
        }
    }

    // The arguments the landings gained (see `thread`): every way into a
    // landing hands over what it was recorded as handing, in the order the
    // arguments were added.
    for (BlockId landing = 0; landing < landing_args.size(); landing++) {
        const string &landing_name = cfg.name(landing);
        for (const LandingArg &arg : landing_args[landing]) {
            for (const auto &[from, value] : arg.from) {
                bool passed = false;
                for (Terminator::Jump *jump : jumps_of(cfg[from])) {
                    if (jump->name == landing_name) {
                        jump->args.push_back(value);
                        passed = true;
                    }
                }
                internal_assert(passed)
                    << cfg.name(from) << " does not go to " << landing_name
                    << ", whose argument it was to hand a value";
            }
        }
    }

    // Predecessor lists are rebuilt from the new terminators, since the
    // rewiring above invalidated them. For the region's blocks and the
    // guards, which is what `cfg` holds by now.
    {
        const Cfg now(func);
        for (BlockId b = 0; b < cfg.size(); b++) {
            const shared_ptr<Block> &block = cfg.block(b);
            block->preds.clear();
            for (BlockId p : now.preds[now.id(*block)]) {
                block->preds.push_back(now.block(p));
            }
        }
    }

    // The slots that stood in for the arguments of partially folded joins
    // become values again, merged wherever the rewired graph merges them.
    // Before predication, which would otherwise mask their stores a second
    // time -- the select each one wraps already did.
    if (slots_made) {
        promote_allocas(func, entry_name);
    }

    //===------------------------------------------------------------===//
    // Predication of side effects
    //===------------------------------------------------------------===//

    for (const BlockId b : by_index) {
        auto mask = mask_of(b);
        if (!mask) {
            continue; // always executed with every lane on
        }
        const string &name = cfg.name(b);
        auto &instrs = cfg[b].instrs;
        for (size_t i = 0; i < instrs.size(); i++) {
            const auto &instr = instrs[i];
            // A read of one element per lane out of memory -- a gather, its
            // index differing between the lanes -- reads only for the lanes
            // that are on: a lane that is off may hold any index at all, a
            // sentinel the program only ever tests, and the original program
            // never read at it. (A read at one shared index is left alone,
            // as ispc leaves a uniform load: the address is one the program
            // reads whenever any lane reaches here.)
            if (instr->op == Instruction::Op::ExtractIdx &&
                instr->operands.size() == 2 &&
                instr->operands[0]->get_type().is_reference() &&
                divergence.is_varying(name, *instr->operands[1])) {
                instr->operands.push_back(*mask);
                continue;
            }
            // An integer division by a divisor the lanes disagree about
            // divides only for the lanes that are on. A lane that is off may
            // hold any divisor, zero included, and where a lane's result is
            // never read the machine still traps on the zero -- so the lanes
            // that are off divide by one instead, which is what ispc does
            // for the same reason. Floating-point division needs nothing: a
            // zero there gives an infinity the lane never looks at.
            if ((instr->op == Instruction::Op::Div ||
                 instr->op == Instruction::Op::Mod) &&
                instr->type.is_int_or_uint() &&
                divergence.is_varying(name, *instr->operands[1]) &&
                !std::holds_alternative<Constant>(instr->operands[1]->data)) {
                const Type &type = instr->type;
                auto one = std::make_shared<Value>(
                    type.is_uint() ? Constant{type, uint64_t(1)}
                                   : Constant{type, int64_t(1)});
                auto safe = std::make_shared<Instruction>(
                    func.get_unique_name(), type, Instruction::Op::Select,
                    vector<shared_ptr<Value>>{*mask, instr->operands[1], one},
                    cfg.block(b));
                instr->operands[1] = std::make_shared<Value>(safe);
                instrs.insert(instrs.begin() + long(i), safe);
                i++;
                continue;
            }
            // Stores, and accumulates, which are stores that read first: a
            // lane that is off must neither write nor add.
            if (instr->op != Instruction::Op::Store &&
                instr->op != Instruction::Op::AccAdd &&
                instr->op != Instruction::Op::AccMul &&
                instr->op != Instruction::Op::AccSub &&
                instr->op != Instruction::Op::AccMin &&
                instr->op != Instruction::Op::AccMax &&
                instr->op != Instruction::Op::AccArgmin &&
                instr->op != Instruction::Op::AccArgmax) {
                continue;
            }
            internal_assert(instr->operands.size() == 2)
                << "Write in " << name << " is already predicated";
            instr->operands.push_back(*mask);
        }
    }

    BlockMasks result;
    for (BlockId b = 0; b < cfg.size(); b++) {
        if (masks.block[b]) {
            result[cfg.name(b)] = masks.block[b];
        }
    }
    return result;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
