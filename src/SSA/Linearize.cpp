#include "SSA/Linearize.h"

#include "IR/Equality.h"
#include "SSA/Analysis.h"
#include "SSA/PromoteAllocas.h"

#include "Utils.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <set>
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
// which is just a pair of block names and does not carry the successor
// index the algorithm needs.)
struct CfgEdge {
    string from;
    size_t index = 0;
    string to;
};

vector<CfgEdge> outgoing_edges(const Block &block) {
    vector<CfgEdge> edges;
    const auto succs = successors(block);
    for (size_t i = 0; i < succs.size(); i++) {
        edges.push_back({block.name, i, succs[i]});
    }
    return edges;
}

// The result of running the paper's figure 5 over one block: the edges it has
// in the linearized graph, in the same (b, i, s) form.
using LinearEdges = map<string, vector<CfgEdge>>;

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
LinearEdges partial_linearize(const BlockMap &blocks,
                              const vector<string> &by_index,
                              const map<string, size_t> &index,
                              const set<string> &divergent_branches,
                              const set<Edge> &back_edges,
                              const LoopForest &loops) {
    LinearEdges linear;
    // The deferral relation, as (block, target) pairs.
    set<std::pair<string, string>> deferred;

    auto least = [&](const set<string> &candidates) {
        internal_assert(!candidates.empty()) << "no successor to pick";
        return *std::min_element(candidates.begin(), candidates.end(),
                                 [&](const string &a, const string &b) {
                                     return index.at(a) < index.at(b);
                                 });
    };

    for (const string &b : by_index) {
        // T: what earlier blocks deferred to this one.
        set<string> T;
        for (const auto &[from, to] : deferred) {
            if (from == b) {
                T.insert(to);
            }
        }

        // Inside a loop, only the targets inside it may travel along its
        // edges; see above.
        const optional<string> in_loop = innermost_loop(loops, b);
        if (in_loop.has_value()) {
            const Loop &loop = loops.at(*in_loop);
            set<string> inside;
            for (const string &t : T) {
                if (loop.blocks.count(t)) {
                    inside.insert(t);
                } else {
                    internal_assert(b == loop.header)
                        << "Block " << b << " inside loop " << loop.header
                        << " was deferred " << t << ", which is outside it, "
                        << "and is not the header where that could be relayed";
                    for (const Edge &exit : loop.exits) {
                        deferred.insert({exit.second, t});
                    }
                }
            }
            T = std::move(inside);
        }

        // Figure 5 wants an acyclic graph, so the back edges are left out and
        // put back at their latches afterwards (section 3.3).
        vector<CfgEdge> edges;
        for (const CfgEdge &e : outgoing_edges(*blocks.at(b))) {
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
                for (const string &name : by_index) {
                    std::cerr << "  " << name
                              << (divergent_branches.count(name) ? " (divergent)"
                                                                  : "")
                              << " ->";
                    for (const CfgEdge &e : outgoing_edges(*blocks.at(name))) {
                        std::cerr << " " << e.to
                                  << (back_edges.count({name, e.to}) ? "^" : "");
                    }
                    std::cerr << "\n";
                }
                std::cerr << "--- deferred to " << b << ":";
                for (const string &t : T) {
                    std::cerr << " " << t;
                }
                std::cerr << "\n";
            }
            internal_assert(T.empty())
                << "Block " << b << " has deferred successors but no way to "
                << "reach them: the region's exits are not post-dominated by "
                << "everything deferred to them";
            continue;
        }

        if (divergent_branches.count(b) == 0) {
            // Uniform: every original successor still gets its own edge, so
            // the branch survives. This is the point of the algorithm.
            for (const CfgEdge &e : edges) {
                set<string> candidates = T;
                candidates.insert(e.to);
                const string next = least(candidates);
                linear[b].push_back({b, e.index, next});
                for (const string &t : candidates) {
                    if (t != next) {
                        deferred.insert({next, t});
                    }
                }
            }
        } else {
            // Divergent: one edge out, everything else deferred.
            set<string> candidates = T;
            for (const CfgEdge &e : edges) {
                candidates.insert(e.to);
            }
            const string next = least(candidates);
            linear[b].push_back({b, 0, next});
            for (const string &t : candidates) {
                if (t != next) {
                    deferred.insert({next, t});
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
    string from;
    // One entry per argument of the target block. A call continuation's
    // result argument has no incoming value here and is left alone.
    vector<shared_ptr<Value>> values;
};

map<string, vector<Incoming>> snapshot_arguments(const BlockMap &blocks,
                                                 const set<string> &region) {
    map<string, vector<Incoming>> incoming;
    for (const string &name : region) {
        const Block &block = *blocks.at(name);
        std::visit(overloads{
                       [&](const std::monostate &) {},
                       [&](const Terminator::Jump &j) {
                           incoming[j.name].push_back({name, j.args});
                       },
                       [&](const Terminator::Dispatch &d) {
                           for (const auto &target : d.targets) {
                               incoming[target.name].push_back(
                                   {name, target.args});
                           }
                       },
                       [&](const Terminator::Return &) {},
                       [&](const Terminator::ParFor &) {
                           internal_error << "Nested ParFor in " << name
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
                           incoming[c.cont.name].push_back({name, c.cont.args});
                       },
                       [&](const Terminator::MultiCall &c) {
                           // Like Call: the run's calls all leave the region,
                           // and the only edge inside it is the one after
                           // them.
                           incoming[c.cont.name].push_back({name, c.cont.args});
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
    // block -> its execution mask, absent when the block is always executed
    // with every lane enabled.
    map<string, shared_ptr<Value>> block;
    // (from, to) -> the mask of that edge, for edges out of divergent
    // branches; other edges carry their source block's mask.
    map<std::pair<string, string>, shared_ptr<Value>> edge;
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

} // namespace

BlockMasks linearize(Function &func, const string &entry,
                     const Divergence &divergence,
                     const shared_ptr<Value> &entry_mask,
                     const vector<UniformLoop> &loops_in) {
    if (divergence.branches.empty() && !entry_mask && loops_in.empty()) {
        return {}; // nothing diverges; the control flow is already uniform
    }

    const BlockMap blocks = make_block_map(func);
    const AdjacencyMap all_succs = compute_successors(func);
    const set<string> region = reachable_from(entry, all_succs);

    AdjacencyMap succs;
    for (const string &name : region) {
        succs[name];
        for (const string &s : all_succs.at(name)) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }
    const AdjacencyMap preds = compute_predecessors(succs);
    const vector<string> rpo = reverse_postorder(entry, succs);
    const DomTree dom = compute_dominator_tree(entry, succs, preds, rpo);
    const DomTree pdom = compute_post_dominator_tree(entry, succs, preds);
    const ControlDependence cdep = compute_control_dependence(succs, pdom);
    const LoopForest loops = compute_loop_forest(succs, preds, dom, rpo);

    // The edges figure 5 is not allowed to see, and that the rewiring below
    // must leave exactly as they are.
    set<Edge> back_edges;
    for (const auto &[header, loop] : loops) {
        for (const string &latch : loop.latches) {
            back_edges.insert({latch, header});
        }
    }

    // Every loop still standing has to be uniform, which after
    // uniformize_loops() means every loop: its exits are folded into the live
    // mask and the only branch left is the one on `any`.
    for (const auto &[header, loop] : loops) {
        for (const Edge &exit : loop.exits) {
            internal_assert(divergence.branches.count(exit.first) == 0)
                << "Loop " << header << " still leaves divergently from "
                << exit.first << "; it has to be uniformized first (Moll & "
                << "Hack section 5)";
        }
    }

    // The live mask governing each block of a uniformized loop, and where each
    // loop's mask is seeded.
    // Innermost first: the uniformization transformed inner loops before the
    // loops around them and lists them in that order, and a block of an inner
    // loop runs under the inner loop's live mask, which was seeded from the
    // outer's and has been narrowed since.
    map<string, shared_ptr<Value>> loop_masks;
    map<string, const UniformLoop *> loop_of;
    map<string, const UniformLoop *> loop_seeds;
    for (const UniformLoop &loop : loops_in) {
        for (const string &block : loop.blocks) {
            loop_masks.emplace(block, loop.live);
            loop_of.emplace(block, &loop);
        }
        loop_seeds[loop.preheader] = &loop;
    }

    // The index the algorithm walks in, which has to be dominance compact and
    // loop compact for the result to be correct (figure 8 of the paper).
    const map<string, size_t> index =
        compute_block_index(entry, succs, dom, loops);
    vector<string> by_index(index.size());
    for (const auto &[name, i] : index) {
        by_index[i] = name;
    }

    // Everything the old edges carried, before they are rewritten.
    const map<string, vector<Incoming>> incoming =
        snapshot_arguments(blocks, region);

    // The conditions of the branches being folded away, kept before their
    // terminators are replaced.
    map<string, Terminator::Dispatch> dispatches;
    for (const string &name : region) {
        if (const auto *d = std::get_if<Terminator::Dispatch>(
                &blocks.at(name)->terminator.data)) {
            dispatches[name] = *d;
        }
    }

    const LinearEdges linear = partial_linearize(
        blocks, by_index, index, divergence.branches, back_edges, loops);

    //===------------------------------------------------------------===//
    // Masks
    //===------------------------------------------------------------===//

    const Type bool_type = Bool_t::make();
    Masks masks;

    // Edge masks first: a divergent branch splits its source's mask by the
    // condition. A dispatch on a bool has target 0 as the false side and
    // target 1 the true side (see the IfElse visitor in SSA/Convert.cpp); a
    // dispatch on an integer is a switch, with target k taken on k (see the
    // SwitchStmt visitor there).
    auto mask_of = [&](const string &name) -> optional<shared_ptr<Value>> {
        const auto it = masks.block.find(name);
        if (it == masks.block.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    for (const string &b : by_index) {
        auto block = blocks.at(b);

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
        const auto loop_mask = loop_masks.find(b);
        const UniformLoop *loop_of_b =
            loop_of.count(b) ? loop_of.at(b) : nullptr;
        vector<Edge> deciding;
        if (divergence.masked.count(b)) {
            for (const auto &[from, to] : cdep.at(b)) {
                if (index.at(from) >= index.at(b)) {
                    continue; // through a back edge
                }
                if (loop_of_b != nullptr && !loop_of_b->blocks.count(from)) {
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
                    << "Control dependence edge " << from << "->" << to
                    << " of " << b << " has no mask yet; the block index is "
                    << "not topological";
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
        } else if (loop_mask != loop_masks.end()) {
            // A block of a uniformized loop runs under that loop's live mask.
            // Control dependence cannot say this: after the transform the
            // only branch deciding whether the loop body runs is the uniform
            // one on `any`, and yet a lane that left on an earlier iteration
            // is still not executing.
            masks.block[b] = loop_mask->second;
        } else if (entry_mask) {
            // Everything in the region runs under the mask the region was
            // entered with, so a block whose own predicate is uniform still
            // carries it.
            masks.block[b] = entry_mask;
        } else {
            internal_assert(!divergence.masked.count(b))
                << "Masked block " << b << " has no control dependences";
        }

        // A loop is entered under the mask of the block that jumps into it,
        // which is only known now. The transform seeded it with `true`,
        // standing for "every lane that got here".
        const auto seed = loop_seeds.find(b);
        if (seed != loop_seeds.end()) {
            if (auto m = mask_of(b)) {
                for (Terminator::Jump *jump : jumps_of(*block)) {
                    if (jump->name != seed->second->header) {
                        continue;
                    }
                    internal_assert(seed->second->seed_arg < jump->args.size())
                        << "Loop " << seed->second->header << " has no live "
                        << "mask seed at index " << seed->second->seed_arg
                        << " in the jump from " << b;
                    jump->args[seed->second->seed_arg] = *m;
                }
            }
        }

        // The masks of the edges leaving this block.
        const auto dispatch = dispatches.find(b);
        const bool folds = divergence.branches.count(b) > 0;
        for (const CfgEdge &e : outgoing_edges(*block)) {
            if (!folds || dispatch == dispatches.end()) {
                // A uniform branch does not split the lanes: every edge out
                // carries the block's own mask.
                if (auto m = mask_of(b)) {
                    masks.edge[{b, e.to}] = *m;
                }
                continue;
            }

            const shared_ptr<Value> &tag = dispatch->second.cond;
            const size_t n = dispatch->second.targets.size();
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
    // Blending
    //===------------------------------------------------------------===//
    //
    // A block argument picks a value based on which predecessor was taken. If
    // those predecessors have been folded into one path then the choice is no
    // longer made by control flow, so it becomes a select over the masks of
    // the folded edges -- every value has been computed by the time the block
    // runs, and the mask says which one this lane wanted.

    // What each blended-away argument was replaced by, so that a later block
    // taking that argument as an incoming value picks up the blend instead of
    // a name that no longer exists. Nested branches need this: the outer
    // join's incoming value is the inner join's argument.
    map<std::pair<string, string>, shared_ptr<Value>> replaced;

    // Blocks that still merge values through their arguments -- a loop header,
    // say. An edge into one of these has to go on carrying what it passes;
    // dropping it would leave the phi with nothing to merge.
    set<string> keeps_args;

    // Whether any argument was turned into a slot (see below), which the
    // promotion pass turns back into values once the graph is rewired.
    bool slots_made = false;
    const auto entry_block = blocks.at(entry);
    auto append_store = [&](const shared_ptr<Block> &into,
                            const shared_ptr<Value> &slot,
                            const shared_ptr<Value> &value) {
        into->instrs.push_back(std::make_shared<Instruction>(
            Instruction::Op::Store, vector<shared_ptr<Value>>{slot, value},
            into));
    };

    for (const string &b : by_index) {
        auto block = blocks.at(b);
        if (block->args.empty() || b == entry) {
            continue;
        }

        const auto it = incoming.find(b);
        internal_assert(it != incoming.end())
            << "Block " << b << " takes arguments but nothing jumps to it";
        const vector<Incoming> &sources = it->second;

        // How many predecessors does this block still have of its own? If the
        // branch into it survived linearization, the argument is still a real
        // phi and has to stay. Back edges count: they were kept out of figure
        // 5 but are still edges, and a loop header reached both from its
        // preheader and from its latch is exactly such a join.
        size_t remaining = 0;
        for (const auto &[from, edges] : linear) {
            for (const CfgEdge &e : edges) {
                if (e.to == b) {
                    remaining++;
                }
            }
        }
        for (const Edge &back : back_edges) {
            if (back.second == b) {
                remaining++;
            }
        }
        // A call continuation is handed the returned value as its first
        // argument, which no predecessor passes. Those leading arguments have
        // nothing to blend and stay as they are.
        size_t leading = block->args.size();
        for (const Incoming &source : sources) {
            internal_assert(source.values.size() <= block->args.size())
                << "Jump from " << source.from << " to " << b << " passes "
                << source.values.size() << " arguments to a block taking "
                << block->args.size();
            leading =
                std::min(leading, block->args.size() - source.values.size());
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
            for (size_t j = leading; j < block->args.size() && threaded; j++) {
                const string &name = block->args[j].name;
                for (const Incoming &source : sources) {
                    const size_t offset =
                        block->args.size() - source.values.size();
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
                for (string at = b;;) {
                    const auto up = dom.idom.find(at);
                    if (up == dom.idom.end() || up->second == at) {
                        break;
                    }
                    at = up->second;
                    const Block &above = *blocks.at(at);
                    if (std::any_of(above.args.begin(), above.args.end(),
                                    [&](const Argument &arg) {
                                        return arg.name == name;
                                    })) {
                        declared_above = true;
                        break;
                    }
                }
                threaded = declared_above;
            }
            if (threaded) {
                block->args.erase(block->args.begin() + leading,
                                  block->args.end());
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
            const size_t offset = block->args.size() - source.values.size();
            internal_assert(j >= offset)
                << "Jump from " << source.from << " to " << b
                << " passes too few arguments";
            shared_ptr<Value> incoming_value = source.values[j - offset];
            if (std::holds_alternative<Argument>(incoming_value->data)) {
                const string &name =
                    std::get<Argument>(incoming_value->data).name;
                for (string at = source.from;;) {
                    const auto substituted = replaced.find({at, name});
                    if (substituted != replaced.end()) {
                        incoming_value = substituted->second;
                        break;
                    }
                    const Block &here = *blocks.at(at);
                    if (std::any_of(here.args.begin(), here.args.end(),
                                    [&](const Argument &arg) {
                                        return arg.name == name;
                                    })) {
                        break;
                    }
                    const auto up = dom.idom.find(at);
                    if (up == dom.idom.end() || up->second == at) {
                        break;
                    }
                    at = up->second;
                }
            }
            return incoming_value;
        };

        vector<shared_ptr<Value>> blended(block->args.size());
        set<const Instruction *> blends;

        if (remaining > 1) {
            // A join that kept some of its edges and lost others: a block
            // several uniform paths still reach, and that the lanes of some
            // folded branch reach along one of those paths as well. No single
            // select chain can stand for its arguments, because the folded
            // values are not all in scope at any one predecessor. What each
            // lane brings is instead treated as a variable: every original
            // predecessor assigns it, under the mask of its edge where the
            // edge was folded -- a lane that did not take that edge keeps
            // what it had -- and the block reads it. Written to a slot, so
            // that the promotion pass (SSA/PromoteAllocas.h) can rebuild the
            // merges over the rewired graph, which is exactly the SSA repair
            // the phi needs; the slot itself is gone by the end of this
            // function. This is how Moll's linearizer repairs a phi whose
            // block keeps several predecessors.
            for (size_t j = leading; j < block->args.size(); j++) {
                const Type &type = block->args[j].type;
                bool same = true;
                for (const Incoming &source : sources) {
                    same = same && same_definition(*value_from(sources[0], j),
                                                   *value_from(source, j));
                }
                if (same) {
                    // Threaded through under its own name: still in scope.
                    blended[j] = value_from(sources[0], j);
                    continue;
                }
                auto slot = append(func, entry_block, Ptr_t::make(type),
                                   Instruction::Op::Alloca, {});
                append_store(entry_block, slot,
                             zero_value(type, func, entry_block));
                for (const Incoming &source : sources) {
                    const auto from = blocks.at(source.from);
                    shared_ptr<Value> value = value_from(source, j);
                    const auto edge = masks.edge.find({source.from, b});
                    if (edge != masks.edge.end()) {
                        auto held = append(func, from, type,
                                           Instruction::Op::Load, {slot});
                        value = append(func, from, type, Instruction::Op::Select,
                                       {edge->second, value, held});
                    }
                    append_store(from, slot, value);
                }
                auto read =
                    append(func, block, type, Instruction::Op::Load, {slot});
                blends.insert(std::get<shared_ptr<Instruction>>(read->data).get());
                blended[j] = read;
                slots_made = true;
            }
        } else {
            // The pure latch of a uniformized loop hands the header its
            // carried values under the header's own names (see UniformLoop);
            // a lane that has left the loop is off in every mask here and
            // must keep the header's value, so that is what the blend of
            // such an argument falls through to.
            const Block *loop_header = nullptr;
            for (const UniformLoop &loop : loops_in) {
                if (loop.latch == b) {
                    loop_header = blocks.at(loop.header).get();
                }
            }
            for (size_t j = leading; j < block->args.size(); j++) {
                // Fold from the last source backwards, so the first source
                // ends up as the outermost condition.
                shared_ptr<Value> value;
                if (loop_header != nullptr) {
                    const auto held =
                        loop_header->lookups.find(block->args[j].name);
                    if (held != loop_header->lookups.end()) {
                        value = held->second;
                    }
                }
                for (size_t s = sources.size(); s-- > 0;) {
                    const Incoming &source = sources[s];
                    shared_ptr<Value> incoming_value = value_from(source, j);

                    if (!value) {
                        value = incoming_value;
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
                    const auto edge = masks.edge.find({source.from, b});
                    internal_assert(edge != masks.edge.end())
                        << "Folded edge " << source.from << "->" << b
                        << " has no mask to blend on";
                    value = append(func, block, block->args[j].type,
                                   Instruction::Op::Select,
                                   {edge->second, incoming_value, value});
                    blends.insert(
                        std::get<shared_ptr<Instruction>>(value->data).get());
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
        for (size_t j = leading; j < block->args.size(); j++) {
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

            for (const string &other : region) {
                if (other == b || !dom.dominates(b, other)) {
                    continue;
                }
                // Does a block between here and `b` -- this one included --
                // declare the name for itself?
                bool shadowed = false;
                for (string at = other; at != b;) {
                    const Block &between = *blocks.at(at);
                    if (std::any_of(between.args.begin(), between.args.end(),
                                    [&](const Argument &arg) {
                                        return arg.name == name;
                                    })) {
                        shadowed = true;
                        break;
                    }
                    const auto up = dom.idom.find(at);
                    if (up == dom.idom.end() || up->second == at) {
                        break;
                    }
                    at = up->second;
                }
                if (!shadowed) {
                    replace_in(*blocks.at(other), /*skip_blends=*/false);
                }
            }

            for (const auto &instr : block->instrs) {
                // Not inside a blend: its operands are the incoming values,
                // one of which may well be an argument of this block with
                // the same name, and rewriting them would make the blend
                // select between itself.
                if (blends.count(instr.get())) {
                    continue;
                }
                for (auto &operand : instr->operands) {
                    replace(operand);
                }
            }
            std::visit(overloads{
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
                       block->terminator.data);
            block->lookups[name] = blended[j];
            replaced[{b, name}] = blended[j];
        }
        // Only the blended arguments go: a call's returned value is still
        // delivered as an argument.
        block->args.erase(block->args.begin() + leading, block->args.end());
    }

    //===------------------------------------------------------------===//
    // Rewiring
    //===------------------------------------------------------------===//

    // Retargets one jump, keeping the values it carries only when the block it
    // now goes to still has arguments to bind them to.
    auto rewire = [&](const string &b, Terminator::Jump &jump,
                      const CfgEdge &edge) {
        if (!keeps_args.count(edge.to)) {
            jump.name = edge.to;
            jump.args.clear();
            return;
        }
        internal_assert(jump.name == edge.to)
            << "Linearization sent " << b << " to " << edge.to
            << ", which kept its arguments, but the edge used to go to "
            << jump.name << ": there are no values for the arguments of "
            << edge.to << " on this path";
        jump.name = edge.to;
    };

    for (const string &b : by_index) {
        auto block = blocks.at(b);
        const auto it = linear.find(b);
        if (it == linear.end()) {
            continue; // an exit, or a latch whose only edge goes back
        }
        const vector<CfgEdge> &edges = it->second;

        if (divergence.branches.count(b)) {
            internal_assert(edges.size() == 1)
                << "A folded branch in " << b << " kept " << edges.size()
                << " edges";
            // A block that still merges values cannot be the target of a fold:
            // the folded branch has no values to hand it.
            internal_assert(!keeps_args.count(edges[0].to))
                << "A folded branch in " << b << " goes to " << edges[0].to
                << ", which still merges values through its arguments";
            block->terminator.data = Terminator::Jump{edges[0].to};
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

    // Predecessor lists are rebuilt from the new terminators, since the
    // rewiring above invalidated them.
    const AdjacencyMap new_preds =
        compute_predecessors(compute_successors(func));
    for (const string &name : region) {
        auto block = blocks.at(name);
        block->preds.clear();
        const auto it = new_preds.find(name);
        if (it == new_preds.end()) {
            continue;
        }
        for (const string &p : it->second) {
            block->preds.push_back(blocks.at(p));
        }
    }

    // The slots that stood in for the arguments of partially folded joins
    // become values again, merged wherever the rewired graph merges them.
    // Before predication, which would otherwise mask their stores a second
    // time -- the select each one wraps already did.
    if (slots_made) {
        promote_allocas(func, entry);
    }

    //===------------------------------------------------------------===//
    // Predication of side effects
    //===------------------------------------------------------------===//

    for (const string &b : by_index) {
        auto mask = mask_of(b);
        if (!mask) {
            continue; // always executed with every lane on
        }
        auto &instrs = blocks.at(b)->instrs;
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
                divergence.is_varying(b, *instr->operands[1])) {
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
                divergence.is_varying(b, *instr->operands[1]) &&
                !std::holds_alternative<Constant>(instr->operands[1]->data)) {
                const Type &type = instr->type;
                auto one = std::make_shared<Value>(
                    type.is_uint() ? Constant{type, uint64_t(1)}
                                   : Constant{type, int64_t(1)});
                auto safe = std::make_shared<Instruction>(
                    func.get_unique_name(), type, Instruction::Op::Select,
                    vector<shared_ptr<Value>>{*mask, instr->operands[1], one},
                    blocks.at(b));
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
                << "Write in " << b << " is already predicated";
            instr->operands.push_back(*mask);
        }
    }

    return masks.block;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
