#include "SSA/Linearize.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <algorithm>
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
LinearEdges partial_linearize(const BlockMap &blocks,
                              const vector<string> &by_index,
                              const map<string, size_t> &index,
                              const set<string> &divergent_branches,
                              const set<Edge> &back_edges) {
    LinearEdges linear;
    // The deferral relation, as (block, target) pairs.
    set<std::pair<string, string>> deferred;

    auto least = [&](const set<string> &candidates) {
        internal_assert(!candidates.empty()) << "no successor to pick";
        return *std::min_element(
            candidates.begin(), candidates.end(),
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
        std::visit(
            overloads{
                [&](const std::monostate &) {},
                [&](const Terminator::Jump &j) {
                    incoming[j.name].push_back({name, j.args});
                },
                [&](const Terminator::Dispatch &d) {
                    for (const auto &target : d.targets) {
                        incoming[target.name].push_back({name, target.args});
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
                    // value arrives as the continuation's first argument and
                    // is not passed here, which the blending accounts for by
                    // matching the values to the *last* arguments.
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
// definition onwards under its own name, so a name is enough to tell.
bool same_definition(const Value &a, const Value &b) {
    if (const auto *ai = std::get_if<shared_ptr<Instruction>>(&a.data)) {
        const auto *bi = std::get_if<shared_ptr<Instruction>>(&b.data);
        return bi != nullptr && (*ai)->name == (*bi)->name;
    }
    if (const auto *aa = std::get_if<Argument>(&a.data)) {
        const auto *ba = std::get_if<Argument>(&b.data);
        return ba != nullptr && aa->name == ba->name;
    }
    return false; // constants: cheap enough to select between
}

// Appends an instruction to `block` without the operand rethreading
// make_instruction does: after linearization the blocks form a chain, so a
// value defined in an earlier block is already available here and must not be
// turned into a block argument.
shared_ptr<Value> append(Function &func, const shared_ptr<Block> &block,
                         Type type, Instruction::Op op,
                         vector<shared_ptr<Value>> operands) {
    auto instr = std::make_shared<Instruction>(func.get_unique_name(),
                                               std::move(type), op,
                                               std::move(operands), block);
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
    map<string, shared_ptr<Value>> loop_masks;
    map<string, const UniformLoop *> loop_seeds;
    for (const UniformLoop &loop : loops_in) {
        for (const string &block : loop.blocks) {
            loop_masks[block] = loop.live;
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
        blocks, by_index, index, divergence.branches, back_edges);

    //===------------------------------------------------------------===//
    // Masks
    //===------------------------------------------------------------===//

    const Type bool_type = Bool_t::make();
    Masks masks;

    // Edge masks first: a divergent branch splits its source's mask by the
    // condition. Dispatch target 0 is the false side, target 1 the true side
    // (see the IfElse visitor in SSA/Convert.cpp).
    auto mask_of = [&](const string &name) -> optional<shared_ptr<Value>> {
        const auto it = masks.block.find(name);
        if (it == masks.block.end()) {
            return std::nullopt;
        }
        return it->second;
    };

    for (const string &b : by_index) {
        auto block = blocks.at(b);

        // A block of a uniformized loop runs under that loop's live mask.
        // Control dependence cannot say this: after the transform the only
        // branch deciding whether the loop body runs is the uniform one on
        // `any`, and yet a lane that left on an earlier iteration is still
        // not executing. Blocks that a divergent branch inside the loop does
        // decide get a narrower mask below, which already has the live mask
        // folded into it -- the edge masks inside the loop all start from it.
        const auto loop_mask = loop_masks.find(b);
        if (loop_mask != loop_masks.end() && !divergence.masked.count(b)) {
            masks.block[b] = loop_mask->second;
        } else if (entry_mask && !divergence.masked.count(b)) {
            // Everything in the region runs under the mask the region was
            // entered with, so a block whose own predicate is uniform still
            // carries it. Blocks that are control dependent on a divergent
            // branch get a narrower mask, computed below; the entry mask is
            // already folded into it, since the edge masks start from the
            // entry's.
            masks.block[b] = entry_mask;
        }

        // The block's own mask, from the edges it is control dependent on.
        if (divergence.masked.count(b)) {
            shared_ptr<Value> mask;
            for (const auto &[from, to] : cdep.at(b)) {
                const auto it = masks.edge.find({from, to});
                internal_assert(it != masks.edge.end())
                    << "Control dependence edge " << from << "->" << to
                    << " of " << b << " has no mask yet; the block index is "
                    << "not topological";
                mask = mask ? append(func, block, bool_type,
                                     Instruction::Op::LOr, {mask, it->second})
                            : it->second;
            }
            internal_assert(mask) << "Masked block " << b
                                  << " has no control dependences";
            masks.block[b] = mask;
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

            shared_ptr<Value> cond = dispatch->second.cond;
            if (e.index == 0) {
                // The false side: the lanes where the condition does not
                // hold. Select rather than a dedicated negation, which the
                // SSA form does not have (see the UnOp visitor in
                // SSA/Convert.cpp).
                auto t = std::make_shared<Value>(Constant{bool_type, true});
                auto f = std::make_shared<Value>(Constant{bool_type, false});
                cond = append(func, block, bool_type, Instruction::Op::Select,
                              {cond, f, t});
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
        if (remaining > 1) {
            internal_assert(remaining == sources.size())
                << "TODO: blend the arguments of " << b
                << ", which keeps several predecessors but had some of its "
                << "incoming edges folded";

            // A genuine join, still selected by control flow. Whether the
            // values it merges have to keep being passed depends on what they
            // are: an argument every edge hands the same definition to is only
            // threading that definition onwards under its own name, and the
            // blocks below can go on naming it whether or not the edges say
            // so. One that really does merge different values -- what a loop
            // carries, above all -- has nowhere else to get them from.
            for (size_t j = 0; j < block->args.size(); j++) {
                for (const Incoming &source : sources) {
                    const size_t offset =
                        block->args.size() - source.values.size();
                    if (j < offset) {
                        continue;
                    }
                    if (!std::holds_alternative<Argument>(
                            source.values[j - offset]->data) ||
                        std::get<Argument>(source.values[j - offset]->data)
                                .name != block->args[j].name) {
                        keeps_args.insert(b);
                    }
                }
            }
            continue;
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
            leading = std::min(leading, block->args.size() - source.values.size());
        }

        vector<shared_ptr<Value>> blended(block->args.size());
        set<const Instruction *> blends;
        for (size_t j = leading; j < block->args.size(); j++) {
            // Fold from the last source backwards, so the first source ends
            // up as the outermost condition.
            shared_ptr<Value> value;
            for (size_t s = sources.size(); s-- > 0;) {
                const Incoming &source = sources[s];
                const size_t offset = block->args.size() - source.values.size();
                internal_assert(j >= offset)
                    << "Jump from " << source.from << " to " << b
                    << " passes too few arguments";
                shared_ptr<Value> incoming_value = source.values[j - offset];
                if (std::holds_alternative<Argument>(incoming_value->data)) {
                    const auto substituted = replaced.find(
                        {source.from,
                         std::get<Argument>(incoming_value->data).name});
                    if (substituted != replaced.end()) {
                        incoming_value = substituted->second;
                    }
                }

                if (!value) {
                    value = incoming_value;
                    continue;
                }
                // Every path passing the same definition is the common case
                // -- a value merely threaded through the region rather than
                // computed on either side -- and selecting between a value
                // and itself is both pointless and, since the argument is
                // about to be replaced by this select, self-referential.
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
        for (size_t j = leading; j < block->args.size(); j++) {
            const string name = block->args[j].name;
            auto replace = [&](shared_ptr<Value> &v) {
                if (v && std::holds_alternative<Argument>(v->data) &&
                    std::get<Argument>(v->data).name == name) {
                    v = blended[j];
                }
            };

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
    // Predication of side effects
    //===------------------------------------------------------------===//

    for (const string &b : by_index) {
        auto mask = mask_of(b);
        if (!mask) {
            continue; // always executed with every lane on
        }
        for (const auto &instr : blocks.at(b)->instrs) {
            if (instr->op != Instruction::Op::Store) {
                continue;
            }
            internal_assert(instr->operands.size() == 2)
                << "Store in " << b << " is already predicated";
            instr->operands.push_back(*mask);
        }
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

        if (auto *d = std::get_if<Terminator::Dispatch>(&block->terminator.data)) {
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

    return masks.block;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
