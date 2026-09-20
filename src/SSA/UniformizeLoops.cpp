#include "SSA/UniformizeLoops.h"

#include "SSA/Analysis.h"
#include "SSA/InsertPreheader.h"
#include "SSA/MergeLatches.h"

#include "Utils.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// The value a tracker holds before any lane has left the loop. Nothing reads
// it -- a lane's tracker is only ever read after that lane has left, and
// leaving is what writes it -- but the header argument has to be given
// something on the way in, and an undefined value says exactly that: the
// backend builds nothing for it, and the select that keeps a tracker where
// its lane has not left has nothing to keep on the first trip.
shared_ptr<Value> seed_of(const Type &type) {
    internal_assert(type.is_bool() || type.is_numeric() ||
                    type.is<Vector_t>() || type.is<Struct_t>())
        << "[unimplemented] a divergent loop carries a value of type " << type
        << " out to a use after the loop; only values a lane can hold one of "
        << "can be captured at the iteration it leaves";
    return undef_value(type);
}

shared_ptr<Value> bool_constant(bool b) {
    return std::make_shared<Value>(Constant{Bool_t::make(), b});
}

bool is_named_argument(const Value &v, const string &name) {
    const auto *a = std::get_if<Argument>(&v.data);
    return a != nullptr && a->name == name;
}

// A name for a block argument that no block of the function declares yet.
// The masks and trackers a loop is given are named for what they are, and a
// loop inside another gets its own: two loops both calling their live mask
// `!live` would have the outer loop's latch, which comes after the inner
// loop, read the inner loop's -- exhausted by then -- as its own.
string fresh_argument(const Function &func, const string &stem) {
    set<string> taken;
    for (const auto &block : func.blocks) {
        for (const Argument &arg : block->args) {
            taken.insert(arg.name);
        }
        for (const auto &[name, _] : block->lookups) {
            taken.insert(name);
        }
    }
    string name = stem;
    for (size_t i = 0; taken.count(name); i++) {
        name = stem + "_" + std::to_string(i);
    }
    return name;
}

// A fresh block, owned by the same function and guaranteed not to collide
// with a name already in use. It is not added to the function: whoever makes
// one has to add it before making the next, since the name is chosen by
// looking at the names the function already has.
shared_ptr<Block> new_block(Function &func, const string &stem) {
    set<string> taken;
    for (const auto &block : func.blocks) {
        taken.insert(block->name);
    }
    string name = stem;
    for (size_t i = 0; taken.count(name); i++) {
        name = stem + "_" + std::to_string(i);
    }
    auto block = std::make_shared<Block>();
    block->name = std::move(name);
    block->owner = func.blocks.front()->owner;
    return block;
}

// A control flow edge by the names of its blocks, (from, to).
using NamedEdge = std::pair<string, string>;

// The loop being uniformized, by the names of its blocks: the transform
// below rewrites the function, and a name is what survives that.
struct NamedLoop {
    string header;
    set<string> latches;
    set<string> blocks;
    // Edges leaving the loop, in the analysis's order.
    vector<NamedEdge> exits;
};

NamedLoop name_loop(const Cfg &cfg, const Loop &loop) {
    NamedLoop named;
    named.header = cfg.name(loop.header);
    for (BlockId latch : loop.latches) {
        named.latches.insert(cfg.name(latch));
    }
    for (BlockId block : loop.blocks) {
        named.blocks.insert(cfg.name(block));
    }
    for (const Edge &exit : loop.exits) {
        named.exits.push_back({cfg.name(exit.first), cfg.name(exit.second)});
    }
    return named;
}

// One destination a loop is left for, and how the values its block expects
// are recovered once the exits have been folded into data flow.
struct ExitTarget {
    string block;
    // Name of the header argument holding the mask of lanes that left this
    // way.
    string mask;
    // One per argument the destination block takes.
    struct Slot {
        // The value is the same on every iteration and on every edge, so it
        // can be handed over as it is rather than captured per lane.
        shared_ptr<Value> invariant;
        // Otherwise, the name of the header argument tracking it.
        string tracker;
        Type type;
    };
    vector<Slot> slots;
};

} // namespace

LoopUniformization uniformize_loops(Function &func, const string &entry,
                                    const Analyzer &analyze) {
    LoopUniformization result;

    // Loops are found again after each transform, and so is what diverges:
    // uniformizing one rewrites the CFG the next one is found in, and the
    // dispatch it leaves over the exit masks may be a divergent exit of the
    // loop around it.
    for (;;) {
        const Divergence divergence =
            analyze(result.varying_args, result.masked_blocks());
        refresh_preds(func);
        const Cfg cfg(func, entry);
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest loops = compute_loop_forest(cfg, dom);
        const DomTree pdom = compute_post_dominator_tree(cfg);
        const ControlDependence cdep = compute_control_dependence(cfg, pdom);

        // A loop is divergent when the lanes disagree about when to leave it:
        // when some exit is taken by some lanes and not others on the same
        // iteration, which is to say the exiting block is control dependent
        // -- through blocks of the loop -- on a divergent branch inside it
        // (Moll & Hack section 5). The exiting block need not be the branch
        // itself: the dispatch over exit masks that folding an inner loop
        // leaves behind branches divergently, and the blocks it dispatches to
        // are what jump out. A branch outside the loop does not count; a loop
        // that runs under an outer condition is left by every lane that
        // entered it together.
        // BONSAI_EXPLAIN_LOOP=<header> says why that loop is being
        // uniformized: the exit found divergent, the branch inside the loop
        // it is control dependent on, and why that branch's condition varies
        // (see explain_varying). For when a loop every lane should leave
        // together is folded anyway, which costs its body a mask and makes
        // its index vary.
        const char *explain = std::getenv("BONSAI_EXPLAIN_LOOP");
        auto report = [&](const Loop &loop, BlockId exiting, BlockId culprit) {
            if (explain == nullptr || cfg.name(loop.header) != explain) {
                return;
            }
            std::cerr << "--- loop " << cfg.name(loop.header)
                      << " leaves divergently at " << cfg.name(exiting)
                      << ", control dependent on the branch in "
                      << cfg.name(culprit)
                      << ", whose condition varies because:\n";
            const auto *d = std::get_if<Terminator::Dispatch>(
                &cfg[culprit].terminator.data);
            if (d != nullptr && d->cond != nullptr) {
                explain_varying(func, divergence, cfg.name(culprit), *d->cond,
                                1);
            }
        };
        auto divergent_exit = [&](const Loop &loop, BlockId exiting) {
            if (divergence.branches.count(cfg.name(exiting))) {
                report(loop, exiting, exiting);
                return true; // the exiting block is the divergent branch
            }
            BlockSet seen(cfg.size());
            vector<BlockId> work{exiting};
            while (!work.empty()) {
                const BlockId b = work.back();
                work.pop_back();
                if (!seen.insert(b)) {
                    continue;
                }
                for (const auto &[from, to] : cdep[b]) {
                    if (!loop.blocks.contains(from)) {
                        continue;
                    }
                    if (divergence.branches.count(cfg.name(from))) {
                        report(loop, exiting, from);
                        return true;
                    }
                    work.push_back(from);
                }
            }
            return false;
        };

        // Innermost first, so that an inner loop is already uniform -- and its
        // exits therefore already folded -- when its parent is looked at:
        // among the divergent loops, one with no divergent loop inside it. (A
        // loop already made uniform still is a loop, nested where it was, so
        // "innermost of all loops" would never reach its parent.)
        vector<const Loop *> divergent;
        for (const Loop &loop : loops.loops()) {
            if (std::any_of(loop.exits.begin(), loop.exits.end(),
                            [&](const Edge &e) {
                                return divergent_exit(loop, e.first);
                            })) {
                divergent.push_back(&loop);
            }
        }
        const Loop *target = nullptr;
        for (const Loop *candidate : divergent) {
            const bool innermost = std::none_of(
                divergent.begin(), divergent.end(), [&](const Loop *other) {
                    // Is `candidate` an enclosing loop of `other`?
                    for (BlockId up = other->parent; up != NO_BLOCK;
                         up = loops.find(up)->parent) {
                        if (up == candidate->header) {
                            return true;
                        }
                    }
                    return false;
                });
            if (innermost) {
                target = candidate;
                break;
            }
        }
        if (target == nullptr) {
            return result;
        }

        const NamedLoop loop = name_loop(cfg, *target);
        // Several ways of going round again become one, through a latch of
        // their own, which the pure latch below then replaces: partial
        // linearization wants a unique back edge (Moll & Hack section 2.1).
        // Found again from the start, since the graph has changed.
        if (loop.latches.size() > 1) {
            merge_latches(func, loop.header, loop.latches);
            continue;
        }
        // A loop nested in another needs nothing more. Its live mask and
        // trackers are header arguments seeded by the jump from its preheader,
        // and the preheader is inside the enclosing loop, so they are seeded
        // afresh on every iteration of it -- with, once mask generation has
        // run, the mask the enclosing loop's body has at that point. An edge
        // that leaves both loops at once becomes an exit of this one here, and
        // an exit of the enclosing one when that is uniformized in turn.

        // The graph as this loop is found, for reading against the dump made
        // once it has been transformed (below).
        if (std::getenv("BONSAI_DUMP_UNIFORMIZE") != nullptr) {
            std::cerr << "--- before uniformizing " << loop.header << ":\n";
            func.dump(std::cerr);
        }

        // A dedicated preheader, so that the masks can be added as header
        // arguments: the header may be the region entry, whose arguments are
        // the function's parameters and cannot grow.
        const string preheader = loop.header;
        const string header = insert_preheader(func, loop.header, loop.blocks);

        // Splitting renamed the block the loop closes on, so anything the
        // analysis recorded under the old name now means the new one.
        auto renamed = [&](const string &block) {
            return block == preheader ? header : block;
        };

        BlockMap blocks = make_block_map(func);
        auto head = blocks.at(header);
        const size_t carried = head->args.size();

        const auto latch = blocks.at(renamed(*loop.latches.begin()));

        // The blocks the loop is made of, under the names they have now.
        set<string> in_loop;
        for (const string &name : loop.blocks) {
            in_loop.insert(renamed(name));
        }
        in_loop.erase(preheader);

        // The names the loop itself gives a value to: what it carries around
        // the back edge, and what its inner joins merge. Everything else an
        // argument may be called inside the loop is the same value threaded
        // through from outside -- an array it writes through, say -- under a
        // name each block on the way happens to repeat.
        set<string> defined_in_loop;
        for (const Argument &arg : head->args) {
            defined_in_loop.insert(arg.name);
        }
        for (const string &name : in_loop) {
            Block &block = *blocks.at(name);
            for (Terminator::Jump *jump : jumps_of(block)) {
                const auto target = blocks.find(jump->name);
                if (target == blocks.end() || !in_loop.count(jump->name)) {
                    continue;
                }
                // A call continuation's leading argument is the result, which
                // no jump passes, so the values line up with the last ones.
                const size_t offset =
                    target->second->args.size() - jump->args.size();
                for (size_t j = 0; j < jump->args.size(); j++) {
                    const string &param = target->second->args[j + offset].name;
                    if (!is_named_argument(*jump->args[j], param)) {
                        defined_in_loop.insert(param);
                    }
                }
            }
        }

        // Is this value the same on every iteration? Such a value needs no
        // per-lane capture on the way out of the loop -- and an array handle
        // had better not get one, since a blend of aggregates is not
        // expressible.
        auto loop_invariant = [&](const Value &v) {
            if (std::holds_alternative<Constant>(v.data)) {
                return true;
            }
            if (const auto *i = std::get_if<shared_ptr<Instruction>>(&v.data)) {
                const auto owner = (*i)->owner.lock();
                return owner == nullptr || in_loop.count(owner->name) == 0;
            }
            return defined_in_loop.count(std::get<Argument>(v.data).name) == 0;
        };

        // What the code after the loop reads of what the loop defines, made
        // explicit first. A block an exit leads to may name a value defined
        // inside the loop directly -- the early `return found` arm of a branch
        // in the body reads the body's `found`, which a join before it
        // declared and which dominated the arm -- and that is sound SSA while
        // the arm hangs off the loop. Uniformized, every exit is reached from
        // the header's test by way of the cascade below, and nothing inside
        // the body dominates it any more: the value has to leave the loop the
        // way an exit edge's arguments do, captured by a tracker at the
        // iteration the lane left on. So each such reference is threaded into
        // its block as an argument (Block::get_value), back along the
        // predecessors to where the value is defined, which makes it an
        // argument of the exit edge that the slots below then see.
        {
            set<string> loop_defined;
            for (const string &name : in_loop) {
                const Block &block = *blocks.at(name);
                for (const Argument &arg : block.args) {
                    loop_defined.insert(arg.name);
                }
                for (const auto &instr : block.instrs) {
                    if (!instr->name.empty()) {
                        loop_defined.insert(instr->name);
                    }
                }
            }
            refresh_preds(func);
            const Cfg now(func);
            set<string> after;
            for (const NamedEdge &e : loop.exits) {
                for (BlockId reached : reachable_from(now, now.id(e.second))) {
                    const string &r = now.name(reached);
                    if (!in_loop.count(r) && r != preheader && r != header) {
                        after.insert(r);
                    }
                }
            }
            for (const string &name : after) {
                Block &block = *blocks.at(name);
                auto thread = [&](shared_ptr<Value> &v) {
                    if (!v) {
                        return;
                    }
                    const string *n = nullptr;
                    Type type;
                    if (const auto *a = std::get_if<Argument>(&v->data)) {
                        n = &a->name;
                        type = a->type;
                    } else if (const auto *i =
                                   std::get_if<shared_ptr<Instruction>>(&v->data)) {
                        n = &(*i)->name;
                        type = (*i)->type;
                    }
                    if (n == nullptr || !loop_defined.count(*n)) {
                        return;
                    }
                    const string wanted = *n;
                    v = block.get_value(wanted, type);
                };
                for (const auto &instr : block.instrs) {
                    for (auto &operand : instr->operands) {
                        thread(operand);
                    }
                }
                std::visit(overloads{
                               [&](std::monostate &) {},
                               [&](Terminator::Jump &t) {
                                   for (auto &a : t.args) {
                                       thread(a);
                                   }
                               },
                               [&](Terminator::Dispatch &t) {
                                   thread(t.cond);
                                   for (auto &target : t.targets) {
                                       for (auto &a : target.args) {
                                           thread(a);
                                       }
                                   }
                               },
                               [&](Terminator::Return &t) { thread(t.value); },
                               [&](Terminator::ParFor &t) {
                                   thread(t.start);
                                   thread(t.end);
                                   thread(t.stride);
                               },
                               [&](Terminator::Yield &) {},
                               [&](Terminator::Call &t) {
                                   for (auto &a : t.call.args) {
                                       thread(a);
                                   }
                                   for (auto &a : t.cont.args) {
                                       thread(a);
                                   }
                               },
                               [&](Terminator::MultiCall &t) {
                                   for (auto &a : t.call.args) {
                                       thread(a);
                                   }
                                   for (auto &vs : t.varying) {
                                       for (auto &a : vs) {
                                           thread(a);
                                       }
                                   }
                                   for (auto &k : t.keys) {
                                       thread(k);
                                   }
                                   for (auto &a : t.cont.args) {
                                       thread(a);
                                   }
                               },
                           },
                           block.terminator.data);
            }
            // Threading adds arguments to blocks and values to jumps, which
            // the analysis of the exits below reads.
            blocks = make_block_map(func);
        }

        // The edges that leave, in a deterministic order.
        vector<NamedEdge> exits;
        for (const NamedEdge &e : loop.exits) {
            exits.push_back({renamed(e.first), e.second});
        }

        // What each destination needs handed to it. Two edges to the same
        // block share its exit mask and its trackers.
        vector<ExitTarget> targets;
        auto target_for = [&](const string &block) -> ExitTarget & {
            for (ExitTarget &t : targets) {
                if (t.block == block) {
                    return t;
                }
            }
            string tag = block;
            if (!tag.empty() && tag[0] == '!') {
                tag.erase(tag.begin());
            }
            targets.push_back({block, fresh_argument(func, "!exit." + tag), {}});
            return targets.back();
        };

        for (const NamedEdge &exit : exits) {
            ExitTarget &t = target_for(exit.second);
            const Block &dest = *blocks.at(exit.second);
            if (t.slots.empty()) {
                t.slots.resize(dest.args.size());
                for (size_t j = 0; j < dest.args.size(); j++) {
                    t.slots[j].type = dest.args[j].type;
                }
            }
            internal_assert(t.slots.size() == dest.args.size())
                << "Exit block " << dest.name << " takes " << dest.args.size()
                << " arguments but " << t.slots.size() << " were recorded";

            for (Terminator::Jump *jump : jumps_of(*blocks.at(exit.first))) {
                if (jump->name != exit.second) {
                    continue;
                }
                internal_assert(jump->args.size() == dest.args.size())
                    << "Exiting edge " << exit.first << "->" << exit.second
                    << " passes " << jump->args.size() << " values to a block "
                    << "taking " << dest.args.size();
                for (size_t j = 0; j < jump->args.size(); j++) {
                    const Value &v = *jump->args[j];
                    // A value the loop does not change is the same whichever
                    // iteration a lane leaves on, so it can be handed to the
                    // destination as it is.
                    const bool fixed = loop_invariant(v);
                    if (fixed && (t.slots[j].invariant == nullptr ||
                                  same_value(*t.slots[j].invariant, v))) {
                        t.slots[j].invariant = jump->args[j];
                    } else {
                        t.slots[j].invariant = nullptr;
                    }
                }
            }
        }

        // Name the trackers only for the slots that really need one.
        for (ExitTarget &t : targets) {
            string tag = t.block;
            if (!tag.empty() && tag[0] == '!') {
                tag.erase(tag.begin());
            }
            for (size_t j = 0; j < t.slots.size(); j++) {
                if (t.slots[j].invariant == nullptr) {
                    t.slots[j].tracker = fresh_argument(
                        func, "!track." + tag + "." + std::to_string(j));
                }
            }
        }

        //===------------------------------------------------------------===//
        // The masks, as header arguments
        //===------------------------------------------------------------===//

        const Type bool_type = Bool_t::make();
        struct Carried {
            string name;
            Type type;
            shared_ptr<Value> seed;
        };
        const string live_name = fresh_argument(func, "!live");
        vector<Carried> added;
        added.push_back({live_name, bool_type, bool_constant(true)});
        for (const ExitTarget &t : targets) {
            added.push_back({t.mask, bool_type, bool_constant(false)});
        }
        for (const ExitTarget &t : targets) {
            for (const ExitTarget::Slot &slot : t.slots) {
                if (!slot.tracker.empty()) {
                    added.push_back(
                        {slot.tracker, slot.type, seed_of(slot.type)});
                }
            }
        }

        // The masks live on a header of their own, ahead of the loop's first
        // block, holding nothing but the test that decides whether the gang
        // goes round again.
        //
        // Putting the test here rather than at the pure latch, where the paper
        // draws it, matters for what happens *after* the loop. A value a lane
        // carries out -- the tracker holding what it had when it left -- is
        // read by the code the loop exits to. Computed at the latch it would
        // be a value defined inside the loop and read outside it, which is
        // fine in SSA but not in the structured form this is lowered back to,
        // where the body of a loop is a scope. Carried into the header it is a
        // loop-carried value, and by the time the test fails it holds what the
        // last iteration put there. The loop is the same loop either way, and
        // testing on the way in also lets a gang with no live lane skip it
        // entirely.
        auto entry_test = new_block(func, "!loop");
        entry_test->args = std::move(head->args);
        head->args.clear();
        for (const Argument &arg : entry_test->args) {
            entry_test->lookups[arg.name] = std::make_shared<Value>(arg);
        }

        shared_ptr<Value> live;
        for (const Carried &c : added) {
            const Argument arg{c.type, c.name};
            entry_test->args.push_back(arg);
            auto value = std::make_shared<Value>(arg);
            entry_test->lookups[c.name] = value;
            result.varying_args.insert({entry_test->name, c.name});
            if (c.name == live_name) {
                live = value;
            }
        }

        // The loop's blocks name all of it directly: the test dominates them,
        // so nothing has to be threaded through arguments to get there.
        for (const auto &[arg_name, value] : entry_test->lookups) {
            head->lookups[arg_name] = value;
        }

        // The preheader seeds them. The live mask is seeded with `true` here
        // and corrected by mask generation to whatever mask the loop is
        // entered under; nothing else can know that yet.
        size_t seed_arg = 0;
        for (Terminator::Jump *jump : jumps_of(*blocks.at(preheader))) {
            if (jump->name != header) {
                continue;
            }
            jump->name = entry_test->name;
            seed_arg = jump->args.size();
            for (const Carried &c : added) {
                jump->args.push_back(c.seed);
            }
        }

        //===------------------------------------------------------------===//
        // The pure latch
        //===------------------------------------------------------------===//

        auto pure = new_block(func, "!latch");
        for (size_t k = 0; k < carried; k++) {
            pure->args.push_back(entry_test->args[k]);
        }
        for (const Carried &c : added) {
            pure->args.push_back(Argument{c.type, c.name + ".upd"});
            result.varying_args.insert({pure->name, c.name + ".upd"});
        }
        for (const Argument &arg : pure->args) {
            pure->lookups[arg.name] = std::make_shared<Value>(arg);
        }
        func.blocks.push_back(pure);

        // The pure latch has one edge, the back edge, and hands the updated
        // values to the test at the top; the updates themselves happen on the
        // edges into here.
        Terminator::Jump again{entry_test->name};
        for (const Argument &arg : pure->args) {
            again.args.push_back(pure->lookups.at(arg.name));
        }
        pure->terminator.data = std::move(again);

        // Going round again is a question about the gang, not about a lane:
        // the loop continues as long as any lane is still live.
        auto any = std::make_shared<Instruction>(
            func.get_unique_name(), bool_type, Instruction::Op::Any,
            vector<shared_ptr<Value>>{entry_test->lookups.at(live_name)},
            entry_test);
        entry_test->instrs.push_back(any);

        // Leaving goes to the one destination, or to a dispatch over the exit
        // masks when there is more than one.
        Terminator::Jump leave;
        auto arguments_for = [&](const ExitTarget &t) {
            vector<shared_ptr<Value>> args;
            for (size_t j = 0; j < t.slots.size(); j++) {
                const ExitTarget::Slot &slot = t.slots[j];
                if (slot.invariant != nullptr) {
                    // Named from the test's own copy of the carried value, so
                    // that it is available here.
                    const auto *arg =
                        std::get_if<Argument>(&slot.invariant->data);
                    args.push_back(arg != nullptr &&
                                           entry_test->lookups.count(arg->name)
                                       ? entry_test->lookups.at(arg->name)
                                       : slot.invariant);
                } else {
                    args.push_back(entry_test->lookups.at(slot.tracker));
                }
            }
            return args;
        };

        if (targets.size() == 1) {
            leave = Terminator::Jump{targets.front().block,
                                     arguments_for(targets.front())};
        } else {
            // An if-cascade, one test per destination but the last, each
            // dispatching on that destination's exit mask straight to the
            // destination or on to the next test; every lane left somewhere,
            // so the last destination needs no test of its own. The tests
            // are divergent branches, but they are outside the loop now and
            // linearization treats them like any other (Moll & Hack section
            // 5).
            //
            // A destination is jumped to from the test itself, with the
            // values it takes on that edge, rather than through a block of
            // its own that names them. This loop may sit inside another that
            // is uniformized after it, and an edge out of here that leaves
            // that loop too is then one of *its* exits: what the edge carries
            // is captured in trackers of the outer loop, whereas a block past
            // the edge that named the values directly would be reading
            // definitions of this loop's from outside the outer one, where
            // nothing of the sort is in scope any more.
            const ExitTarget &last = targets.back();
            Terminator::Jump onward{last.block, arguments_for(last)};
            for (size_t i = targets.size() - 1; i-- > 0;) {
                const ExitTarget &t = targets[i];
                // Registered as it is made: a fresh name is picked by looking
                // at the names already in the function.
                auto test = new_block(func, "!exit_to");
                func.blocks.push_back(test);
                test->terminator.data = Terminator::Dispatch{
                    entry_test->lookups.at(t.mask),
                    {std::move(onward),
                     Terminator::Jump{t.block, arguments_for(t)}}};
                onward = Terminator::Jump{test->name};
            }
            leave = std::move(onward);
        }

        // Target 0 is the false side of a dispatch, target 1 the true side
        // (see the IfElse visitor in SSA/Convert.cpp): with no lane left the
        // gang leaves the loop, otherwise it runs the body again.
        entry_test->terminator.data =
            Terminator::Dispatch{std::make_shared<Value>(any),
                                 {std::move(leave), Terminator::Jump{header}}};
        const auto at = std::find(func.blocks.begin(), func.blocks.end(), head);
        func.blocks.insert(at, entry_test);

        //===------------------------------------------------------------===//
        // Rebinding the edges
        //===------------------------------------------------------------===//

        // The blocks the live mask governs, collected as the edges are rebound.
        set<string> inside;

        // The back edge goes through the pure latch, carrying the masks along
        // unchanged: reaching here without exiting means no lane left on this
        // iteration.
        refresh_preds(func);
        blocks = make_block_map(func);
        for (Terminator::Jump *jump : jumps_of(*blocks.at(latch->name))) {
            if (jump->name != header) {
                continue;
            }
            jump->name = pure->name;
            for (const Carried &c : added) {
                jump->args.push_back(latch->get_value(c.name, c.type));
            }
        }

        // Each exiting edge is rebound through a break block, whose predicate
        // is the edge's own: what the block records is which lanes left along
        // that edge, and it can only record it where it is the only thing
        // being said about them.
        for (const NamedEdge &exit : exits) {
            blocks = make_block_map(func);
            auto from = blocks.at(exit.first);
            auto brk = new_block(func, "!break");
            inside.insert(brk->name);
            // Terminated up front: the predecessor lists are rebuilt from the
            // terminators below, before there is anything to say about the
            // values this block hands on.
            brk->terminator.data = Terminator::Jump{pure->name};
            func.blocks.push_back(brk);
            const ExitTarget &t = target_for(exit.second);

            // Redirect first, and only then ask for values: threading a value
            // into the break block walks its predecessors' jumps, so the edge
            // has to point here before there is anything to thread along.
            // What the edge carried is captured on the way past -- that is
            // the value of a tracker, taken at the iteration these lanes
            // leave on.
            vector<shared_ptr<Value>> carried_out;
            for (Terminator::Jump *jump : jumps_of(*from)) {
                if (jump->name != exit.second) {
                    continue;
                }
                carried_out = jump->args;
                jump->name = brk->name;
                jump->args.clear();
            }
            refresh_preds(func);

            vector<shared_ptr<Value>> out;
            // What the loop would have carried on with. No lane that arrives
            // here goes round again, so any of it will do; the current values
            // are what a tracker would capture anyway.
            for (size_t k = 0; k < carried; k++) {
                out.push_back(brk->get_value(entry_test->args[k].name,
                                             entry_test->args[k].type));
            }
            for (const Carried &c : added) {
                if (c.name == live_name) {
                    // These lanes are done.
                    out.push_back(bool_constant(false));
                } else if (c.name == t.mask) {
                    // ...and they left this way. Blending against the edge's
                    // mask turns this into `exited |= mask`.
                    out.push_back(bool_constant(true));
                } else {
                    out.push_back(brk->get_value(c.name, c.type));
                }
            }

            for (size_t j = 0; j < t.slots.size(); j++) {
                if (t.slots[j].invariant != nullptr) {
                    continue;
                }
                size_t at = carried;
                for (const Carried &c : added) {
                    if (c.name == t.slots[j].tracker) {
                        break;
                    }
                    at++;
                }
                internal_assert(j < carried_out.size())
                    << "Exiting edge " << exit.first << "->" << exit.second
                    << " carried " << carried_out.size() << " values but slot "
                    << j << " needs one";
                out[at] = carried_out[j];
            }

            std::get<Terminator::Jump>(brk->terminator.data).args =
                std::move(out);
            refresh_preds(func);
        }

        refresh_preds(func);

        // Everything the live mask governs: the loop's own blocks, and the
        // ones the transform added inside it.
        inside.insert(loop.blocks.begin(), loop.blocks.end());
        inside.erase(preheader);
        inside.insert(header);
        inside.insert(entry_test->name);
        inside.insert(pure->name);

        result.loops.push_back({entry_test->name, live, preheader, seed_arg,
                                std::move(inside), pure->name});

        // A loop inside another is transformed first and the outer one then
        // sees the result, so what goes wrong at the outer one is only
        // readable in the graph as it stands between the two.
        if (std::getenv("BONSAI_DUMP_UNIFORMIZE") != nullptr) {
            std::cerr << "--- after uniformizing " << header << ":\n";
            func.dump(std::cerr);
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
