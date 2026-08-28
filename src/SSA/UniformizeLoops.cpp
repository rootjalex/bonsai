#include "SSA/UniformizeLoops.h"

#include "SSA/Analysis.h"
#include "SSA/InsertPreheader.h"

#include "Utils.h"

#include <algorithm>
#include <map>
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
// something on the way in, and this way the generated code has no undefined
// values in it.
shared_ptr<Value> zero_of(const Type &type) {
    if (type.is_bool()) {
        return std::make_shared<Value>(Constant{type, false});
    }
    if (type.is_float()) {
        return std::make_shared<Value>(Constant{type, 0.0});
    }
    if (type.is_uint()) {
        return std::make_shared<Value>(Constant{type, uint64_t(0)});
    }
    if (type.is_int_or_uint()) {
        return std::make_shared<Value>(Constant{type, int64_t(0)});
    }
    internal_error << "[unimplemented] a divergent loop carries a value of "
                   << "type " << type << " out to a use after the loop; only "
                   << "values a lane can hold one of can be captured at the "
                   << "iteration it leaves";
    return nullptr;
}

shared_ptr<Value> bool_constant(bool b) {
    return std::make_shared<Value>(Constant{Bool_t::make(), b});
}

// Do two values name the same definition? A name is enough: this SSA form
// threads a definition onwards through block arguments under its own name.
bool same_value(const Value &a, const Value &b) {
    if (const auto *ai = std::get_if<shared_ptr<Instruction>>(&a.data)) {
        const auto *bi = std::get_if<shared_ptr<Instruction>>(&b.data);
        return bi != nullptr && (*ai)->name == (*bi)->name;
    }
    if (const auto *aa = std::get_if<Argument>(&a.data)) {
        const auto *ba = std::get_if<Argument>(&b.data);
        return ba != nullptr && aa->name == ba->name;
    }
    if (const auto *ac = std::get_if<Constant>(&a.data)) {
        const auto *bc = std::get_if<Constant>(&b.data);
        return bc != nullptr && ac->data == bc->data;
    }
    return false;
}

bool is_named_argument(const Value &v, const string &name) {
    const auto *a = std::get_if<Argument>(&v.data);
    return a != nullptr && a->name == name;
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
                                    const Divergence &divergence) {
    LoopUniformization result;

    // Loops are found again after each transform: uniformizing one rewrites
    // the CFG the next one is found in.
    for (;;) {
        refresh_preds(func);
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
        const LoopForest loops = compute_loop_forest(succs, preds, dom, rpo);

        // A loop is divergent when the lanes disagree about when to leave it,
        // which is exactly a loop with an exiting branch that is divergent.
        // Innermost first, so that an inner loop is already uniform -- and its
        // exits therefore already folded -- when its parent is looked at.
        const Loop *target = nullptr;
        for (const auto &[header, loop] : loops) {
            const bool divergent =
                std::any_of(loop.exits.begin(), loop.exits.end(),
                            [&](const Edge &e) {
                                return divergence.branches.count(e.first) > 0;
                            });
            if (!divergent) {
                continue;
            }
            const string outer = header;
            const bool innermost =
                std::none_of(loops.begin(), loops.end(),
                             [&](const auto &other) {
                                 return other.second.parent.has_value() &&
                                        *other.second.parent == outer;
                             });
            if (innermost) {
                target = &loop;
                break;
            }
        }
        if (target == nullptr) {
            return result;
        }

        const Loop loop = *target;
        internal_assert(loop.latches.size() == 1)
            << "[unimplemented] a divergent loop with " << loop.latches.size()
            << " latches; partial linearization needs a unique one (Moll & "
            << "Hack section 2.1), so the latches have to be merged first";
        internal_assert(!loop.parent.has_value())
            << "[unimplemented] a divergent loop nested in another loop: the "
            << "live mask of " << loop.header << " would have to be reset on "
            << "every iteration of " << *loop.parent;

        // A dedicated preheader, so that the masks can be added as header
        // arguments: the header may be the region entry, whose arguments are
        // the function's parameters and cannot grow.
        const string preheader = loop.header;
        const string header =
            insert_preheader(func, loop.header, loop.blocks);

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

        // The edges that leave, in a deterministic order.
        vector<Edge> exits;
        for (const Edge &e : loop.exits) {
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
            targets.push_back({block, "!exit." + tag, {}});
            return targets.back();
        };

        for (const Edge &exit : exits) {
            ExitTarget &t = target_for(exit.second);
            const Block &dest = *blocks.at(exit.second);
            if (t.slots.empty()) {
                t.slots.resize(dest.args.size());
                for (size_t j = 0; j < dest.args.size(); j++) {
                    t.slots[j].type = dest.args[j].type;
                }
            }
            internal_assert(t.slots.size() == dest.args.size())
                << "Exit block " << dest.name << " takes "
                << dest.args.size() << " arguments but " << t.slots.size()
                << " were recorded";

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
                    t.slots[j].tracker =
                        "!track." + tag + "." + std::to_string(j);
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
        vector<Carried> added;
        added.push_back({"!live", bool_type, bool_constant(true)});
        for (const ExitTarget &t : targets) {
            added.push_back({t.mask, bool_type, bool_constant(false)});
        }
        for (const ExitTarget &t : targets) {
            for (const ExitTarget::Slot &slot : t.slots) {
                if (!slot.tracker.empty()) {
                    added.push_back(
                        {slot.tracker, slot.type, zero_of(slot.type)});
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
            if (c.name == "!live") {
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
            vector<shared_ptr<Value>>{entry_test->lookups.at("!live")},
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
                    const auto *arg = std::get_if<Argument>(
                        &slot.invariant->data);
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
            // An if-cascade, one block per destination, testing that
            // destination's exit mask. The tests are divergent branches, but
            // they are outside the loop now and linearization treats them
            // like any other (Moll & Hack section 5).
            string next;
            for (size_t i = targets.size(); i-- > 0;) {
                const ExitTarget &t = targets[i];
                if (i + 1 == targets.size()) {
                    // Every lane left somewhere, so the last destination
                    // needs no test of its own.
                    auto tail = new_block(func, "!exit_to");
                    tail->terminator.data =
                        Terminator::Jump{t.block, arguments_for(t)};
                    func.blocks.push_back(tail);
                    next = tail->name;
                    continue;
                }
                // Registered as they are made: a fresh name is picked by
                // looking at the names already in the function, so two blocks
                // made before either is added would be given the same one.
                auto test = new_block(func, "!exit_to");
                func.blocks.push_back(test);
                auto taken = new_block(func, "!exit_to");
                func.blocks.push_back(taken);

                taken->terminator.data =
                    Terminator::Jump{t.block, arguments_for(t)};
                test->terminator.data = Terminator::Dispatch{
                    entry_test->lookups.at(t.mask),
                    {Terminator::Jump{next}, Terminator::Jump{taken->name}}};
                next = test->name;
            }
            leave = Terminator::Jump{next};
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
        for (const Edge &exit : exits) {
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
                if (c.name == "!live") {
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

        result.loops.push_back(
            {entry_test->name, live, preheader, seed_arg, std::move(inside)});
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
