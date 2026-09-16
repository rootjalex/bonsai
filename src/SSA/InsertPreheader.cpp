#include "SSA/InsertPreheader.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// Renames one argument, and every reference to it, throughout `blocks`.
void rename_argument_in(const vector<shared_ptr<Block>> &blocks,
                        const string &from, const string &to) {
    auto rename = [&](const shared_ptr<Value> &v) {
        if (!v) {
            return;
        }
        if (auto *a = std::get_if<Argument>(&v->data); a && a->name == from) {
            a->name = to;
        }
    };

    for (const auto &block : blocks) {
        for (Argument &arg : block->args) {
            if (arg.name == from) {
                arg.name = to;
            }
        }
        const auto found = block->lookups.find(from);
        if (found != block->lookups.end()) {
            auto value = found->second;
            block->lookups.erase(found);
            block->lookups[to] = std::move(value);
        }
        for (const auto &instr : block->instrs) {
            for (const auto &operand : instr->operands) {
                rename(operand);
            }
        }
        for (const auto &[_, value] : block->lookups) {
            rename(value);
        }
        for (Terminator::Jump *jump : jumps_of(*block)) {
            for (const auto &arg : jump->args) {
                rename(arg);
            }
        }
        std::visit(overloads{
                       [&](std::monostate &) {},
                       [&](Terminator::Jump &) {},
                       [&](Terminator::Dispatch &d) { rename(d.cond); },
                       [&](Terminator::Return &r) { rename(r.value); },
                       [&](Terminator::ParFor &p) {
                           rename(p.start);
                           rename(p.end);
                           rename(p.stride);
                       },
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &c) {
                           for (const auto &a : c.call.args) {
                               rename(a);
                           }
                       },
                       [&](Terminator::MultiCall &c) {
                           for (const auto &a : c.call.args) {
                               rename(a);
                           }
                           for (const auto &vs : c.varying) {
                               for (const auto &a : vs) {
                                   rename(a);
                               }
                           }
                           for (const auto &k : c.keys) {
                               rename(k);
                           }
                       },
                   },
                   block->terminator.data);
    }
}

} // namespace

void rename_argument(Function &func, const string &region, const string &from,
                     const string &to) {
    vector<shared_ptr<Block>> blocks;
    const set<string> reach = reachable_from(region, compute_successors(func));
    for (const auto &block : func.blocks) {
        if (reach.count(block->name)) {
            blocks.push_back(block);
        }
    }
    rename_argument_in(blocks, from, to);
}

string insert_preheader(Function &func, const string &header,
                        const set<string> &back_edges) {
    const BlockMap blocks = make_block_map(func);
    auto old = blocks.at(header);

    set<string> taken;
    for (const auto &block : func.blocks) {
        taken.insert(block->name);
    }
    auto body = std::make_shared<Block>();
    body->name = header + "!loop";
    for (size_t i = 0; taken.count(body->name); i++) {
        body->name = header + "!loop" + std::to_string(i);
    }
    body->owner = old->owner;

    body->instrs = std::move(old->instrs);
    old->instrs.clear();
    for (const auto &instr : body->instrs) {
        instr->owner = body;
    }
    body->terminator = std::move(old->terminator);
    body->lookups = old->lookups;

    // Only the values the loop actually changes are carried: an argument every
    // back edge hands straight back is the same on every iteration, so the
    // blocks inside can go on naming the preheader's copy of it, which
    // dominates them. That keeps things the loop merely reads -- an array it
    // writes through, an index it does not move -- out of the loop's state.
    //
    // Handed straight back means the same value, not merely one of the same
    // name. A local the loop assigns is promoted to a block argument called
    // after the local in every block that merges it (see
    // SSA/PromoteAllocas.h), the latch included, and what the latch then
    // passes under the header's name is its own merge of the new values --
    // the very thing the loop carries. A name the loop gives a value to is
    // one some jump inside the loop passes something other than the name
    // itself for; a name every jump passes on as itself, from block to block,
    // is one value throughout, whatever redeclares it on the way.
    set<string> defined_in_loop;
    for (const string &name : back_edges) {
        Block &block = name == header ? *body : *blocks.at(name);
        for (Terminator::Jump *jump : jumps_of(block)) {
            const auto target = blocks.find(jump->name);
            if (target == blocks.end() || !back_edges.count(jump->name)) {
                continue;
            }
            // A call continuation's leading argument is the result, which no
            // jump passes, so the values line up with the last ones.
            const Block &to = jump->name == header ? *old : *target->second;
            const size_t offset = to.args.size() - jump->args.size();
            for (size_t j = 0; j < jump->args.size(); j++) {
                const string &param = to.args[j + offset].name;
                const auto *a = std::get_if<Argument>(&jump->args[j]->data);
                if (a == nullptr || a->name != param) {
                    defined_in_loop.insert(param);
                }
            }
        }
    }
    vector<bool> carried(old->args.size(), false);
    for (const string &name : back_edges) {
        Block &block = name == header ? *body : *blocks.at(name);
        for (Terminator::Jump *jump : jumps_of(block)) {
            if (jump->name != header) {
                continue;
            }
            internal_assert(jump->args.size() == old->args.size())
                << "Back edge from " << name << " passes " << jump->args.size()
                << " values to a header taking " << old->args.size();
            for (size_t k = 0; k < carried.size(); k++) {
                const auto *a = std::get_if<Argument>(&jump->args[k]->data);
                carried[k] = carried[k] || a == nullptr ||
                             a->name != old->args[k].name ||
                             defined_in_loop.count(a->name) > 0;
            }
        }
    }
    for (size_t k = 0; k < carried.size(); k++) {
        if (carried[k]) {
            body->args.push_back(old->args[k]);
        }
    }

    // The preheader keeps every argument -- they may be the function's
    // parameters -- and hands on the ones the loop carries.
    Terminator::Jump into{body->name};
    old->lookups.clear();
    for (size_t k = 0; k < old->args.size(); k++) {
        auto value = std::make_shared<Value>(old->args[k]);
        old->lookups[old->args[k].name] = value;
        if (carried[k]) {
            into.args.push_back(std::move(value));
        }
    }
    old->terminator.data = std::move(into);

    // The back edges now close on the new header; anything entering the loop
    // from outside still arrives at the preheader. The new block is included:
    // a loop whose header is its own latch has its back edge in the code that
    // was just moved.
    for (const string &name : back_edges) {
        Block &block = name == header ? *body : *blocks.at(name);
        for (Terminator::Jump *jump : jumps_of(block)) {
            if (jump->name != header) {
                continue;
            }
            jump->name = body->name;
            vector<shared_ptr<Value>> kept;
            for (size_t k = 0; k < carried.size(); k++) {
                if (carried[k]) {
                    kept.push_back(jump->args[k]);
                }
            }
            jump->args = std::move(kept);
        }
    }

    const auto at = std::find(func.blocks.begin(), func.blocks.end(), old);
    func.blocks.insert(at + 1, body);
    refresh_preds(func);

    // The preheader and the header now both have an argument for each value
    // carried into the loop, under the same name -- and they are two different
    // values, since what the loop carries changes on every iteration while
    // what the preheader holds does not. Widening tells them apart most
    // sharply: a loop-carried value that becomes per-lane leaves the
    // preheader's copy uniform. So the header's get fresh names.
    //
    // The header's rather than the preheader's, because for a function that
    // loopify() turned into a loop the preheader is the entry, and its
    // argument names are the ones the rest of the compiler refers to the
    // function's parameters by.
    //
    // Renamed in the blocks the header dominates, which is where a reference
    // to its argument can mean the header's: a name resolves to the nearest
    // declaration that dominates the use. Not in everything reachable from
    // the header, which is more -- for a loop inside another it takes in the
    // enclosing loop's header, whose own argument of that name is a different
    // value, and renaming that one alike left the outer back edge passing the
    // inner value under the outer argument's name, which reads as the loop
    // handing the value straight back and so not carrying it at all.
    const AdjacencyMap all_succs = compute_successors(func);
    const AdjacencyMap all_preds = compute_predecessors(all_succs);
    const string entry = func.blocks.front()->name;
    const DomTree dom = compute_dominator_tree(
        entry, all_succs, all_preds, reverse_postorder(entry, all_succs));
    vector<shared_ptr<Block>> dominated;
    for (const auto &block : func.blocks) {
        if (dom.idom.count(block->name) &&
            dom.dominates(body->name, block->name)) {
            dominated.push_back(block);
        }
    }
    for (const auto &block : func.blocks) {
        for (const Argument &arg : block->args) {
            taken.insert(arg.name);
        }
        for (const auto &[name, _] : block->lookups) {
            taken.insert(name);
        }
    }
    const vector<Argument> original = body->args;
    for (const Argument &arg : original) {
        string renamed = arg.name + "!loop";
        for (size_t i = 0; taken.count(renamed); i++) {
            renamed = arg.name + "!loop" + std::to_string(i);
        }
        taken.insert(renamed);
        rename_argument_in(dominated, arg.name, renamed);
    }
    return body->name;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
