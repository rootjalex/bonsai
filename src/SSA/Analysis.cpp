#include "SSA/Analysis.h"

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <algorithm>
#include <iostream>
#include <optional>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::tuple;
using std::vector;

namespace {

struct ValueOrigin {
    enum class Kind {
        Known = 0,
        Merged = 1,
        Unknown = 2,
    } kind = Kind::Unknown;
    // only filled for Known:
    // constant = to_string, instruction = name, argument = name
    std::string value;

    static ValueOrigin MakeArgument(const ssa::Argument &a) {
        return ValueOrigin(Kind::Known, a.name);
    }
    static ValueOrigin MakeConstant(const ssa::Constant &c) {
        std::string s = std::visit(
            overloads{
                [](const std::string &v) -> std::string { return v; },
                [](bool v) -> std::string { return v ? "true" : "false"; },
                [](auto v) -> std::string { return std::to_string(v); }},
            c.data);
        return ValueOrigin(Kind::Known, std::move(s));
    }
    static ValueOrigin MakeInstruction(const ssa::Instruction &i) {
        return ValueOrigin(Kind::Known, i.name);
    }
    static ValueOrigin MakeUnknown() { return ValueOrigin(Kind::Unknown, ""); }

    static ValueOrigin MakeMerge(const ValueOrigin &a, const ValueOrigin &b) {
        if (a.kind == Kind::Unknown) {
            return b;
        } else if (b.kind == Kind::Unknown) {
            return a;
        } else if (a != b) {
            // TODO: make unique merge here to rm constraint below.
            return ValueOrigin(Kind::Merged, "");
        }
        // Equal.
        return a;
    }
    // For call results and parfor loop indexes
    static ValueOrigin MakeMerge() {
        // TODO: make unique merge here to rm constraint below.
        return ValueOrigin(Kind::Merged, "");
    }

    bool operator==(const ValueOrigin &o) const {
        // TODO: relax Merged != restriction
        return (kind != Kind::Merged) && (kind == o.kind && value == o.value);
    }
    bool operator!=(const ValueOrigin &o) const { return !(*this == o); }

  private:
    ValueOrigin(Kind kind, std::string value)
        : kind(kind), value(std::move(value)) {}
};

// block -> [origin for arg in block.args]
using OriginMap = map<string, vector<ValueOrigin>>;

OriginMap make_origin_map(const ssa::Function &func) {
    OriginMap omap;

    const auto bmap = make_block_map(func);

    // Insert empties
    bool entry_block = true;
    for (const auto &block : func.blocks) {
        omap[block->name] =
            vector<ValueOrigin>(block->args.size(), ValueOrigin::MakeUnknown());
        if (entry_block) {
            for (size_t i = 0; i < block->args.size(); i++) {
                // These are arguments in the entry.
                omap[block->name][i] =
                    ValueOrigin::MakeArgument(block->args[i]);
            }
        }
        entry_block = false;
    }

    // Now iterate over all terminators and insert.
    // TODO: how to track uniqueness through cycles?
    // Should this be DFS, linear order, backwards order?

    for (const auto &block : func.blocks) {
        auto handle_jump = [&](const Terminator::Jump &j,
                               const bool non_drop_call) {
            // Don't care about individual jumps (except to entry!)
            internal_assert(bmap.contains(j.name)) << j.name;
            if (j.name != func.blocks[0]->name &&
                bmap.at(j.name)->preds.size() < 2) {
                return;
            }

            const size_t offset = non_drop_call ? 1 : 0;

            internal_assert(omap.contains(j.name) &&
                            omap[j.name].size() == offset + j.args.size())
                << "Bad argument count in jump to " << j.name
                << " in terminator of " << block->name;

            auto &om = omap[j.name];

            if (non_drop_call) {
                // *always* a phi node.
                om[0] = ValueOrigin::MakeMerge();
            }

            for (size_t i = 0; i < j.args.size(); i++) {
                auto v =
                    std::visit(overloads{
                                   [&](const std::shared_ptr<Instruction> &i) {
                                       return ValueOrigin::MakeInstruction(*i);
                                   },
                                   [&](const Constant &c) {
                                       return ValueOrigin::MakeConstant(c);
                                   },
                                   [&](const Argument &a) {
                                       return ValueOrigin::MakeArgument(a);
                                   },
                               },
                               j.args[i]->data);
                om[offset + i] = ValueOrigin::MakeMerge(om[offset + i], v);
            }
        };

        std::visit(
            overloads{
                [&](const std::monostate &m) {
                    internal_error
                        << "Monostate terminator found in make_origin_map";
                },
                [&](const Terminator::Jump &j) { handle_jump(j, false); },
                [&](const Terminator::Dispatch &d) {
                    for (const auto &t : d.targets) {
                        handle_jump(t, false);
                    }
                },
                [&](const Terminator::Return &r) {},
                [&](const Terminator::ParFor &p) {
                    handle_jump(p.body, true);
                    handle_jump(p.cont, false);
                },
                [&](const Terminator::Yield &y) {},
                [&](const Terminator::Call &call) {
                    handle_jump(call.cont, !call.drop);
                },
            },
            block->terminator.data);
    }

    // TODO: iterate to a fixed point or no??

    return omap;
}

} // namespace

BlockMap make_block_map(const shared_ptr<Function> &func) {
    BlockMap bmap;
    for (const auto &block : func->blocks) {
        bmap[block->name] = block;
    }
    return bmap;
}

BlockMap make_block_map(const Function &func) {
    BlockMap bmap;
    for (const auto &block : func.blocks) {
        bmap[block->name] = block;
    }
    return bmap;
}

ArgMutabilityMap get_mutability_map(const ssa::Function &func) {
    OriginMap omap = make_origin_map(func);
    /*
    for (const auto &[bname, vec] : omap) {

        std::cout << bname << " = {";
        bool first = true;
        for (const auto &v : vec) {
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << static_cast<int>(v.kind);
        }
        std::cout << "}\n";
    }
    */

    ArgMutabilityMap result;

    for (const auto &block : func.blocks) {
        const size_t num_args = block->args.size();
        if (num_args == 0) {
            result[block->name] = {};
            continue;
        }

        internal_assert(omap.contains(block->name)) << block->name;
        const auto om = omap.at(block->name);
        internal_assert(om.size() == num_args)
            << block->name << " has " << num_args
            << " but origin map stores: " << om.size() << " entries";

        std::vector<bool> is_mutable(num_args, false);

        for (size_t i = 0; i < num_args; i++) {
            is_mutable[i] = om[i].kind == ValueOrigin::Kind::Merged;
        }
        result[block->name] = std::move(is_mutable);
    }
    return result;
}

//===--------------------------------------------------------------------===//
// Control flow graph
//===--------------------------------------------------------------------===//

vector<string> successors(const Block &block) {
    return std::visit(
        overloads{
            [&](const std::monostate &) -> vector<string> {
                internal_error << "successors() on block without terminator: "
                               << block.name;
            },
            [](const Terminator::Jump &j) -> vector<string> {
                return {j.name};
            },
            [](const Terminator::Dispatch &d) -> vector<string> {
                vector<string> ss;
                ss.reserve(d.targets.size());
                for (const auto &t : d.targets) {
                    ss.push_back(t.name);
                }
                return ss;
            },
            [](const Terminator::Return &) -> vector<string> { return {}; },
            [](const Terminator::ParFor &p) -> vector<string> {
                return {p.body.name, p.cont.name};
            },
            [](const Terminator::Yield &) -> vector<string> { return {}; },
            [](const Terminator::Call &c) -> vector<string> {
                return {c.cont.name};
            },
        },
        block.terminator.data);
}

void refresh_preds(Function &func) {
    const BlockMap blocks = make_block_map(func);
    const AdjacencyMap preds = compute_predecessors(compute_successors(func));
    for (const auto &block : func.blocks) {
        block->preds.clear();
        const auto it = preds.find(block->name);
        if (it == preds.end()) {
            continue;
        }
        for (const string &p : it->second) {
            block->preds.push_back(blocks.at(p));
        }
    }
}

vector<Terminator::Jump *> jumps_of(Block &block) {
    vector<Terminator::Jump *> jumps;
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) { jumps.push_back(&j); },
                   [&](Terminator::Dispatch &d) {
                       for (auto &t : d.targets) {
                           jumps.push_back(&t);
                       }
                   },
                   [&](Terminator::Return &) {},
                   [&](Terminator::ParFor &p) {
                       jumps.push_back(&p.body);
                       jumps.push_back(&p.cont);
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) { jumps.push_back(&c.cont); },
               },
               block.terminator.data);
    return jumps;
}

AdjacencyMap compute_successors(const Function &func) {
    AdjacencyMap succs;
    for (const auto &block : func.blocks) {
        succs[block->name] = successors(*block);
    }
    return succs;
}

AdjacencyMap compute_predecessors(const AdjacencyMap &succs) {
    AdjacencyMap preds;
    for (const auto &[name, _] : succs) {
        preds[name]; // ensure every block has an entry, even if unreachable
    }
    for (const auto &[name, ss] : succs) {
        for (const auto &s : ss) {
            // Guard against terminators naming blocks that do not exist; that
            // is a malformed function, but this analysis should not crash
            // before the verifier can say so.
            if (preds.count(s)) {
                preds[s].push_back(name);
            }
        }
    }
    return preds;
}

vector<string> reverse_postorder(const string &entry,
                                 const AdjacencyMap &succs) {
    vector<string> postorder;
    set<string> visited;

    // Iterative DFS so deep CFGs cannot overflow the stack. The second element
    // is the index of the next successor to visit.
    vector<pair<string, size_t>> stack;
    if (!succs.count(entry)) {
        return {};
    }
    stack.emplace_back(entry, 0);
    visited.insert(entry);

    while (!stack.empty()) {
        auto &[name, next] = stack.back();
        const auto it = succs.find(name);
        if (it == succs.end() || next >= it->second.size()) {
            postorder.push_back(name);
            stack.pop_back();
            continue;
        }
        const string &s = it->second[next++];
        if (succs.count(s) && visited.insert(s).second) {
            stack.emplace_back(s, 0);
        }
    }

    std::reverse(postorder.begin(), postorder.end());
    return postorder;
}

set<string> reachable_from(const string &from, const AdjacencyMap &succs) {
    const vector<string> rpo = reverse_postorder(from, succs);
    return set<string>(rpo.begin(), rpo.end());
}

//===--------------------------------------------------------------------===//
// Dominance
//===--------------------------------------------------------------------===//

bool DomTree::dominates(const string &a, const string &b) const {
    if (!idom.count(a) || !idom.count(b)) {
        return false;
    }
    string cur = b;
    while (true) {
        if (cur == a) {
            return true;
        }
        const auto it = idom.find(cur);
        if (it == idom.end() || it->second == cur) {
            return false; // reached the root without finding `a`
        }
        cur = it->second;
    }
}

string DomTree::nearest_common_ancestor(const string &a,
                                        const string &b) const {
    // Walk `a`'s ancestors into a set, then walk `b`'s until one hits it.
    set<string> ancestors;
    string cur = a;
    while (true) {
        ancestors.insert(cur);
        const auto it = idom.find(cur);
        if (it == idom.end() || it->second == cur) {
            break;
        }
        cur = it->second;
    }
    cur = b;
    while (!ancestors.count(cur)) {
        const auto it = idom.find(cur);
        internal_assert(it != idom.end())
            << "nearest_common_ancestor: " << b << " is not in the tree";
        if (it->second == cur) {
            return cur; // root
        }
        cur = it->second;
    }
    return cur;
}

AdjacencyMap DomTree::children() const {
    AdjacencyMap kids;
    for (const auto &[name, _] : idom) {
        kids[name];
    }
    for (const auto &[name, parent] : idom) {
        if (parent != name) {
            kids[parent].push_back(name);
        }
    }
    return kids;
}

DomTree compute_dominator_tree(const string &entry, const AdjacencyMap &succs,
                               const AdjacencyMap &preds,
                               const vector<string> &rpo) {
    // Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm" (2001).
    DomTree tree;
    tree.root = entry;
    if (rpo.empty()) {
        return tree;
    }

    map<string, size_t> rpo_number;
    for (size_t i = 0; i < rpo.size(); i++) {
        rpo_number[rpo[i]] = i;
    }

    // `intersect` walks two nodes up the partially-built tree until they meet,
    // always advancing whichever is deeper (larger RPO number).
    map<string, string> idom;
    idom[entry] = entry;

    auto intersect = [&](string a, string b) {
        while (a != b) {
            while (rpo_number.at(a) > rpo_number.at(b)) {
                a = idom.at(a);
            }
            while (rpo_number.at(b) > rpo_number.at(a)) {
                b = idom.at(b);
            }
        }
        return a;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto &name : rpo) {
            if (name == entry) {
                continue;
            }
            const auto pit = preds.find(name);
            if (pit == preds.end()) {
                continue;
            }

            std::optional<string> new_idom;
            for (const auto &p : pit->second) {
                if (!idom.count(p)) {
                    continue; // not yet processed on this pass
                }
                new_idom = new_idom.has_value() ? intersect(*new_idom, p) : p;
            }
            if (new_idom.has_value() &&
                (!idom.count(name) || idom.at(name) != *new_idom)) {
                idom[name] = *new_idom;
                changed = true;
            }
        }
    }

    tree.idom = std::move(idom);
    return tree;
}

AdjacencyMap compute_dominance_frontier(const AdjacencyMap &preds,
                                        const DomTree &dom) {
    // Cytron, Ferrante, Rosen, Wegman & Zadeck (1991): every join block ends
    // the dominance of each of its predecessors' ancestors, up to (but not
    // including) its own immediate dominator.
    AdjacencyMap frontier;
    for (const auto &[name, _] : dom.idom) {
        frontier[name];
    }

    for (const auto &[name, ps] : preds) {
        if (ps.size() < 2 || !dom.idom.count(name)) {
            continue;
        }
        const string &stop = dom.idom.at(name);
        for (const auto &p : ps) {
            string cur = p;
            while (cur != stop && dom.idom.count(cur)) {
                auto &fs = frontier[cur];
                if (std::find(fs.begin(), fs.end(), name) == fs.end()) {
                    fs.push_back(name);
                }
                const string &parent = dom.idom.at(cur);
                if (parent == cur) {
                    break; // root
                }
                cur = parent;
            }
        }
    }
    return frontier;
}

set<string> iterated_dominance_frontier(const set<string> &defs,
                                        const AdjacencyMap &frontier) {
    set<string> result;
    vector<string> worklist(defs.begin(), defs.end());
    while (!worklist.empty()) {
        const string name = worklist.back();
        worklist.pop_back();

        const auto it = frontier.find(name);
        if (it == frontier.end()) {
            continue;
        }
        for (const string &f : it->second) {
            // A block that gains a definition this way is itself a
            // definition, so its frontier joins the worklist.
            if (result.insert(f).second) {
                worklist.push_back(f);
            }
        }
    }
    return result;
}

string virtual_exit() { return "!!exit"; }

DomTree compute_post_dominator_tree(const string &entry,
                                    const AdjacencyMap &succs,
                                    const AdjacencyMap &preds) {
    // Post-dominance is dominance on the reverse CFG. A function usually has
    // several exits, so root the reverse graph at a synthetic node that every
    // real exit flows into.
    const string exit = virtual_exit();
    const set<string> live = reachable_from(entry, succs);

    AdjacencyMap rsuccs; // reverse-CFG successors == forward predecessors
    AdjacencyMap rpreds; // reverse-CFG predecessors == forward successors
    rsuccs[exit];
    rpreds[exit];
    for (const auto &name : live) {
        rsuccs[name];
        rpreds[name];
    }

    for (const auto &name : live) {
        const auto sit = succs.find(name);
        const bool is_exit = sit == succs.end() || sit->second.empty();
        if (is_exit) {
            // Reversing (name -> exit) makes the synthetic exit the *source*,
            // so that a traversal rooted at it reaches the whole function.
            rsuccs[exit].push_back(name);
            rpreds[name].push_back(exit);
            continue;
        }
        for (const auto &s : sit->second) {
            if (!live.count(s)) {
                continue;
            }
            rsuccs[s].push_back(name);
            rpreds[name].push_back(s);
        }
    }

    // An infinite loop has no path to the exit, which would leave its blocks
    // outside the tree. Attach any block that cannot reach the exit so that
    // post-dominance stays total.
    const set<string> reaches_exit = reachable_from(exit, rsuccs);
    for (const auto &name : live) {
        if (!reaches_exit.count(name)) {
            rsuccs[exit].push_back(name);
            rpreds[name].push_back(exit);
        }
    }

    const vector<string> rpo = reverse_postorder(exit, rsuccs);
    return compute_dominator_tree(exit, rsuccs, rpreds, rpo);
}

//===--------------------------------------------------------------------===//
// Control dependence
//===--------------------------------------------------------------------===//

ControlDependence compute_control_dependence(const AdjacencyMap &succs,
                                             const DomTree &pdom) {
    // Ferrante, Ottenstein & Warren: for each edge (a, b) where b does not
    // post-dominate a, walk up the post-dominator tree from b to ipdom(a),
    // marking every node on the way as control dependent on (a, b).
    ControlDependence cdep;
    for (const auto &[name, _] : succs) {
        cdep[name];
    }

    for (const auto &[a, ss] : succs) {
        if (!pdom.idom.count(a)) {
            continue;
        }
        for (const auto &b : ss) {
            if (!pdom.idom.count(b) || pdom.dominates(b, a)) {
                continue; // b post-dominates a: not a control dependence
            }
            const string stop = pdom.idom.at(a);
            string cur = b;
            while (cur != stop && pdom.idom.count(cur)) {
                if (cdep.count(cur)) {
                    cdep[cur].insert({a, b});
                }
                const string &parent = pdom.idom.at(cur);
                if (parent == cur) {
                    break; // root
                }
                cur = parent;
            }
        }
    }
    return cdep;
}

//===--------------------------------------------------------------------===//
// Loops
//===--------------------------------------------------------------------===//

LoopForest compute_loop_forest(const AdjacencyMap &succs,
                               const AdjacencyMap &preds, const DomTree &dom,
                               const vector<string> &rpo) {
    LoopForest loops;

    // A back edge is (latch -> header) where header dominates latch. In a
    // reducible CFG these are exactly the edges that close a natural loop.
    for (const auto &name : rpo) {
        const auto sit = succs.find(name);
        if (sit == succs.end()) {
            continue;
        }
        for (const auto &s : sit->second) {
            if (!dom.dominates(s, name)) {
                continue;
            }
            Loop &loop = loops[s];
            loop.header = s;
            loop.latches.insert(name);
        }
    }

    // The body is everything that reaches a latch without passing through the
    // header, found by walking predecessors backwards from each latch.
    for (auto &[header, loop] : loops) {
        loop.blocks.insert(header);
        vector<string> stack(loop.latches.begin(), loop.latches.end());
        while (!stack.empty()) {
            const string name = stack.back();
            stack.pop_back();
            if (!loop.blocks.insert(name).second) {
                continue;
            }
            const auto pit = preds.find(name);
            if (pit == preds.end()) {
                continue;
            }
            for (const auto &p : pit->second) {
                if (p != header) {
                    stack.push_back(p);
                }
            }
        }
    }

    // Exits, and nesting. A loop's parent is the innermost other loop that
    // contains its header.
    for (auto &[header, loop] : loops) {
        for (const auto &name : loop.blocks) {
            const auto sit = succs.find(name);
            if (sit == succs.end()) {
                continue;
            }
            for (const auto &s : sit->second) {
                if (!loop.blocks.count(s)) {
                    loop.exits.insert({name, s});
                }
            }
        }

        for (const auto &[other_header, other] : loops) {
            if (other_header == header || !other.blocks.count(header)) {
                continue;
            }
            if (!loop.parent.has_value() ||
                loops.at(*loop.parent).blocks.count(other_header)) {
                loop.parent = other_header;
            }
        }
    }

    return loops;
}

std::optional<string> innermost_loop(const LoopForest &loops,
                                     const string &block) {
    std::optional<string> best;
    for (const auto &[header, loop] : loops) {
        if (!loop.blocks.count(block)) {
            continue;
        }
        // Prefer the loop nested inside all other candidates.
        if (!best.has_value() || loops.at(*best).blocks.count(header)) {
            best = header;
        }
    }
    return best;
}

//===--------------------------------------------------------------------===//
// Block index
//===--------------------------------------------------------------------===//

map<string, size_t> compute_block_index(const string &entry,
                                        const AdjacencyMap &succs,
                                        const DomTree &dom,
                                        const LoopForest &loops) {
    // Partial linearization requires a topological order (over the CFG with
    // back edges removed) in which every dominance region and every loop is a
    // contiguous range.
    //
    // Emitting blocks in dominator-tree order gives dominance compactness for
    // free: a node's whole subtree is emitted before any sibling. Among the
    // children that are ready, we pick by RPO so the result is also a valid
    // topological order, and we hold back any child outside the current loop
    // until the loop is exhausted, which gives loop compactness.
    const vector<string> rpo = reverse_postorder(entry, succs);
    map<string, size_t> rpo_number;
    for (size_t i = 0; i < rpo.size(); i++) {
        rpo_number[rpo[i]] = i;
    }

    // Forward (non-back) edge predecessor counts, so we only emit a block once
    // everything that can reach it without a back edge has been emitted.
    map<string, size_t> pending;
    for (const auto &name : rpo) {
        pending[name] = 0;
    }
    for (const auto &name : rpo) {
        const auto sit = succs.find(name);
        if (sit == succs.end()) {
            continue;
        }
        for (const auto &s : sit->second) {
            if (!pending.count(s) || dom.dominates(s, name)) {
                continue; // back edge, or unreachable
            }
            pending[s]++;
        }
    }

    const AdjacencyMap kids = dom.children();

    map<string, size_t> index;
    set<string> emitted;
    // Ready blocks, ordered by (enclosing-loop depth, RPO) so that we finish a
    // loop before leaving it.
    auto loop_depth = [&](const string &name) {
        size_t depth = 0;
        auto cur = innermost_loop(loops, name);
        while (cur.has_value()) {
            depth++;
            cur = loops.at(*cur).parent;
        }
        return depth;
    };

    set<string> ready;
    for (const auto &name : rpo) {
        if (pending.at(name) == 0) {
            ready.insert(name);
        }
    }

    while (!ready.empty()) {
        // Pick the ready block that is deepest in the loop nest, breaking ties
        // by RPO. Staying at maximum depth is what keeps loops contiguous.
        string best;
        size_t best_depth = 0;
        for (const auto &name : ready) {
            const size_t depth = loop_depth(name);
            if (best.empty() || depth > best_depth ||
                (depth == best_depth &&
                 rpo_number.at(name) < rpo_number.at(best))) {
                best = name;
                best_depth = depth;
            }
        }

        ready.erase(best);
        index[best] = index.size();
        emitted.insert(best);

        const auto sit = succs.find(best);
        if (sit == succs.end()) {
            continue;
        }
        for (const auto &s : sit->second) {
            if (!pending.count(s) || dom.dominates(s, best)) {
                continue; // back edge
            }
            if (--pending[s] == 0) {
                ready.insert(s);
            }
        }
    }

    internal_assert(index.size() == rpo.size())
        << "compute_block_index: irreducible control flow (" << index.size()
        << " of " << rpo.size() << " blocks ordered)";

    return index;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
