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
#include <queue>
#include <tuple>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::pair;
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
                [](const ssa::Undefined &) -> std::string { return "undef"; },
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

            if (!omap.contains(j.name) ||
                omap[j.name].size() != offset + j.args.size()) {
                // The whole function, since which rewrite left the edge and
                // the block disagreeing is only readable off it.
                func.dump(std::cerr);
            }
            internal_assert(omap.contains(j.name) &&
                            omap[j.name].size() == offset + j.args.size())
                << "Bad argument count in jump to " << j.name
                << " in terminator of " << block->name << " of "
                << func.blocks[0]->name << ": the block takes "
                << (omap.contains(j.name) ? omap[j.name].size() : 0)
                << " arguments and the jump passes " << j.args.size()
                << (non_drop_call ? " plus the call's result" : "");

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
                [&](const Terminator::MultiCall &call) {
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
    return make_block_map(*func);
}

BlockMap make_block_map(const Function &func) {
    BlockMap bmap;
    bmap.reserve(func.blocks.size());
    for (const auto &block : func.blocks) {
        bmap[block->name] = block;
    }
    return bmap;
}

ArgMutabilityMap get_mutability_map(const ssa::Function &func) {
    OriginMap omap = make_origin_map(func);

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
// Blocks and edges, by name
//===--------------------------------------------------------------------===//

shared_ptr<Value> passed_to(const Block &pred, const Block &block, size_t k) {
    shared_ptr<Value> result;
    std::visit(
        overloads{
            [&](const std::monostate &) {},
            [&](const Terminator::Jump &j) {
                if (j.name == block.name && k < j.args.size()) {
                    result = j.args[k];
                }
            },
            [&](const Terminator::Dispatch &d) {
                for (const auto &t : d.targets) {
                    if (t.name == block.name && k < t.args.size()) {
                        result = t.args[k];
                    }
                }
            },
            [&](const Terminator::Return &) {},
            [&](const Terminator::ParFor &p) {
                if (p.body.name == block.name) {
                    // The body's first argument is the index, which the
                    // loop defines.
                    if (k >= 1 && k - 1 < p.body.args.size()) {
                        result = p.body.args[k - 1];
                    }
                } else if (p.cont.name == block.name &&
                           k < p.cont.args.size()) {
                    result = p.cont.args[k];
                }
            },
            [&](const Terminator::Yield &) {},
            [&](const Terminator::Call &c) {
                if (c.cont.name != block.name) {
                    return;
                }
                // A kept result is the continuation's first argument, and
                // the call defines it.
                const size_t offset = c.drop ? 0 : 1;
                if (k >= offset && k - offset < c.cont.args.size()) {
                    result = c.cont.args[k - offset];
                }
            },
            [&](const Terminator::MultiCall &c) {
                if (c.cont.name == block.name && k < c.cont.args.size()) {
                    result = c.cont.args[k];
                }
            },
        },
        pred.terminator.data);
    return result;
}

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
            [](const Terminator::MultiCall &c) -> vector<string> {
                // Like Call: the callee is a separate function, so within this
                // CFG the only successor is what comes after the run.
                return {c.cont.name};
            },
        },
        block.terminator.data);
}

void refresh_preds(Function &func) {
    const Cfg cfg(func);
    for (BlockId b = 0; b < cfg.size(); b++) {
        Block &block = cfg[b];
        block.preds.clear();
        for (BlockId p : cfg.preds[b]) {
            block.preds.push_back(cfg.block(p));
        }
    }
}

size_t remove_unreachable_blocks(Function &func) {
    internal_assert(!func.blocks.empty()) << "A function with no blocks";
    // Reachable along every kind of edge, a parfor's body included: the
    // regions of one function are all its own.
    const Cfg live(func, func.blocks.front()->name);
    const size_t before = func.blocks.size();
    std::erase_if(func.blocks, [&](const std::shared_ptr<Block> &block) {
        return !live.contains(*block);
    });
    refresh_preds(func);
    return before - func.blocks.size();
}

bool is_recursive(const Function &func) {
    if (func.blocks.empty()) {
        return false;
    }
    for (const auto &block : func.blocks) {
        const auto *call = block->terminator.callee();
        if (call != nullptr && call->name == func.blocks.front()->name) {
            return true;
        }
    }
    return false;
}

void for_each_value(Terminator &terminator,
                    const std::function<void(shared_ptr<Value> &)> &fn) {
    const auto each = [&](shared_ptr<Value> &v) {
        if (v != nullptr) {
            fn(v);
        }
    };
    const auto each_jump = [&](Terminator::Jump &jump) {
        for (auto &arg : jump.args) {
            each(arg);
        }
    };
    std::visit(overloads{
                   [](std::monostate &) {},
                   [&](Terminator::Jump &j) { each_jump(j); },
                   [&](Terminator::Dispatch &d) {
                       each(d.cond);
                       for (auto &t : d.targets) {
                           each_jump(t);
                       }
                   },
                   [&](Terminator::Return &r) { each(r.value); },
                   [&](Terminator::ParFor &p) {
                       each(p.start);
                       each(p.end);
                       each(p.stride);
                       each_jump(p.body);
                       each_jump(p.cont);
                   },
                   [](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       each_jump(c.call);
                       each_jump(c.cont);
                   },
                   [&](Terminator::MultiCall &c) {
                       each_jump(c.call);
                       each_jump(c.cont);
                       for (auto &vs : c.varying) {
                           for (auto &v : vs) {
                               each(v);
                           }
                       }
                       for (auto &k : c.keys) {
                           each(k);
                       }
                   },
               },
               terminator.data);
}

void for_each_value(Block &block,
                    const std::function<void(shared_ptr<Value> &)> &fn) {
    for (const auto &instr : block.instrs) {
        for (auto &operand : instr->operands) {
            if (operand != nullptr) {
                fn(operand);
            }
        }
    }
    for_each_value(block.terminator, fn);
    for (auto &[_, value] : block.lookups) {
        if (value != nullptr) {
            fn(value);
        }
    }
}

void for_each_value(Function &func,
                    const std::function<void(shared_ptr<Value> &)> &fn) {
    for (const auto &block : func.blocks) {
        for_each_value(*block, fn);
    }
}

void replace_uses(Function &func, const Instruction *of,
                  const shared_ptr<Value> &with) {
    for_each_value(func, [&](shared_ptr<Value> &v) {
        const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
        if (held != nullptr && held->get() == of) {
            v = with;
        }
    });
    for (const auto &block : func.blocks) {
        std::erase_if(block->instrs, [&](const shared_ptr<Instruction> &in) {
            return in.get() == of;
        });
    }
}

bool has_uses(const Function &func, const Instruction *of) {
    const auto is_of = [&](const shared_ptr<Value> &v) {
        if (v == nullptr) {
            return false;
        }
        const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
        return held != nullptr && held->get() == of;
    };
    const auto in_jump = [&](const Terminator::Jump &jump) {
        return std::any_of(jump.args.begin(), jump.args.end(), is_of);
    };
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (std::any_of(instr->operands.begin(), instr->operands.end(),
                            is_of)) {
                return true;
            }
        }
        const bool used = std::visit(
            overloads{
                [](const std::monostate &) { return false; },
                [&](const Terminator::Jump &j) { return in_jump(j); },
                [&](const Terminator::Dispatch &d) {
                    return is_of(d.cond) ||
                           std::any_of(d.targets.begin(), d.targets.end(),
                                       in_jump);
                },
                [&](const Terminator::Return &r) { return is_of(r.value); },
                [&](const Terminator::ParFor &p) {
                    return is_of(p.start) || is_of(p.end) ||
                           is_of(p.stride) || in_jump(p.body) ||
                           in_jump(p.cont);
                },
                [](const Terminator::Yield &) { return false; },
                [&](const Terminator::Call &c) {
                    return in_jump(c.call) || in_jump(c.cont);
                },
                [&](const Terminator::MultiCall &c) {
                    if (in_jump(c.call) || in_jump(c.cont) ||
                        std::any_of(c.keys.begin(), c.keys.end(), is_of)) {
                        return true;
                    }
                    return std::any_of(
                        c.varying.begin(), c.varying.end(), [&](const auto &vs) {
                            return std::any_of(vs.begin(), vs.end(), is_of);
                        });
                },
            },
            block->terminator.data);
        if (used) {
            return true;
        }
    }
    return false;
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
                   [&](Terminator::MultiCall &c) { jumps.push_back(&c.cont); },
               },
               block.terminator.data);
    return jumps;
}

//===--------------------------------------------------------------------===//
// Control flow graph
//===--------------------------------------------------------------------===//

vector<BlockId> reverse_postorder(const Graph &g, BlockId from) {
    vector<BlockId> postorder;
    if (from >= g.size()) {
        return postorder;
    }
    vector<bool> visited(g.size(), false);

    // Iterative DFS so deep CFGs cannot overflow the stack. The second element
    // is the index of the next successor to visit.
    vector<pair<BlockId, size_t>> stack;
    stack.emplace_back(from, 0);
    visited[from] = true;

    while (!stack.empty()) {
        auto &[b, next] = stack.back();
        const vector<BlockId> &succs = g.succs[b];
        if (next >= succs.size()) {
            postorder.push_back(b);
            stack.pop_back();
            continue;
        }
        const BlockId s = succs[next++];
        if (!visited[s]) {
            visited[s] = true;
            stack.emplace_back(s, 0);
        }
    }

    std::reverse(postorder.begin(), postorder.end());
    return postorder;
}

BlockSet reachable_from(const Graph &g, BlockId from) {
    BlockSet reached(g.size());
    for (BlockId b : reverse_postorder(g, from)) {
        reached.insert(b);
    }
    return reached;
}

Graph Graph::from_successors(vector<vector<BlockId>> succs, BlockId entry) {
    Graph g;
    g.succs = std::move(succs);
    g.entry = entry;
    g.preds.resize(g.succs.size());
    for (BlockId a = 0; a < g.size(); a++) {
        for (BlockId s : g.succs[a]) {
            internal_assert(s < g.size())
                << "Graph::from_successors: node " << a << " has a successor "
                << s << " outside the graph of " << g.size() << " nodes";
            g.preds[s].push_back(a);
        }
    }
    g.rpo = reverse_postorder(g, entry);
    return g;
}

Cfg::Cfg(const Function &func) {
    build(func.blocks, func.blocks.empty() ? "" : func.blocks.front()->name);
}

Cfg::Cfg(const Function &func, const string &entry_name) {
    // The blocks reachable from the entry, by a walk over the terminators.
    std::unordered_map<string, const shared_ptr<Block> *> all;
    all.reserve(func.blocks.size());
    for (const auto &block : func.blocks) {
        all.emplace(block->name, &block);
    }
    vector<shared_ptr<Block>> reached;
    std::unordered_map<const Block *, bool> seen;
    vector<const shared_ptr<Block> *> stack;
    if (const auto at = all.find(entry_name); at != all.end()) {
        stack.push_back(at->second);
        seen.emplace(at->second->get(), true);
    }
    while (!stack.empty()) {
        const shared_ptr<Block> &block = *stack.back();
        stack.pop_back();
        reached.push_back(block);
        for (const string &s : successors(*block)) {
            const auto at = all.find(s);
            if (at == all.end()) {
                continue; // a jump to a block that does not exist; see build
            }
            if (seen.emplace(at->second->get(), true).second) {
                stack.push_back(at->second);
            }
        }
    }
    build(std::move(reached), entry_name);
}

void Cfg::build(vector<shared_ptr<Block>> blocks, const string &entry_name) {
    std::sort(blocks.begin(), blocks.end(),
              [](const shared_ptr<Block> &a, const shared_ptr<Block> &b) {
                  return a->name < b->name;
              });
    held = std::move(blocks);
    by_name.reserve(held.size());
    by_pointer.reserve(held.size());
    for (BlockId b = 0; b < held.size(); b++) {
        internal_assert(by_name.emplace(held[b]->name, b).second)
            << "Two blocks named " << held[b]->name;
        by_pointer.emplace(held[b].get(), b);
    }

    succs.resize(held.size());
    preds.resize(held.size());
    for (BlockId a = 0; a < held.size(); a++) {
        for (const string &s : successors(*held[a])) {
            const BlockId to = find(s);
            if (to == NO_BLOCK) {
                // A terminator naming a block that does not exist: a
                // malformed function, which the verifier reports; the
                // analyses leave the edge out rather than crash before it
                // can say so.
                continue;
            }
            succs[a].push_back(to);
            preds[to].push_back(a);
        }
    }
    entry = find(entry_name);
    rpo = reverse_postorder(*this, entry);
}

BlockId Cfg::find(const string &name) const {
    const auto it = by_name.find(name);
    return it == by_name.end() ? NO_BLOCK : it->second;
}

BlockId Cfg::find(const Block &block) const {
    const auto it = by_pointer.find(&block);
    return it == by_pointer.end() ? NO_BLOCK : it->second;
}

BlockId Cfg::id(const string &name) const {
    const BlockId b = find(name);
    internal_assert(b != NO_BLOCK) << "No block " << name << " in the graph";
    return b;
}

BlockId Cfg::id(const Block &block) const {
    const BlockId b = find(block);
    internal_assert(b != NO_BLOCK)
        << "Block " << block.name << " is not in the graph";
    return b;
}

BlockId Cfg::add_block(const shared_ptr<Block> &block) {
    internal_assert(!contains(block->name))
        << "Block " << block->name << " is already in the graph";
    const BlockId b = BlockId(held.size());
    held.push_back(block);
    by_name.emplace(block->name, b);
    by_pointer.emplace(block.get(), b);
    succs.emplace_back();
    preds.emplace_back();
    return b;
}

//===--------------------------------------------------------------------===//
// Dominance
//===--------------------------------------------------------------------===//

void DomTree::number() {
    kids.assign(idom.size(), {});
    range.assign(idom.size(), {0, 0});
    for (BlockId b = 0; b < idom.size(); b++) {
        if (idom[b] != NO_BLOCK && idom[b] != b) {
            kids[idom[b]].push_back(b);
        }
    }
    if (!contains(root)) {
        return;
    }
    // Iterative preorder walk, so a deep tree cannot overflow the stack. The
    // second element is the index of the next child to visit.
    uint32_t next_number = 0;
    vector<pair<BlockId, size_t>> stack;
    range[root].first = next_number++;
    stack.emplace_back(root, 0);
    while (!stack.empty()) {
        const BlockId b = stack.back().first;
        const size_t next = stack.back().second++;
        const vector<BlockId> &children_of = kids[b];
        if (next < children_of.size()) {
            const BlockId child = children_of[next];
            range[child].first = next_number++;
            stack.emplace_back(child, 0);
        } else {
            range[b].second = next_number;
            stack.pop_back();
        }
    }
}

vector<BlockId> DomTree::subtree(BlockId a) const {
    vector<BlockId> blocks;
    if (!contains(a)) {
        return blocks;
    }
    vector<BlockId> stack{a};
    while (!stack.empty()) {
        const BlockId b = stack.back();
        stack.pop_back();
        blocks.push_back(b);
        const vector<BlockId> &children_of = kids[b];
        // Pushed in reverse so that they are visited in id order.
        for (auto it = children_of.rbegin(); it != children_of.rend(); ++it) {
            stack.push_back(*it);
        }
    }
    return blocks;
}

DomTree compute_dominator_tree(const Graph &g) {
    // Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm" (2001).
    DomTree tree;
    tree.root = g.entry;
    tree.idom.assign(g.size(), NO_BLOCK);
    if (g.rpo.empty()) {
        tree.number();
        return tree;
    }

    vector<size_t> rpo_number(g.size(), 0);
    for (size_t i = 0; i < g.rpo.size(); i++) {
        rpo_number[g.rpo[i]] = i;
    }

    // `intersect` walks two nodes up the partially-built tree until they meet,
    // always advancing whichever is deeper (larger RPO number).
    vector<BlockId> &idom = tree.idom;
    idom[g.entry] = g.entry;

    auto intersect = [&](BlockId a, BlockId b) {
        while (a != b) {
            while (rpo_number[a] > rpo_number[b]) {
                a = idom[a];
            }
            while (rpo_number[b] > rpo_number[a]) {
                b = idom[b];
            }
        }
        return a;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (BlockId b : g.rpo) {
            if (b == g.entry) {
                continue;
            }
            BlockId new_idom = NO_BLOCK;
            for (BlockId p : g.preds[b]) {
                if (idom[p] == NO_BLOCK) {
                    continue; // not yet processed on this pass
                }
                new_idom = new_idom == NO_BLOCK ? p : intersect(new_idom, p);
            }
            if (new_idom != NO_BLOCK && idom[b] != new_idom) {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    tree.number();
    return tree;
}

bool known_nonempty(const Cfg &cfg, const DomTree &dom, BlockId block,
                    const Value &mask) {
    for (BlockId at = block; dom.contains(at) && dom.idom[at] != at;
         at = dom.idom[at]) {
        const BlockId above = dom.idom[at];
        const auto *d =
            std::get_if<Terminator::Dispatch>(&cfg[above].terminator.data);
        if (d == nullptr || d->targets.size() != 2) {
            continue;
        }
        // The true side, and only that side: a block both sides go to, or
        // one reached some other way as well, says nothing about the test.
        if (d->targets[1].name != cfg.name(at) ||
            d->targets[0].name == cfg.name(at) || cfg.preds[at].size() != 1) {
            continue;
        }
        const auto *any = std::get_if<shared_ptr<Instruction>>(&d->cond->data);
        if (any != nullptr && (*any)->op == Instruction::Op::Any &&
            (*any)->operands.size() == 1 &&
            same_value(*(*any)->operands[0], mask)) {
            return true;
        }
    }
    return false;
}

DominanceFrontier compute_dominance_frontier(const Graph &g,
                                             const DomTree &dom) {
    // Cytron, Ferrante, Rosen, Wegman & Zadeck (1991): every join block ends
    // the dominance of each of its predecessors' ancestors, up to (but not
    // including) its own immediate dominator.
    DominanceFrontier frontier(g.size());
    for (BlockId join = 0; join < g.size(); join++) {
        const vector<BlockId> &ps = g.preds[join];
        if (ps.size() < 2 || !dom.contains(join)) {
            continue;
        }
        const BlockId stop = dom.idom[join];
        for (BlockId p : ps) {
            BlockId cur = p;
            while (cur != stop && dom.contains(cur)) {
                vector<BlockId> &fs = frontier[cur];
                if (std::find(fs.begin(), fs.end(), join) == fs.end()) {
                    fs.push_back(join);
                }
                const BlockId parent = dom.idom[cur];
                if (parent == cur) {
                    break; // root
                }
                cur = parent;
            }
        }
    }
    return frontier;
}

BlockSet iterated_dominance_frontier(const BlockSet &defs,
                                     const DominanceFrontier &frontier) {
    BlockSet result(frontier.size());
    vector<BlockId> worklist(defs.begin(), defs.end());
    while (!worklist.empty()) {
        const BlockId b = worklist.back();
        worklist.pop_back();
        if (b >= frontier.size()) {
            continue;
        }
        for (BlockId f : frontier[b]) {
            // A block that gains a definition this way is itself a
            // definition, so its frontier joins the worklist.
            if (result.insert(f)) {
                worklist.push_back(f);
            }
        }
    }
    return result;
}

string virtual_exit() { return "!!exit"; }

DomTree compute_post_dominator_tree(const Graph &g) {
    // Post-dominance is dominance on the reverse CFG. A function usually has
    // several exits, so root the reverse graph at a synthetic node that every
    // real exit flows into.
    const BlockId exit = BlockId(g.size());
    const BlockSet live = reachable_from(g, g.entry);

    // Reverse-CFG successors are forward predecessors, and the other way
    // about.
    vector<vector<BlockId>> rsuccs(g.size() + 1);
    for (BlockId b : live) {
        if (g.succs[b].empty()) {
            // Reversing (b -> exit) makes the synthetic exit the *source*,
            // so that a traversal rooted at it reaches the whole function.
            rsuccs[exit].push_back(b);
            continue;
        }
        for (BlockId s : g.succs[b]) {
            if (live.contains(s)) {
                rsuccs[s].push_back(b);
            }
        }
    }

    // An infinite loop has no path to the exit, which would leave its blocks
    // outside the tree. Attach any block that cannot reach the exit so that
    // post-dominance stays total.
    {
        Graph so_far;
        so_far.succs = rsuccs;
        const BlockSet reaches_exit = reachable_from(so_far, exit);
        for (BlockId b : live) {
            if (!reaches_exit.contains(b)) {
                rsuccs[exit].push_back(b);
            }
        }
    }

    return compute_dominator_tree(Graph::from_successors(std::move(rsuccs), exit));
}

//===--------------------------------------------------------------------===//
// Control dependence
//===--------------------------------------------------------------------===//

ControlDependence compute_control_dependence(const Graph &g,
                                             const DomTree &pdom) {
    // Ferrante, Ottenstein & Warren: for each edge (a, b) where b does not
    // post-dominate a, walk up the post-dominator tree from b to ipdom(a),
    // marking every node on the way as control dependent on (a, b).
    ControlDependence cdep(g.size());
    for (BlockId a = 0; a < g.size(); a++) {
        if (!pdom.contains(a)) {
            continue;
        }
        for (BlockId b : g.succs[a]) {
            if (!pdom.contains(b) || pdom.dominates(b, a)) {
                continue; // b post-dominates a: not a control dependence
            }
            const BlockId stop = pdom.idom[a];
            BlockId cur = b;
            while (cur != stop && pdom.contains(cur)) {
                if (cur < g.size()) {
                    cdep[cur].push_back({a, b});
                }
                const BlockId parent = pdom.idom[cur];
                if (parent == cur) {
                    break; // root
                }
                cur = parent;
            }
        }
    }
    for (vector<Edge> &edges : cdep) {
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }
    return cdep;
}

//===--------------------------------------------------------------------===//
// Loops
//===--------------------------------------------------------------------===//

const Loop *LoopForest::find(BlockId header) const {
    if (header >= position_of.size() || position_of[header] == NO_BLOCK) {
        return nullptr;
    }
    return &all[position_of[header]];
}

const Loop *LoopForest::innermost(BlockId b) const {
    if (b >= innermost_of.size() || innermost_of[b] == NO_BLOCK) {
        return nullptr;
    }
    return find(innermost_of[b]);
}

size_t LoopForest::depth(BlockId b) const {
    size_t depth = 0;
    for (const Loop *loop = innermost(b); loop != nullptr;
         loop = find(loop->parent)) {
        depth++;
    }
    return depth;
}

LoopForest compute_loop_forest(const Graph &g, const DomTree &dom) {
    LoopForest forest;
    forest.innermost_of.assign(g.size(), NO_BLOCK);
    forest.position_of.assign(g.size(), NO_BLOCK);

    // A back edge is (latch -> header) where header dominates latch. In a
    // reducible CFG these are exactly the edges that close a natural loop.
    map<BlockId, Loop> by_header;
    for (BlockId b : g.rpo) {
        for (BlockId s : g.succs[b]) {
            if (!dom.dominates(s, b)) {
                continue;
            }
            Loop &loop = by_header[s];
            loop.header = s;
            loop.latches.push_back(b);
        }
    }

    vector<Loop> &loops = forest.all;
    loops.reserve(by_header.size());
    for (auto &[header, loop] : by_header) {
        std::sort(loop.latches.begin(), loop.latches.end());
        loop.latches.erase(
            std::unique(loop.latches.begin(), loop.latches.end()),
            loop.latches.end());
        loops.push_back(std::move(loop));
    }
    for (size_t i = 0; i < loops.size(); i++) {
        forest.position_of[loops[i].header] = BlockId(i);
    }

    // The body is everything that reaches a latch without passing through the
    // header, found by walking predecessors backwards from each latch.
    for (Loop &loop : loops) {
        loop.blocks = BlockSet(g.size());
        loop.blocks.insert(loop.header);
        vector<BlockId> stack(loop.latches.begin(), loop.latches.end());
        while (!stack.empty()) {
            const BlockId b = stack.back();
            stack.pop_back();
            if (!loop.blocks.insert(b)) {
                continue;
            }
            for (BlockId p : g.preds[b]) {
                if (p != loop.header) {
                    stack.push_back(p);
                }
            }
        }
    }

    // Exits, and nesting. A loop's parent is the innermost other loop that
    // contains its header.
    for (Loop &loop : loops) {
        for (BlockId b : loop.blocks) {
            for (BlockId s : g.succs[b]) {
                if (!loop.blocks.contains(s)) {
                    loop.exits.push_back({b, s});
                }
            }
        }
        std::sort(loop.exits.begin(), loop.exits.end());
        loop.exits.erase(std::unique(loop.exits.begin(), loop.exits.end()),
                         loop.exits.end());

        for (const Loop &other : loops) {
            if (other.header == loop.header ||
                !other.blocks.contains(loop.header)) {
                continue;
            }
            if (loop.parent == NO_BLOCK ||
                forest.find(loop.parent)->blocks.contains(other.header)) {
                loop.parent = other.header;
            }
        }
    }

    // The innermost loop of each block: the loops are painted over their
    // blocks outermost first, so that the deepest to contain a block is the
    // one left standing.
    const auto nesting = [&](const Loop &loop) {
        size_t depth = 0;
        for (const Loop *l = &loop; l != nullptr; l = forest.find(l->parent)) {
            depth++;
        }
        return depth;
    };
    vector<size_t> by_depth(loops.size());
    for (size_t i = 0; i < loops.size(); i++) {
        by_depth[i] = i;
    }
    std::stable_sort(by_depth.begin(), by_depth.end(),
                     [&](size_t a, size_t b) {
                         return nesting(loops[a]) < nesting(loops[b]);
                     });
    for (size_t i : by_depth) {
        for (BlockId b : loops[i].blocks) {
            forest.innermost_of[b] = loops[i].header;
        }
    }

    return forest;
}

//===--------------------------------------------------------------------===//
// Block index
//===--------------------------------------------------------------------===//

BlockIndex compute_block_index(const Graph &g, const DomTree &dom,
                               const LoopForest &loops, const BlockSet &last) {
    // Partial linearization requires a topological order (over the CFG with
    // back edges removed) in which every dominance region and every loop is a
    // contiguous range.
    //
    // Emitting blocks in dominator-tree order gives dominance compactness for
    // free: a node's whole subtree is emitted before any sibling. Among the
    // children that are ready, we pick by RPO so the result is also a valid
    // topological order, and we hold back any child outside the current loop
    // until the loop is exhausted, which gives loop compactness.
    vector<size_t> rpo_number(g.size(), 0);
    vector<bool> reachable(g.size(), false);
    for (size_t i = 0; i < g.rpo.size(); i++) {
        rpo_number[g.rpo[i]] = i;
        reachable[g.rpo[i]] = true;
    }

    // Forward (non-back) edge predecessor counts, so we only emit a block once
    // everything that can reach it without a back edge has been emitted.
    vector<size_t> pending(g.size(), 0);
    for (BlockId b : g.rpo) {
        for (BlockId s : g.succs[b]) {
            if (!reachable[s] || dom.dominates(s, b)) {
                continue; // back edge, or unreachable
            }
            pending[s]++;
        }
    }

    // The ready blocks, the one to emit next on top: deepest in the loop nest
    // first, then the blocks not asked to go last, then by RPO. Staying at
    // maximum depth is what keeps loops contiguous.
    using Key = tuple<size_t, bool, size_t, BlockId>;
    const auto key = [&](BlockId b) {
        return Key{loops.depth(b), !last.contains(b), g.size() - rpo_number[b],
                   b};
    };
    std::priority_queue<Key> ready;
    for (BlockId b : g.rpo) {
        if (pending[b] == 0) {
            ready.push(key(b));
        }
    }

    BlockIndex index;
    index.of.assign(g.size(), 0);
    while (!ready.empty()) {
        const BlockId best = std::get<3>(ready.top());
        ready.pop();
        index.of[best] = index.order.size();
        index.order.push_back(best);

        for (BlockId s : g.succs[best]) {
            if (!reachable[s] || dom.dominates(s, best)) {
                continue; // back edge
            }
            if (--pending[s] == 0) {
                ready.push(key(s));
            }
        }
    }

    internal_assert(index.order.size() == g.rpo.size())
        << "compute_block_index: irreducible control flow ("
        << index.order.size() << " of " << g.rpo.size() << " blocks ordered)";

    return index;
}

std::set<string> called_inside_loops(
    const std::map<string, shared_ptr<Function>> &funcs) {
    // The calls each function makes, and those it makes from inside a loop:
    // a block in a natural loop of the function, or in a parfor's body.
    std::map<string, std::set<string>> calls, calls_in_loops;
    for (const auto &[name, func] : funcs) {
        if (func->blocks.empty()) {
            continue;
        }
        const Cfg cfg(*func);
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest loops = compute_loop_forest(cfg, dom);
        BlockSet in_body(cfg.size());
        for (BlockId b : cfg.rpo) {
            if (const auto *p = std::get_if<Terminator::ParFor>(&cfg[b].terminator.data)) {
                for (BlockId inside : reachable_from(cfg, cfg.id(p->body.name))) {
                    in_body.insert(inside);
                }
            }
        }
        for (BlockId b : cfg.rpo) {
            const Terminator::Jump *call = cfg[b].terminator.callee();
            if (call == nullptr) {
                continue;
            }
            calls[name].insert(call->name);
            if (in_body.contains(b) || loops.innermost(b) != nullptr) {
                calls_in_loops[name].insert(call->name);
            }
        }
    }
    // A function called from a loop, and everything it calls, runs per
    // iteration; to a fixed point over the call graph.
    std::set<string> looped;
    for (const auto &[_, callees] : calls_in_loops) {
        looped.insert(callees.begin(), callees.end());
    }
    for (bool grew = true; grew;) {
        grew = false;
        for (const string &f : std::vector<string>(looped.begin(), looped.end())) {
            const auto it = calls.find(f);
            if (it == calls.end()) {
                continue;
            }
            for (const string &g : it->second) {
                grew = looped.insert(g).second || grew;
            }
        }
    }
    return looped;
}

bool binds_to_gpu(const Function &func) {
    for (const auto &block : func.blocks) {
        const auto *loop =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (loop != nullptr && loop->binding.has_value() &&
            (*loop->binding == Resource::GPUBlock ||
             *loop->binding == Resource::GPUThread)) {
            return true;
        }
    }
    return false;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
