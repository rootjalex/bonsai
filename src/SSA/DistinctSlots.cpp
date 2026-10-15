#include "SSA/DistinctSlots.h"
#include "SSA/Analysis.h"

#include "Error.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using FuncMap = std::map<std::string, std::shared_ptr<Function>>;

// More pushes than one activity may make: a loop around one, a recursion
// through one, or anything that could not be counted.
constexpr size_t kMany = std::numeric_limits<size_t>::max();

size_t plus(size_t a, size_t b) {
    return a == kMany || b == kMany ? kMany : a + b;
}

size_t times(size_t a, size_t n) {
    if (a == 0 || n == 0) {
        return 0;
    }
    return a == kMany || n > 1 ? kMany : a;
}

// The queue a Push instruction pushes onto, by the name its type carries
// (`Queue_<name>`, SSA/Defer.cpp), or empty when the operand is not one.
std::string pushed_queue(const Instruction &push) {
    if (push.operands.empty() || !push.operands[0]) {
        return "";
    }
    const Type type = push.operands[0]->get_type();
    const auto *ptr = type.as<Ptr_t>();
    const auto *queue = ptr != nullptr ? ptr->etype.as<Struct_t>() : nullptr;
    const std::string prefix = "Queue_";
    if (queue == nullptr || queue->name.rfind(prefix, 0) != 0) {
        return "";
    }
    return queue->name.substr(prefix.size());
}

struct Counter {
    const FuncMap &funcs;
    const Function &owner;
    const std::string &queue;
    const std::string &callee;
    std::string why;

    // Per function: the greatest number of pushes onto the queue one run of
    // it makes, once counted. The functions being counted are the call
    // chain's, and a call back into one of them is a recursion, answered
    // with nothing for the moment (`hit`) and settled when that function's
    // own count is done: a recursion that passes a push pushes again every
    // time round, so any push at all through it is more than one, and a
    // recursion that passes none -- a tree traversal -- pushes nothing. A
    // count taken while a function further up the chain was still being
    // counted leaned on its unsettled answer and is not kept; that
    // function's count, which is kept, has it right.
    std::map<std::string, size_t> counted;
    std::set<std::string> counting;
    std::set<std::string> hit;

    void note(const std::string &reason) {
        if (why.empty()) {
            why = reason;
        }
    }

    // What one Push instruction adds: the queue itself, one entry; a queue
    // of the owner drained later in the round, what its drained function
    // makes of the entry (the greatest over the drain's callees, one per arm
    // of a split); a queue drained in rounds, nothing, since the entry runs
    // in the next round; a queue of someone else's, more than can be said.
    size_t push_weight(const Instruction &push) {
        const std::string target = pushed_queue(push);
        if (target.empty()) {
            note("a push onto something whose queue could not be named");
            return kMany;
        }
        if (target == queue) {
            return 1;
        }
        const auto drained = owner.queue_sizes.find(target);
        if (drained == owner.queue_sizes.end()) {
            note("a push onto " + target + ", which " +
                 owner.blocks.front()->name + " does not drain");
            return kMany;
        }
        if (drained->second.rounds) {
            return 0;
        }
        if (drained->second.callees.empty()) {
            note("a push onto " + target + ", whose drain calls nothing known");
            return kMany;
        }
        size_t most = 0;
        for (const std::string &c : drained->second.callees) {
            most = std::max(most, function_count(c));
        }
        return most;
    }

    // What a block's terminator adds: a call of the function being deferred
    // is the push the deferral makes of it; a call of anything else is what
    // that function makes, once per call of a run; a parallel loop's body
    // is every iteration's, so anything it pushes is more than one.
    size_t terminator_weight(const Function &f, const Block &block) {
        const Terminator &t = block.terminator;
        if (const auto *p = std::get_if<Terminator::ParFor>(&t.data)) {
            const size_t inside = region_count(f, p->body.name);
            if (inside > 0) {
                note("the loop over " + p->index + " in " +
                     f.blocks.front()->name + " pushes in every iteration");
                return kMany;
            }
            return 0;
        }
        const Terminator::Jump *called = t.callee();
        if (called == nullptr) {
            return 0;
        }
        size_t calls = 1;
        if (const auto *run = std::get_if<Terminator::MultiCall>(&t.data)) {
            calls = run->varying.size();
        }
        if (!callee.empty() && called->name == callee) {
            // A run of calls of it would be a run of pushes.
            return calls > 1 ? kMany : 1;
        }
        return times(function_count(called->name), calls);
    }

    // The greatest number of pushes along any path through the region of
    // `f` entered at `entry`: a longest path over the region's graph with
    // its back edges dropped, which is a DAG for the reducible graphs the
    // SSA has. A block inside a loop that pushes at all pushes more than
    // once, since the loop comes round to it for the same slot.
    size_t region_count(const Function &f, const std::string &entry) {
        const Cfg cfg(f, entry);
        if (cfg.rpo.empty()) {
            return 0;
        }
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest loops = compute_loop_forest(cfg, dom);
        std::set<std::pair<BlockId, BlockId>> back_edges;
        for (const Loop &loop : loops.loops()) {
            for (BlockId latch : loop.latches) {
                back_edges.insert({latch, loop.header});
            }
        }
        std::vector<size_t> weight(cfg.size(), 0), best(cfg.size(), 0);
        for (BlockId b : cfg.rpo) {
            const Block &block = *cfg.block(b);
            size_t w = 0;
            for (const auto &instr : block.instrs) {
                if (instr->op == Instruction::Op::Push) {
                    w = plus(w, push_weight(*instr));
                }
            }
            w = plus(w, terminator_weight(f, block));
            if (w > 0 && loops.innermost(b) != nullptr) {
                note("a push inside a loop of " + f.blocks.front()->name +
                     " (" + block.name + "), made again for the same slot");
                w = kMany;
            }
            weight[b] = w;
        }
        // Postorder: every successor but a back edge's target is done first.
        for (auto it = cfg.rpo.rbegin(); it != cfg.rpo.rend(); ++it) {
            const BlockId b = *it;
            const Block &block = *cfg.block(b);
            size_t onward = 0;
            const auto *p = std::get_if<Terminator::ParFor>(&block.terminator.data);
            for (BlockId s : cfg.succs[b]) {
                if (back_edges.contains({b, s})) {
                    continue;
                }
                // A parallel loop's body is counted by the terminator's
                // weight, not as a path onward; the path goes on after it.
                if (p != nullptr && cfg.name(s) == p->body.name) {
                    continue;
                }
                onward = std::max(onward, best[s]);
            }
            best[b] = plus(weight[b], onward);
        }
        return best[cfg.id(entry)];
    }

    size_t function_count(const std::string &g) {
        const auto done = counted.find(g);
        if (done != counted.end()) {
            return done->second;
        }
        const auto f = funcs.find(g);
        if (f == funcs.end() || f->second->blocks.empty()) {
            return 0; // an extern: pushes nothing of ours
        }
        if (!counting.insert(g).second) {
            hit.insert(g);
            return 0;
        }
        size_t n = region_count(*f->second, f->second->blocks.front()->name);
        counting.erase(g);
        if (hit.erase(g) > 0 && n > 0) {
            note("a recursion through " + g + ", which pushes");
            n = kMany;
        }
        if (hit.empty()) {
            counted[g] = n;
        }
        return n;
    }
};

} // namespace

bool pushes_at_most_once(const FuncMap &funcs, const Function &owner,
                         const std::string &queue, const std::string &callee,
                         std::string *why) {
    Counter counter{funcs, owner, queue, callee};
    // The roots: the owner's innermost parallel loops, whose iterations are
    // the activities -- a sample of the sample loop, an entry of a drain.
    bool once = true;
    for (const auto &block : owner.blocks) {
        const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (p == nullptr) {
            continue;
        }
        const Cfg body(owner, p->body.name);
        bool innermost = true;
        for (BlockId b : body.rpo) {
            if (std::holds_alternative<Terminator::ParFor>(
                    body.block(b)->terminator.data)) {
                innermost = false;
                break;
            }
        }
        if (!innermost) {
            continue;
        }
        const size_t n = counter.region_count(owner, p->body.name);
        if (n > 1) {
            once = false;
            if (why != nullptr) {
                std::ostringstream os;
                os << "an iteration of the loop over " << p->index << " may push ";
                if (n == kMany) {
                    os << "more than once" << (counter.why.empty() ? "" : ": " + counter.why);
                } else {
                    os << n << " times";
                }
                *why = os.str();
            }
            break;
        }
    }
    return once;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
