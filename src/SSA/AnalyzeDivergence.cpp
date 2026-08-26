#include "SSA/AnalyzeDivergence.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <algorithm>
#include <map>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

bool Divergence::is_varying(const string &block, const Value &v) const {
    return std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &i) {
                return instrs.count(i.get()) > 0;
            },
            [&](const Constant &) { return false; },
            [&](const Argument &a) { return args.count({block, a.name}) > 0; },
        },
        v.data);
}

namespace {

// Values are identified by name: the SSA builder threads a definition through
// block arguments under its own name, so two references to the same name --
// in whatever blocks -- are two references to the same definition.
string value_key(const Value &v) {
    return std::visit(
        overloads{
            [](const shared_ptr<Instruction> &i) { return "%" + i->name; },
            [](const Constant &c) {
                std::ostringstream os;
                c.dump(os);
                return os.str();
            },
            [](const Argument &a) { return "%" + a.name; },
        },
        v.data);
}

// One edge's worth of argument passing: the values `from`'s terminator hands
// to `target`, and the index of the target argument the first of them binds
// to.
struct ArgumentFlow {
    string from;
    size_t first_arg = 0;
    const vector<shared_ptr<Value>> *values = nullptr;
};

// The argument flows out of `block` that stay inside the region, keyed by
// target block. The callee of a Call is not one of them: it belongs to
// another function's CFG, and its arguments are bound by that function's own
// analysis.
void collect_argument_flows(const Block &block,
                            map<string, vector<ArgumentFlow>> &into) {
    auto add = [&](const Terminator::Jump &j, size_t first_arg) {
        into[j.name].push_back({block.name, first_arg, &j.args});
    };

    std::visit(
        overloads{
            [&](const std::monostate &) {
                internal_error << "Block " << block.name
                               << " has no terminator.";
            },
            [&](const Terminator::Jump &j) { add(j, 0); },
            [&](const Terminator::Dispatch &d) {
                for (const auto &target : d.targets) {
                    add(target, 0);
                }
            },
            [&](const Terminator::Return &) {},
            [&](const Terminator::ParFor &) {
                internal_error << "TODO: nested ParFor in " << block.name
                               << " during divergence analysis.";
            },
            [&](const Terminator::Yield &) {},
            [&](const Terminator::Call &c) {
                // A returned value is prepended to the continuation's
                // arguments, so the explicitly passed ones start at 1.
                add(c.cont, c.drop ? 0 : 1);
            },
        },
        block.terminator.data);
}

template <typename T>
const T &lookup_or(const map<string, T> &m, const string &key, const T &fallback) {
    const auto it = m.find(key);
    return it == m.end() ? fallback : it->second;
}

} // namespace

Divergence analyze_divergence(const Function &func, const string &entry,
                              const set<string> &varying_seeds) {
    const BlockMap all_blocks = make_block_map(func);
    const AdjacencyMap all_succs = compute_successors(func);

    // Restrict the CFG to the region: a ParFor body ends at its Yield, and a
    // Call's callee is outside it, so this is exactly the set of blocks whose
    // lanes execute together.
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
    const DomTree pdom = compute_post_dominator_tree(entry, succs, preds);
    const ControlDependence cdep = compute_control_dependence(succs, pdom);

    map<string, vector<ArgumentFlow>> flows;
    for (const string &name : region) {
        collect_argument_flows(*all_blocks.at(name), flows);
    }

    Divergence result;
    for (const string &seed : varying_seeds) {
        result.args.insert({entry, seed});
    }

    const set<Edge> no_edges;
    const vector<string> no_names;
    const vector<ArgumentFlow> no_flows;

    // Monotone: every rule only ever adds to the sets, so this terminates.
    bool changed = true;
    while (changed) {
        changed = false;
        auto mark = [&](auto &s, const auto &key) {
            changed |= s.insert(key).second;
        };

        for (const string &name : rpo) {
            const Block &block = *all_blocks.at(name);

            // A block executes with a partial mask if it is control dependent
            // on a divergent branch, or on a branch in a block that is itself
            // only partially executed: a block's mask is always a subset of
            // the masks of the blocks it is control dependent on.
            for (const auto &[from, to] : lookup_or(cdep, name, no_edges)) {
                if (result.branches.count(from) || result.masked.count(from)) {
                    mark(result.masked, name);
                }
            }

            // At a join whose predecessors are only partially executed, lanes
            // arrive along different edges, so an argument that is not passed
            // the same definition along every edge disagrees between lanes.
            const vector<string> &block_preds = lookup_or(preds, name, no_names);
            const bool divergent_join =
                block_preds.size() > 1 &&
                std::any_of(
                    block_preds.begin(), block_preds.end(),
                    [&](const string &p) { return result.masked.count(p) > 0; });

            const vector<ArgumentFlow> &incoming =
                lookup_or(flows, name, no_flows);
            for (size_t j = 0; j < block.args.size(); j++) {
                std::optional<string> common_key;
                bool varying = false;
                for (const ArgumentFlow &flow : incoming) {
                    if (j < flow.first_arg ||
                        j - flow.first_arg >= flow.values->size()) {
                        continue;
                    }
                    const Value &v = *(*flow.values)[j - flow.first_arg];
                    varying |= result.is_varying(flow.from, v);
                    const string key = value_key(v);
                    varying |= divergent_join && common_key.has_value() &&
                               *common_key != key;
                    common_key = key;
                }
                if (varying) {
                    mark(result.args, std::pair{name, block.args[j].name});
                }
            }

            for (const shared_ptr<Instruction> &instr : block.instrs) {
                const bool varying =
                    std::any_of(instr->operands.begin(), instr->operands.end(),
                                [&](const shared_ptr<Value> &v) {
                                    return result.is_varying(name, *v);
                                });
                if (varying) {
                    mark(result.instrs, instr.get());
                }
            }

            if (const auto *d =
                    std::get_if<Terminator::Dispatch>(&block.terminator.data)) {
                if (result.is_varying(name, *d->cond)) {
                    mark(result.branches, name);
                }
            }

            if (const auto *c =
                    std::get_if<Terminator::Call>(&block.terminator.data)) {
                // Until divergence is solved across functions, a call's
                // result is varying whenever an argument is: a call on
                // uniform arguments cannot produce a varying result, but a
                // call on varying ones may produce anything.
                const bool varying_call =
                    std::any_of(c->call.args.begin(), c->call.args.end(),
                                [&](const shared_ptr<Value> &v) {
                                    return result.is_varying(name, *v);
                                });
                if (!c->drop && varying_call) {
                    const Block &cont = *all_blocks.at(c->cont.name);
                    internal_assert(!cont.args.empty())
                        << "Call continuation " << cont.name
                        << " takes no result argument.";
                    mark(result.args, std::pair{cont.name, cont.args[0].name});
                }
            }
        }
    }

    return result;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
