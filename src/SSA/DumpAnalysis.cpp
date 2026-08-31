#include "SSA/DumpAnalysis.h"

#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
#include "SSA/Convert.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Utils.h"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

void print_list(std::ostream &os, const std::vector<std::string> &names) {
    os << "[";
    for (size_t i = 0; i < names.size(); i++) {
        if (i != 0) {
            os << ", ";
        }
        os << names[i];
    }
    os << "]";
}

void print_list(std::ostream &os, const std::set<std::string> &names) {
    print_list(os, std::vector<std::string>(names.begin(), names.end()));
}

void print_edges(std::ostream &os, const std::set<Edge> &edges) {
    os << "[";
    bool first = true;
    for (const auto &[from, to] : edges) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << from << "->" << to;
    }
    os << "]";
}

// Dumps every control-flow analysis for the subgraph rooted at `entry`.
//
// Rooting at a ParFor's body block yields exactly that loop body: Yield has no
// successors, so the traversal stops at the end of the body. This is the same
// region `vectorize()` transforms, and within it the body block is (correctly)
// always executed, which is not true when the ParFor is viewed from the
// enclosing function as a two-way branch.
void dump_region(std::ostream &os, const std::string &indent,
                 const std::string &entry, const AdjacencyMap &all_succs) {
    // Restrict the graph to the blocks of this region.
    const std::set<std::string> region = reachable_from(entry, all_succs);
    AdjacencyMap succs;
    for (const auto &name : region) {
        succs[name];
        const auto it = all_succs.find(name);
        if (it == all_succs.end()) {
            continue;
        }
        for (const auto &s : it->second) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }

    const AdjacencyMap preds = compute_predecessors(succs);
    const std::vector<std::string> rpo = reverse_postorder(entry, succs);
    const DomTree dom = compute_dominator_tree(entry, succs, preds, rpo);
    const DomTree pdom = compute_post_dominator_tree(entry, succs, preds);
    const LoopForest loops = compute_loop_forest(succs, preds, dom, rpo);
    const ControlDependence cdep = compute_control_dependence(succs, pdom);
    const std::map<std::string, size_t> index =
        compute_block_index(entry, succs, dom, loops);

    os << indent << "rpo: ";
    print_list(os, rpo);
    os << "\n";

    // Print in block-index order: this is the order partial linearization
    // visits blocks in, and the order whose compactness it depends on.
    std::vector<std::string> by_index(index.size());
    for (const auto &[name, i] : index) {
        by_index[i] = name;
    }

    for (size_t i = 0; i < by_index.size(); i++) {
        const std::string &name = by_index[i];
        os << indent << "[" << i << "] " << name << "\n";

        os << indent << "  succs: ";
        print_list(os, succs.count(name) ? succs.at(name)
                                         : std::vector<std::string>{});
        os << "\n";
        os << indent << "  preds: ";
        print_list(os, preds.count(name) ? preds.at(name)
                                         : std::vector<std::string>{});
        os << "\n";
        os << indent << "  idom: "
           << (dom.idom.count(name) ? dom.idom.at(name) : "<none>") << "\n";
        os << indent << "  ipdom: "
           << (pdom.idom.count(name) ? pdom.idom.at(name) : "<none>") << "\n";
        os << indent << "  cdep: ";
        print_edges(os, cdep.count(name) ? cdep.at(name) : std::set<Edge>{});
        os << "\n";

        // A block with no control dependences executes whenever the region
        // does, so it can never need an execution mask no matter which
        // branches turn out to be divergent.
        os << indent << "  always-executed: "
           << ((cdep.count(name) && cdep.at(name).empty()) ? "yes" : "no")
           << "\n";
    }

    if (loops.empty()) {
        os << indent << "loops: none\n";
        return;
    }
    os << indent << "loops:\n";
    for (const auto &[header, loop] : loops) {
        os << indent << "  header " << header;
        if (loop.parent.has_value()) {
            os << " (nested in " << *loop.parent << ")";
        }
        os << "\n";
        os << indent << "    latches: ";
        print_list(os, loop.latches);
        os << "\n";
        os << indent << "    blocks: ";
        print_list(os, loop.blocks);
        os << "\n";
        os << indent << "    exits: ";
        print_edges(os, loop.exits);
        os << "\n";
    }
}

// Dumps the uniform/varying classification of the region rooted at `entry`,
// per block, so it can be read against the SSA dump above it.
void dump_divergence(std::ostream &os, const std::string &indent,
                     const ssa::Function &func, const std::string &entry,
                     const std::string &index) {
    const Divergence div = analyze_divergence(func, entry, {index});
    const BlockMap blocks = make_block_map(func);

    for (const auto &name :
         reverse_postorder(entry, compute_successors(func))) {
        const Block &block = *blocks.at(name);
        os << indent << name << ": "
           << (div.masked.count(name) ? "masked" : "unmasked");
        if (div.branches.count(name)) {
            os << ", divergent branch";
        }
        os << "\n";

        std::vector<std::string> varying;
        for (const auto &arg : block.args) {
            if (div.args.count({name, arg.name})) {
                varying.push_back(arg.name);
            }
        }
        for (const auto &instr : block.instrs) {
            if (!div.instrs.count(instr.get())) {
                continue;
            }
            // A side-effecting instruction has no name; print it whole, since
            // a varying store is what will become a scatter.
            if (instr->name.empty()) {
                std::ostringstream text;
                instr->dump(text);
                varying.push_back(text.str());
            } else {
                varying.push_back(instr->name);
            }
        }
        os << indent << "  varying: ";
        print_list(os, varying);
        os << "\n";
    }
}

void dump(std::ostream &os, const std::string &fname, ssa::Function &func) {
    internal_assert(!func.blocks.empty()) << fname << " has no blocks";

    // Promote first: this is what `vectorize()` does, and the divergence of a
    // mutable local is only visible once it is a value rather than memory.
    os << "promoted " << promote_allocas(func, func.blocks[0]->name)
       << " allocation(s) in " << fname << "\n";

    const AdjacencyMap succs = compute_successors(func);

    os << "function " << fname << ":\n";
    func.dump(os);
    dump_region(os, "  ", func.blocks[0]->name, succs);

    // Then each ParFor body region, which is the unit `vectorize()` works on.
    for (const auto &block : func.blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor == nullptr) {
            continue;
        }
        os << "  parfor " << parfor->index << " body region ("
           << parfor->body.name << "):\n";
        dump_region(os, "    ", parfor->body.name, succs);
        os << "    divergence (varying: " << parfor->index << "):\n";
        dump_divergence(os, "      ", func, parfor->body.name, parfor->index);
    }
}

} // namespace

ir::Program DumpSSAAnalysis::run(ir::Program program,
                                 const CompilerOptions &options) const {
    FuncMap fmap;
    for (const auto &[name, func] : program.funcs) {
        fmap[name] = build(func);
    }

    // Loopify before dumping, since it is what puts a back edge in a function
    // at all: without it there is no loop for these analyses to report, and
    // the loop forest of a traversal is exactly what a reader comes here for.
    // Nothing else in the schedule changes the CFG at this level.
    if (const auto it = program.schedules.find(ir::Target::Host);
        it != program.schedules.end()) {
        for (const auto &[name, ts] : it->second.func_transforms) {
            if (!fmap.contains(name)) {
                continue;
            }
            for (const auto &t : ts) {
                if (const auto *l = std::get_if<ir::Loopify>(&t)) {
                    int size = 0;
                    if (l->queue_size.has_value()) {
                        const auto n =
                            get_constant_value<int64_t>(*l->queue_size);
                        internal_assert(n.has_value() && *n > 0)
                            << "loopify(" << *l->queue_size << ") on " << name
                            << " needs a constant, positive stack depth";
                        size = int(*n);
                    }
                    loopify(fmap, name, size);
                }
            }
        }
    }

    // std::map iteration is ordered, so the dump is deterministic.
    for (const auto &[name, func] : fmap) {
        dump(std::cout, name, *func);
    }
    return program;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
